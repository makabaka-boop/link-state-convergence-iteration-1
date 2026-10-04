"""逐 tick 的链路状态网络模拟器。

每个 tick 依次执行：
  1. 处理本 tick 的外部事件（涨价 / 切断 / 恢复 / 注入过期通告 /
     设置丢弃概率窗口）；
  2. 周期性补发：各路由器把当前 LSDB 的全部已知通告发给每个
     活动邻居（消息可能丢失时收敛的最终保障）；
  3. 传输层投递所有到期消息，接收端决定 new / dup / stale，
     新通告按水平分割继续向邻居转发；
  4. 生成快照：逐消息日志、各节点 LSDB 与路由表。

网络真值（活动拓扑）只被传输层与事件层使用；路由器计算路由时
只能访问自己的 LSDB。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .lsa import LSA
from .router import Router
from .transport import Transport


class Network:
    def __init__(
        self,
        *,
        refresh_interval: int = 5,
        seed: int = 1234,
        base_delay: int = 1,
        jitter: int = 1,
        duplicate_prob: float = 0.0,
        max_age: int = 12,
    ):
        self.routers: Dict[str, Router] = {}
        # 无向链路：frozenset({a,b}) -> {"cost": int, "active": bool}
        self._links: Dict[frozenset, dict] = {}
        self.refresh_interval = refresh_interval
        self.max_age = max_age
        self.transport = Transport(
            self,
            seed=seed,
            base_delay=base_delay,
            jitter=jitter,
            duplicate_prob=duplicate_prob,
        )
        self.tick = 0
        self._boots_done = False
        self._records: List[dict] = []
        # 已排队但尚未执行的事件（多次调用 run 时持续保留）
        self._pending_events: List[dict] = []

    def _require_router(self, rid: str) -> None:
        if rid not in self.routers:
            raise ValueError(f"未知路由器 {rid}")

    # ---- 拓扑搭建（boot 之前） ----------------------------------------

    def add_router(self, router_id: str) -> None:
        if router_id in self.routers:
            raise ValueError(f"路由器 {router_id} 已存在")
        self.routers[router_id] = Router(router_id, max_age=self.max_age)

    def add_link(self, a: str, b: str, cost: int) -> None:
        if a == b:
            raise ValueError("不允许自环")
        if cost <= 0:
            raise ValueError("链路费用必须为正整数")
        key = frozenset({a, b})
        if key in self._links:
            raise ValueError(f"链路 {a}-{b} 已存在")
        self._links[key] = {"cost": cost, "active": True}
        self.routers[a].set_local_link(b, cost)
        self.routers[b].set_local_link(a, cost)

    def _link_key(self, a: str, b: str) -> frozenset:
        return frozenset({a, b})

    def link_active(self, a: str, b: str) -> bool:
        rec = self._links.get(self._link_key(a, b))
        return rec is not None and rec["active"]

    def active_neighbors(self, rid: str) -> List[str]:
        return [
            nb
            for nb in self.routers[rid].links
            if self.link_active(rid, nb)
        ]

    def truth_graph(self) -> Dict[str, Dict[str, int]]:
        """当前活动拓扑真值（仅供测试核对，路由器无权访问）。"""
        g: Dict[str, Dict[str, int]] = {rid: {} for rid in self.routers}
        for key, rec in self._links.items():
            if not rec["active"]:
                continue
            a, b = tuple(key)
            g[a][b] = rec["cost"]
            g[b][a] = rec["cost"]
        return g

    def truth_drained(self) -> set:
        """当前已宣告过境排空的节点集合（仅供测试核对；路由器算路
        时只能依据各自 LSDB 中随 LSA 泛洪来的 transit 位，不读本集
        合）。"""
        return {rid for rid, r in self.routers.items() if not r.transit}

    # ---- 事件：过境排空 / 撤销 ----------------------------------------

    def drain_router(self, tick: int, rid: str) -> str:
        """让 ``rid`` 在本地 LSA 中以递增序号宣告过境排空。

        非法调用（未知节点、重复排空）在 Router 内修改任何状态前
        抛出，因此通告不会签发、消息不会发送、轨迹不会记录。
        """
        self._require_router(rid)
        r = self.routers[rid]
        if not r.transit:
            # 与 Router.drain 的守卫保持一致，保证网络层也无副作用
            raise ValueError(f"路由器 {rid} 已处于排空状态，不能重复排空")
        lsa = r.drain(tick)
        for nb in self.active_neighbors(rid):
            self.transport.send(tick, rid, nb, lsa)
        return f"drain {rid}（过境排空：可达为目的地，但禁止中转）"

    def undrain_router(self, tick: int, rid: str) -> str:
        """撤销 ``rid`` 的过境排空，恢复其中转资格。非法（未排空
        却撤销）时同样无任何副作用。"""
        self._require_router(rid)
        r = self.routers[rid]
        if r.transit:
            raise ValueError(f"路由器 {rid} 当前未排空，不能撤销排空")
        lsa = r.undrain(tick)
        for nb in self.active_neighbors(rid):
            self.transport.send(tick, rid, nb, lsa)
        return f"undrain {rid}（撤销排空：恢复过境资格）"

    # ---- 事件：链路变化与通告注入 -------------------------------------

    def cut_set(self, tick: int, links: List[tuple]) -> str:
        """一次性切断一组边：先改完全部链路状态，再让每个受影响
        节点只签发一次新 LSA，避免“切了一半”的中间状态被泛洪。"""
        affected = set()
        descs = []
        for a, b in links:
            key = self._link_key(a, b)
            if not self._links[key]["active"]:
                continue
            self._links[key]["active"] = False
            self.routers[a].remove_local_link(b)
            self.routers[b].remove_local_link(a)
            affected.update((a, b))
            descs.append(f"cut {a}-{b}（进入分区/不可达）")
        for rid in sorted(affected):
            self._originate(tick, rid)
        return "；".join(descs)

    def restore_set(
        self, tick: int, links: List[tuple], costs: Optional[Dict[tuple, int]] = None
    ) -> str:
        costs = costs or {}
        affected = set()
        descs = []
        for a, b in links:
            key = self._link_key(a, b)
            c = costs.get((a, b), costs.get((b, a), self._links[key]["cost"]))
            self._links[key]["cost"] = c
            self._links[key]["active"] = True
            self.routers[a].set_local_link(b, c)
            self.routers[b].set_local_link(a, c)
            affected.update((a, b))
            descs.append(f"restore {a}-{b}（费用 {c}，分区恢复）")
        for rid in sorted(affected):
            self._originate(tick, rid)
        return "；".join(descs)

    def change_cost(self, tick: int, a: str, b: str, cost: int) -> str:
        if cost <= 0:
            raise ValueError("链路费用必须为正整数")
        key = self._link_key(a, b)
        old = self._links[key]["cost"]
        self._links[key]["cost"] = cost
        self.routers[a].set_local_link(b, cost)
        self.routers[b].set_local_link(a, cost)
        desc = f"link-cost {a}-{b}: {old} -> {cost}"
        # 两端各自签发递增序号的新通告并立即泛洪
        self._originate(tick, a)
        self._originate(tick, b)
        return desc

    def cut_link(self, tick: int, a: str, b: str) -> str:
        return self.cut_set(tick, [(a, b)])

    def restore_link(self, tick: int, a: str, b: str, cost: Optional[int] = None) -> str:
        costs = {(a, b): cost} if cost is not None else None
        return self.restore_set(tick, [(a, b)], costs)

    def inject_lsa(
        self,
        tick: int,
        origin: str,
        seq: int,
        links: Dict[str, int],
        *,
        transit: bool = True,
        src: Optional[str] = None,
        deliver_now: bool = False,
    ) -> str:
        """注入一条“外部到达”的通告（测试过期通告用）。

        默认经传输层送达（可被延迟/丢弃）；``deliver_now`` 时
        立即交给目标路由器裁决。``transit`` 可伪造过境资格位，
        用于验证旧序号/同序号异内容（含排空位被篡改）的通告
        不能恢复旧资格。
        """
        lsa = LSA(origin, seq, dict(links), transit)
        target = src or origin
        if deliver_now:
            result = self.routers[target].receive(lsa, tick)
            self._log_message(
                tick,
                "deliver",
                {
                    "id": -1,
                    "src": origin,
                    "dst": target,
                    "lsa": lsa,
                },
                result=result,
                injected=True,
            )
            return f"直接注入 {lsa.describe()} 到 {target} -> {result}"
        self.transport.send(tick, origin, target, lsa)
        return f"注入过期通告 {lsa.describe()} 到 {target}"

    # ---- 签发与泛洪 ---------------------------------------------------

    def _originate(self, tick: int, rid: str) -> LSA:
        lsa = self.routers[rid].originate(tick)
        for nb in self.active_neighbors(rid):
            self.transport.send(tick, rid, nb, lsa)
        return lsa

    def deliver(self, tick: int, msg: dict) -> None:
        """传输层把到期消息交给接收端路由器。"""
        r = self.routers[msg["dst"]]
        lsa = msg["lsa"]
        if lsa.origin == r.id:
            # 自己签发的通告绕了一圈回来：只能是旧序号或完全相同，
            # 照常走裁决逻辑（旧序号/同序号异内容拒绝），不转发。
            if lsa.seq < r.current_seq:
                result = "stale"
            elif (
                lsa.seq == r.current_seq
                and lsa.content_key() == r.lsdb[r.id].lsa.content_key()
            ):
                result = "dup"
            else:
                result = "stale"
        else:
            result = r.receive(lsa, tick, neighbor=msg["src"])
        self._log_message(tick, "deliver", msg, result=result)

        if result == "new":
            # 水平分割：只不向入边回送；向其余所有活动邻居继续泛洪。
            for nb in self.active_neighbors(r.id):
                if nb == msg["src"]:
                    continue
                self.transport.send(tick, r.id, nb, lsa)

    # ---- 消息日志 -----------------------------------------------------

    def _log_message(self, tick: int, kind: str, msg: dict, **extra) -> None:
        rec = {
            "kind": kind,
            "id": msg["id"],
            "src": msg["src"],
            "dst": msg["dst"],
            "lsa": msg["lsa"].to_dict(),
        }
        if msg.get("copy"):
            rec["copy"] = True
        rec.update(extra)
        self._records[-1]["messages"].append(rec)

    # ---- 快照 ---------------------------------------------------------

    def _snapshot(self) -> dict:
        routers = {}
        for rid in sorted(self.routers):
            r = self.routers[rid]
            routers[rid] = {
                "lsdb": [lsa.to_dict() for lsa in r.known_lsas()],
                "routes": [
                    route.to_dict()
                    for _, route in sorted(r.routing_table(self.tick).items())
                ],
            }
        return {"tick": self.tick, "routers": routers}

    def routing_snapshot(self) -> Dict[str, Dict[str, tuple]]:
        """供测试：{router: {dest: (next_hop, cost)}}（含不可达项）。"""
        out: Dict[str, Dict[str, tuple]] = {}
        for rid, r in self.routers.items():
            out[rid] = {
                d: (rt.next_hop, rt.cost)
                for d, rt in r.routing_table(self.tick).items()
            }
        return out

    # ---- 主循环 -------------------------------------------------------

    def _process_event(self, ev: dict) -> str:
        t = ev["tick"]
        typ = ev["type"]
        if typ == "cost":
            return self.change_cost(t, ev["a"], ev["b"], ev["cost"])
        if typ == "cut":
            return self.cut_link(t, ev["a"], ev["b"])
        if typ == "restore":
            return self.restore_link(t, ev["a"], ev["b"], ev.get("cost"))
        if typ == "cut_set":
            return self.cut_set(t, ev["links"])
        if typ == "restore_set":
            return self.restore_set(t, ev["links"])
        if typ == "drain":
            return self.drain_router(t, ev["router"])
        if typ == "undrain":
            return self.undrain_router(t, ev["router"])
        if typ == "inject":
            return self.inject_lsa(
                t,
                ev["origin"],
                ev["seq"],
                ev.get("links", {}),
                transit=ev.get("transit", True),
                src=ev.get("src"),
                deliver_now=ev.get("deliver_now", False),
            )
        if typ == "drop_window":
            self.transport.add_drop_window(
                ev["start"], ev["end"], ev["prob"]
            )
            return (
                f"丢弃窗口 [{ev['start']},{ev['end']}) "
                f"p={ev['prob']}"
            )
        raise ValueError(f"未知事件类型: {typ}")

    def _periodic_refresh(self) -> None:
        for rid in sorted(self.routers):
            r = self.routers[rid]
            for nb in sorted(self.active_neighbors(rid)):
                for lsa in r.known_lsas():
                    self._refresh_send(rid, nb, lsa)

    def _refresh_send(self, rid: str, nb: str, lsa: LSA) -> None:
        # 与普通发送一致，仅在日志中标注 refresh
        self.transport.msg_counter += 1
        msg = {
            "id": self.transport.msg_counter,
            "sent_tick": self.tick,
            "src": rid,
            "dst": nb,
            "lsa": lsa,
            "refresh": True,
        }
        reason = self.transport._pre_send_check(msg, self.tick)
        if reason is not None:
            self._log_message(self.tick, "drop", msg, reason=reason, refresh=True)
            return
        self.transport._enqueue(self.tick, msg)

    def schedule(self, event: dict) -> None:
        """追加一个事件（tick 为绝对时刻，可在 run 之前或之间追加）。"""
        self._pending_events.append(event)

    def run(self, until_tick: int, events: Optional[List[dict]] = None) -> List[dict]:
        """从当前 tick 继续模拟到 ``until_tick``（含该 tick）。

        新传入的事件追加到持久队列；未到执行时刻的事件在多次
        ``run`` 调用之间保留。
        """
        if events:
            self._pending_events.extend(events)
        self._pending_events.sort(key=lambda e: e["tick"])

        if not self._boots_done:
            # tick 0：所有路由器签发序号 1 的初始通告并泛洪
            self._records.append({"tick": 0, "events": [], "messages": []})
            for rid in sorted(self.routers):
                self._originate(0, rid)
            self._boots_done = True
            self.tick = 0
            # boot tick 上可能已排队 tick0 事件
            try:
                while self._pending_events and self._pending_events[0]["tick"] == 0:
                    ev = self._pending_events.pop(0)
                    self._records[-1]["events"].append(self._process_event(ev))
            except Exception:
                # 非法事件不得在轨迹中留下“执行了一半”的 tick：
                # 此时该 tick 尚未投递任何消息，整条记录回滚。
                self._records.pop()
                raise
            self.transport.deliver_due(0)
            self._records[-1].update(self._snapshot())
            first_tick = 1
        else:
            first_tick = self.tick + 1

        for t in range(first_tick, until_tick + 1):
            self.tick = t
            self._records.append({"tick": t, "events": [], "messages": []})

            # 1. 外部事件（含 boot tick 上排队的 tick0 事件）
            try:
                while self._pending_events and self._pending_events[0]["tick"] == t:
                    ev = self._pending_events.pop(0)
                    desc = self._process_event(ev)
                    self._records[-1]["events"].append(desc)
            except Exception:
                # 事件非法时：它在产生任何通告/消息之前即已拒绝，
                # 回滚本 tick 尚为空的记录，不留部分轨迹。
                self._records.pop()
                raise

            # 2. 周期性补发（boot 那 tick 已全量首发，从下一周期开始）
            if t > 0 and t % self.refresh_interval == 0:
                self._periodic_refresh()

            # 3. 投递到期消息
            self.transport.deliver_due(t)

            # 4. 快照
            self._records[-1].update(self._snapshot())

        return self._records

    @property
    def records(self) -> List[dict]:
        return self._records
