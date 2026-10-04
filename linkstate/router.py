"""路由器：只持有本地链路状态库 (LSDB)，不读取任何全局真值。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .lsa import LSA
from .dijkstra import compute_routes, Route


@dataclass
class LSAEntry:
    """库中的一条状态。

    * ``lsa`` —— 最新 LSA（旧序号不得覆盖，由 :meth:`Router.receive`
      严格保证）；
    * ``heard_at`` —— ``邻居 -> 最近一次从该邻居收到此 LSA（相同
      或更新版本）的 tick``。周期补发会持续刷新它。

    老化在 :meth:`Router._fresh_set` 中沿“逐跳心跳边”传播：每条
    库中声明的边 X-Y，只有当我近期从 X 方向听到过 X 的 LSA、且
    从 Y 方向听到过 Y 的 LSA 时才算心跳有效；从自己出发沿有效
    心跳边做闭包，闭包外即不可达。分区断链后，断点任一端的
    新 LSA 已删除该边，且对端心跳无法跨分区送达，闭包必然在
    断口处停止（对应真实链路状态协议的 LSA aging/MaxAge）。
    """

    lsa: LSA
    heard_at: Dict[str, int] = field(default_factory=dict)


class Router:
    def __init__(self, router_id: str, max_age: int = 12):
        self.id = router_id
        # 本地直连链路的当前配置（邻居 -> 费用），仅用于签发自身 LSA
        self._links: Dict[str, int] = {}
        self._seq = 0
        # 链路状态库：origin -> LSAEntry
        self.lsdb: Dict[str, LSAEntry] = {}
        self.max_age = max_age

    # ---- 本地链路配置 -------------------------------------------------

    def set_local_link(self, neighbor: str, cost: int) -> None:
        if cost <= 0:
            raise ValueError("链路费用必须为正整数")
        self._links[neighbor] = cost

    def remove_local_link(self, neighbor: str) -> None:
        self._links.pop(neighbor, None)

    @property
    def links(self) -> Dict[str, int]:
        return dict(self._links)

    # ---- 签发通告 -----------------------------------------------------

    def originate(self, tick: int = 0) -> LSA:
        """链路变化时调用：序号严格递增，并通告当前链路的完整快照。"""
        self._seq += 1
        lsa = LSA(self.id, self._seq, dict(self._links))
        self.lsdb[self.id] = LSAEntry(lsa, heard_at={self.id: tick})
        return lsa

    @property
    def current_seq(self) -> int:
        return self._seq

    # ---- 接收通告 -----------------------------------------------------

    def receive(self, lsa: LSA, tick: int = 0, neighbor: str = "") -> str:
        """处理从 ``neighbor`` 到达的通告，返回处置结果：

        "new"   —— 首次见到或序号更新：记录，并记下从该邻居听到；
        "stale" —— 旧序号或同序号内容冲突：拒绝，绝不覆盖新状态，
                   也不刷新任何听到时间；
        "dup"   —— 完全相同：只刷新从该邻居听到的时间。
        """
        old = self.lsdb.get(lsa.origin)
        if old is not None:
            if lsa.seq < old.lsa.seq:
                return "stale"
            if lsa.seq == old.lsa.seq:
                if lsa.links != old.lsa.links:
                    return "stale"
                old.heard_at[neighbor] = tick
                return "dup"
        self.lsdb[lsa.origin] = LSAEntry(lsa, heard_at={neighbor: tick})
        return "new"

    def known_lsas(self) -> List[LSA]:
        """当前已知的全部状态，供周期性向邻居完整补发。"""
        return [self.lsdb[k].lsa for k in sorted(self.lsdb)]

    # ---- 老化判定 -----------------------------------------------------

    def _heard_via_fresh(
        self, origin: str, fresh: Set[str], tick: int
    ) -> bool:
        """origin 是否有“可信的近期心跳证据”。

        heard_at 的键是送达该 LSA 的直接邻居。证据成立当且仅当
        存在一个送达邻居 nb，满足：
          1. 时间在 MaxAge 内；
          2. nb 就是 origin 自身（我与来源直连，直接听到其心跳），
             或 nb 已在新鲜闭包内（沿可信方向传来）。
        自签发的 LSA 永远视为新鲜。
        """
        if origin == self.id:
            return True
        entry = self.lsdb.get(origin)
        if entry is None:
            return False
        for nb, ts in entry.heard_at.items():
            if tick - ts > self.max_age:
                continue
            if nb == origin or nb in fresh:
                return True
        return False

    def _fresh_set(self, tick: int) -> Set[str]:
        """从自己出发，沿“双向声明一致且两端心跳近期有效”的边做
        可达闭包；闭包内为当前新鲜可达的来源。心跳方向必须来自
        闭包内，因此分区对端的旧 LSA 无法靠本侧转发互相保活。"""
        fresh: Set[str] = {self.id}
        changed = True
        while changed:
            changed = False
            for x in list(fresh):
                ex = self.lsdb.get(x)
                if ex is None:
                    continue
                for y, cost in ex.lsa.links.items():
                    if y in fresh or y not in self.lsdb:
                        continue
                    ey = self.lsdb[y]
                    # 双向声明一致（防止旧 LSA 拼出幽灵边）
                    if ey.lsa.links.get(x) != cost:
                        continue
                    # 两端心跳都必须能从新鲜方向近期听到
                    if self._heard_via_fresh(x, fresh, tick) and \
                            self._heard_via_fresh(y, fresh, tick):
                        fresh.add(y)
                        changed = True
        return fresh

    # ---- 路由计算 -----------------------------------------------------

    def routing_table(self, tick: int = 0) -> Dict[str, Route]:
        """只根据本地 LSDB 计算下一跳。

        * 老化闭包外的来源视为不可达，其 LSA 仍保留在库中；
        * 已知但当前无路径的节点显式标记不可达；
        * 从未学到过的节点不产生表项。
        """
        fresh = self._fresh_set(tick)
        fresh_lsdb: Dict[str, LSA] = {
            origin: self.lsdb[origin].lsa
            for origin in fresh
            if origin in self.lsdb
        }
        return compute_routes(self.id, fresh_lsdb, all_known=sorted(self.lsdb))
