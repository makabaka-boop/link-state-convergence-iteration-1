"""命令行演示：构造一个含涨价、分区、重连、过期通告的场景，
逐 tick 输出消息日志、各节点 LSDB 与路由表。

用法：
    python -m linkstate.cli                 # 文本输出（精简）
    python -m linkstate.cli --verbose       # 文本输出（含每条消息）
    python -m linkstate.cli --json out.json # 同时导出完整结构化记录
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .network import Network


def build_scenario() -> tuple:
    """6 节点环 + 两条弦：
          R1=1 R2=2
        /  \\    /  \\
      R6    R3--R4
        \\  /      \\
          R5-------
    """
    nw = Network(
        refresh_interval=5,
        seed=2026,
        base_delay=1,
        jitter=1,            # 延迟 0~2 tick，产生乱序
        duplicate_prob=0.15,  # 15% 复制
    )
    for rid in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        nw.add_router(rid)
    # 环
    nw.add_link("R1", "R2", 1)
    nw.add_link("R2", "R3", 2)
    nw.add_link("R3", "R4", 1)
    nw.add_link("R4", "R5", 2)
    nw.add_link("R5", "R6", 1)
    nw.add_link("R6", "R1", 2)
    # 弦
    nw.add_link("R3", "R5", 3)
    nw.add_link("R1", "R4", 4)

    events = [
        # 早期有较高丢弃概率，靠周期补发收敛
        {"tick": 1, "type": "drop_window", "start": 1, "end": 4, "prob": 0.5},
        # 链路涨价：R1-R2 1 -> 8，R1 去往 R2/R3 方向应改走其他路径
        {"tick": 8, "type": "cost", "a": "R1", "b": "R2", "cost": 8},
        # 分区：切成 {R1,R6,R5} 与 {R2,R3,R4} 两侧。
        # 跨区边共四条：R1-R2、R1-R4、弦 R3-R5、环边 R4-R5。
        {
            "tick": 14,
            "type": "cut_set",
            "links": [("R1", "R2"), ("R1", "R4"), ("R3", "R5"), ("R4", "R5")],
        },
        # 恢复前不再设置新的随机丢失（早期 [1,4) 已演示过有损传输；
        # 分区期持续丢包的收敛保证由单元测试 test_partition_* 覆盖）。
        # 注入一条过期通告：R1 的旧序号（涨价/切链前都已超过此序号），
        # 直接送达对侧 R3，模拟绕开传输层的旧副本回灌
        {
            "tick": 17,
            "type": "inject",
            "origin": "R1",
            "seq": 1,
            "links": {"R2": 1, "R6": 2, "R4": 4},
            "src": "R3",
            "deliver_now": True,
        },
        # 分区重连（消息停止丢失后必须收敛）
        {
            "tick": 20,
            "type": "restore_set",
            "links": [
                ("R1", "R2"),
                ("R1", "R4"),
                ("R3", "R5"),
                ("R4", "R5"),
            ],
        },
        {"tick": 20, "type": "drop_window", "start": 20, "end": 10_000, "prob": 0.0},
    ]
    return nw, events


def build_drain_scenario() -> tuple:
    """过境排空专用演示：6 节点环 + 弦，中途维护 R3，随后撤销。

    拓扑与 :func:`build_scenario` 相同，默认演示仍保持旧输出不变；
    本场景用 ``--drain-demo`` 触发，展示：排空 R3 后全网仍能到达
    R3，但所有路径不再以 R3 为中间点；撤销后恢复最短路。
    """
    nw, events = build_scenario()
    events = events + [
        {"tick": 24, "type": "drain", "router": "R3"},
        {"tick": 28, "type": "undrain", "router": "R3"},
    ]
    return nw, events


def _fmt_lsa(lsa: dict) -> str:
    links = ",".join(f"{k}:{v}" for k, v in lsa["links"].items())
    suffix = "" if lsa.get("transit", True) else ",!transit"
    return f"{lsa['origin']}#{lsa['seq']}({links}{suffix})"


def _fmt_route(rt: dict) -> str:
    if rt["next_hop"] is None:
        if rt["cost"] == 0:
            return f"{rt['dest']}: 本地"
        return f"{rt['dest']}: 不可达"
    return (
        f"{rt['dest']}: ->{rt['next_hop']} "
        f"费用{rt['cost']} 路径{'-'.join(rt['path'])}"
    )


def render_text(records: List[dict], verbose: bool) -> str:
    lines: List[str] = []
    for rec in records:
        head = f"===== tick {rec['tick']} ====="
        lines.append(head)
        for ev in rec["events"]:
            lines.append(f"  [事件] {ev}")

        msgs = rec["messages"]
        sends = [m for m in msgs if m["kind"] == "send"]
        delivered = [m for m in msgs if m["kind"] == "deliver"]
        drops = [m for m in msgs if m["kind"] == "drop"]
        lines.append(
            f"  [消息] 发出 {len(sends)}，送达 {len(delivered)}，丢弃 {len(drops)}"
        )
        if verbose:
            for m in msgs:
                tag = "补发" if m.get("refresh") else ("副本" if m.get("copy") else "")
                lsa = _fmt_lsa(m["lsa"])
                if m["kind"] == "send":
                    extra = f" 延迟{m['delay']}"
                    lines.append(
                        f"    SEND {m['src']}->{m['dst']} {lsa}{tag and ' ['+tag+']'}{extra}"
                    )
                elif m["kind"] == "deliver":
                    lines.append(
                        f"    RECV {m['src']}->{m['dst']} {lsa} => {m['result']}"
                        + (f" [{tag}]" if tag else "")
                    )
                else:
                    lines.append(
                        f"    DROP {m['src']}->{m['dst']} {lsa} ({m['reason']})"
                    )

        for rid, data in rec["routers"].items():
            lsdb = " | ".join(_fmt_lsa(l) for l in data["lsdb"]) or "(空)"
            lines.append(f"  {rid} LSDB: {lsdb}")
            reach = [_fmt_route(rt) for rt in data["routes"] if rt["next_hop"] is not None]
            unreach = [
                f"{rt['dest']}=不可达"
                for rt in data["routes"]
                if rt["next_hop"] is None and rt["cost"] != 0
            ]
            local = [
                f"{rt['dest']}=本地"
                for rt in data["routes"]
                if rt["next_hop"] is None and rt["cost"] == 0
            ]
            routes = " | ".join(reach + unreach + local)
            lines.append(f"  {rid} 路由: {routes}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="链路状态路由模拟器")
    parser.add_argument("--ticks", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", help="打印每条消息")
    parser.add_argument(
        "--drain-demo",
        action="store_true",
        help="运行过境排空演示（tick24 排空 R3，tick28 撤销）",
    )
    parser.add_argument("--json", dest="json_path", help="导出完整 JSON 记录")
    args = parser.parse_args(argv)

    if args.drain_demo:
        nw, events = build_drain_scenario()
    else:
        nw, events = build_scenario()
    records = nw.run(args.ticks, events)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(render_text(records, args.verbose))


if __name__ == "__main__":
    main()
