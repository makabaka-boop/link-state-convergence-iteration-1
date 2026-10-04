"""测试辅助：从记录中取最终状态、按参照真值核对收敛。"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkstate.network import Network  # noqa: E402
from tests.reference import (  # noqa: E402
    brute_force_routes,
    reference_routes,
    transit_graph,
)

RouteT = Tuple[Optional[str], int, Tuple[str, ...]]


def route_view(nw: Network) -> Dict[str, Dict[str, RouteT]]:
    """把模拟器各路由器的实际路由表转成 (next_hop, cost, path)。"""
    view: Dict[str, Dict[str, RouteT]] = {}
    for rid, r in nw.routers.items():
        view[rid] = {
            d: (rt.next_hop, rt.cost, tuple(rt.path))
            for d, rt in r.routing_table().items()
        }
    return view


def assert_routes_match_truth(
    nw: Network,
    *,
    reachable_only: bool = False,
    drained: Optional[set] = None,
    msg: str = "",
) -> None:
    """每个路由器只凭自身 LSDB 的结果，必须与独立参照在“活动拓扑真值”
    上算出的最短路完全一致（下一跳、费用、完整路径）。

    分区后两侧各自只应“知道”本侧节点，所以默认对照节点集取
    各路由器路由表里实际出现的节点；``reachable_only`` 进一步
    只比较可达项。``drained`` 给出当前处于过境排空的节点集合，
    参照侧用独立的“删点”实现 (:func:`transit_graph`) 计算期望。
    """
    truth = nw.truth_graph()
    nodes = sorted(nw.routers)
    drained = drained or set()
    for rid in nodes:
        # 真值侧：以网络活动拓扑计算（真值知道全部节点，含不可达）
        expected_full = reference_routes(rid, transit_graph(truth, rid, drained), nodes)
        actual = route_view(nw)[rid]
        for dest in sorted(actual):
            exp = expected_full[dest]
            act = actual[dest]
            if reachable_only and exp[1] < 0:
                continue
            assert act == exp, (
                f"{msg} 节点 {rid} 到 {dest} 路由不一致："
                f"实际 next={act[0]},cost={act[1]},path={'-'.join(act[2])}；"
                f"期望 next={exp[0]},cost={exp[1]},path={'-'.join(exp[2])}"
            )


def assert_lsdb_fresh(nw: Network) -> None:
    """各节点 LSDB 中持有的每个 origin 的序号必须等于该 origin
    当前真实序号（防止过期通告回灌）。"""
    current = {rid: r.current_seq for rid, r in nw.routers.items()}
    for rid, r in nw.routers.items():
        for origin, entry in r.lsdb.items():
            assert entry.lsa.seq == current[origin], (
                f"{rid} 的 LSDB 中 {origin} 序号为 {entry.lsa.seq}，"
                f"应为 {current[origin]}（过期通告污染）"
            )


def assert_unreachable_both_ways(view, side_a, side_b) -> None:
    """两个分区之间任何节点对都应显式不可达（next_hop 为 None）。"""
    for x in side_a:
        for y in side_b:
            if y in view[x]:
                assert view[x][y][0] is None, f"分区中 {x} 到 {y} 却有下一跳"
            if x in view[y]:
                assert view[y][x][0] is None, f"分区中 {y} 到 {x} 却有下一跳"


def cross_check_algorithms_random(trials: int = 60) -> None:
    """在随机小图上交叉验证两种独立参照写法一致。"""
    import random

    rng = random.Random(99)
    for t in range(trials):
        n = rng.randint(2, 8)
        ids = [f"R{i}" for i in range(1, n + 1)]
        g = {rid: {} for rid in ids}
        for i in ids:
            for j in ids:
                if i < j and rng.random() < 0.4:
                    c = rng.randint(1, 9)
                    g[i][j] = c
                    g[j][i] = c
        src = rng.choice(ids)
        a = reference_routes(src, g, ids)
        b = brute_force_routes(src, g, ids)
        assert a == b, f"参照算法分歧（trial {t}）:\n{a}\n{b}"
