"""测试辅助：从记录中取最终状态、按参照真值核对收敛。"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkstate.network import Network  # noqa: E402
from tests.reference import (  # noqa: E402
    reference_routes,
    brute_force_routes,
    reference_constrained_routes,
    brute_force_constrained_routes,
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
    msg: str = "",
) -> None:
    """每个路由器只凭自身 LSDB 的结果，必须与独立参照在“活动拓扑真值”
    上算出的最短路完全一致（下一跳、费用、完整路径）。

    分区后两侧各自只应“知道”本侧节点，所以默认对照节点集取
    各路由器路由表里实际出现的节点；``reachable_only`` 进一步
    只比较可达项。
    """
    truth = nw.truth_graph()
    nodes = sorted(nw.routers)
    for rid in nodes:
        # 真值侧：以网络活动拓扑计算（真值知道全部节点，含不可达）
        expected_full = reference_routes(rid, truth, nodes)
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


def assert_routes_match_truth_constrained(
    nw: Network,
    *,
    drained: Optional[set] = None,
    msg: str = "",
) -> None:
    """过境排空下：每个路由器只凭自身 LSDB 算出的路由，必须与
    “活动拓扑 + 排空集合”上的两种独立受构搜索结果逐项一致。

    ``drained`` 为当前真正排空的节点（网络真值）。
    """
    truth = nw.truth_graph()
    drained = set(drained if drained is not None else nw.truth_drained())
    nodes = sorted(nw.routers)
    for rid in nodes:
        exp_relax = reference_constrained_routes(rid, truth, drained, nodes)
        exp_dfs = brute_force_constrained_routes(rid, truth, drained, nodes)
        assert exp_relax == exp_dfs, (
            f"{msg} 排空参照自相矛盾（{rid}）:\n{exp_relax}\n{exp_dfs}"
        )
        actual = route_view(nw)[rid]
        for dest in sorted(actual):
            assert actual[dest] == exp_relax[dest], (
                f"{msg} 节点 {rid} 到 {dest} 路由不一致："
                f"实际 next={actual[dest][0]},cost={actual[dest][1]},"
                f"path={'-'.join(actual[dest][2])}；"
                f"期望 next={exp_relax[dest][0]},cost={exp_relax[dest][1]},"
                f"path={'-'.join(exp_relax[dest][2])}"
            )


def assert_no_route_transits_drained(nw: Network, *, msg: str = "") -> None:
    """不变量：任何可达路由的中间点（path[1:-1]）都不含排空节点；
    排空节点自身作为终点（path 末元素）或起点（path 首元素）允许。
    """
    drained = nw.truth_drained()
    for rid, table in route_view(nw).items():
        for dest, (nxt, cost, path) in table.items():
            if nxt is None or len(path) <= 2:
                continue
            offenders = [p for p in path[1:-1] if p in drained]
            assert not offenders, (
                f"{msg} 路由 {rid}->{dest} 经过排空节点 {offenders}："
                f"{'-'.join(path)}"
            )


def cross_check_constrained_random(trials: int = 120) -> None:
    """随机连通小图 + 随机排空集合上，两种受构参照必须一致。"""
    import random

    rng = random.Random(202)
    for t in range(trials):
        n = rng.randint(2, 8)
        ids = [f"R{i}" for i in range(1, n + 1)]
        g = {rid: {} for rid in ids}
        # 先保证连通（生成环），再随机补边
        for i in range(n):
            c = rng.randint(1, 9)
            a, b = ids[i], ids[(i + 1) % n]
            g[a][b] = c
            g[b][a] = c
        for i in ids:
            for j in ids:
                if i < j and j not in g[i] and rng.random() < 0.25:
                    c = rng.randint(1, 9)
                    g[i][j] = c
                    g[j][i] = c
        k = rng.randint(1, min(3, n))
        drained = set(rng.sample(ids, k))
        src = rng.choice(ids)
        a = reference_constrained_routes(src, g, drained, ids)
        b = brute_force_constrained_routes(src, g, drained, ids)
        assert a == b, f"排空参照分歧（trial {t}）drained={drained}:\n{a}\n{b}"
