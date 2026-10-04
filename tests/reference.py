"""独立参照实现（仅供测试核对）。

刻意与模拟代码 ``linkstate/dijkstra.py`` 采用不同写法，避免
“同一份算法验证它自己”：

* :func:`reference_routes` —— 对 (费用, 完整ID路径) 标签做定点
  松弛（Bellman-Ford 形态，不使用堆、不使用 Dijkstra 结构）；
* :func:`brute_force_routes` —— DFS 枚举全部简单路径后直接按
  (费用, 路径字典序) 取最小。

过境排空的语义由**另外两个独立参照**核对，二者再次互相交叉验证：

* :func:`reference_constrained_routes` —— 先按“排空节点删除出向边、
  保留入向边（源节点自身排空时例外）”剪枝，再走定点松弛；
* :func:`brute_force_constrained_routes` —— DFS 枚举全部简单路径，
  仅接受**中间点（path[1:-1]）不含任何排空节点**的路径后取最小，
  完全不构造剪枝图。

两种写法对“可到达排空节点 / 必须绕行 / 并列路线 / 排空点之后
不可达”的回答必须一致。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

Label = Tuple[int, Tuple[str, ...]]
_INF: Label = (10**18, ())


def reference_routes(
    source: str, graph: Dict[str, Dict[str, int]], all_nodes: Optional[List[str]] = None
) -> Dict[str, Tuple[Optional[str], int, Tuple[str, ...]]]:
    nodes = all_nodes if all_nodes is not None else sorted(graph)
    labels: Dict[str, Label] = {n: _INF for n in nodes}
    if source in labels:
        labels[source] = (0, (source,))

    # 定点松弛：反复扫描全部有向边直到标签不再变化
    changed = True
    while changed:
        changed = False
        for u in sorted(graph):
            if labels[u] is _INF or labels[u] >= _INF:
                continue
            u_cost, u_path = labels[u]
            for v in sorted(graph[u]):
                cand = (u_cost + graph[u][v], u_path + (v,))
                if cand < labels[v]:
                    labels[v] = cand
                    changed = True

    result = {}
    for dest in nodes:
        cost, path = labels[dest]
        if cost >= 10**17:
            result[dest] = (None, -1, ())
        elif dest == source:
            result[dest] = (None, 0, (source,))
        else:
            result[dest] = (path[1], cost, path)
    return result


def brute_force_routes(
    source: str, graph: Dict[str, Dict[str, int]], all_nodes: Optional[List[str]] = None
) -> Dict[str, Tuple[Optional[str], int, Tuple[str, ...]]]:
    nodes = all_nodes if all_nodes is not None else sorted(graph)
    best: Dict[str, Label] = {}

    def dfs(node: str, path: Tuple[str, ...], cost: int, seen: set) -> None:
        old = best.get(node)
        label = (cost, path)
        if old is None or label < old:
            best[node] = label
        for nxt in sorted(graph.get(node, {})):
            if nxt not in seen:
                seen.add(nxt)
                dfs(nxt, path + (nxt,), cost + graph[node][nxt], seen)
                seen.remove(nxt)

    dfs(source, (source,), 0, {source})
    result = {}
    for dest in nodes:
        if dest not in best:
            result[dest] = (None, -1, ())
        elif dest == source:
            result[dest] = (None, 0, (source,))
        else:
            cost, path = best[dest]
            result[dest] = (path[1], cost, path)
    return result


def _pruned_graph(
    graph: Dict[str, Dict[str, int]],
    source: str,
    drained: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    """过境排空约束下的**有向**剪枝图。

    对每个排空节点 d：删除 d 的全部出向边（路径不能再经 d 前往
    第三台设备）；其它节点指向 d 的边保留（d 仍可作终点）。
    当 ``source`` 自身排空时，它仍是自己路径的起点，其出向边保留。
    """
    blocked: Set[str] = set(drained)
    pruned: Dict[str, Dict[str, int]] = {u: {} for u in graph}
    for u, nbrs in graph.items():
        if u in blocked and u != source:
            continue
        for v, c in nbrs.items():
            pruned[u][v] = c
    return pruned


def reference_constrained_routes(
    source: str,
    graph: Dict[str, Dict[str, int]],
    drained: Iterable[str] = (),
    all_nodes: Optional[List[str]] = None,
) -> Dict[str, Tuple[Optional[str], int, Tuple[str, ...]]]:
    """定点松弛写法：排空节点删除出边、保留入边后求最短路。"""
    pruned = _pruned_graph(graph, source, drained)
    return reference_routes(source, pruned, all_nodes)


def brute_force_constrained_routes(
    source: str,
    graph: Dict[str, Dict[str, int]],
    drained: Iterable[str] = (),
    all_nodes: Optional[List[str]] = None,
) -> Dict[str, Tuple[Optional[str], int, Tuple[str, ...]]]:
    """DFS 全枚举写法：只接受“中间点不含排空节点”的简单路径。

    与剪枝写法刻意不共享图构造：终点是排空节点时仍允许到达，
    起点排空时其作为首元素也允许；仅 path[1:-1] 受约束。
    """
    blocked = set(drained)
    nodes = all_nodes if all_nodes is not None else sorted(graph)
    best: Dict[str, Label] = {}

    def dfs(node: str, path: Tuple[str, ...], cost: int, seen: set) -> None:
        # 新到达的 node：若它是中间点（后面还有后续），必须未排空；
        # 它作为终点时无此限制。起点是否排空也不影响。
        old = best.get(node)
        label = (cost, path)
        if old is None or label < old:
            best[node] = label
        for nxt in sorted(graph.get(node, {})):
            if nxt in seen:
                continue
            if node in blocked and node != source:
                # node 在本扩展中充当中间点：排空节点不得继续中转
                continue
            seen.add(nxt)
            dfs(nxt, path + (nxt,), cost + graph[node][nxt], seen)
            seen.remove(nxt)

    dfs(source, (source,), 0, {source})
    result = {}
    for dest in nodes:
        if dest not in best:
            result[dest] = (None, -1, ())
        elif dest == source:
            result[dest] = (None, 0, (source,))
        else:
            cost, path = best[dest]
            result[dest] = (path[1], cost, path)
    return result
