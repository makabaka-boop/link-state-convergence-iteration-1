"""独立参照实现（仅供测试核对）。

刻意与模拟代码 ``linkstate/dijkstra.py`` 采用不同写法，避免
“同一份算法验证它自己”：

* :func:`reference_routes` —— 对 (费用, 完整ID路径) 标签做定点
  松弛（Bellman-Ford 形态，不使用堆、不使用 Dijkstra 结构）；
* :func:`brute_force_routes` —— DFS 枚举全部简单路径后直接按
  (费用, 路径字典序) 取最小；
* :func:`transit_graph` —— 过境排空的独立语义实现：把排空节点
  从图中**删点**（源点豁免），与模拟器“删进入边”的写法互相核对。

两者在随机小图上必须彼此一致，随后测试再以 ``reference_routes``
对照各路由器 LSDB 实际收敛出的路由表。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

Label = Tuple[int, Tuple[str, ...]]
_INF: Label = (10**18, ())


def transit_graph(
    graph: Dict[str, Dict[str, int]],
    source: str,
    drained: Iterable[str] = (),
) -> Dict[str, Dict[str, int]]:
    """按过境排空语义改造真值图（与模拟器 ``_build_graph`` 的“删边”
    写法刻意不同：这里直接**删点**再补回）。

    * 排空节点不作为中间点：先从图中整体删除，最短路自然不会经过它；
    * 若 ``source`` 自身在排空集合中，它作为起点保留（其出边完整）；
    * 其余排空节点仍作为目的地出现在节点集里——由调用方传入的
      ``all_nodes`` 体现，此处只负责边集。
    """
    blocked: Set[str] = set(drained) - {source}
    g: Dict[str, Dict[str, int]] = {}
    for u, nbrs in graph.items():
        if u in blocked:
            continue
        g[u] = {v: c for v, c in nbrs.items() if v not in blocked}
    return g


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
