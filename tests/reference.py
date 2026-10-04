"""独立参照实现（仅供测试核对）。

刻意与模拟代码 ``linkstate/dijkstra.py`` 采用不同写法，避免
“同一份算法验证它自己”：

* :func:`reference_routes` —— 对 (费用, 完整ID路径) 标签做定点
  松弛（Bellman-Ford 形态，不使用堆、不使用 Dijkstra 结构）；
* :func:`brute_force_routes` —— DFS 枚举全部简单路径后直接按
  (费用, 路径字典序) 取最小。

两者在随机小图上必须彼此一致，随后测试再以 ``reference_routes``
对照各路由器 LSDB 实际收敛出的路由表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
