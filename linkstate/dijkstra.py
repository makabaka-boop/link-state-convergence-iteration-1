"""最短路算法（路由器侧使用）。

裁决规则：
  1. 路径总费用最小者胜出；
  2. 费用相同时，比较“完整路由器 ID 路径”的字典序
     （逐跳比较 ID，例如 ("R1","R2") < ("R1","R3")，
      前缀更短的路径更小）。

实现上给每个节点维护标签 ``(总费用, 完整路径元组)``，
标签按 (费用, 路径) 组合比较后做松弛。所有链路费用均为正，
因此该比较次序满足最优子结构：最优路径的每条前缀也必是
同一序下到该中间点的最优路径，Dijkstra 直接成立。

路由器只能拿到自己 LSDB 里的信息——分区中对端节点完全缺失时，
它既不会出现在图里，也不会得到任何路由（上层标记为不可达）。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Route:
    dest: str
    next_hop: Optional[str]     # None 表示不可达
    cost: int                  # 不可达时为 -1
    path: Tuple[str, ...]      # 完整路径；不可达时为空元组

    def to_dict(self) -> dict:
        return {
            "dest": self.dest,
            "next_hop": self.next_hop,
            "cost": self.cost,
            "path": list(self.path),
        }


def _build_graph(lsdb: Dict[str, "LSA"]) -> Dict[str, Dict[str, int]]:
    """从 LSDB 重建无向图：只有两端互相声明（且费用一致）的边才有效。

    为什么必须双向确认——网络分区时，一侧路由器可能长期持有对侧
    邻居的旧 LSA（对端新 LSA 物理上无法跨分区送达）。双向确认
    保证：任一端的新 LSA 删除该边后，该边立即失效，旧副本无法
    凭空拼出跨越分区的幽灵路径。调用方传入的 ``lsdb`` 只包含
    老化闭包内仍新鲜的条目。
    """
    graph: Dict[str, Dict[str, int]] = {origin: {} for origin in lsdb}
    for origin, lsa in lsdb.items():
        for neighbor, cost in lsa.links.items():
            if neighbor not in lsdb:
                continue
            remote = lsdb[neighbor].links.get(origin)
            if remote is not None and remote == cost:
                graph[origin][neighbor] = cost
    return graph


def dijkstra(
    source: str,
    graph: Dict[str, Dict[str, int]],
    all_nodes: Optional[List[str]] = None,
) -> Dict[str, Route]:
    """以 (费用, 完整ID路径) 为标签求最短路。"""
    if source not in graph:
        graph = dict(graph)
        graph[source] = {}

    # label[node] = (累计费用, 从 source 到 node 的完整路径)
    labels: Dict[str, Tuple[int, Tuple[str, ...]]] = {
        source: (0, (source,))
    }
    pq: List[Tuple[Tuple[int, Tuple[str, ...]], str]] = [((0, (source,)), source)]
    done: set = set()

    while pq:
        (cost, path), node = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        for neighbor, edge_cost in graph.get(node, {}).items():
            new_label = (cost + edge_cost, path + (neighbor,))
            old_label = labels.get(neighbor)
            if old_label is None or new_label < old_label:
                labels[neighbor] = new_label
                heapq.heappush(pq, (new_label, neighbor))

    nodes = all_nodes if all_nodes is not None else sorted(labels)
    routes: Dict[str, Route] = {}
    for dest in nodes:
        if dest == source:
            routes[dest] = Route(source, None, 0, (source,))
        elif dest in labels:
            dist, path = labels[dest]
            routes[dest] = Route(dest, path[1], dist, path)
        else:
            # LSDB 里存在该节点（如通过邻居声明得知），但图中无路径：
            # 明确给出不可达，而不是悄悄省略。
            routes[dest] = Route(dest, None, -1, ())
    return routes


def compute_routes(
    source: str, lsdb: Dict[str, "LSA"], all_known: Optional[List[str]] = None
) -> Dict[str, Route]:
    """路由器入口：仅凭自身（未老化的）LSDB 计算路由。

    ``lsdb`` 只含仍新鲜的条目；``all_known`` 为路由器知道存在的
    全部节点（含已老化/不可达），这些节点会得到显式不可达表项。
    """
    graph = _build_graph(lsdb)
    nodes = all_known if all_known is not None else sorted(lsdb)
    return dijkstra(source, graph, nodes)
