"""链路状态通告 (LSA)。

每条通告由其来源路由器签发，并带一个单调递增的序号；
``links`` 是该路由器当前全部直连链路的完整快照（邻居 -> 费用），
链路断开时邻居直接从快照中消失，而不是表示成零费用。

``transit`` 为该路由器的**过境资格**：

* ``True``（缺省）——普通节点，可作为路径中间点转发；
* ``False``——该节点正在维护（过境排空）：其它路由器算路时
  仍可把它作为目的地，却不得让路径经它中转到第三台设备
  （该约束在 :mod:`linkstate.dijkstra` 中实现）。

资格位与链路快照同属一条 LSA 的内容：迟到的低序号通告或同序号
但内容（链路快照/资格位）不一致的通告一律不能覆盖较新状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class LSA:
    origin: str                 # 签发该通告的路由器 ID
    seq: int                    # 递增序号，越大越新
    links: Dict[str, int] = field(default_factory=dict)
    transit: bool = True        # 过境资格：False=排空，只可作为目的地

    def describe(self) -> str:
        links = ",".join(f"{nb}:{cost}" for nb, cost in sorted(self.links.items()))
        suffix = "" if self.transit else ",!transit"
        return f"{self.origin}#{self.seq}({links}{suffix})"

    def to_dict(self) -> dict:
        # 资格为缺省值 True 时不写出该键：不使用排空事件的旧场景
        # 结构化输出与旧版本逐字节一致。
        d = {
            "origin": self.origin,
            "seq": self.seq,
            "links": dict(sorted(self.links.items())),
        }
        if not self.transit:
            d["transit"] = False
        return d

    def content_key(self) -> Tuple[Dict[str, int], bool]:
        """同序号下需要逐项核对的内容（链路快照 + 过境资格）。"""
        return (self.links, self.transit)
