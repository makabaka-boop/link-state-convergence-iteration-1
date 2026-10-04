"""链路状态通告 (LSA)。

每条通告由其来源路由器签发，并带一个单调递增的序号；
``links`` 是该路由器当前全部直连链路的完整快照（邻居 -> 费用），
链路断开时邻居直接从快照中消失，而不是表示成零费用。

``transit`` 是签发者宣告的**过境资格**：``False`` 表示该路由器
正在排空（维护），其他节点算路时不得把它用作路径中间点，但仍可
把它作为目的地；宣告者自身作为起点不受限制。该标志与链路快照
一样随序号递增的通告传播，旧序号/同序号异内容的通告同样不得
覆盖它（见 ``Router.receive``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class LSA:
    origin: str                 # 签发该通告的路由器 ID
    seq: int                    # 递增序号，越大越新
    links: Dict[str, int] = field(default_factory=dict)
    transit: bool = True        # 是否允许其他节点经本节点中转到第三台设备

    def same_content(self, other: "LSA") -> bool:
        """同序号下的内容一致性：链路快照与过境资格都必须相同。"""
        return self.links == other.links and self.transit == other.transit

    def describe(self) -> str:
        links = ",".join(f"{nb}:{cost}" for nb, cost in sorted(self.links.items()))
        tag = "" if self.transit else " [排空]"
        return f"{self.origin}#{self.seq}({links}){tag}"

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "seq": self.seq,
            "links": dict(sorted(self.links.items())),
            "transit": self.transit,
        }
