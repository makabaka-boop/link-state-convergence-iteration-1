"""链路状态通告 (LSA)。

每条通告由其来源路由器签发，并带一个单调递增的序号；
``links`` 是该路由器当前全部直连链路的完整快照（邻居 -> 费用），
链路断开时邻居直接从快照中消失，而不是表示成零费用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class LSA:
    origin: str                 # 签发该通告的路由器 ID
    seq: int                    # 递增序号，越大越新
    links: Dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        links = ",".join(f"{nb}:{cost}" for nb, cost in sorted(self.links.items()))
        return f"{self.origin}#{self.seq}({links})"

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "seq": self.seq,
            "links": dict(sorted(self.links.items())),
        }
