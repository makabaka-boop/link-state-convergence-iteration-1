"""传输层：在节点之间搬运通告，可延迟、复制、乱序或丢弃。

所有行为都是“逐消息”的：
  * delay   —— 每条消息独立随机延迟，延迟有正有负时同一批发出的
               消息到达顺序被打乱（乱序）；
  * duplicate —— 按概率额外产生一份副本，副本延迟独立抽取；
  * drop    —— 按 tick 区间内的概率随机丢弃，或由精确过滤器
               丢弃指定消息（测试用）；
  * 分区    —— 链路不通时，经过该链路的消息一律丢弃并记录。

传输层不知道 LSA 内容的“新旧”，它只负责搬运；新旧裁决完全发生
在接收端路由器上（这样过期通告即使穿过传输层也无法污染状态库）。
"""

from __future__ import annotations

import heapq
import random
from typing import Callable, Dict, List, Optional


# 精确过滤器：给定消息与当前 tick，返回丢弃原因字符串或 None
FilterFn = Callable[[dict, int], Optional[str]]


class Transport:
    def __init__(
        self,
        network,
        *,
        seed: int = 1234,
        base_delay: int = 1,
        jitter: int = 1,
        duplicate_prob: float = 0.0,
    ):
        self.nw = network
        self.rng = random.Random(seed)
        self.base_delay = base_delay
        self.jitter = jitter              # delay ∈ [base-jitter, base+jitter]
        self.duplicate_prob = duplicate_prob
        # (起始tick, 结束tick, 丢弃概率) 的区间列表
        self.drop_windows: List[tuple] = []
        self.filter: Optional[FilterFn] = None
        # 最小堆：(送达tick, 入队序号, 消息)
        self._queue: List[tuple] = []
        self._counter = 0
        self.msg_counter = 0

    def reset_rng(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def add_drop_window(self, start: int, end: int, prob: float) -> None:
        self.drop_windows.append((start, end, prob))

    def _drop_prob(self, tick: int) -> float:
        p = 0.0
        for start, end, prob in self.drop_windows:
            if start <= tick < end:
                p = max(p, prob)
        return p

    def send(self, tick: int, src: str, dst: str, lsa) -> None:
        """发送一条通告；是否入队在此刻决定。"""
        self.msg_counter += 1
        msg = {
            "id": self.msg_counter,
            "sent_tick": tick,
            "src": src,
            "dst": dst,
            "lsa": lsa,
        }
        reason = self._pre_send_check(msg, tick)
        if reason is not None:
            self.nw._log_message(tick, "drop", msg, reason=reason)
            return
        self._enqueue(tick, msg)
        # 复制：再独立产生一份延迟不同的副本
        if self.rng.random() < self.duplicate_prob:
            self._enqueue(tick, msg, copy=True)

    def _pre_send_check(self, msg: dict, tick: int) -> Optional[str]:
        if not self.nw.link_active(msg["src"], msg["dst"]):
            return "partitioned"
        if self.filter is not None:
            reason = self.filter(msg, tick)
            if reason:
                return reason
        prob = self._drop_prob(tick)
        if prob > 0.0 and self.rng.random() < prob:
            return "random-drop"
        return None

    def _enqueue(self, tick: int, msg: dict, copy: bool = False) -> None:
        delay = self.base_delay + self.rng.randint(-self.jitter, self.jitter)
        delay = max(0, delay)
        sent = dict(msg)
        if copy:
            sent["copy"] = True
        self._counter += 1
        heapq.heappush(self._queue, (tick + delay, self._counter, sent))
        self.nw._log_message(tick, "send", sent, delay=delay, copy=copy)

    def deliver_due(self, tick: int) -> None:
        """送达所有在本 tick 到期的消息（按入队顺序出堆即体现乱序）。"""
        while self._queue and self._queue[0][0] <= tick:
            due_tick, _, msg = heapq.heappop(self._queue)
            # 发送后链路才被切断：投递前再检查一次，分区期间不可达
            if not self.nw.link_active(msg["src"], msg["dst"]):
                self.nw._log_message(tick, "drop", msg, reason="partitioned")
                continue
            self.nw.deliver(tick, msg)
