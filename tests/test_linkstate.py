"""端到端测试。

覆盖：
  1. 2~8 节点小图，有损传输下最终收敛，并与独立最短路参照逐项核对；
  2. 并列费用时按完整路由器 ID 路径字典序裁决（含非本地并列）；
  3. 过期通告：直接裁决拒绝 + 经传输层回灌均不能覆盖新状态；
  4. 链路涨价后所有节点改用次优路径，且之后再注入旧通告无效；
  5. 分区期间显式不可达；重连且消息停止丢失后收敛；
  6. 初始通告被全部丢弃时，周期补发仍能保证收敛。
"""

from __future__ import annotations

import unittest

from linkstate import LSA, Network, Router
from tests.helpers import (
    assert_lsdb_fresh,
    assert_routes_match_truth,
    assert_unreachable_both_ways,
    cross_check_algorithms_random,
    route_view,
)
from tests.reference import brute_force_routes, reference_routes
from linkstate.dijkstra import compute_routes


class TestAlgorithm(unittest.TestCase):
    def test_two_independent_references_agree(self):
        cross_check_algorithms_random()

    def test_lexicographic_full_path_tiebreak(self):
        # R1 到 R4 有两条同费用路径：
        #   R1-R2-R4 费用 1+3=4；R1-R3-R4 费用 2+2=4
        # 完整路径 ("R1","R2","R4") < ("R1","R3","R4")，选 R2
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R4", 3)
        nw.add_link("R1", "R3", 2)
        nw.add_link("R3", "R4", 2)
        nw.run(6)
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"], ("R2", 4, ("R1", "R2", "R4")))

    def test_lexicographic_tiebreak_beyond_first_hop(self):
        # 到 R5：两条路径第一跳都是 R2，在第二跳才分出字典序：
        #   R1-R2-R3-R5 费用 2+1+1=4
        #   R1-R2-R4-R5 费用 1+2+1=4
        # ("R1","R2","R3","R5") < ("R1","R2","R4","R5")，选 R3
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3", "R4", "R5"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 2)
        nw.add_link("R2", "R3", 1)
        nw.add_link("R3", "R5", 1)
        nw.add_link("R2", "R4", 2)
        nw.add_link("R4", "R5", 1)
        nw.run(8)
        view = route_view(nw)
        self.assertEqual(
            view["R1"]["R5"], ("R2", 4, ("R1", "R2", "R3", "R5"))
        )

    def test_unknown_node_has_no_route_known_node_is_none(self):
        # 从一开始就不连通且从未交换过 LSA：对端节点不在路由表中；
        # 同分区内“已知”的节点若无路径则显式标记不可达（见分区测试）。
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R3", "R4", 1)
        nw.run(6)
        view = route_view(nw)
        self.assertNotIn("R3", view["R1"])  # 从未学到，不产生表项
        self.assertEqual(view["R1"]["R2"][:2], ("R2", 1))


class TestConvergence(unittest.TestCase):
    def _mesh(self, n: int, edges, *, seed=1, drop_end=3, prob=0.4):
        nw = Network(base_delay=1, jitter=1, duplicate_prob=0.1, seed=seed)
        for i in range(1, n + 1):
            nw.add_router(f"R{i}")
        for a, b, c in edges:
            nw.add_link(a, b, c)
        events = [
            {"tick": 1, "type": "drop_window",
             "start": 1, "end": drop_end, "prob": prob},
        ]
        nw.run(12, events)
        return nw

    def test_2_to_8_routers_converge(self):
        # 2 节点
        nw = self._mesh(2, [("R1", "R2", 3)], seed=2)
        assert_routes_match_truth(nw, msg="2节点")

        # 8 节点环 + 交叉边
        edges8 = [(f"R{i}", f"R{i + 1}", i % 3 + 1) for i in range(1, 8)]
        edges8.append(("R8", "R1", 2))
        edges8 += [("R1", "R5", 6), ("R2", "R7", 7), ("R3", "R6", 5)]
        nw = self._mesh(8, edges8, seed=3, drop_end=5, prob=0.45)
        assert_routes_match_truth(nw, msg="8节点")
        assert_lsdb_fresh(nw)

    def test_random_graphs_3_to_7(self):
        import random

        rng = random.Random(77)
        for trial in range(8):
            n = rng.randint(3, 7)
            ids = [f"R{i}" for i in range(1, n + 1)]
            edges = []
            # 保证连通：先生成环，再随机补边
            for i in range(n):
                edges.append((ids[i], ids[(i + 1) % n], rng.randint(1, 6)))
            for i in range(n):
                for j in range(i + 2, n):
                    if (i, j) == (0, n - 1):
                        continue
                    if rng.random() < 0.3:
                        edges.append((ids[i], ids[j], rng.randint(1, 9)))
            nw = self._mesh(n, edges, seed=100 + trial, drop_end=4)
            assert_routes_match_truth(nw, msg=f"随机图 trial={trial} n={n}")


class TestStaleAdvertisements(unittest.TestCase):
    def test_receive_directly_rejects_old_seq(self):
        r = Router("R1")
        r.set_local_link("R2", 1)
        r.originate()                 # seq=1
        r.set_local_link("R2", 9)
        latest = r.originate()        # seq=2
        self.assertEqual(latest.seq, 2)

        stale = LSA("R1", 1, {"R2": 1})
        self.assertEqual(r.receive(stale), "stale")
        # 旧序号不得覆盖新状态
        self.assertIs(r.lsdb["R1"].lsa, latest)
        self.assertEqual(r.lsdb["R1"].lsa.links["R2"], 9)

        self.assertEqual(r.receive(latest), "dup")  # 完全相同视为重复

    def test_same_seq_different_content_rejected(self):
        r = Router("R1")
        r.set_local_link("R2", 1)
        r.originate()
        # 同序号但内容被篡改：保守视为过期，不得覆盖
        forged = LSA("R1", 1, {"R2": 1, "R9": 1})
        self.assertEqual(r.receive(forged), "stale")
        self.assertNotIn("R9", r.lsdb["R1"].lsa.links)

    def test_stale_lsa_via_transport_ignored(self):
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 1)
        nw.run(4)

        # R1 涨价后 seq=2（第二段从 tick5 起，事件用绝对时刻）
        nw.run(8, [{"tick": 5, "type": "cost", "a": "R1", "b": "R2", "cost": 9}])
        self.assertEqual(nw.routers["R1"].current_seq, 2)

        records = nw.run(
            12,
            [
                {
                    "tick": 9,
                    "type": "inject",
                    "origin": "R1",
                    "seq": 1,
                    "links": {"R2": 1},
                    "src": "R2",
                    "deliver_now": True,
                },
                {
                    "tick": 9,
                    "type": "inject",
                    "origin": "R1",
                    "seq": 1,
                    "links": {"R2": 1},
                    "src": "R3",
                },
            ],
        )
        # 直接注入：日志中裁决结果必须是 stale
        verdicts = [
            m["result"]
            for m in records[9]["messages"]
            if m["kind"] == "deliver" and m["lsa"]["origin"] == "R1"
        ]
        self.assertIn("stale", verdicts)
        # 任何节点都不能持有旧状态
        assert_lsdb_fresh(nw)
        assert_routes_match_truth(nw, msg="过期通告后")


class TestLinkCostIncrease(unittest.TestCase):
    def test_cost_increase_reroutes_everywhere(self):
        # 初始：R1-R4 直连费用 2（R1->R4 最优为直连）
        # 备选：R1-R2-R3-R4 费用 1+1+1=3
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 1)
        nw.add_link("R3", "R4", 1)
        nw.add_link("R1", "R4", 2)
        nw.run(4)
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"][:2], ("R4", 2))

        # 涨价 2 -> 9（绝对 tick5）：R1 到 R4 应改走 R2-R3
        nw.run(8, [{"tick": 5, "type": "cost", "a": "R1", "b": "R4", "cost": 9}])
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"], ("R2", 3, ("R1", "R2", "R3", "R4")))
        assert_routes_match_truth(nw, msg="涨价后")

        # 涨价期间仍有丢消息 + 旧价通告回灌，最终依旧正确
        # （第三段从 tick9 起；旧 seq=1 已被 seq=2 盖过）
        nw.run(
            18,
            [
                {"tick": 10, "type": "drop_window",
                 "start": 10, "end": 13, "prob": 0.6},
                {
                    "tick": 11,
                    "type": "inject",
                    "origin": "R1",
                    "seq": 1,                       # 涨价前的旧序号
                    "links": {"R2": 1, "R4": 2},
                    "src": "R4",
                },
            ],
        )
        assert_lsdb_fresh(nw)
        assert_routes_match_truth(nw, msg="涨价+丢消息+过期通告后")


class TestPartitionAndReconnect(unittest.TestCase):
    def _four_nodes(self):
        nw = Network(base_delay=1, jitter=1, duplicate_prob=0.1, seed=5)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        # 环：R1-R2-R3-R4-R1，外加 R2-R4
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 2)
        nw.add_link("R3", "R4", 1)
        nw.add_link("R4", "R1", 3)
        nw.add_link("R2", "R4", 2)
        return nw

    def test_aging_requires_fresh_direction(self):
        # 周期补发相同 LSA 能为直连来源续命；但仅经中继转发的副本
        # 不能在来源被切断后无限保活。
        nw = Network(base_delay=0, jitter=0, seed=1, refresh_interval=3, max_age=4)
        for rid in ["R1", "R2", "R3"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 1)
        nw.run(10)  # 全网稳定，心跳充足
        view = route_view(nw)
        self.assertEqual(view["R1"]["R3"][0], "R2")

        # 切断 R2-R3：R3 的心跳只能到 R2，R1 侧 R3 应在 MaxAge 后不可达
        nw.run(20, [{"tick": 11, "type": "cut", "a": "R2", "b": "R3"}])
        view = route_view(nw)
        self.assertIsNone(view["R1"]["R3"][0])
        self.assertIsNone(view["R3"]["R1"][0])
        # R1-R2 仍连通（直连心跳不受影响）
        self.assertEqual(view["R1"]["R2"][:2], ("R2", 1))

    def test_partition_shows_unreachable_then_converges(self):
        nw = self._four_nodes()
        nw.run(5)
        assert_routes_match_truth(nw, msg="分区前")

        # 切成 {R1} 与 {R2,R3,R4}：切断 R1 的两条边（第二段 tick6 起）
        records = nw.run(
            11,
            [
                {"tick": 6, "type": "cut", "a": "R1", "b": "R2"},
                {"tick": 6, "type": "cut", "a": "R1", "b": "R4"},
                # 分区期间高丢包
                {"tick": 6, "type": "drop_window",
                 "start": 6, "end": 14, "prob": 0.5},
                # 分区中直接向 R3 注入 R1 的旧序号通告，但伪造链路内容，
                # 即使绕开传输层也必须被接收端拒绝（同序号内容不同 => stale）
                {
                    "tick": 9,
                    "type": "inject",
                    "origin": "R1",
                    "seq": 1,
                    "links": {"R2": 1, "R4": 3, "R3": 1},
                    "src": "R3",
                    "deliver_now": True,
                },
            ],
        )
        view = route_view(nw)
        # 分区两侧仍保留彼此旧 LSA，因此节点“已知但不可达”
        self.assertIsNone(view["R1"]["R2"][0])
        self.assertIsNone(view["R1"]["R3"][0])
        self.assertIsNone(view["R3"]["R1"][0])
        assert_unreachable_both_ways(view, ["R1"], ["R2", "R3", "R4"])
        # R3 持有的分区前 R1#1 旧快照不能被“同样旧但内容不同”的注入改写，
        # 注入裁决必须是 stale（R3 已有的 seq 就是 1）
        verdicts = [
            m["result"]
            for m in records[9]["messages"]
            if m["kind"] == "deliver" and m["lsa"]["origin"] == "R1"
        ]
        self.assertIn("stale", verdicts)
        self.assertEqual(nw.routers["R3"].lsdb["R1"].lsa.seq, 1)
        # 分区内 {R2,R3,R4} 的路由仍然正确
        # R2-R3 直连费用 2；R2-R4-R3 费用 2+1=3，直连更优
        self.assertEqual(view["R2"]["R3"], ("R3", 2, ("R2", "R3")))
        # 注意：分区期间 R1 物理上学不到对侧新序号，不能做全局新鲜度断言；
        # 但任何可达接收端收到的旧通告都必须被拒绝（见上方注入）。

        # 重连，且消息从此停止丢失 -> 必须收敛回全局最短路（第三段 tick12 起）
        nw.run(
            22,
            [
                {"tick": 14, "type": "restore", "a": "R1", "b": "R4"},
                {"tick": 14, "type": "drop_window",
                 "start": 14, "end": 10_000, "prob": 0.0},
            ],
        )
        assert_routes_match_truth(nw, msg="重连后")
        assert_lsdb_fresh(nw)
        view = route_view(nw)
        # R1 经 R4 到 R3：费用 3+1=4；经 R4-R2-R3=3+2+2 更差
        self.assertEqual(view["R1"]["R3"], ("R4", 4, ("R1", "R4", "R3")))

    def test_initially_dropped_lsas_recovered_by_periodic_refresh(self):
        # tick 0 的初始通告全部丢弃（含转发），只靠周期补发收敛
        nw = Network(base_delay=1, jitter=0, seed=11, refresh_interval=4)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 2)
        nw.add_link("R3", "R4", 1)
        nw.add_link("R4", "R1", 5)

        nw.transport.filter = lambda msg, tick: (
            "boot-drop" if tick <= 1 else None
        )
        nw.run(3)  # 前两 tick 全丢，几乎什么都学不到
        view = route_view(nw)
        # R1 此时还不知道远端
        self.assertNotIn("R3", view["R1"])

        nw.transport.filter = None
        nw.run(12)  # 等待周期补发
        assert_routes_match_truth(nw, msg="补发后收敛")


class TestRecordsShape(unittest.TestCase):
    def test_per_tick_record_contents(self):
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        records = nw.run(3)
        self.assertEqual([r["tick"] for r in records], [0, 1, 2, 3])
        for rec in records:
            self.assertIn("messages", rec)
            self.assertIn("routers", rec)
            for rid, data in rec["routers"].items():
                self.assertIn("lsdb", data)
                self.assertIn("routes", data)
        # tick 0 两个首发 LSA
        sends = [m for m in records[0]["messages"] if m["kind"] == "send"]
        self.assertEqual(len(sends), 2)
        # tick 1 到期交付（base_delay=0 => tick 0 即到期，允许在 tick0/1）
        delivered = [
            m for rec in records for m in rec["messages"] if m["kind"] == "deliver"
        ]
        self.assertTrue(any(m["result"] == "new" for m in delivered))

    def test_message_sequence_increases_with_source(self):
        nw = Network(base_delay=0, jitter=0)
        nw.add_router("R1")
        nw.add_router("R2")
        nw.add_link("R1", "R2", 1)
        nw.run(2)
        # 每条通告都带 origin 与递增 seq
        for rid, r in nw.routers.items():
            self.assertEqual(r.lsdb[rid].lsa.origin, rid)
            self.assertGreaterEqual(r.lsdb[rid].lsa.seq, 1)


class TestDemoScenario(unittest.TestCase):
    def test_cli_scenario_full_lifecycle(self):
        # 直接驱动 CLI 的 6 节点场景：早期丢包、涨价、分区、
        # 过期通告注入、重连停丢 -> 收敛
        from linkstate.cli import build_scenario

        nw, events = build_scenario()
        nw.run(30, events)

        # 最终所有节点路由必须与独立参照真值完全一致
        assert_routes_match_truth(nw, msg="演示场景最终")
        assert_lsdb_fresh(nw)

        # 分区窗口（tick 17~19）两侧显式不可达
        recs = nw.records
        for t in (17, 18, 19):
            r1_routes = {r["dest"]: r for r in recs[t]["routers"]["R1"]["routes"]}
            for dest in ("R2", "R3", "R4"):
                self.assertIsNone(r1_routes[dest]["next_hop"])
            r2_routes = {r["dest"]: r for r in recs[t]["routers"]["R2"]["routes"]}
            for dest in ("R1", "R6"):
                self.assertIsNone(r2_routes[dest]["next_hop"])

        # tick17 注入的过期 R1#1 在 R3 处裁决为 stale
        verdicts = [
            m["result"]
            for m in recs[17]["messages"]
            if m["kind"] == "deliver"
            and m["lsa"]["origin"] == "R1"
            and m["lsa"]["seq"] == 1
        ]
        self.assertTrue(
            all(v == "stale" for v in verdicts) and verdicts,
            f"过期通告应全部被判 stale，实际：{verdicts}",
        )

        # 重连且停丢后（tick22 起）收敛并保持稳定
        for t in range(22, 31):
            r1_routes = {r["dest"]: r for r in recs[t]["routers"]["R1"]["routes"]}
            # R1->R2 应走 R4-R3（费用 7），不再走涨价后的直连
            self.assertEqual(r1_routes["R2"]["cost"], 7)
            self.assertEqual(r1_routes["R2"]["next_hop"], "R4")


if __name__ == "__main__":
    unittest.main()
