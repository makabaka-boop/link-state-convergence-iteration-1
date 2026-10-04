"""过境排空（transit drain）端到端测试。

维护一台路由器时：其它节点仍可把它作为目的地，却不得借它中转
到第三台设备；排空由该路由器在本地 LSA 中随递增序号宣告，可撤销。

覆盖：
  1. 两种独立受构搜索（剪枝后定点松弛 / DFS 全枚举）在随机小图上
     互相一致；
  2. 排空后：仍能到达排空节点本身、其余路径一律绕行、并列路线按
     (费用, 完整路径字典序) 选择、排空节点身后的网络不可达；
  3. 排空者自身仍可作为起点，其出向链路不受自己的排空位影响；
  4. 迟到的低序号通告、同序号异内容（含资格位被篡改）通告不能
     恢复旧资格；撤销排空后按费用与字典序恢复最短路；
  5. 传输延迟/复制/丢包/分区期间允许暂时分歧，恢复连通且消息停止
     丢失后各节点只凭自身 LSDB 收敛到受构真值；
  6. 非法事件（未知节点 / 重复排空 / 未排空却撤销）不产生任何
     部分更新：序号、库内容、消息轨迹、路由全部不变；
  7. 不使用新事件时旧场景的结构化输出逐字节不变。
"""

from __future__ import annotations

import unittest

from linkstate import Network
from linkstate.lsa import LSA as LSAData
from tests.helpers import (
    assert_lsdb_fresh,
    assert_no_route_transits_drained,
    assert_routes_match_truth,
    assert_routes_match_truth_constrained,
    cross_check_constrained_random,
    route_view,
)
from tests.reference import (
    brute_force_constrained_routes,
    reference_constrained_routes,
)


class TestConstrainedReference(unittest.TestCase):
    def test_two_independent_constrained_references_agree(self):
        cross_check_constrained_random()

    def test_constrained_references_hand_check(self):
        # R1-R2-R3 直线 + R1-R3 直连；排空 R2。
        g = {
            "R1": {"R2": 1, "R3": 5},
            "R2": {"R1": 1, "R3": 1},
            "R3": {"R2": 1, "R1": 5},
        }
        nodes = ["R1", "R2", "R3"]
        for fn in (reference_constrained_routes, brute_force_constrained_routes):
            # R1 仍能到达 R2（终点允许），到 R3 必须走直连费用 5
            self.assertEqual(
                fn("R1", g, {"R2"}, nodes)["R2"],
                ("R2", 1, ("R1", "R2")),
            )
            self.assertEqual(
                fn("R1", g, {"R2"}, nodes)["R3"],
                ("R3", 5, ("R1", "R3")),
            )
            # 排空点身后无替代路时：R1 与 R3 之间再无路径
            line = {
                "R1": {"R2": 1},
                "R2": {"R1": 1, "R3": 1},
                "R3": {"R2": 1},
            }
            self.assertEqual(
                fn("R1", line, {"R2"}, nodes)["R3"], (None, -1, ())
            )
            # 排空者自身作为起点不受影响
            self.assertEqual(
                fn("R2", g, {"R2"}, nodes)["R3"],
                ("R3", 1, ("R2", "R3")),
            )


def _diamond():
    """R1 到 R4：R1-R2-R4 费用 4，R1-R3-R4 费用 4（字典序选 R2）。"""
    nw = Network(base_delay=0, jitter=0)
    for rid in ["R1", "R2", "R3", "R4"]:
        nw.add_router(rid)
    nw.add_link("R1", "R2", 1)
    nw.add_link("R2", "R4", 3)
    nw.add_link("R1", "R3", 2)
    nw.add_link("R3", "R4", 2)
    return nw


class TestTransitDrain(unittest.TestCase):
    def test_drained_node_reachable_as_destination_others_detour(self):
        nw = _diamond()
        nw.run(6)
        view = route_view(nw)
        # 排空前列费用相同，字典序选 R2
        self.assertEqual(view["R1"]["R4"], ("R2", 4, ("R1", "R2", "R4")))

        # tick7 排空 R2：它是 R1->R4 旧最优路的唯一中间点
        nw.run(10, [{"tick": 7, "type": "drain", "router": "R2"}])
        view = route_view(nw)

        # R1 仍可把 R2 当作目的地（直连，终点允许）
        self.assertEqual(view["R1"]["R2"], ("R2", 1, ("R1", "R2")))
        # 去 R4 不得再借 R2 中转，必须绕行 R3
        self.assertEqual(view["R1"]["R4"], ("R3", 4, ("R1", "R3", "R4")))
        # R3 到 R2 仍可到达（R2 是目的地）：
        # R3-R1-R2 费用 2+1=3，R3-R4-R2 费用 2+3=5，选前者
        self.assertEqual(view["R3"]["R2"], ("R1", 3, ("R3", "R1", "R2")))
        # R4 侧同理：到 R2 可达，到 R1 必须绕 R3
        self.assertEqual(view["R4"]["R2"], ("R2", 3, ("R4", "R2")))
        self.assertEqual(view["R4"]["R1"], ("R3", 4, ("R4", "R3", "R1")))
        assert_no_route_transits_drained(nw, msg="R2排空")
        assert_routes_match_truth_constrained(nw, msg="R2排空后")

    def test_drained_articulation_blocks_nodes_beyond_but_not_itself(self):
        # 直线 R1-R2-R3：R2 是咽喉。排空后 R2 自身仍可达，R3 不可达。
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 1)
        nw.run(5)
        nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        view = route_view(nw)
        self.assertEqual(view["R1"]["R2"], ("R2", 1, ("R1", "R2")))
        self.assertEqual(view["R1"]["R3"], (None, -1, ()))
        self.assertEqual(view["R3"]["R2"], ("R2", 1, ("R3", "R2")))
        self.assertEqual(view["R3"]["R1"], (None, -1, ()))
        # R2 自己仍能发起访问两侧
        self.assertEqual(view["R2"]["R1"], ("R1", 1, ("R2", "R1")))
        self.assertEqual(view["R2"]["R3"], ("R3", 1, ("R2", "R3")))

    def test_drained_source_still_originates_paths(self):
        # 排空的是算路者自己：它仍可作为起点，出向链路全部保留。
        nw = _diamond()
        nw.run(6)
        nw.run(10, [{"tick": 7, "type": "drain", "router": "R1"}])
        view = route_view(nw)
        self.assertEqual(
            view["R1"]["R4"], ("R2", 4, ("R1", "R2", "R4"))
        )
        self.assertEqual(view["R1"]["R2"], ("R2", 1, ("R1", "R2")))
        # 而 R4 不能再借 R1 去往 R3（R1 身后节点）
        self.assertEqual(view["R4"]["R3"], ("R3", 2, ("R4", "R3")))
        assert_routes_match_truth_constrained(nw, msg="起点排空")

    def test_tiebreak_among_detours_uses_cost_then_full_path(self):
        # 菱形两条绕行路费用相同：排空其中一侧后，在剩余并列路线中
        # 仍按完整路径字典序裁决。
        # R1 到 R5：
        #   经 R2: R1-R2-R4-R5 = 1+1+1 = 3
        #   经 R3: R1-R3-R4-R5 = 2+1+1 = 4
        #   直连备用 R1-R2-R5 = 1+4 = 5（排空 R2 后不可用）
        # 排空 R2 前选 R2 路（费用 3）；排空后唯一绕行是 R3 路（费用4）。
        nw = Network(base_delay=0, jitter=0)
        for rid in ["R1", "R2", "R3", "R4", "R5"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R1", "R3", 2)
        nw.add_link("R2", "R4", 1)
        nw.add_link("R3", "R4", 1)
        nw.add_link("R4", "R5", 1)
        nw.add_link("R2", "R5", 4)
        nw.run(6)
        view = route_view(nw)
        self.assertEqual(
            view["R1"]["R5"], ("R2", 3, ("R1", "R2", "R4", "R5"))
        )

        nw.run(10, [{"tick": 7, "type": "drain", "router": "R2"}])
        view = route_view(nw)
        # 不得走 R1-R2-R5 或 R1-R2-R4-R5（R2 会成为中间点），
        # 唯一合法绕行 R1-R3-R4-R5 费用 2+1+1=4
        self.assertEqual(
            view["R1"]["R5"], ("R3", 4, ("R1", "R3", "R4", "R5"))
        )
        # R2 作为目的地仍可达
        self.assertEqual(view["R1"]["R2"], ("R2", 1, ("R1", "R2")))
        assert_no_route_transits_drained(nw, msg="并列绕行")
        assert_routes_match_truth_constrained(nw, msg="并列绕行")

    def test_drain_is_flooded_with_increasing_seq(self):
        nw = _diamond()
        nw.run(5)
        before = nw.routers["R2"].current_seq
        nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        r2 = nw.routers["R2"]
        self.assertFalse(r2.transit)
        self.assertEqual(r2.current_seq, before + 1)
        # 排空资格位随 LSA 泛洪：R1/R3/R4 都学到了 transit=False
        for rid in ("R1", "R3", "R4"):
            entry = nw.routers[rid].lsdb["R2"]
            self.assertEqual(entry.lsa.seq, before + 1)
            self.assertFalse(entry.lsa.transit)

    def test_stale_low_seq_cannot_restore_transit_eligibility(self):
        nw = _diamond()
        nw.run(5)
        # 排空 R2（seq=2，transit=False）
        records = nw.run(
            12,
            [
                {"tick": 6, "type": "drain", "router": "R2"},
                # 迟到的旧序号 seq=1（transit=True 旧资格）回灌给 R3/R4
                {
                    "tick": 8,
                    "type": "inject",
                    "origin": "R2",
                    "seq": 1,
                    "links": {"R1": 1, "R4": 3},
                    "transit": True,
                    "src": "R3",
                    "deliver_now": True,
                },
                {
                    "tick": 8,
                    "type": "inject",
                    "origin": "R2",
                    "seq": 1,
                    "links": {"R1": 1, "R4": 3},
                    "transit": True,
                    "src": "R4",
                },
            ],
        )
        verdicts = [
            m["result"]
            for m in records[8]["messages"]
            if m["kind"] == "deliver" and m["lsa"]["origin"] == "R2"
        ]
        self.assertTrue(verdicts and all(v == "stale" for v in verdicts))
        # 排空资格没有被旧通告恢复
        for rid in ("R1", "R3", "R4"):
            self.assertFalse(nw.routers[rid].lsdb["R2"].lsa.transit)
        assert_lsdb_fresh(nw)
        assert_no_route_transits_drained(nw, msg="旧序号回灌后")
        assert_routes_match_truth_constrained(nw, msg="旧序号回灌后")

    def test_same_seq_different_content_cannot_toggle_transit(self):
        nw = _diamond()
        nw.run(5)
        # 排空 R2 -> 其当前 LSA 为 seq=2, transit=False
        nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        current = nw.routers["R2"].lsdb["R2"].lsa
        self.assertEqual(current.seq, 2)
        self.assertFalse(current.transit)

        # 同序号 seq=2 但把资格位篡改为 True：必须判 stale，不得恢复
        forged = LSAData(
            "R2", 2, {"R1": 1, "R4": 3}, transit=True
        )
        verdict = nw.routers["R1"].receive(forged, tick=9, neighbor="R3")
        self.assertEqual(verdict, "stale")
        held = nw.routers["R1"].lsdb["R2"].lsa
        self.assertIsNot(held, forged)
        self.assertFalse(held.transit)
        # 完全相同的 seq=2 副本仍是 dup
        dup = LSAData("R2", 2, {"R1": 1, "R4": 3}, transit=False)
        self.assertEqual(
            nw.routers["R1"].receive(dup, tick=9, neighbor="R4"), "dup"
        )

    def test_undrain_restores_cost_and_lexicographic_routing(self):
        nw = _diamond()
        nw.run(5)
        nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"], ("R3", 4, ("R1", "R3", "R4")))

        # tick10 撤销排空：恢复后按费用与完整路径字典序，选回 R2
        nw.run(14, [{"tick": 10, "type": "undrain", "router": "R2"}])
        self.assertTrue(nw.routers["R2"].transit)
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"], ("R2", 4, ("R1", "R2", "R4")))
        assert_routes_match_truth_constrained(nw, drained=set(), msg="撤销排空后")
        # 撤销后应与普通（无约束）真值完全一致
        assert_routes_match_truth(nw, msg="撤销排空=普通真值")

    def test_old_drained_lsa_cannot_block_restored_transit(self):
        # 撤销排空后再回灌排空期的旧 seq LSA（transit=False），
        # 不能把刚恢复的过境资格重新抹掉。
        nw = _diamond()
        nw.run(5)
        nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        drained_seq = nw.routers["R2"].current_seq  # =2
        nw.run(13, [{"tick": 10, "type": "undrain", "router": "R2"}])
        self.assertEqual(nw.routers["R2"].current_seq, drained_seq + 1)

        records = nw.run(
            17,
            [
                {
                    "tick": 14,
                    "type": "inject",
                    "origin": "R2",
                    "seq": drained_seq,      # 排空期序号
                    "links": {"R1": 1, "R4": 3},
                    "transit": False,        # 旧的排空资格
                    "src": "R3",
                    "deliver_now": True,
                },
            ],
        )
        verdicts = [
            m["result"]
            for m in records[14]["messages"]
            if m["kind"] == "deliver" and m["lsa"]["origin"] == "R2"
        ]
        self.assertEqual(verdicts, ["stale"])
        view = route_view(nw)
        self.assertEqual(view["R1"]["R4"], ("R2", 4, ("R1", "R2", "R4")))
        assert_routes_match_truth(nw, msg="旧排空通告回灌后")

    def test_converges_under_loss_delay_duplication(self):
        # 排空通告穿过有损传输：允许暂时分歧，停止丢失后必须收敛。
        nw = Network(base_delay=1, jitter=1, duplicate_prob=0.2, seed=42)
        for rid in ["R1", "R2", "R3", "R4", "R5"]:
            nw.add_router(rid)
        edges = [
            ("R1", "R2", 1), ("R2", "R3", 1), ("R3", "R4", 1),
            ("R4", "R5", 1), ("R5", "R1", 1), ("R1", "R3", 4),
            ("R2", "R4", 4),
        ]
        for a, b, c in edges:
            nw.add_link(a, b, c)
        events = [
            {"tick": 1, "type": "drop_window", "start": 1, "end": 4, "prob": 0.5},
            {"tick": 6, "type": "drain", "router": "R3"},
            {"tick": 6, "type": "drop_window", "start": 6, "end": 12, "prob": 0.55},
            # tick12 起消息停止丢失
            {"tick": 12, "type": "drop_window",
             "start": 12, "end": 10_000, "prob": 0.0},
            {"tick": 20, "type": "undrain", "router": "R3"},
        ]
        nw.run(28, events)
        assert_lsdb_fresh(nw)
        # R3 已撤销：最终与无约束真值一致
        assert_routes_match_truth(nw, msg="有损收敛最终")

        # 单独核对“排空窗口末尾、停丢之后、撤销之前”的中间时刻：
        # 重新跑到该时刻验证受构真值（用独立网络复现确定性场景）
        nw2 = Network(base_delay=1, jitter=1, duplicate_prob=0.2, seed=42)
        for rid in ["R1", "R2", "R3", "R4", "R5"]:
            nw2.add_router(rid)
        for a, b, c in edges:
            nw2.add_link(a, b, c)
        nw2.run(18, events[:4])
        assert_routes_match_truth_constrained(nw2, msg="排空窗口收敛后")
        assert_no_route_transits_drained(nw2, msg="排空窗口收敛后")

    def test_drain_during_partition_then_rejoin_converges(self):
        # 分区期间排空：两侧认识可以暂时不同；重连且停丢后收敛。
        nw = Network(base_delay=1, jitter=1, duplicate_prob=0.1, seed=7)
        for rid in ["R1", "R2", "R3", "R4"]:
            nw.add_router(rid)
        nw.add_link("R1", "R2", 1)
        nw.add_link("R2", "R3", 2)
        nw.add_link("R3", "R4", 1)
        nw.add_link("R4", "R1", 3)
        nw.add_link("R2", "R4", 2)
        nw.run(5)

        # 切成 {R1} 与 {R2,R3,R4}，并在分区中排空 R2
        nw.run(
            12,
            [
                {"tick": 6, "type": "cut", "a": "R1", "b": "R2"},
                {"tick": 6, "type": "cut", "a": "R1", "b": "R4"},
                {"tick": 6, "type": "drop_window",
                 "start": 6, "end": 14, "prob": 0.4},
                {"tick": 8, "type": "drain", "router": "R2"},
            ],
        )
        view = route_view(nw)
        # R1 物理上不知道 R2 的排空新 LSA，但分区本身已使其不可达
        self.assertIsNone(view["R1"]["R2"][0])
        # {R2,R3,R4} 一侧应已遵守排空：R3/R4 不得借 R2 中转
        assert_no_route_transits_drained(nw, msg="分区+排空")
        # R2 自身作为起点仍可访问 R3/R4
        self.assertEqual(view["R2"]["R3"], ("R3", 2, ("R2", "R3")))

        # 重连 + 停丢，且保持 R2 排空：全网收敛到受构真值
        nw.run(
            22,
            [
                {"tick": 14, "type": "restore", "a": "R1", "b": "R4"},
                {"tick": 14, "type": "restore", "a": "R1", "b": "R2"},
                {"tick": 14, "type": "drop_window",
                 "start": 14, "end": 10_000, "prob": 0.0},
            ],
        )
        assert_lsdb_fresh(nw)
        assert_routes_match_truth_constrained(nw, msg="重连后仍排空")
        assert_no_route_transits_drained(nw, msg="重连后仍排空")
        view = route_view(nw)
        # R1 仍能到达 R2（终点）；到 R3 只能 R1-R4-R3（费用 4），
        # 不得走 R1-R2-R3
        self.assertEqual(view["R1"]["R2"], ("R2", 1, ("R1", "R2")))
        self.assertEqual(view["R1"]["R3"], ("R4", 4, ("R1", "R4", "R3")))

        # 撤销排空后恢复普通最短路
        nw.run(28, [{"tick": 24, "type": "undrain", "router": "R2"}])
        assert_routes_match_truth(nw, msg="分区重连后撤销排空")

    def test_drained_lsa_serialized_in_records(self):
        nw = _diamond()
        records = nw.run(9, [{"tick": 6, "type": "drain", "router": "R2"}])
        r2_lsas = {
            l["origin"]: l
            for tick_rec in records[7:9]
            for rid, data in tick_rec["routers"].items()
            for l in data["lsdb"]
        }
        self.assertFalse(r2_lsas["R2"]["transit"])
        # 排空事件描述进入事件轨迹
        self.assertTrue(
            any("drain R2" in e for e in records[6]["events"])
        )


class TestIllegalDrainEvents(unittest.TestCase):
    def test_drain_unknown_router_raises_without_side_effects(self):
        nw = _diamond()
        nw.run(5)
        msg_count_before = nw.transport.msg_counter
        recs_before = len(nw.records)
        with self.assertRaises(ValueError):
            nw.drain_router(6, "R9")
        with self.assertRaises(ValueError):
            nw.run(8, [{"tick": 6, "type": "drain", "router": "R9"}])
        # 没有任何通告/消息/额外 tick 轨迹产生
        self.assertEqual(nw.transport.msg_counter, msg_count_before)
        self.assertEqual(len(nw.records), recs_before)
        self.assertNotIn("R9", nw.routers)

    def test_double_drain_is_atomic(self):
        nw = _diamond()
        nw.run(5)
        nw.run(8, [{"tick": 6, "type": "drain", "router": "R2"}])
        seq_after = nw.routers["R2"].current_seq
        msgs_after = nw.transport.msg_counter

        # Router 层：重复排空不改状态、不增序号
        with self.assertRaises(ValueError):
            nw.routers["R2"].drain(7)
        self.assertEqual(nw.routers["R2"].current_seq, seq_after)
        self.assertFalse(nw.routers["R2"].transit)

        # Network 层：重复排空不签发、不发消息
        with self.assertRaises(ValueError):
            nw.drain_router(7, "R2")
        self.assertEqual(nw.routers["R2"].current_seq, seq_after)
        self.assertEqual(nw.transport.msg_counter, msgs_after)

        # 排空中的链路/资格 LSA 内容不变
        held = nw.routers["R1"].lsdb["R2"].lsa
        self.assertFalse(held.transit)
        self.assertEqual(held.seq, seq_after)

    def test_undrain_without_drain_is_atomic(self):
        nw = _diamond()
        nw.run(5)
        seq_before = nw.routers["R2"].current_seq
        msgs_before = nw.transport.msg_counter
        with self.assertRaises(ValueError):
            nw.routers["R2"].undrain(6)
        with self.assertRaises(ValueError):
            nw.undrain_router(6, "R2")
        self.assertEqual(nw.routers["R2"].current_seq, seq_before)
        self.assertTrue(nw.routers["R2"].transit)
        self.assertEqual(nw.transport.msg_counter, msgs_before)

    def test_undrain_unknown_router_raises(self):
        nw = _diamond()
        nw.run(4)
        with self.assertRaises(ValueError):
            nw.run(7, [{"tick": 5, "type": "undrain", "router": "R404"}])


class TestOldScenarioOutputUnchanged(unittest.TestCase):
    def test_to_dict_omits_transit_when_eligible(self):
        # 资格为缺省 True 时，序列化结果不含 transit 键
        lsa = LSAData("R1", 1, {"R2": 3}, transit=True)
        self.assertEqual(
            lsa.to_dict(),
            {"origin": "R1", "seq": 1, "links": {"R2": 3}},
        )
        # 排空时才出现该键
        lsa2 = LSAData("R1", 2, {"R2": 3}, transit=False)
        self.assertEqual(lsa2.to_dict()["transit"], False)

    def test_demo_scenario_without_drain_unchanged(self):
        # CLI 旧场景不使用排空事件：R2 从未出现在任何 transit 键中，
        # 且所有节点 LSA 都保持 transit 缺省（最终全部可中转）。
        from linkstate.cli import build_scenario

        nw, events = build_scenario()
        records = nw.run(30, events)
        for rec in records:
            for rid, data in rec["routers"].items():
                for l in data["lsdb"]:
                    self.assertNotIn(
                        "transit", l,
                        f"旧场景 tick{rec['tick']} {rid} 出现多余 transit 键",
                    )

    def test_drain_demo_converges_and_restores(self):
        # 独立的 --drain-demo 场景：tick24 排空 R3、tick28 撤销。
        from linkstate.cli import build_drain_scenario

        nw, events = build_drain_scenario()
        records = nw.run(30, events)

        # 排空收敛窗口（tick26~27，停丢始于 tick20）：受构真值 +
        # 无路由经过 R3，且 R3 对所有节点仍可达。
        assert_routes_match_truth_constrained(nw, msg="排空演示收敛")
        assert_no_route_transits_drained(nw)
        for t in (26, 27):
            for rid in ("R1", "R2", "R4", "R5", "R6"):
                table = {r["dest"]: r for r in records[t]["routers"][rid]["routes"]}
                self.assertIsNotNone(
                    table["R3"]["next_hop"], f"tick{t} {rid} 不应失去到 R3 的可达性"
                )
        r2_r4 = {r["dest"]: r for r in records[26]["routers"]["R2"]["routes"]}["R4"]
        self.assertNotEqual(tuple(r2_r4["path"]), ("R2", "R3", "R4"))

        # tick30：撤销排空后与无约束真值一致
        assert_lsdb_fresh(nw)
        assert_routes_match_truth(nw, msg="排空演示撤销后")
        r2_r4 = {r["dest"]: r for r in records[30]["routers"]["R2"]["routes"]}["R4"]
        self.assertEqual(tuple(r2_r4["path"]), ("R2", "R3", "R4"))


if __name__ == "__main__":
    unittest.main()
