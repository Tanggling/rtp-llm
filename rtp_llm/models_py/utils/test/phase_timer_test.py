"""Tests for the MoE phase timer (rtp_llm/models_py/utils/phase_timer.py).

Covers: gating (armed / token threshold), strict checkpoint ordering, phase
additivity, forward rollback on failure, dispatch sub-checkpoint split, and the
report / export shapes.
"""

import json
import os
import tempfile
import unittest

import torch

from rtp_llm.models_py.utils import phase_timer as pt


def _burn(x: torch.Tensor, iters: int = 4) -> torch.Tensor:
    """Enqueue enough GPU work that consecutive events are measurably apart."""
    for _ in range(iters):
        x = x @ x
    return x


class PhaseTimerTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.timer = pt.PhaseTimer()
        self.timer.arm()
        self.x = torch.randn(512, 512, device="cuda")

    def _record_layer(
        self,
        layer_idx: int,
        *,
        split: bool = False,
        route: bool = False,
        gemm: bool = False,
    ) -> None:
        t = self.timer
        t.moe_block_begin(layer_idx)
        t.record_moe_checkpoint("moe_start", layer_idx)
        _burn(self.x)
        if route:
            t.record_route_split("topk_done")
            _burn(self.x)
            t.record_route_split("policy_done")
            _burn(self.x)
        t.record_moe_checkpoint("route_done", layer_idx)
        if split:
            _burn(self.x)
            t.record_dispatch_split("quant_done")
            _burn(self.x)
            t.record_dispatch_split("layout_done")
        _burn(self.x)
        t.record_moe_checkpoint("dispatch_done")
        _burn(self.x)
        if gemm:
            t.record_gemm_split("align_done")
            _burn(self.x)
            t.record_gemm_split("kernels_done")
            _burn(self.x)
        t.record_moe_checkpoint("gemm_done")
        _burn(self.x)
        t.record_moe_checkpoint("combine_done")
        t.moe_block_end(layer_idx)

    def _record_forward(self, num_layers: int, tokens: int, **kw) -> None:
        t = self.timer
        t.begin_forward(tokens)
        for i in range(num_layers):
            t.layer_begin(i)
            t.attn_begin(i)
            _burn(self.x)
            t.attn_end(i)
            self._record_layer(i, **kw)
            t.layer_end(i)
        t.end_forward(success=True)

    # ------------------------------------------------------------------ gating

    def test_disarmed_records_nothing(self):
        t = pt.PhaseTimer()
        t.disarm()
        t.begin_forward(4096)
        self.assertFalse(t.recording)
        t.record_moe_checkpoint("moe_start", 0)
        t.end_forward(success=True)
        self.assertEqual(t.report(), {})

    def test_short_forward_is_gated_out(self):
        t = self.timer
        self._record_forward(2, tokens=4096)
        self.assertEqual(t.report()["forwards"], 1)
        # A tail chunk below half the first forward's tokens must not record.
        t.begin_forward(16)
        self.assertFalse(t.recording)
        t.end_forward(success=True)
        self.assertEqual(t.report()["forwards"], 1)
        # ...but a comparable forward still does.
        t.begin_forward(4096)
        self.assertTrue(t.recording)
        t.end_forward(success=True)
        self.assertEqual(t.report()["forwards"], 2)

    def test_checkpoint_outside_forward_is_dropped(self):
        # FusedMoe reached from a model_desc that never opens a window.
        self.timer.record_moe_checkpoint("dispatch_done")
        self.assertEqual(self.timer.report(), {})

    # ------------------------------------------------------------- correctness

    def test_phases_are_additive(self):
        self._record_forward(3, tokens=4096)
        pl = self.timer.per_layer()
        self.assertEqual(pl["layer_index"], [0, 1, 2])
        for i in range(3):
            parts = pl["gate_route"][i] + pl["dispatch"][i]
            self.assertAlmostEqual(parts, pl["gate_dispatch"][i], places=2)
            total = parts + pl["gemm"][i] + pl["combine"][i]
            self.assertAlmostEqual(total, pl["moe_total"][i], places=2)
            # every phase must have taken a measurable, positive amount of time
            for key in ("gate_route", "dispatch", "gemm", "combine"):
                self.assertGreater(pl[key][i], 0.0, f"{key} not advancing")
            # layer >= attn + moe_block, so overhead is a non-negative residual
            self.assertGreaterEqual(
                pl["layer"][i] + 1e-3, pl["attn"][i] + pl["moe_block"][i]
            )

    def test_checkpoints_are_monotonic_from_epoch(self):
        self._record_forward(2, tokens=4096)
        pl = self.timer.per_layer()
        prev = -1.0
        for ckpts in pl["checkpoints"]:
            offsets = [ckpts[n] for n in pt.MOE_CHECKPOINTS]
            self.assertEqual(offsets, sorted(offsets))
            self.assertGreater(offsets[0], prev)
            prev = offsets[-1]

    def test_report_sums_over_forwards(self):
        self._record_forward(2, tokens=4096)
        one = self.timer.report()
        self._record_forward(2, tokens=4096)
        two = self.timer.report()
        self.assertEqual(one["forwards"], 1)
        self.assertEqual(two["forwards"], 2)
        self.assertEqual(two["layers"], 4)
        self.assertEqual(two["moe_layers"], 4)
        self.assertEqual(two["tokens"], 8192)
        self.assertGreater(two["moe_total_ms"], one["moe_total_ms"])

    def test_dense_layer_leaves_moe_phases_zero(self):
        t = self.timer
        t.begin_forward(4096)
        # layer 0 dense: attn + layer only, no MoE checkpoints
        t.layer_begin(0)
        t.attn_begin(0)
        _burn(self.x)
        t.attn_end(0)
        t.layer_end(0)
        # layer 1 sparse
        t.layer_begin(1)
        t.attn_begin(1)
        _burn(self.x)
        t.attn_end(1)
        self._record_layer(1)
        t.layer_end(1)
        t.end_forward(success=True)

        pl = t.per_layer()
        self.assertEqual(pl["layer_index"], [0, 1])
        self.assertEqual(pl["moe_total"][0], 0.0)
        self.assertEqual(pl["checkpoints"][0], {})
        self.assertGreater(pl["moe_total"][1], 0.0)
        self.assertEqual(t.report()["moe_layers"], 1)

    # --------------------------------------------------------------- rollback

    def test_failed_forward_is_rolled_back(self):
        t = self.timer
        self._record_forward(2, tokens=4096)
        good = t.report()

        t.begin_forward(4096)
        self._record_layer(0)
        t.end_forward(success=False)

        after = t.report()
        self.assertEqual(after["forwards"], good["forwards"])
        self.assertEqual(after["layers"], good["layers"])
        self.assertAlmostEqual(after["moe_total_ms"], good["moe_total_ms"], places=3)

    def test_incomplete_sequence_does_not_corrupt_neighbours(self):
        t = self.timer
        t.begin_forward(4096)
        t.record_moe_checkpoint("moe_start", 0)
        t.record_moe_checkpoint("route_done", 0)
        # layer 0 aborted mid-sequence; layer 1 restarts cleanly
        self._record_layer(1)
        t.end_forward(success=True)
        pl = t.per_layer()
        # the incomplete record is dropped, so only layer 1 reports MoE phases
        self.assertEqual(pl["layer_index"], [1])
        self.assertGreater(pl["moe_total"][0], 0.0)

    def test_strict_mode_raises_on_out_of_order(self):
        t = self.timer
        t.begin_forward(4096)
        t.record_moe_checkpoint("moe_start", 0)
        pt._STRICT = True
        try:
            with self.assertRaises(RuntimeError):
                t.record_moe_checkpoint("gemm_done")
        finally:
            pt._STRICT = False
            t.end_forward(success=False)

    def test_unknown_checkpoint_name_raises(self):
        t = self.timer
        t.begin_forward(4096)
        with self.assertRaises(ValueError):
            t.record_moe_checkpoint("nope")
        with self.assertRaises(ValueError):
            t.record_dispatch_split("nope")
        t.end_forward(success=False)

    # ---------------------------------------------------------- dispatch split

    def test_dispatch_split_is_additive(self):
        self.timer.arm(dispatch_split=True)
        self._record_forward(2, tokens=4096, split=True)
        pl = self.timer.per_layer()
        for i in range(2):
            split = pl["dispatch_split"][i]
            self.assertTrue(split, "dispatch split not recorded")
            total = sum(split[n] for n in pt.DISPATCH_SUB_PHASE_NAMES)
            self.assertAlmostEqual(total, pl["dispatch"][i], places=2)

    def test_report_excludes_absolute_offsets(self):
        # *_from_epoch offsets are per-layer timestamps; summing them is
        # meaningless, so they must not leak into the aggregate report.
        self.timer.arm(dispatch_split=True)
        self._record_forward(2, tokens=4096, split=True)
        rep = self.timer.report()
        self.assertTrue(any(k.startswith("dispatch_quant") for k in rep))
        for key in rep:
            self.assertNotIn("_from_epoch", key)
        # ...but they are still exported per layer for cross-rank correction
        split = self.timer.per_layer()["dispatch_split"][0]
        self.assertIn("quant_done_from_epoch", split)

    def test_dispatch_split_off_by_default(self):
        self._record_forward(1, tokens=4096, split=True)
        pl = self.timer.per_layer()
        self.assertEqual(pl["dispatch_split"][0], {})

    # ------------------------------------------------------------- route split

    def test_route_split_is_additive(self):
        self.timer.arm(route_split=True)
        self._record_forward(2, tokens=4096, route=True)
        pl = self.timer.per_layer()
        for i in range(2):
            split = pl["route_split"][i]
            self.assertTrue(split, "route split not recorded")
            total = sum(split[n] for n in pt.ROUTE_SUB_PHASE_NAMES)
            self.assertAlmostEqual(total, pl["gate_route"][i], places=2)
            self.assertGreater(split["route_policy"], 0.0)

    def test_route_split_off_by_default(self):
        self._record_forward(1, tokens=4096, route=True)
        self.assertEqual(self.timer.per_layer()["route_split"][0], {})

    def test_route_split_is_independent_of_dispatch_split(self):
        # The two channels must not interfere: enabling one alone still yields a
        # complete, additive decomposition for that one.
        self.timer.arm(route_split=True, dispatch_split=False)
        self._record_forward(1, tokens=4096, route=True, split=True)
        pl = self.timer.per_layer()
        self.assertTrue(pl["route_split"][0])
        self.assertEqual(pl["dispatch_split"][0], {})
        rep = self.timer.report()
        self.assertIn("route_policy_ms", rep)
        self.assertNotIn("dispatch_quant_ms", rep)

    def test_unknown_route_checkpoint_raises(self):
        t = self.timer
        t.begin_forward(4096)
        with self.assertRaises(ValueError):
            t.record_route_split("nope")
        t.end_forward(success=False)

    # ------------------------------------------------------------- GEMM split

    def test_gemm_split_is_additive(self):
        self.timer.arm(gemm_split=True)
        self._record_forward(2, tokens=4096, gemm=True)
        pl = self.timer.per_layer()
        for i in range(2):
            split = pl["gemm_split"][i]
            self.assertTrue(split, "GEMM split not recorded")
            total = sum(split[n] for n in pt.GEMM_SUB_PHASE_NAMES)
            self.assertAlmostEqual(total, pl["gemm"][i], places=2)
            self.assertGreater(split["gemm_kernels"], 0.0)

    def test_gemm_split_off_by_default(self):
        self._record_forward(1, tokens=4096, gemm=True)
        self.assertEqual(self.timer.per_layer()["gemm_split"][0], {})

    def test_gemm_split_is_independent(self):
        self.timer.arm(
            route_split=False, dispatch_split=False, gemm_split=True
        )
        self._record_forward(
            1, tokens=4096, route=True, split=True, gemm=True
        )
        pl = self.timer.per_layer()
        self.assertEqual(pl["route_split"][0], {})
        self.assertEqual(pl["dispatch_split"][0], {})
        self.assertTrue(pl["gemm_split"][0])
        rep = self.timer.report()
        self.assertIn("gemm_kernels_ms", rep)
        self.assertNotIn("route_policy_ms", rep)
        self.assertNotIn("dispatch_quant_ms", rep)

    def test_unknown_gemm_checkpoint_raises(self):
        t = self.timer
        t.begin_forward(4096)
        with self.assertRaises(ValueError):
            t.record_gemm_split("nope")
        t.end_forward(success=False)

    # ------------------------------------------------------------ flush/export

    def test_flush_resets_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            pt._EXPORT_DIR = tmp
            try:
                self._record_forward(2, tokens=4096)
                self.timer.flush(rank=3, tag="unittest")
            finally:
                pt._EXPORT_DIR = ""
            files = sorted(os.listdir(tmp))
            self.assertEqual(len(files), 1)
            self.assertIn("rank003", files[0])
            with open(os.path.join(tmp, files[0])) as f:
                payload = json.load(f)
            self.assertEqual(payload["schema"], "rtp_llm.phase_timer/v2")
            self.assertEqual(payload["rank"], 3)
            self.assertEqual(payload["tag"], "unittest")
            # arm identity must travel with the numbers
            self.assertIn("dispatch_policy", payload["meta"])
            self.assertIn("placement", payload["meta"])
            self.assertEqual(len(payload["forwards"]), 1)
            self.assertEqual(payload["forwards"][0]["layer_index"], [0, 1])
            self.assertGreater(payload["report"]["gemm_ms"], 0.0)
        # flush must clear state so events do not accumulate across reports
        self.assertEqual(self.timer.report(), {})

    def test_flush_accumulates_until_report_step(self):
        pt._REPORT_STEP = 3
        try:
            for i in range(2):
                self._record_forward(1, tokens=4096)
                self.timer.flush(rank=0)
                self.assertEqual(self.timer.report()["forwards"], i + 1)
            self._record_forward(1, tokens=4096)
            self.timer.flush(rank=0)
            self.assertEqual(self.timer.report(), {})
        finally:
            pt._REPORT_STEP = 1

    def test_singleton(self):
        self.assertIs(pt.get_phase_timer(), pt.get_phase_timer())


if __name__ == "__main__":
    unittest.main()
