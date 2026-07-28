import os
import unittest
from unittest import mock

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

from rtp_llm.models_py.triton_kernels.moe import ep_kernels
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    _record_expert_stats_torch,
    logical_to_physical_experts,
    record_expert_stats,
)


def reference_expert_stats(
    topk_ids: torch.Tensor,
    layer_num: int,
    log_exp_num: int,
    ep_size: int,
    layer_idx: int,
):
    """Pure-torch reference of the triton kernel semantics."""
    log_stats = torch.zeros(layer_num, log_exp_num, dtype=torch.int32)
    gpu_loads = torch.zeros(layer_num, ep_size, dtype=torch.int32)
    experts_per_rank = -(-log_exp_num // ep_size)  # ceil_div
    for expert_id in topk_ids.flatten().tolist():
        if expert_id < 0 or expert_id >= log_exp_num:
            continue
        log_stats[layer_idx, expert_id] += 1
        gpu_loads[layer_idx, min(expert_id // experts_per_rank, ep_size - 1)] += 1
    return log_stats, gpu_loads


def run_kernel(
    topk_ids: torch.Tensor,
    layer_num: int,
    log_exp_num: int,
    ep_size: int,
    layer_idx: int,
    log_stats: torch.Tensor = None,
    gpu_loads: torch.Tensor = None,
):
    if log_stats is None:
        log_stats = torch.zeros(layer_num, log_exp_num, dtype=torch.int32)
    if gpu_loads is None:
        gpu_loads = torch.zeros(layer_num, ep_size, dtype=torch.int32)
    record_expert_stats(topk_ids, log_stats, gpu_loads, layer_idx)
    return log_stats, gpu_loads


class RecordExpertStatsTest(unittest.TestCase):
    """Runs the triton kernel on CPU via TRITON_INTERPRET=1 and compares
    against a pure-torch reference."""

    def check(self, topk_ids, layer_num, log_exp_num, ep_size, layer_idx):
        got_log, got_gpu = run_kernel(
            topk_ids, layer_num, log_exp_num, ep_size, layer_idx
        )
        want_log, want_gpu = reference_expert_stats(
            topk_ids, layer_num, log_exp_num, ep_size, layer_idx
        )
        self.assertTrue(
            torch.equal(got_log, want_log),
            f"log_stats mismatch:\n got {got_log}\nwant {want_log}",
        )
        self.assertTrue(
            torch.equal(got_gpu, want_gpu),
            f"gpu_loads mismatch:\n got {got_gpu}\nwant {want_gpu}",
        )

    def test_basic_counting(self):
        topk_ids = torch.tensor([[0, 1], [1, 3], [3, 3]], dtype=torch.int32)
        self.check(topk_ids, layer_num=2, log_exp_num=4, ep_size=2, layer_idx=0)

    def test_layer_offset(self):
        topk_ids = torch.tensor([[2, 5, 7]], dtype=torch.int32)
        got_log, got_gpu = run_kernel(
            topk_ids, layer_num=4, log_exp_num=8, ep_size=4, layer_idx=2
        )
        # only the target layer row is touched
        self.assertEqual(int(got_log.sum()), 3)
        self.assertEqual(int(got_log[2].sum()), 3)
        self.assertEqual(int(got_gpu.sum()), 3)
        self.assertEqual(int(got_gpu[2].sum()), 3)

    def test_int64_topk_ids(self):
        topk_ids = torch.tensor([[0, 63], [17, 42]], dtype=torch.int64)
        self.check(topk_ids, layer_num=1, log_exp_num=64, ep_size=8, layer_idx=0)

    def test_accumulates_on_nonzero_buffers(self):
        topk_ids = torch.tensor([[1, 1]], dtype=torch.int32)
        log_stats = torch.full((1, 4), 5, dtype=torch.int32)
        gpu_loads = torch.full((1, 2), 7, dtype=torch.int32)
        run_kernel(topk_ids, 1, 4, 2, 0, log_stats=log_stats, gpu_loads=gpu_loads)
        self.assertEqual(int(log_stats[0, 1]), 7)
        self.assertEqual(int(log_stats[0, 0]), 5)
        self.assertEqual(int(gpu_loads[0, 0]), 9)
        self.assertEqual(int(gpu_loads[0, 1]), 7)

    def test_out_of_range_ids_are_ignored(self):
        topk_ids = torch.tensor([[-1, 4, 100, 2]], dtype=torch.int32)
        got_log, got_gpu = run_kernel(
            topk_ids, layer_num=1, log_exp_num=4, ep_size=2, layer_idx=0
        )
        self.assertEqual(int(got_log.sum()), 1)
        self.assertEqual(int(got_log[0, 2]), 1)
        self.assertEqual(int(got_gpu.sum()), 1)
        self.assertEqual(int(got_gpu[0, 1]), 1)

    def test_empty_input_is_noop(self):
        topk_ids = torch.empty(0, 8, dtype=torch.int32)
        got_log, got_gpu = run_kernel(
            topk_ids, layer_num=1, log_exp_num=8, ep_size=2, layer_idx=0
        )
        self.assertEqual(int(got_log.sum()), 0)
        self.assertEqual(int(got_gpu.sum()), 0)

    def test_randomized_against_reference(self):
        gen = torch.Generator().manual_seed(0)
        for _ in range(20):
            num_tokens = int(torch.randint(1, 64, (1,), generator=gen))
            top_k = int(torch.randint(1, 9, (1,), generator=gen))
            log_exp_num = int(torch.randint(2, 65, (1,), generator=gen))
            ep_size = int(torch.randint(1, 9, (1,), generator=gen))
            layer_num = int(torch.randint(1, 5, (1,), generator=gen))
            layer_idx = int(torch.randint(0, layer_num, (1,), generator=gen))
            topk_ids = torch.randint(
                0, log_exp_num, (num_tokens, top_k), generator=gen, dtype=torch.int32
            )
            self.check(topk_ids, layer_num, log_exp_num, ep_size, layer_idx)

    def test_ep_rank_partitioning_uneven(self):
        # 10 experts over 4 ranks: experts_per_rank = 3, expert 9 -> rank 3
        topk_ids = torch.tensor([[0, 3, 6, 9]], dtype=torch.int32)
        got_log, got_gpu = run_kernel(
            topk_ids, layer_num=1, log_exp_num=10, ep_size=4, layer_idx=0
        )
        self.assertTrue(
            torch.equal(got_gpu, torch.tensor([[1, 1, 1, 1]], dtype=torch.int32))
        )
        self.assertEqual(int(got_log.sum()), 4)


class TorchFallbackTest(unittest.TestCase):
    """Covers the pure-torch implementation used when triton is unavailable
    (e.g. PPU without a working nvidia backend)."""

    def check_torch(self, topk_ids, layer_num, log_exp_num, ep_size, layer_idx):
        got_log = torch.zeros(layer_num, log_exp_num, dtype=torch.int32)
        got_gpu = torch.zeros(layer_num, ep_size, dtype=torch.int32)
        _record_expert_stats_torch(
            topk_ids, topk_ids, got_log, got_gpu, layer_idx, log_exp_num
        )
        want_log, want_gpu = reference_expert_stats(
            topk_ids, layer_num, log_exp_num, ep_size, layer_idx
        )
        self.assertTrue(torch.equal(got_log, want_log))
        self.assertTrue(torch.equal(got_gpu, want_gpu))

    def test_basic_counting(self):
        topk_ids = torch.tensor([[0, 1], [1, 3], [3, 3]], dtype=torch.int32)
        self.check_torch(topk_ids, layer_num=2, log_exp_num=4, ep_size=2, layer_idx=1)

    def test_out_of_range_and_uneven_ranks(self):
        topk_ids = torch.tensor([[-1, 100, 9, 0]], dtype=torch.int64)
        self.check_torch(topk_ids, layer_num=1, log_exp_num=10, ep_size=4, layer_idx=0)

    def test_randomized_against_reference(self):
        gen = torch.Generator().manual_seed(1)
        for _ in range(20):
            num_tokens = int(torch.randint(1, 64, (1,), generator=gen))
            top_k = int(torch.randint(1, 9, (1,), generator=gen))
            log_exp_num = int(torch.randint(2, 65, (1,), generator=gen))
            ep_size = int(torch.randint(1, 9, (1,), generator=gen))
            layer_num = int(torch.randint(1, 5, (1,), generator=gen))
            layer_idx = int(torch.randint(0, layer_num, (1,), generator=gen))
            topk_ids = torch.randint(
                0, log_exp_num, (num_tokens, top_k), generator=gen, dtype=torch.int32
            )
            self.check_torch(topk_ids, layer_num, log_exp_num, ep_size, layer_idx)

    def test_wrapper_falls_back_when_triton_launch_fails(self):
        class BrokenKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    raise RuntimeError("simulated triton backend failure")

                return launch

        topk_ids = torch.tensor([[0, 1, 1]], dtype=torch.int32)
        log_stats = torch.zeros(1, 4, dtype=torch.int32)
        gpu_loads = torch.zeros(1, 2, dtype=torch.int32)
        with mock.patch.object(
            ep_kernels, "_record_expert_stats_kernel", BrokenKernel()
        ), mock.patch.object(ep_kernels, "_record_expert_stats_use_triton", True):
            record_expert_stats(topk_ids, log_stats, gpu_loads, 0)
            # fallback decision is cached after the first failure
            self.assertFalse(ep_kernels._record_expert_stats_use_triton)
            record_expert_stats(topk_ids, log_stats, gpu_loads, 0)

        self.assertTrue(
            torch.equal(log_stats, torch.tensor([[2, 4, 0, 0]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(gpu_loads, torch.tensor([[6, 0]], dtype=torch.int32))
        )


class ExpertStatsWiringTest(unittest.TestCase):
    """Checks GptModelBase.initialize picks up expert_stats from init resources."""

    def _import_base(self):
        try:
            from rtp_llm.models_py.model_desc.module_base import GptModelBase
        except ImportError as e:
            self.skipTest(f"module_base unavailable in this environment: {e}")
        return GptModelBase

    def test_initialize_receives_expert_stats(self):
        GptModelBase = self._import_base()

        class FakeStats:
            log_stats_buf = torch.zeros(2, 4, dtype=torch.int32)
            gpu_loads_buf = torch.zeros(2, 2, dtype=torch.int32)

        class FakeInitResource:
            kv_cache = None
            expert_stats = FakeStats()

        model = GptModelBase.__new__(GptModelBase)
        self.assertTrue(GptModelBase.initialize(model, FakeInitResource()))
        self.assertIs(model.expert_stats, FakeInitResource.expert_stats)

    def test_initialize_without_expert_stats(self):
        GptModelBase = self._import_base()

        class FakeInitResource:
            kv_cache = None

        model = GptModelBase.__new__(GptModelBase)
        self.assertTrue(GptModelBase.initialize(model, FakeInitResource()))
        self.assertIsNone(model.expert_stats)


class Log2PhyDispatchTest(unittest.TestCase):
    def test_mock_plan_update_is_applied_to_next_dispatch(self):
        logical_ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
        logic_expert_cnt = torch.tensor([2, 2], dtype=torch.int32)
        log2phy = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)

        first = logical_to_physical_experts(logical_ids, log2phy, logic_expert_cnt)
        self.assertTrue(
            torch.equal(first, torch.tensor([[0, 2], [1, 3]], dtype=torch.int32))
        )

        # Mimic ExpertBalancer::applyPlanWeights: update the existing tensor
        # object so a layer that already captured it sees the new plan.
        captured_mapping = log2phy
        mock_plan = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
        log2phy.copy_(mock_plan)
        self.assertIs(captured_mapping, log2phy)

        second = logical_to_physical_experts(
            logical_ids, captured_mapping, logic_expert_cnt
        )
        self.assertTrue(
            torch.equal(second, torch.tensor([[2, 0], [3, 1]], dtype=torch.int32))
        )
        self.assertFalse(torch.equal(first, second))

    def test_stats_use_logical_heat_and_physical_ep_placement(self):
        logical_ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.int32)
        physical_ids = torch.tensor([[0, 4], [1, 5]], dtype=torch.int32)
        log_stats = torch.zeros(1, 2, dtype=torch.int32)
        gpu_loads = torch.zeros(1, 2, dtype=torch.int32)

        _record_expert_stats_torch(
            logical_ids,
            physical_ids,
            log_stats,
            gpu_loads,
            layer_idx=0,
            phy_exp_num=6,
        )

        self.assertTrue(
            torch.equal(log_stats, torch.tensor([[2, 2]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(gpu_loads, torch.tensor([[2, 2]], dtype=torch.int32))
        )


if __name__ == "__main__":
    unittest.main()
