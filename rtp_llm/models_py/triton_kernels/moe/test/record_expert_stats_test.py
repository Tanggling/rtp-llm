import os
import unittest

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import record_expert_stats


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


if __name__ == "__main__":
    unittest.main()
