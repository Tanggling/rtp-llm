import os
import unittest

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    _fill_k,
    _largest_remainder,
    wf_k_dispatch,
)


class FillKTest(unittest.TestCase):
    def test_fill_two_matches_closed_form(self):
        cases = [([0, 10], 5), ([0, 10], 20), ([12, 3], 17), ([4, 4], 9)]
        for loads, demand in cases:
            s0, s1 = loads
            gap = abs(s0 - s1)
            level = (s0 + s1 + demand) * 0.5
            if demand > gap:
                expected = np.array([level - s0, level - s1])
            elif s0 <= s1:
                expected = np.array([min(demand, gap), 0.0])
            else:
                expected = np.array([0.0, min(demand, gap)])
            np.testing.assert_allclose(_fill_k(loads, demand), expected)

    def test_fill_arbitrary_k_equalizes_final_load(self):
        loads = np.array([0.0, 10.0, 20.0, 30.0])
        allocation = _fill_k(loads, 100)
        np.testing.assert_allclose(loads + allocation, np.full(4, 40.0))
        self.assertAlmostEqual(float(allocation.sum()), 100.0)

    def test_largest_remainder_conserves_total(self):
        rounded = _largest_remainder([1.2, 2.7, 3.1], 7)
        self.assertEqual(int(rounded.sum()), 7)
        np.testing.assert_array_equal(rounded, [1, 3, 3])


class WfKDispatchTest(unittest.TestCase):
    def test_k_one_is_identity(self):
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        l2p = torch.tensor([[2], [5]], dtype=torch.int32)
        count = torch.ones(2, dtype=torch.int32)
        got = wf_k_dispatch(ids, l2p, count, num_gpus=2, num_physical=6)
        expected = torch.tensor([[2, 5], [5, 2]], dtype=torch.int32)
        self.assertTrue(torch.equal(got, expected))

    def test_all_replicas_are_usable_and_demand_is_conserved(self):
        ids = torch.zeros((120, 1), dtype=torch.int32)
        l2p = torch.tensor([[0, 2, 4, 6]], dtype=torch.int32)
        count = torch.tensor([4], dtype=torch.int32)
        got = wf_k_dispatch(ids, l2p, count, num_gpus=4, num_physical=8)
        physical_counts = torch.bincount(got.flatten().long(), minlength=8)
        self.assertEqual(int(physical_counts.sum()), 120)
        self.assertTrue(
            torch.equal(physical_counts[[0, 2, 4, 6]], torch.full((4,), 30))
        )

    def test_multiple_replicas_on_same_gpu(self):
        ids = torch.zeros((60, 1), dtype=torch.int32)
        l2p = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        count = torch.tensor([3], dtype=torch.int32)
        got = wf_k_dispatch(ids, l2p, count, num_gpus=2, num_physical=4)
        physical_counts = torch.bincount(got.flatten().long(), minlength=4)
        gpu_counts = physical_counts.view(2, 2).sum(1)
        self.assertTrue(torch.equal(gpu_counts, torch.tensor([30, 30])))
        self.assertEqual(int(physical_counts[0]), 15)
        self.assertEqual(int(physical_counts[1]), 15)

    def test_deterministic_for_mixed_experts(self):
        ids = torch.tensor([[0, 1, 2, 0, 2, 1, 0, 2]] * 20, dtype=torch.int32)
        l2p = torch.tensor([[0, 3, 6], [1, 4, -1], [2, 5, 7]], dtype=torch.int32)
        count = torch.tensor([3, 2, 3], dtype=torch.int32)
        first = wf_k_dispatch(ids, l2p, count, num_gpus=4, num_physical=8)
        second = wf_k_dispatch(ids, l2p, count, num_gpus=4, num_physical=8)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), ids.numel())
        self.assertTrue(bool(((first >= 0) & (first < 8)).all()))


if __name__ == "__main__":
    unittest.main()
