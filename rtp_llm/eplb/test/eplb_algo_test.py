import itertools
import unittest

import torch

from rtp_llm.eplb.eplb import (
    balanced_packing,
    inverse,
    rebalance_experts,
    rebalance_experts_hierarchical,
    replicate_experts,
)


def check_plan_invariants(
    testcase: unittest.TestCase,
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    num_layers: int,
    num_log: int,
    num_replicas: int,
):
    """Formal invariants every valid EPLB plan must satisfy.

    I1 (shape):        phy2log [L, P], logcnt [L, N], log2phy [L, N, maxcnt]
    I2 (surjective):   every logical expert has >= 1 physical replica
    I3 (conservation): logcnt sums to num_replicas per layer, and matches
                       the histogram of phy2log
    I4 (consistency):  log2phy and phy2log are mutually inverse mappings
    I5 (range):        all physical ids in [0, P), all logical ids in [0, N)
    """
    testcase.assertEqual(tuple(phy2log.shape), (num_layers, num_replicas))
    testcase.assertEqual(tuple(logcnt.shape), (num_layers, num_log))
    testcase.assertEqual(log2phy.shape[0], num_layers)
    testcase.assertEqual(log2phy.shape[1], num_log)

    # I5: ranges
    testcase.assertTrue(bool((phy2log >= 0).all()))
    testcase.assertTrue(bool((phy2log < num_log).all()))

    for layer in range(num_layers):
        hist = torch.bincount(phy2log[layer], minlength=num_log)
        # I2: surjective
        testcase.assertTrue(
            bool((hist >= 1).all()),
            f"layer {layer}: some logical expert has no physical replica",
        )
        # I3: conservation
        testcase.assertTrue(
            torch.equal(hist, logcnt[layer]),
            f"layer {layer}: logcnt does not match phy2log histogram",
        )
        testcase.assertEqual(int(logcnt[layer].sum()), num_replicas)

        # I4: log2phy consistency with phy2log
        for log_id in range(num_log):
            phys = log2phy[layer, log_id]
            valid = phys[phys >= 0]
            testcase.assertEqual(
                valid.numel(),
                int(logcnt[layer, log_id]),
                f"layer {layer}, expert {log_id}: replica count mismatch",
            )
            for phy_id in valid.tolist():
                testcase.assertTrue(0 <= phy_id < num_replicas)
                testcase.assertEqual(
                    int(phy2log[layer, phy_id]),
                    log_id,
                    f"layer {layer}: log2phy[{log_id}] -> {phy_id} "
                    f"but phy2log[{phy_id}] = {int(phy2log[layer, phy_id])}",
                )


class BalancedPackingTest(unittest.TestCase):
    def check_invariants(self, weight: torch.Tensor, num_packs: int):
        pack_index, rank_in_pack = balanced_packing(weight, num_packs)
        num_layers, num_groups = weight.shape
        groups_per_pack = num_groups // num_packs

        self.assertTrue(bool((pack_index >= 0).all()))
        self.assertTrue(bool((pack_index < num_packs).all()))

        for layer in range(num_layers):
            # each pack holds exactly n/m items, ranks form 0..groups_per_pack-1
            for pack in range(num_packs):
                members = (pack_index[layer] == pack).nonzero().flatten()
                self.assertEqual(members.numel(), groups_per_pack)
                ranks = sorted(rank_in_pack[layer, members].tolist())
                self.assertEqual(ranks, list(range(groups_per_pack)))

    def test_identity_when_one_group_per_pack(self):
        weight = torch.tensor([[5, 1, 3, 7]])
        pack_index, rank_in_pack = balanced_packing(weight, 4)
        self.assertTrue(torch.equal(pack_index, torch.tensor([[0, 1, 2, 3]])))
        self.assertTrue(torch.equal(rank_in_pack, torch.zeros(1, 4, dtype=torch.int64)))

    def test_exhaustive_small_space(self):
        # bounded model check: all weight assignments of 4 items over values 0..3,
        # packed into 2 packs
        for values in itertools.product(range(4), repeat=4):
            weight = torch.tensor([values], dtype=torch.int64)
            self.check_invariants(weight, 2)

    def test_randomized_invariants(self):
        gen = torch.Generator().manual_seed(0)
        for _ in range(200):
            num_packs = int(torch.randint(1, 5, (1,), generator=gen))
            groups_per_pack = int(torch.randint(1, 5, (1,), generator=gen))
            num_layers = int(torch.randint(1, 4, (1,), generator=gen))
            weight = torch.randint(
                0, 1000, (num_layers, num_packs * groups_per_pack), generator=gen
            )
            self.check_invariants(weight, num_packs)

    def test_balance_quality_two_packs(self):
        # greedy LPT bound: for 2 packs the heavier pack never exceeds
        # total/2 + max_item
        gen = torch.Generator().manual_seed(1)
        for _ in range(100):
            weight = torch.randint(0, 100, (1, 8), generator=gen)
            pack_index, _ = balanced_packing(weight, 2)
            load0 = int(weight[0][pack_index[0] == 0].sum())
            load1 = int(weight[0][pack_index[0] == 1].sum())
            total = int(weight.sum())
            self.assertLessEqual(max(load0, load1), total / 2 + int(weight.max()))


class ReplicateExpertsTest(unittest.TestCase):
    def check_invariants(self, weight: torch.Tensor, num_phy: int):
        num_layers, num_log = weight.shape
        phy2log, rank, logcnt = replicate_experts(weight, num_phy)

        self.assertTrue(bool((phy2log >= 0).all()))
        self.assertTrue(bool((phy2log < num_log).all()))

        for layer in range(num_layers):
            hist = torch.bincount(phy2log[layer], minlength=num_log)
            self.assertTrue(torch.equal(hist, logcnt[layer]))
            self.assertTrue(bool((logcnt[layer] >= 1).all()))
            self.assertEqual(int(logcnt[layer].sum()), num_phy)
            # replica ranks of each logical expert form 0..cnt-1
            for log_id in range(num_log):
                replica_ranks = sorted(rank[layer][phy2log[layer] == log_id].tolist())
                self.assertEqual(replica_ranks, list(range(int(hist[log_id]))))

    def test_no_redundancy_is_identity(self):
        weight = torch.tensor([[3, 1, 4, 1]])
        phy2log, rank, logcnt = replicate_experts(weight, 4)
        self.assertTrue(torch.equal(phy2log, torch.tensor([[0, 1, 2, 3]])))
        self.assertTrue(torch.equal(logcnt, torch.ones(1, 4, dtype=torch.int64)))

    def test_replicates_heaviest_expert_first(self):
        weight = torch.tensor([[100, 1, 1, 1]])
        phy2log, _, logcnt = replicate_experts(weight, 5)
        self.assertEqual(int(phy2log[0, 4]), 0)
        self.assertEqual(int(logcnt[0, 0]), 2)

    def test_exhaustive_small_space(self):
        # bounded model check: 3 logical experts, weights in 0..2, up to 3 redundant
        for values in itertools.product(range(3), repeat=3):
            weight = torch.tensor([values], dtype=torch.int64)
            for num_phy in range(3, 7):
                self.check_invariants(weight, num_phy)

    def test_randomized_invariants(self):
        gen = torch.Generator().manual_seed(2)
        for _ in range(200):
            num_log = int(torch.randint(1, 9, (1,), generator=gen))
            num_redundant = int(torch.randint(0, 5, (1,), generator=gen))
            num_layers = int(torch.randint(1, 4, (1,), generator=gen))
            weight = torch.randint(0, 1000, (num_layers, num_log), generator=gen)
            self.check_invariants(weight, num_log + num_redundant)


class InverseTest(unittest.TestCase):
    def test_inverse_of_permutation(self):
        gen = torch.Generator().manual_seed(3)
        for _ in range(50):
            n = int(torch.randint(1, 17, (1,), generator=gen))
            perm = torch.stack([torch.randperm(n, generator=gen) for _ in range(3)])
            inv = inverse(perm)
            identity = torch.arange(n).expand(3, n)
            self.assertTrue(torch.equal(torch.gather(perm, 1, inv), identity))
            self.assertTrue(torch.equal(torch.gather(inv, 1, perm), identity))


class RebalanceExpertsTest(unittest.TestCase):
    def test_global_policy_invariants(self):
        # num_groups % num_nodes != 0 -> global policy
        gen = torch.Generator().manual_seed(4)
        for _ in range(50):
            num_log = 8
            num_replicas = 12
            weight = torch.randint(0, 1000, (2, num_log), generator=gen)
            phy2log, log2phy, logcnt = rebalance_experts(
                weight, num_replicas, num_groups=3, num_nodes=2, num_gpus=4
            )
            check_plan_invariants(
                self, phy2log, log2phy, logcnt, 2, num_log, num_replicas
            )

    def test_hierarchical_policy_invariants(self):
        gen = torch.Generator().manual_seed(5)
        for _ in range(50):
            num_log = 16
            num_replicas = 24
            weight = torch.randint(0, 1000, (2, num_log), generator=gen)
            phy2log, log2phy, logcnt = rebalance_experts(
                weight, num_replicas, num_groups=4, num_nodes=2, num_gpus=4
            )
            check_plan_invariants(
                self, phy2log, log2phy, logcnt, 2, num_log, num_replicas
            )

    def test_force_repack_invariants(self):
        gen = torch.Generator().manual_seed(6)
        for _ in range(50):
            num_log = 8
            num_replicas = 12
            weight = torch.randint(0, 1000, (1, num_log), generator=gen)
            phy2log, log2phy, logcnt = rebalance_experts(
                weight,
                num_replicas,
                num_groups=3,
                num_nodes=2,
                num_gpus=4,
                force_repack=True,
            )
            check_plan_invariants(
                self, phy2log, log2phy, logcnt, 1, num_log, num_replicas
            )

    def test_exhaustive_tiny_space(self):
        # bounded model check over every workload of 4 experts with loads 0..2,
        # hierarchical path (2 groups over 1 node, 2 gpus, 2 redundant experts)
        for values in itertools.product(range(3), repeat=4):
            weight = torch.tensor([values], dtype=torch.int64)
            phy2log, log2phy, logcnt = rebalance_experts(
                weight, 6, num_groups=2, num_nodes=1, num_gpus=2
            )
            check_plan_invariants(self, phy2log, log2phy, logcnt, 1, 4, 6)

    def test_uniform_load_yields_uniform_replication(self):
        weight = torch.full((1, 8), 10)
        phy2log, log2phy, logcnt = rebalance_experts(
            weight, 16, num_groups=8, num_nodes=2, num_gpus=4
        )
        check_plan_invariants(self, phy2log, log2phy, logcnt, 1, 8, 16)
        self.assertTrue(bool((logcnt == 2).all()))

    def test_hierarchical_matches_rebalance_dispatch(self):
        weight = torch.randint(
            0, 100, (2, 12), generator=torch.Generator().manual_seed(7)
        )
        phy2log_h, _, logcnt_h = rebalance_experts_hierarchical(
            weight.float(), 18, num_groups=4, num_nodes=2, num_gpus=6
        )
        phy2log, _, logcnt = rebalance_experts(
            weight, 18, num_groups=4, num_nodes=2, num_gpus=6
        )
        self.assertTrue(torch.equal(phy2log_h, phy2log))
        self.assertTrue(torch.equal(logcnt_h, logcnt))


if __name__ == "__main__":
    unittest.main()
