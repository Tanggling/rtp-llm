import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.eplb.plan_policies import build_circulant_plan
from rtp_llm.model_loader.load_config import LoadConfig


class LoadConfigPlacementTest(unittest.TestCase):
    def test_circulant_layout_is_used_for_every_layer(self):
        expected, _, counts = build_circulant_plan(16, 24, 4)
        phy2log = LoadConfig.create_redundant_expert(
            layer_num=3,
            expert_num=16,
            phy_exp_num=24,
            ep_size=4,
            num_nodes=1,
            placement_policy="circulant",
        )

        self.assertEqual(phy2log, [expected.tolist()] * 3)
        for layer in phy2log:
            histogram = torch.bincount(torch.tensor(layer), minlength=16)
            self.assertTrue(torch.equal(histogram, torch.from_numpy(counts)))
            for rank in range(4):
                rank_slice = layer[rank * 6 : (rank + 1) * 6]
                self.assertEqual(len(rank_slice), 6)

    def test_no_redundancy_circulant_is_valid(self):
        phy2log = LoadConfig.create_redundant_expert(
            layer_num=1,
            expert_num=8,
            phy_exp_num=8,
            ep_size=4,
            num_nodes=1,
            placement_policy="circulant",
        )[0]
        self.assertEqual(sorted(phy2log), list(range(8)))

    def test_explicit_layout_file_takes_precedence(self):
        layout = [list(reversed(range(8)))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phy2log.json"
            path.write_text(json.dumps(layout))
            actual = LoadConfig.create_redundant_expert(
                layer_num=1,
                expert_num=8,
                phy_exp_num=8,
                ep_size=4,
                num_nodes=1,
                phy2log_path=str(path),
                placement_policy="circulant",
            )
        self.assertEqual(actual, layout)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported EPLB placement policy"):
            LoadConfig.create_redundant_expert(
                layer_num=1,
                expert_num=8,
                phy_exp_num=8,
                ep_size=4,
                num_nodes=1,
                placement_policy="unknown",
            )

    def test_default_layout_is_unchanged(self):
        actual = LoadConfig.create_redundant_expert(
            layer_num=1,
            expert_num=8,
            phy_exp_num=12,
            ep_size=4,
            num_nodes=1,
        )[0]
        self.assertEqual(actual, [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0])


if __name__ == "__main__":
    unittest.main()
