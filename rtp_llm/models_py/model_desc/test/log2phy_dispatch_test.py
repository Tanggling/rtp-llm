import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer


class FakeGate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros(
            hidden_states.shape[0], 2, dtype=torch.float32, device=hidden_states.device
        )


class FakeSelectTopk(nn.Module):
    def forward(self, router_logits, topk_ids, topk_weights):
        topk_ids.copy_(torch.tensor([[0, 0], [1, 1]], dtype=topk_ids.dtype))
        topk_weights.fill_(0.5)


class CapturingFusedMoe(nn.Module):
    topk_ids_dtype = torch.int32

    def __init__(self):
        super().__init__()
        self.dispatched_ids = []

    def forward(self, hidden_states, topk_weights, topk_ids, activation):
        self.dispatched_ids.append(topk_ids.clone())
        return torch.zeros_like(hidden_states)


class Log2PhyEndToEndTest(unittest.TestCase):
    def test_updated_mapping_reaches_fused_moe_dispatch(self):
        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        layer.gate = FakeGate()
        layer.select_topk = FakeSelectTopk()
        layer.fused_moe = CapturingFusedMoe()
        layer.top_k = 2
        layer.correction_bias = None
        layer.fake_balance_expert = None
        layer.expert_stats = None
        layer.shared_expert = None
        layer.shared_expert_gate = None
        layer.ffn_tp_size = 1
        layer.ep_size = 2
        layer.num_experts = 4
        layer.parallelism_config = SimpleNamespace(dp_rank=0)
        layer.log2phy = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
        layer.logic_expert_cnt = torch.tensor([2, 2], dtype=torch.int32)

        hidden_states = torch.zeros(2, 4)
        layer(hidden_states)
        self.assertTrue(
            torch.equal(
                layer.fused_moe.dispatched_ids[-1],
                torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
            )
        )

        # Mock a new plan without invoking EPLB plan generation or weight loading.
        layer.log2phy.copy_(torch.tensor([[2, 0], [3, 1]], dtype=torch.int32))
        layer(hidden_states)
        self.assertTrue(
            torch.equal(
                layer.fused_moe.dispatched_ids[-1],
                torch.tensor([[2, 0], [3, 1]], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()
