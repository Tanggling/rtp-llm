import logging
import os
from typing import Any, Dict, Optional

import torch
import triton.language as tl

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.models_py.utils.phase_timer import get_phase_timer
from rtp_llm.utils.model_weight import W


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k

        # w1: [E, 2*inter_size, hidden_size], w2: [E, hidden_size, inter_size]
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.E = self.w1.size(0)
        self.N = self.w1.size(1)  # 2 * inter_size
        self.K = self.w1.size(2)  # hidden_size
        self.inter_size = self.N // 2

        # Under EP this executor sees every token but owns only E/ep_size experts.
        # By default the rows routed elsewhere are clamped into the local range and
        # computed anyway (their routed weight is zeroed), which makes the GEMM cost
        # ep_size times the useful work and independent of the routing distribution.
        # With MOE_EP_SKIP_NONLOCAL=1 those rows are dropped before the GEMM, so the
        # cost tracks the tokens actually routed to this rank's experts.
        self.skip_nonlocal_rows = os.environ.get("MOE_EP_SKIP_NONLOCAL", "0") == "1"
        # align / kernels / tail split inside the T2..T3 window; no-op when off
        self._phase_timer = get_phase_timer()
        if self.skip_nonlocal_rows and self.ep_size > 1:
            logging.info(
                "TritonFusedMoeExecutor: MOE_EP_SKIP_NONLOCAL=1, non-local rows are "
                "excluded from the GEMM (ep_size=%d, local experts=%d)",
                self.ep_size,
                self.num_experts_per_partition,
            )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden_states = payload.expert_x  # [M, K]
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]

        assert topk_ids is not None
        assert topk_weights is not None

        skip_nonlocal = self.skip_nonlocal_rows and self.ep_size > 1
        if self.ep_size > 1:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_experts_per_partition)
            if skip_nonlocal:
                # self.E is a sentinel expert id: moe_align_block_size_torch lays
                # those rows out past num_tokens_post_padded, so no block computes
                # them and `out` keeps the zeros written below.
                topk_ids = torch.where(
                    local_mask, local_ids, torch.full_like(local_ids, self.E)
                )
            else:
                topk_ids = local_ids.clamp(
                    min=0, max=self.num_experts_per_partition - 1
                )
            topk_weights = topk_weights * local_mask

        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        device = hidden_states.device
        dtype = hidden_states.dtype
        compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

        # Config selection for both GEMMs. When non-local rows are skipped only
        # ~1/ep_size of the rows survive, so the block sizes must be picked for
        # that smaller problem.
        config_m = max(1, -(-M // self.ep_size)) if skip_nonlocal else M
        config1 = get_default_config(config_m, self.E, self.N, K, top_k)
        config2 = get_default_config(config_m, self.E, K, self.inter_size, top_k)

        # Flat routing tensors for the kernel
        flat_topk_weights = topk_weights.view(-1)  # [M * top_k]
        flat_topk_ids = topk_ids.view(-1)  # [M * top_k]

        # Token-expert alignment: compute once if block sizes match, else twice
        block_m1 = config1["BLOCK_SIZE_M"]
        block_m2 = config2["BLOCK_SIZE_M"]
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_torch(
                topk_ids, block_m1, self.E, skip_invalid=skip_nonlocal
            )
        )
        if block_m1 == block_m2:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            )
        else:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                moe_align_block_size_torch(
                    topk_ids, block_m2, self.E, skip_invalid=skip_nonlocal
                )
            )

        # GEMM2's output buffer is allocated here, before the align_done marker:
        # when non-local rows are skipped the kernel leaves their rows untouched,
        # so the buffer must start at zero to contribute nothing to the top_k
        # reduction, and that memset is load-independent overhead that belongs to
        # `gemm_align` rather than to the GEMMs themselves.
        out = (
            torch.zeros(M * top_k, K, device=device, dtype=dtype)
            if skip_nonlocal
            else torch.empty(M * top_k, K, device=device, dtype=dtype)
        )

        self._phase_timer.record_moe_scheduled_rows(
            local_mask.sum() if skip_nonlocal else torch.tensor(
                M * top_k, device=device, dtype=torch.int64
            )
        )
        self._phase_timer.record_gemm_split("align_done")

        # GEMM1: hidden_states @ w1.T → intermediate1 [M*top_k, 2*inter]
        intermediate1 = torch.empty(M * top_k, self.N, device=device, dtype=dtype)
        invoke_fused_moe_kernel(
            hidden_states,
            self.w1,
            intermediate1,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config1,
            compute_type=compute_type,
        )

        # Activation: silu_and_mul(intermediate1) → intermediate2 [M*top_k, inter]
        intermediate2 = torch.empty(
            M * top_k, self.inter_size, device=device, dtype=dtype
        )
        silu_and_mul(intermediate2, intermediate1)

        # GEMM2: intermediate2 @ w2.T → out [M*top_k, K]
        invoke_fused_moe_kernel(
            intermediate2,
            self.w2,
            out,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids2,
            expert_ids2,
            num_tokens_post_padded2,
            mul_routed_weight=True,
            top_k=1,
            config=config2,
            compute_type=compute_type,
        )

        self._phase_timer.record_gemm_split("kernels_done")

        # Sum reduce over top_k: out is [M*top_k, K], reshape and sum
        output = out.view(M, top_k, K).sum(dim=1)

        return CombineForwardPayload(fused_expert_output=output)
