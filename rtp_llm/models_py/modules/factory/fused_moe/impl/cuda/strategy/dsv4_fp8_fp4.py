"""DeepSeek-V4 FP8-activation/FP4-weight EP strategy for Hopper."""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    _transform_scale_ue8m0,
    quant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.kernels.cuda.quant_layouts import dequantize_fp4_weight
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import StrategyAttributes
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import normalize_moe_w13_gate_up
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import DeepepNormalRouterBase
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import silu_mul_fp8_quant_packed
from rtp_llm.utils.model_weight import W


class Dsv4DeepepNormalRouter(DeepepNormalRouterBase):
    def __init__(self, config: MoEConfigAdapter, quant_config: FusedMoEQuantConfig):
        super().__init__(config, quant_config, expert_alignment=1)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.model_config.model_type == "deepseek_v4")
        checker.check(config.moe_quant_method == "FP8_FP4")

    def _do_quant_fp8_per_block(self, a1: torch.Tensor):
        return sgl_per_token_group_quant_fp8(
            a1,
            group_size=128,
            eps=1.0e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class Dsv4HopperFp8Fp4Executor(FusedMoeExpertExecutor):
    strategy_name = "dsv4_deepep_fp8_fp4"

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.FP8_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.model_config.model_type == "deepseek_v4")
        checker.check(config.moe_quant_method == "FP8_FP4")
        checker.check(config.ep_size > 1)
        checker.check(torch.cuda.is_available())
        checker.check(torch.cuda.get_device_capability()[0] == 9)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__(config, quant_config, weights)
        self.inter_dim = int(config.moe_inter_dim)
        self.swiglu_limit = float(config.swiglu_limit)
        w13, s13 = normalize_moe_w13_gate_up(
            weights[W.moe_w1],
            weights[W.moe_s1],
            self.inter_dim,
            config.moe_w1_layout,
        )
        w2 = weights[W.moe_w2]
        s2 = weights[W.moe_s2]
        expected = int(config.n_local_experts)
        if w13.size(0) != expected or w2.size(0) != expected:
            raise ValueError(
                "DSV4 local FP4 weight count mismatch: "
                f"w13={w13.size(0)} w2={w2.size(0)} expected={expected}"
            )
        self.w13, self.s13 = self._requant_fp4_stack(w13, s13)
        self.w2, self.s2 = self._requant_fp4_stack(w2, s2)

    @staticmethod
    def _requant_fp4_stack(weight: torch.Tensor, scale: torch.Tensor):
        fp8_weights = []
        fp8_scales = []
        for expert_id in range(weight.size(0)):
            dequant = dequantize_fp4_weight(
                weight[expert_id], scale[expert_id]
            ).to(torch.bfloat16)
            weight_fp8, scale_raw = quant_weight_ue8m0(dequant, [128, 128])
            fp8_weights.append(weight_fp8)
            fp8_scales.append(_transform_scale_ue8m0(scale_raw, weight_fp8.size(0)))
        return torch.stack(fp8_weights), torch.stack(fp8_scales)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if expert_map is not None or a2_scale is not None:
            raise ValueError("DSV4 FP8/FP4 executor does not accept expert_map/a2_scale")
        if apply_router_weight_on_input:
            raise ValueError("DeepEP combine owns DSV4 router-weight application")
        if payload.expert_x_scale is None or payload.expert_tokens_meta is None:
            raise ValueError("DSV4 DeepEP payload is missing FP8 scales/token metadata")
        counts = payload.expert_tokens_meta.expert_num_tokens_cpu
        if counts is None:
            counts = payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
        elif isinstance(counts, torch.Tensor):
            counts = counts.cpu().tolist()

        xq = payload.expert_x
        xs = payload.expert_x_scale
        output = torch.empty(
            (xq.size(0), self.w2.size(1)),
            dtype=torch.bfloat16,
            device=xq.device,
        )
        offset = 0
        for expert_id, count in enumerate(counts):
            count = int(count)
            if count == 0:
                continue
            end = offset + count
            gate_up = torch.empty(
                (count, 2 * self.inter_dim),
                dtype=torch.bfloat16,
                device=xq.device,
            )
            fp8_gemm_nt(
                (xq[offset:end].contiguous(), xs[offset:end].contiguous()),
                (self.w13[expert_id], self.s13[expert_id]),
                gate_up,
                disable_ue8m0_cast=False,
            )
            hq, hs = silu_mul_fp8_quant_packed(
                gate_up, clamp_limit=self.swiglu_limit, group_size=128
            )
            fp8_gemm_nt(
                (hq, hs),
                (self.w2[expert_id], self.s2[expert_id]),
                output[offset:end],
                disable_ue8m0_cast=False,
            )
            offset = end
        if offset != xq.size(0):
            raise RuntimeError(f"DSV4 expert token counts cover {offset}/{xq.size(0)} rows")
        return CombineForwardPayload(fused_expert_output=output)


class CudaDsv4DeepepFp8Fp4Strategy(MoeStrategy):
    strategy_name = "dsv4_deepep_fp8_fp4"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy in ("auto", cls.strategy_name))
        checker.check(config.model_config.model_type == "deepseek_v4")
        checker.check(config.moe_quant_method == "FP8_FP4")
        checker.check(config.ep_size > 1)

    def get_attributes(self) -> StrategyAttributes:
        return StrategyAttributes(
            router_class=Dsv4DeepepNormalRouter,
            executor_class=Dsv4HopperFp8Fp4Executor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )
