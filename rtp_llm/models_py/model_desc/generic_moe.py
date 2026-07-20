import logging
import os
from typing import Any, Dict, Optional

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)

# env-gated prefill cudaEvent timing probe (PREFILL_PROF=1)
_PREFILL_PROF = os.environ.get("PREFILL_PROF", "0") == "1"

_MOE_TOPK_EXPORT_DIR = os.environ.get("MOE_TOPK_EXPORT_DIR", "")

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import (
    get_fmha_params,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FakeBalanceExpert,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    WfDispatcher,
    logical_to_physical_experts,
    record_expert_stats,
    wf_dispatch,
    wf_k_dispatch,
)
from rtp_llm.models_py.utils.expert_stats_probe import get_expert_stats_probe
from rtp_llm.models_py.utils.phase_timer import get_phase_timer
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.layer_idx = layer_idx
        self._topk_call_count = 0
        # ExpertStatsBuffer, assigned by the model in initialize() when EPLB is enabled
        self.expert_stats = None
        # MoE phase checkpoints for PHASE_TIMER=1; no-op when off
        self._phase_timer = get_phase_timer()
        # Per-forward logical/physical expert counters for EXPERT_STATS_PROBE=1;
        # no-op when off. Independent of self.expert_stats (the EPLB buffer).
        self._expert_stats_probe = get_expert_stats_probe()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(config=config)
        if moe_config.fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=config.expert_num,
                moe_k=config.moe_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=parallelism_config.ep_size,
            )
        else:
            self.fake_balance_expert = None
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

        self.log2phy = weights.get(W.log2phy, None)
        self.logic_expert_cnt = weights.get(W.logic_expert_cnt, None)
        if (self.log2phy is None) != (self.logic_expert_cnt is None):
            raise ValueError("log2phy and logic_expert_cnt must be provided together")

        # EPLB log2phy debug: sampled dump of layout and dispatch.
        # EPLB_LOG2PHY_DEBUG=N prints once every N forwards (0 disables).
        self._log2phy_dbg_interval = int(os.environ.get("EPLB_LOG2PHY_DEBUG", "0"))
        self._log2phy_dbg_cnt = 0
        self._logical_expert_num = config.expert_num
        # Dispatch policy: "rr", "wf"/"wf_iter" (optimized for <=2 replicas),
        # or "wf_k" (general replica count, deliberately CPU-oriented).
        self._dispatch_policy = os.environ.get("EPLB_DISPATCH_POLICY", "rr")
        # Internal EPLB recording is separate from the external probe. Dynamic
        # ALL/EPLB arms need this buffer even when the run is otherwise nostats,
        # while the observation-only arm can disable it independently.
        # Unset preserves the historical recording behavior for callers outside
        # this experiment. Formal launchers always set 0/1 explicitly.
        internal_stats = os.environ.get("EPLB_INTERNAL_STATS", "1") == "1"
        eplb_mode = os.environ.get("EPLB_MODE", "NONE").strip().upper()
        if eplb_mode in ("EPLB", "ALL") and not internal_stats:
            raise ValueError(
                "EPLB_INTERNAL_STATS=1 is required for EPLB_MODE="
                f"{eplb_mode}; disabling internal stats would make dynamic "
                "placement consume empty solver buffers"
            )
        # Keep the historical circulant optimization as a second guard. The
        # explicit internal gate is what distinguishes armA nostats from armB
        # nostats under the new pure-DP experiment.
        self._skip_stats_recording = (
            not internal_stats
            or (
                os.environ.get("EPLB_PLACEMENT", "eplb") == "circulant"
                and os.environ.get("EXPERT_STATS", "0") != "1"
            )
        )
        # Loader-preloaded circulant placement is final before model construction.
        self._circulant_preloaded = (
            weights.get(W.circulant_preloaded, None) is not None
            or os.environ.get("EPLB_CIRCULANT_PRELOADED", "0") == "1"
        )
        # Caching placement-derived candidates requires the placement to be
        # final, not merely circulant: the runtime-fallback path still rewrites
        # log2phy in place after the first forward, and because the weights are
        # inference tensors that revision is unobservable. So gate the static
        # path on the loader preload and leave the fallback on live prep.
        self._static_wf_placement = (
            self._circulant_preloaded
            and os.environ.get("EPLB_PLACEMENT", "eplb") == "circulant"
            and self._dispatch_policy in ("wf", "wf_iter")
        )
        self._wf_dispatcher = None

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.add_shared_expert = config.moe_style == 2
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        self.ep_size = parallelism_config.ep_size

        if self._static_wf_placement and self.log2phy is not None:
            self._wf_dispatcher = WfDispatcher(
                self._logical_expert_num,
                self.ep_size,
                self.log2phy.device,
            )
            self._wf_dispatcher.prepare_candidates(
                self.log2phy,
                self.logic_expert_cnt,
                max(1, self.num_experts // self.ep_size),
            )

        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights,
                W.shared_expert_gate,
                None,
                None,
                quant_config=quant_config,
                # For ROCm devices shared_expert_gate is not pre-swizzled during weight
                # loading and its single output column does not satisfy the SwizzleA
                # layout. Keep this scalar projection on the no-swizzle backend.
                hw_kernel_config=None,
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

        if (
            self._log2phy_dbg_interval > 0
            and self.log2phy is not None
            and self.layer_idx == 0
        ):
            self._dbg_dump_layout()

    def _dbg_rank(self) -> str:
        pc = self.parallelism_config
        parts = []
        for name in ("world_rank", "tp_rank", "dp_rank", "ep_rank"):
            v = getattr(pc, name, None)
            if v is not None:
                parts.append(f"{name}={v}")
        return " ".join(parts)

    def _dbg_dump_layout(self) -> None:
        log2phy = self.log2phy.detach().cpu()
        cnt = self.logic_expert_cnt.detach().cpu()
        logical = min(8, self._logical_expert_num)
        lines = [
            f"[EPLB-DBG][{self._dbg_rank()}] layer {self.layer_idx} layout: "
            f"phy_exp_num={self.num_experts}, logical_exp_num={self._logical_expert_num}, "
            f"ep_size={self.ep_size}, local_experts={self.num_local_experts}"
        ]
        for l in range(logical):
            c = int(cnt[l])
            replicas = log2phy[l, :c].tolist()
            lines.append(f"  logical {l}: cnt={c}, physical={replicas}")
        print("\n".join(lines), flush=True)

    def _dbg_dump_dispatch(self, logical_ids, physical_ids) -> None:
        k = min(4, logical_ids.shape[0])
        logical = logical_ids[:k].detach().cpu().tolist()
        physical = physical_ids[:k].detach().cpu().tolist()
        print(
            f"[EPLB-DBG][{self._dbg_rank()}] layer {self.layer_idx} "
            f"step {self._log2phy_dbg_cnt} dispatch (first {k} tokens)\n"
            f"  logical : {logical}\n"
            f"  physical: {physical}",
            flush=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._phase_timer.moe_block_begin(self.layer_idx)
        self._phase_timer.record_moe_checkpoint("moe_start", self.layer_idx)
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        self._phase_timer.record_route_split("topk_done")

        logical_topk_ids = topk_ids
        if self.log2phy is not None:
            if self._dispatch_policy == "wf_k":
                topk_ids = wf_k_dispatch(
                    logical_topk_ids,
                    self.log2phy,
                    self.logic_expert_cnt,
                    self.ep_size,
                    self.num_experts,
                )
            elif self._dispatch_policy in ("wf", "wf_iter"):
                if self._static_wf_placement and self._wf_dispatcher is None:
                    self._wf_dispatcher = WfDispatcher(
                        self._logical_expert_num,
                        self.ep_size,
                        logical_topk_ids.device,
                    )
                topk_ids = wf_dispatch(
                    logical_topk_ids,
                    self.log2phy,
                    self.logic_expert_cnt,
                    self.ep_size,
                    self.num_experts,
                    policy=self._dispatch_policy,
                    dispatcher=self._wf_dispatcher,
                    static_placement=self._static_wf_placement,
                )
            else:
                topk_ids = logical_to_physical_experts(
                    logical_topk_ids,
                    self.log2phy,
                    self.logic_expert_cnt,
                    self.parallelism_config.dp_rank,
                )

        # Closes the policy window: rr / wf / wf_iter / wf_k and nothing else.
        self._phase_timer.record_route_split("policy_done")

        if (
            self._log2phy_dbg_interval > 0
            and self.log2phy is not None
            and self.layer_idx == 0
        ):
            self._log2phy_dbg_cnt += 1
            if self._log2phy_dbg_cnt % self._log2phy_dbg_interval == 0:
                self._dbg_dump_dispatch(logical_topk_ids, topk_ids)

        # Closes the gate+topk+EPLB-remap window: everything after this point is
        # the dispatch / GEMM / combine pipeline inside self.fused_moe.
        self._phase_timer.record_moe_checkpoint("route_done", self.layer_idx)

        is_ep_mode = self.ep_size > 1
        # EP mode: routed expert output is already complete (EP combine handles it).
        # Shared expert output is TP-partial and needs separate allreduce.
        use_ep_shared_allreduce = (
            self.shared_expert is not None and self.ffn_tp_size > 1 and is_ep_mode
        )

        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(
                hidden_states,
                skip_allreduce=use_ep_shared_allreduce,
            )
            if use_ep_shared_allreduce:
                # EP mode: routed expert output is already complete
                # (EP combine via all_to_all / all_gather aggregated across ranks).
                # Only the shared expert output is TP-partial and needs all_reduce.
                if self.shared_expert_gate is not None:
                    gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
                    shared_expert_output = (
                        torch.sigmoid(gate_output) * shared_expert_output
                    )
                shared_expert_output = all_reduce(shared_expert_output, group=Group.TP)
                experts_output = experts_output + shared_expert_output
            else:
                # TP-only mode: same as main - each component does its own allreduce.
                if self.shared_expert_gate is not None:
                    gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
                    # Fused: experts_output += sigmoid(gate_output) * shared_expert_output
                    self.sigmoid_gate_scale_add(
                        gate_output, shared_expert_output, experts_output
                    )
                else:
                    experts_output = experts_output + shared_expert_output

        self._phase_timer.moe_block_end(self.layer_idx)

        # Stats recording deferred past every phase_timer window: the recording
        # kernel cost ~0.2-0.8ms/layer (up to 30% of gate_route at bs32) and was
        # inflating the route window. Safe to defer because the EPLB consumer
        # (ExpertBalancer::excuteEplbPlan) drains these buffers between forwards,
        # and same-stream ordering guarantees the increments land before then.
        if self.expert_stats is not None and not self._skip_stats_recording:
            record_expert_stats(
                logical_topk_ids,
                self.expert_stats.log_stats_buf,
                self.expert_stats.gpu_loads_buf,
                self.layer_idx,
                topk_ids,
                self.num_experts,
            )

        # The only point where the logical and physical id tensors are both live.
        self._expert_stats_probe.record(self.layer_idx, logical_topk_ids, topk_ids)

        # export topk activation
        if _MOE_TOPK_EXPORT_DIR and self.parallelism_config.dp_rank == 0:
            step = self._topk_call_count
            self._topk_call_count += 1
            export_path = os.path.join(
                _MOE_TOPK_EXPORT_DIR,
                f"layer_{self.layer_idx:03d}_step_{step:06d}.pt",
            )
            torch.save(
                {
                    "layer_idx": self.layer_idx,
                    "step": step,
                    "logical_topk_ids": logical_topk_ids.cpu(),
                    "topk_weights": topk_weights.cpu(),
                    "num_tokens": num_tokens,
                    "num_experts": self.num_experts,
                    "top_k": self.top_k,
                },
                export_path,
            )

        return experts_output


class DecodeLayerOutput:
    def __init__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        self.hidden_states = hidden_states
        self.residual = residual


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        # attn / layer checkpoints for PHASE_TIMER=1; no-op when off
        self._phase_timer = get_phase_timer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> DecodeLayerOutput:
        timer = self._phase_timer
        timer.layer_begin(self.layer_idx)

        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        timer.attn_begin(self.layer_idx)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        timer.attn_end(self.layer_idx)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        timer.layer_end(self.layer_idx)
        return DecodeLayerOutput(hidden_states, residual)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        # Owns the recording window for the whole forward; no-op unless PHASE_TIMER=1
        self._phase_timer = get_phase_timer()
        # Owns the per-forward expert counter export; no-op unless
        # EXPERT_STATS_PROBE=1. Configured in initialize() once the MoE layers exist.
        self._expert_stats_probe = get_expert_stats_probe()

    def initialize(self, init_resource) -> bool:
        result = super().initialize(init_resource)
        moe_layers = [
            layer.mlp for layer in self.layers if isinstance(layer.mlp, GenericMoeLayer)
        ]
        if self.expert_stats is not None:
            for mlp in moe_layers:
                mlp.expert_stats = self.expert_stats
        if moe_layers:
            # Layer axis spans every model layer so dense layers keep their index
            # (they simply stay all-zero); the expert axis comes from the MoE layers.
            self._expert_stats_probe.configure(
                layer_num=self.layer_num,
                log_exp_num=moe_layers[0]._logical_expert_num,
                phy_exp_num=moe_layers[0].num_experts,
                ep_size=moe_layers[0].ep_size,
                rank=self.parallelism_config.world_rank,
            )
        return result

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        prof = _PREFILL_PROF and inputs.attention_inputs.is_prefill
        if prof:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        residual = torch.zeros_like(hidden_states)
        # Phase timing covers prefill only: decode may run under CUDA graph
        # replay, where host-side event records are not captured.
        timer = self._phase_timer
        is_prefill = inputs.attention_inputs.is_prefill
        if is_prefill:
            timer.begin_forward(input_ids.numel())
        # Expert counters follow the same prefill-only window as the phase timer.
        probe = self._expert_stats_probe
        probe.begin_forward(is_prefill, input_ids.numel())
        recorded = False
        try:
            for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
                layer_fmha_impl = select_fmha_impl_for_layer(
                    fmha_impl, self.kv_cache, i
                )
                output = decoder_layer(
                    hidden_states,
                    residual,
                    layer_fmha_impl,
                    kv_cache=(
                        self.kv_cache.get_layer_cache(i) if self.kv_cache else None
                    ),
                )
                hidden_states = output.hidden_states
                residual = output.residual

            hidden_states, _ = self.norm(hidden_states, residual)
            recorded = True
        finally:
            # A forward that raised is discarded whole, never half-recorded.
            timer.end_forward(success=recorded)
            # Exports one NPZ per successful prefill forward, then zeroes.
            probe.end_forward(success=recorded)

        if is_prefill and timer.capture_armed:
            timer.flush(rank=self.parallelism_config.world_rank)

        if prof:
            end_ev.record()
            torch.cuda.synchronize()
            _LOGGER.warning(
                "[PREFILL_PROF] rank=%d prefill_ms=%.3f tokens=%d",
                self.parallelism_config.world_rank,
                start_ev.elapsed_time(end_ev),
                input_ids.numel(),
            )
        return PyModelOutputs(hidden_states, get_fmha_params(fmha_impl))


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
