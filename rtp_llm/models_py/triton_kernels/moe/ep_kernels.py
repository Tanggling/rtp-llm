# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# Licensed under the Apache License, Version 2.0

import logging

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.math import ceil_div

logger = logging.getLogger(__name__)


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    expert_mask = offset_cumsum == cur_expert
    cur_expert_start = tl.sum(tl.where(expert_mask, cumsum, tl.zeros_like(cumsum)))
    cur_expert_token_num = tl.sum(
        tl.where(expert_mask, tokens_per_expert, tl.zeros_like(tokens_per_expert))
    )
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    num_experts: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    index_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = index_in_s < SCALE_HIDDEN_SIZE
    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale
            + token_id * recv_x_scale_stride0
            + index_in_s * recv_x_scale_stride1,
            mask=mask_s,
        )
        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0 and expert_id < num_experts:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(
                    output_tensor_scale_ptr + index_in_s * output_tensor_scale_stride1,
                    to_copy_s,
                    mask=mask_s,
                )


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py
@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    scale_ue8m0: bool = False,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts
    scale_hidden_size = hidden_size // BLOCK_D
    if scale_ue8m0:
        # ue8m0 scales are packed here (4 scales per int32),
        # hence the effective size of this dimension is divided by 4.
        scale_hidden_size = ceil_div(scale_hidden_size, 4)

    assert m_indices.shape[0] % BLOCK_E == 0
    assert recv_x_scale.dtype == output_tensor_scale.dtype
    assert recv_x_scale.shape[1] == output_tensor_scale.shape[1] == scale_hidden_size
    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
    )
    return


@triton.jit
def _fwd_kernel_ep_scatter_1_v2(
    alignment,
    expert_start_loc,
    num_experts: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_EXPERT_NUM)
    mask = offset < num_experts
    tl.store(expert_start_loc + offset, offset * alignment, mask=mask)


@triton.jit
def _fwd_kernel_ep_scatter_2_v2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_tensor_scale_stride2,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    num_experts: tl.constexpr,
    alignment: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    index_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = index_in_s < SCALE_HIDDEN_SIZE
    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale
            + token_id * recv_x_scale_stride0
            + index_in_s * recv_x_scale_stride1,
            mask=mask_s,
        )
        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0 and expert_id < num_experts:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                token_idx = dest_token_index % alignment
                output_tensor_scale_ptr = (
                    output_tensor_scale
                    + expert_id * output_tensor_scale_stride0
                    + token_idx * output_tensor_scale_stride1
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(
                    output_tensor_scale_ptr + index_in_s * output_tensor_scale_stride2,
                    to_copy_s,
                    mask=mask_s,
                )


@torch.no_grad()
def ep_scatter_v2(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    alignment: int,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    output_index: torch.Tensor,
    scale_ue8m0: bool = False,
):
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = expert_start_loc.shape[0]
    hidden_size = recv_x.shape[1]
    scale_hidden_size = hidden_size // BLOCK_D
    if scale_ue8m0:
        # ue8m0 scales are packed here (4 scales per int32),
        # hence the effective size of this dimension is divided by 4.
        scale_hidden_size = ceil_div(scale_hidden_size, 4)

    assert recv_x_scale.dtype == output_tensor_scale.dtype
    assert recv_x_scale.shape[1] == output_tensor_scale.shape[2] == scale_hidden_size
    _fwd_kernel_ep_scatter_1_v2[(1,)](
        alignment,
        expert_start_loc,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_2_v2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_tensor_scale.stride(2),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_experts=num_experts,
        alignment=alignment,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    total_input_tokens,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)
    start_cur_token_int32 = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index = source_token_index_int32.to(tl.int64)
                if source_token_index >= 0 and source_token_index < total_input_tokens:
                    acc_weight = tl.load(
                        recv_topk_weight
                        + cur_token * recv_topk_weight_stride0
                        + topk_index
                    )
                    tmp = tl.load(
                        input_tensor
                        + source_token_index * input_tensor_stride0
                        + cur_block * BLOCK_D
                        + off_d
                    )
                    accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 512  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor.shape[0],
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.
    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.
    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    g,
    m,
    k_div_block_size,
    input_scale_stride_g,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_g,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    for m_base in range(pid_m, m, grid_m):
        input_offset = (
            input_scale_ptr
            + pid_g * input_scale_stride_g
            + m_base * input_scale_stride_m
            + k_offsets * input_scale_stride_k
        )
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)
        output_offset = (
            output_ptr
            + pid_g * output_stride_g
            + k_offsets * output_stride_k
            + m_base * output_stride_m
        )
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/quantization/triton_quant/fp8/fp8act_quant_kernel.py
def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() in [2, 3], "Input must be 2D or 3D tensor"

    if input_scale.dim() == 2:
        m, k_div_block_size = input_scale.shape
        g = 1
        input_view = input_scale.unsqueeze(0)
    else:
        g, m, k_div_block_size = input_scale.shape
        input_view = input_scale

    padded_m = get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty(
        (g, k_div_block_size, padded_m),
        dtype=input_scale.dtype,
        device=input_scale.device,
    )
    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)
    _tma_align_input_scale_kernel[(grid_m, g)](
        input_scale_ptr=input_view,
        output_ptr=output,
        g=g,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_g=input_view.stride(0),
        input_scale_stride_m=input_view.stride(1),
        input_scale_stride_k=input_view.stride(2),
        output_stride_g=output.stride(0),
        output_stride_m=output.stride(2),  # Note: these are swapped
        output_stride_k=output.stride(1),  # for column-major
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    if input_scale.dim() == 2:
        output = output.squeeze(0)
        return output.t()[:m].contiguous()

    return output.transpose(1, 2)[:, :m, :].contiguous()


@triton.jit
def recompute_topk_ids_triton_kernel(
    topk_ids_ptr,
    adjusted_topk_ids_ptr,
    expert_count_ptr,
    current_expert_start_id,
    num_local_experts,
    num_total,
    BLOCK_SIZE: tl.constexpr,
):
    token_indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_indices < num_total  # Mask out-of-bounds threads

    # 1. Load
    expert_id = tl.load(topk_ids_ptr + token_indices, mask=mask, other=-1)

    # 2. Adjust expert index
    adjusted = expert_id - current_expert_start_id
    valid = mask & (adjusted >= 0) & (adjusted < num_local_experts)

    # 3. Store
    out = tl.where(valid, adjusted, -1)
    tl.store(adjusted_topk_ids_ptr + token_indices, out, mask=mask)

    # 4. Atomic add - use scalar value for efficiency
    tl.atomic_add(expert_count_ptr + adjusted, 1, mask=valid)


def recompute_topk_ids_sum_expert_count(
    topk_ids: torch.Tensor, current_expert_start_id: int, num_local_experts: int
):
    """
    Recompute topk_ids by subtracting current_expert_start_id and count expert tokens.

    Args:
        topk_ids: Tensor of shape [num_tokens, topk] containing expert IDs.
            Sentinel value -1 is allowed and treated as a padding/invalid slot.
        current_expert_start_id: Starting expert ID to subtract
        num_local_experts: Number of local experts

    Returns:
        tuple: (adjusted_topk_ids, expert_count)

    Sentinel contract (relied on by PureDpRouter padding):
        - Input slots equal to -1 are NOT counted in expert_count.
        - Output adjusted_topk_ids preserves -1 in those slots (no remap).
        - Out-of-range expert ids (after subtraction) also collapse to -1.
    """
    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    num_total = num_tokens * topk

    # Create output tensors
    adjusted_topk_ids = torch.empty_like(topk_ids)
    expert_count = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

    # Configure triton kernel parameters
    # Use smaller block size for better vectorization when topk is large
    # Ensure BLOCK_SIZE is a power of 2 for triton compatibility
    base_block_size = min(256, 1024 // max(topk, 1))
    BLOCK_SIZE = triton.next_power_of_2(base_block_size)

    # Launch recompute kernel
    grid_recompute = (triton.cdiv(num_total, BLOCK_SIZE),)
    recompute_topk_ids_triton_kernel[grid_recompute](
        topk_ids,
        adjusted_topk_ids,
        expert_count,
        current_expert_start_id,
        num_local_experts,
        num_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return adjusted_topk_ids, expert_count


@triton.jit
def pre_reorder_moe_tokenwise_fp8_triton_kernel(
    input_ptr,
    permuted_input_ptr,
    input_scale_ptr,
    permuted_scale_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):

    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size
    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)

    for src_idx_int32 in tl.range(
        start_src_idx, num_tokens, step, num_stages=NUM_STAGES
    ):
        src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk

        src_ptr_offs = input_ptr + src_idx * hidden_size + offset
        dst_ptr_offs = permuted_input_ptr + offset
        # Load input data
        in_data = tl.load(src_ptr_offs, mask=mask)

        if tl.program_id(1) == 0:
            a1_scale = tl.load(input_scale_ptr + src_idx)
        else:
            a1_scale = 1.0

        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id >= 0 and expert_id < num_local_experts:
                dst_idx = tl.load(token_src2dst_ptr + idx)
                # Store reordered input data
                tl.store(dst_ptr_offs + dst_idx * hidden_size, in_data, mask=mask)
                # Store reordered scale
                if tl.program_id(1) == 0:
                    tl.store(permuted_scale_ptr + dst_idx, a1_scale)


@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


# moe pre reorder with token-wise fp8 scale
def cutlass_moe_pre_reorder(
    input,  # [num_tokens, hidden_size]
    permuted_input,  # [num_tokens * topk, hidden_size]
    input_scale,  # [num_tokens]
    permuted_scale,  # [num_tokens * topk]
    topk_ids,  # [num_tokens, topk]
    num_local_experts: int,
    topk: int,
    num_tokens: int,
    hidden_size: int,
):
    assert num_tokens == input.shape[0]
    assert hidden_size == input.shape[1]

    _, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)

    device = input.device
    # 1. get src to dst map
    grid = (triton.cdiv(topk_ids.numel(), 512),)
    src2dst = torch.empty(topk_ids.numel(), device=device, dtype=torch.int32)
    compute_src2dst_triton_kernel[grid](reorder_ids, src2dst, topk_ids.numel(), 512)

    # 2. reorder input tokens and per token input scale
    MAX_THREADS_PER_BLOCK = 1024
    MIN_THREADS_PER_BLOCK = 512
    MAX_WAVES = 8

    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    max_threads_per_sm = props.max_threads_per_multi_processor
    max_num_blocks = sm_count * max_threads_per_sm // MAX_THREADS_PER_BLOCK

    min_block_dim_needed = triton.cdiv(num_tokens * hidden_size, max_num_blocks)
    block_dim = triton.next_power_of_2(min_block_dim_needed)
    block_dim = max(MIN_THREADS_PER_BLOCK, min(block_dim, MAX_THREADS_PER_BLOCK))

    grid_dim_x = triton.cdiv(hidden_size, block_dim)
    grid_dim_y = max(min(num_tokens, max_num_blocks * MAX_WAVES // grid_dim_x), 1)

    grid = (grid_dim_y, grid_dim_x)

    pre_reorder_moe_tokenwise_fp8_triton_kernel[grid](
        input_ptr=input,
        permuted_input_ptr=permuted_input,
        input_scale_ptr=input_scale,
        permuted_scale_ptr=permuted_scale,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        num_local_experts=num_local_experts,
        topk=topk,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_dim,
        NUM_STAGES=3,
    )

    return src2dst


# A generic post reorder implementation where invalid expert_ids are padded with -1
@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    num_local_experts,
    total_dst_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    src_idx = tl.program_id(0).to(tl.int64)

    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    store_ptr = output_ptr + src_idx * hidden_size

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            if expert_id >= 0 and expert_id < num_local_experts:
                dst_idx = tl.load(src2dst_ptr + idx).to(tl.int64)
                if dst_idx >= 0 and dst_idx < total_dst_tokens:
                    weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                    load_ptr = down_output_ptr + dst_idx * hidden_size
                    in_data = tl.load(load_ptr + offset, mask=mask)
                    sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)


@triton.jit
def _compute_problem_sizes_kernel(
    topk_ids_ptr,
    problem_sizes1_ptr,
    problem_sizes2_ptr,
    topk_length,
    n,
    k,
    problem_1_swap_ab: tl.constexpr,
    problem_2_swap_ab: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)

    occurrences = 0
    for start in range(0, topk_length, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < topk_length
        ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1)
        occurrences += tl.sum((ids == expert_id).to(tl.int32))

    base = expert_id * 3
    n2 = 2 * n
    if problem_1_swap_ab:
        tl.store(problem_sizes1_ptr + base, n2)
        tl.store(problem_sizes1_ptr + base + 1, occurrences)
        tl.store(problem_sizes1_ptr + base + 2, k)
    else:
        tl.store(problem_sizes1_ptr + base, occurrences)
        tl.store(problem_sizes1_ptr + base + 1, n2)
        tl.store(problem_sizes1_ptr + base + 2, k)

    if problem_2_swap_ab:
        tl.store(problem_sizes2_ptr + base, k)
        tl.store(problem_sizes2_ptr + base + 1, occurrences)
        tl.store(problem_sizes2_ptr + base + 2, n)
    else:
        tl.store(problem_sizes2_ptr + base, occurrences)
        tl.store(problem_sizes2_ptr + base + 1, k)
        tl.store(problem_sizes2_ptr + base + 2, n)


@triton.jit
def _compute_expert_offsets_kernel(
    problem_sizes1_ptr,
    expert_offsets_ptr,
    num_experts,
    swap_ab: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_E)
    mask = offsets < num_experts

    if swap_ab:
        sizes = tl.load(problem_sizes1_ptr + offsets * 3 + 1, mask=mask, other=0)
    else:
        sizes = tl.load(problem_sizes1_ptr + offsets * 3, mask=mask, other=0)

    cum_sizes = tl.cumsum(sizes)
    expert_starts = cum_sizes - sizes
    tl.store(expert_offsets_ptr + offsets, expert_starts, mask=mask)


@torch.no_grad()
def get_cutlass_moe_mm_without_permute_info(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    problem_1_swap_ab: bool,
    problem_2_swap_ab: bool,
):
    topk_length = topk_ids.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(topk_length))

    _compute_problem_sizes_kernel[(num_experts,)](
        topk_ids,
        problem_sizes1,
        problem_sizes2,
        topk_length,
        n,
        k,
        problem_1_swap_ab=problem_1_swap_ab,
        problem_2_swap_ab=problem_2_swap_ab,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    BLOCK_E = triton.next_power_of_2(num_experts)
    _compute_expert_offsets_kernel[(1,)](
        problem_sizes1,
        expert_offsets,
        num_experts,
        swap_ab=problem_1_swap_ab,
        BLOCK_E=BLOCK_E,
    )


@triton.jit
def _record_expert_stats_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    log_stats_ptr,
    gpu_loads_ptr,
    total_ids,
    layer_idx,
    log_exp_num,
    phy_exp_num,
    ep_size,
    experts_per_rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_ids

    logical_ids = tl.load(logical_topk_ids_ptr + offsets, mask=mask, other=0).to(
        tl.int64
    )
    physical_ids = tl.load(physical_topk_ids_ptr + offsets, mask=mask, other=0).to(
        tl.int64
    )
    # masked-out lanes and padded/invalid expert ids must not touch the buffers
    valid_logical = mask & (logical_ids >= 0) & (logical_ids < log_exp_num)

    tl.atomic_add(
        log_stats_ptr + layer_idx * log_exp_num + logical_ids,
        1,
        mask=valid_logical,
    )

    valid_physical = mask & (physical_ids >= 0) & (physical_ids < phy_exp_num)
    ep_ranks = tl.minimum(physical_ids // experts_per_rank, ep_size - 1)
    tl.atomic_add(
        gpu_loads_ptr + layer_idx * ep_size + ep_ranks,
        1,
        mask=valid_physical,
    )


def _record_expert_stats_torch(
    logical_topk_ids: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    log_stats_buf: torch.Tensor,
    gpu_loads_buf: torch.Tensor,
    layer_idx: int,
    phy_exp_num: int,
) -> None:
    """Pure-torch fallback for platforms without a working triton backend (e.g. PPU)."""
    log_exp_num = log_stats_buf.shape[1]
    ep_size = gpu_loads_buf.shape[1]
    experts_per_rank = ceil_div(phy_exp_num, ep_size)

    logical_ids = logical_topk_ids.reshape(-1).long()
    logical_ids = logical_ids[(logical_ids >= 0) & (logical_ids < log_exp_num)]
    if logical_ids.numel() != 0:
        ones = torch.ones_like(logical_ids, dtype=log_stats_buf.dtype)
        log_stats_buf[layer_idx].scatter_add_(0, logical_ids, ones)

    physical_ids = physical_topk_ids.reshape(-1).long()
    physical_ids = physical_ids[(physical_ids >= 0) & (physical_ids < phy_exp_num)]
    if physical_ids.numel() == 0:
        return

    ranks = torch.clamp(physical_ids // experts_per_rank, max=ep_size - 1)
    gpu_loads_buf[layer_idx].scatter_add_(
        0, ranks, torch.ones_like(ranks, dtype=gpu_loads_buf.dtype)
    )


_record_expert_stats_use_triton = True


def record_expert_stats(
    logical_topk_ids: torch.Tensor,
    log_stats_buf: torch.Tensor,
    gpu_loads_buf: torch.Tensor,
    layer_idx: int,
    physical_topk_ids: torch.Tensor = None,
    phy_exp_num: int = None,
) -> None:
    """Accumulate per-expert activation counts and per-EP-rank loads for EPLB.

    Device-side ops only (no host sync), so it is safe under CUDA Graph capture.
    Falls back to a pure-torch implementation when the triton backend cannot
    compile for the current device (e.g. PPU without ptxas).

    Args:
        logical_topk_ids: [num_tokens, top_k] logical routed expert ids.
        log_stats_buf: [layer_num, log_exp_num] INT32 buffer, incremented per expert hit.
        gpu_loads_buf: [layer_num, ep_size] INT32 buffer, incremented per token-expert
            pair on the EP rank hosting the expert.
    """
    global _record_expert_stats_use_triton

    if physical_topk_ids is None:
        physical_topk_ids = logical_topk_ids
    if phy_exp_num is None:
        phy_exp_num = log_stats_buf.shape[1]

    total_ids = logical_topk_ids.numel()
    if total_ids == 0:
        return

    if _record_expert_stats_use_triton:
        log_exp_num = log_stats_buf.shape[1]
        ep_size = gpu_loads_buf.shape[1]
        experts_per_rank = ceil_div(phy_exp_num, ep_size)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(total_ids))
        grid = (ceil_div(total_ids, BLOCK_SIZE),)
        try:
            _record_expert_stats_kernel[grid](
                logical_topk_ids.contiguous(),
                physical_topk_ids.contiguous(),
                log_stats_buf,
                gpu_loads_buf,
                total_ids,
                layer_idx,
                log_exp_num,
                phy_exp_num,
                ep_size,
                experts_per_rank,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return
        except Exception:
            _record_expert_stats_use_triton = False
            logger.warning(
                "triton record_expert_stats kernel unavailable on this device, "
                "falling back to torch implementation",
                exc_info=True,
            )

    _record_expert_stats_torch(
        logical_topk_ids,
        physical_topk_ids,
        log_stats_buf,
        gpu_loads_buf,
        layer_idx,
        phy_exp_num,
    )


def logical_to_physical_experts(
    logical_ids: torch.Tensor,
    log2phy: torch.Tensor,
    logic_expert_cnt: torch.Tensor,
    route_offset: int = 0,
) -> torch.Tensor:
    """Map logical routes to physical replicas without mutating logical IDs."""
    if log2phy.dtype != torch.int32 or logic_expert_cnt.dtype != torch.int32:
        raise TypeError("log2phy and logic_expert_cnt must be int32 tensors")
    if log2phy.ndim != 2 or logic_expert_cnt.ndim != 1:
        raise ValueError("log2phy must be 2-D and logic_expert_cnt must be 1-D")
    if log2phy.shape[0] != logic_expert_cnt.shape[0]:
        raise ValueError("log2phy and logic_expert_cnt logical dimensions differ")

    flat_ids = logical_ids.reshape(-1).long()
    counts = logic_expert_cnt.index_select(0, flat_ids)
    positions = torch.arange(flat_ids.numel(), device=flat_ids.device)
    replica_slots = torch.remainder(positions + route_offset, counts.long())
    physical_ids = log2phy[flat_ids, replica_slots]
    return physical_ids.to(logical_ids.dtype).view_as(logical_ids)


# ---------------------------------------------------------------------------
# Water-filling dispatch (ported from vLLM expander dispatch.py)
#
# GPU-side runtime quota computation: per-forward, per-layer, each rank
# independently water-fills its local token demand across each expert's
# replica GPUs.  Zero host synchronization.  Assumes <= 2 replicas per
# expert (circulant placement guarantees this for x <= 2.0).
# ---------------------------------------------------------------------------

_wf_dispatch_use_triton = True


@triton.jit
def _wf_count_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_logical,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (e >= 0) & (e < num_logical)
    safe_e = tl.where(valid, e, 0)
    tl.atomic_add(counts_ptr + safe_e, 1.0, mask=valid)


@triton.jit
def _wf_quota_kernel(
    counts_ptr,
    hot_order_ptr,
    cand_gpu_ptr,
    rep_count_ptr,
    quota_ptr,
    alloc_ptr,
    num_logical,
    MAX_PASSES: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    idx = tl.arange(0, BLOCK_G)
    recv = tl.zeros([BLOCK_G], dtype=tl.float32)

    for i in range(num_logical):
        e = tl.load(hot_order_ptr + i)
        c = tl.load(counts_ptr + e)
        k = tl.load(rep_count_ptr + e)
        g0 = tl.load(cand_gpu_ptr + e * 2 + 0)
        g1 = tl.load(cand_gpu_ptr + e * 2 + 1)
        s0 = tl.sum(tl.where(idx == g0, recv, 0.0))
        s1 = tl.sum(tl.where(idx == g1, recv, 0.0))
        # _fill2: closed-form 2-way water-fill
        gap = tl.abs(s0 - s1)
        level = (s0 + s1 + c) * 0.5
        two_a0 = tl.where(s0 <= s1, tl.minimum(c, gap), 0.0)
        two_a1 = tl.where(s0 <= s1, 0.0, tl.minimum(c, gap))
        split0 = level - s0
        split1 = level - s1
        use_split = c > gap
        a0_2 = tl.where(use_split, split0, two_a0)
        a1_2 = tl.where(use_split, split1, two_a1)
        a0 = tl.where(k == 2, a0_2, c)
        a1 = tl.where(k == 2, a1_2, 0.0)
        recv = recv + tl.where(idx == g0, a0, 0.0) + tl.where(idx == g1, a1, 0.0)
        # _round2: largest-remainder integer rounding
        ci = (c + 0.5).to(tl.int32)
        f0 = tl.floor(a0)
        f1 = tl.floor(a1)
        i0 = f0.to(tl.int32)
        i1 = f1.to(tl.int32)
        rem = ci - i0 - i1
        frac0 = a0 - f0
        frac1 = a1 - f1
        give0 = (rem == 1) & (frac0 >= frac1)
        give1 = (rem == 1) & (frac1 > frac0)
        q0 = i0 + give0.to(tl.int32)
        q1 = i1 + give1.to(tl.int32)
        q0 = tl.where(k == 2, q0, ci)
        q1 = tl.where(k == 2, q1, 0)
        tl.store(quota_ptr + e * 2 + 0, q0)
        tl.store(quota_ptr + e * 2 + 1, q1)
        tl.store(alloc_ptr + e * 2 + 0, a0)
        tl.store(alloc_ptr + e * 2 + 1, a1)

    for _p in range(MAX_PASSES - 1):
        for i in range(num_logical):
            e = tl.load(hot_order_ptr + i)
            k = tl.load(rep_count_ptr + e)
            c = tl.load(counts_ptr + e)
            do = (k == 2) & (c > 0.0)
            if do:
                g0 = tl.load(cand_gpu_ptr + e * 2 + 0)
                g1 = tl.load(cand_gpu_ptr + e * 2 + 1)
                oa0 = tl.load(alloc_ptr + e * 2 + 0)
                oa1 = tl.load(alloc_ptr + e * 2 + 1)
                recv = (
                    recv - tl.where(idx == g0, oa0, 0.0) - tl.where(idx == g1, oa1, 0.0)
                )
                s0 = tl.sum(tl.where(idx == g0, recv, 0.0))
                s1 = tl.sum(tl.where(idx == g1, recv, 0.0))
                gap = tl.abs(s0 - s1)
                level = (s0 + s1 + c) * 0.5
                two_a0 = tl.where(s0 <= s1, tl.minimum(c, gap), 0.0)
                two_a1 = tl.where(s0 <= s1, 0.0, tl.minimum(c, gap))
                split0 = level - s0
                split1 = level - s1
                use_split = c > gap
                a0_2 = tl.where(use_split, split0, two_a0)
                a1_2 = tl.where(use_split, split1, two_a1)
                a0 = tl.where(k == 2, a0_2, c)
                a1 = tl.where(k == 2, a1_2, 0.0)
                recv = (
                    recv + tl.where(idx == g0, a0, 0.0) + tl.where(idx == g1, a1, 0.0)
                )
                ci = (c + 0.5).to(tl.int32)
                f0 = tl.floor(a0)
                f1 = tl.floor(a1)
                i0 = f0.to(tl.int32)
                i1 = f1.to(tl.int32)
                rem = ci - i0 - i1
                frac0 = a0 - f0
                frac1 = a1 - f1
                give0 = (rem == 1) & (frac0 >= frac1)
                give1 = (rem == 1) & (frac1 > frac0)
                q0 = i0 + give0.to(tl.int32)
                q1 = i1 + give1.to(tl.int32)
                q0 = tl.where(k == 2, q0, ci)
                q1 = tl.where(k == 2, q1, 0)
                tl.store(quota_ptr + e * 2 + 0, q0)
                tl.store(quota_ptr + e * 2 + 1, q1)
                tl.store(alloc_ptr + e * 2 + 0, a0)
                tl.store(alloc_ptr + e * 2 + 1, a1)


@triton.jit
def _wf_scatter_kernel(
    topk_ids_ptr,
    quota_ptr,
    cand_phys_ptr,
    rank_counter_ptr,
    out_ids_ptr,
    num_logical,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    valid = mask & (e >= 0) & (e < num_logical)
    safe_e = tl.where(valid, e, 0)
    rank = tl.atomic_add(rank_counter_ptr + safe_e, 1, mask=valid)
    q0 = tl.load(quota_ptr + safe_e * 2 + 0, mask=valid, other=0)
    p0 = tl.load(cand_phys_ptr + safe_e * 2 + 0, mask=valid, other=-1)
    p1 = tl.load(cand_phys_ptr + safe_e * 2 + 1, mask=valid, other=-1)
    phys = tl.where(rank < q0, p0, p1)
    phys = tl.where(valid, phys, -1)
    tl.store(out_ids_ptr + offs, phys, mask=mask)


# ---- torch fallback (PPU without triton compilation) ----


def _wf_dispatch_torch(
    flat_ids: torch.Tensor,
    counts: torch.Tensor,
    cand_gpu: torch.Tensor,
    cand_phys: torch.Tensor,
    rep_count: torch.Tensor,
    quota: torch.Tensor,
    rank_counter: torch.Tensor,
    num_logical: int,
    numel: int,
    max_passes: int,
) -> torch.Tensor:
    """Pure-torch WF dispatch fallback (CPU waterfill, GPU scatter)."""
    import numpy as np

    device = flat_ids.device

    # count on GPU → CPU
    counts.zero_()
    counts.scatter_add_(
        0,
        flat_ids.long().clamp(0, num_logical - 1),
        torch.ones(numel, dtype=torch.float32, device=device),
    )
    counts_cpu = counts.cpu().numpy().astype(np.float64)

    # waterfill on CPU (128 experts, microseconds). Recompute hot order from the
    # freshly counted demand (the passed-in hot_order is stale).
    hot_cpu = np.argsort(-counts_cpu, kind="stable")
    cand_gpu_cpu = cand_gpu.cpu().numpy()
    rep_cpu = rep_count.cpu().numpy()
    alloc_cpu = np.zeros((num_logical, 2), dtype=np.float64)
    quota_cpu = np.zeros((num_logical, 2), dtype=np.int32)
    G = int(cand_gpu_cpu.max()) + 1 if cand_gpu_cpu.size > 0 else 1
    recv = np.zeros(max(G, 1), dtype=np.float64)

    for i in range(num_logical):
        e = int(hot_cpu[i])
        c = float(counts_cpu[e])
        k = int(rep_cpu[e])
        g0 = int(cand_gpu_cpu[e, 0])
        g1 = int(cand_gpu_cpu[e, 1])
        s0 = recv[g0] if g0 >= 0 else 0.0
        s1 = recv[g1] if g1 >= 0 else 0.0
        if k == 2 and c > 0:
            gap = abs(s0 - s1)
            level = (s0 + s1 + c) * 0.5
            if c > gap:
                a0 = level - s0
                a1 = level - s1
            elif s0 <= s1:
                a0 = min(c, gap)
                a1 = 0.0
            else:
                a0 = 0.0
                a1 = min(c, gap)
        else:
            a0 = c
            a1 = 0.0
        if g0 >= 0:
            recv[g0] += a0
        if g1 >= 0:
            recv[g1] += a1
        alloc_cpu[e, 0] = a0
        alloc_cpu[e, 1] = a1
        ci = int(round(c))
        f0, f1 = int(np.floor(a0)), int(np.floor(a1))
        rem = ci - f0 - f1
        if rem == 1:
            if (a0 - f0) >= (a1 - f1):
                f0 += 1
            else:
                f1 += 1
        if k == 2:
            quota_cpu[e, 0] = f0
            quota_cpu[e, 1] = f1
        else:
            quota_cpu[e, 0] = ci
            quota_cpu[e, 1] = 0

    # coordinate descent passes
    for _p in range(max_passes - 1):
        for i in range(num_logical):
            e = int(hot_cpu[i])
            k = int(rep_cpu[e])
            c = float(counts_cpu[e])
            if k != 2 or c <= 0:
                continue
            g0 = int(cand_gpu_cpu[e, 0])
            g1 = int(cand_gpu_cpu[e, 1])
            oa0, oa1 = alloc_cpu[e, 0], alloc_cpu[e, 1]
            if g0 >= 0:
                recv[g0] -= oa0
            if g1 >= 0:
                recv[g1] -= oa1
            s0 = recv[g0] if g0 >= 0 else 0.0
            s1 = recv[g1] if g1 >= 0 else 0.0
            gap = abs(s0 - s1)
            level = (s0 + s1 + c) * 0.5
            if c > gap:
                a0 = level - s0
                a1 = level - s1
            elif s0 <= s1:
                a0 = min(c, gap)
                a1 = 0.0
            else:
                a0 = 0.0
                a1 = min(c, gap)
            if g0 >= 0:
                recv[g0] += a0
            if g1 >= 0:
                recv[g1] += a1
            alloc_cpu[e, 0] = a0
            alloc_cpu[e, 1] = a1
            ci = int(round(c))
            f0, f1 = int(np.floor(a0)), int(np.floor(a1))
            rem = ci - f0 - f1
            if rem == 1:
                if (a0 - f0) >= (a1 - f1):
                    f0 += 1
                else:
                    f1 += 1
            quota_cpu[e, 0] = f0
            quota_cpu[e, 1] = f1

    quota.copy_(torch.from_numpy(quota_cpu).to(device))

    # scatter on GPU (sort-based, deterministic)
    sorted_ids, sort_idx = torch.sort(flat_ids.long())
    seg_counts = torch.zeros(num_logical, dtype=torch.int64, device=device)
    seg_counts.scatter_add_(
        0,
        sorted_ids.clamp(0, num_logical - 1),
        torch.ones(numel, dtype=torch.int64, device=device),
    )
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=device), seg_counts.cumsum(0)[:-1]]
    )
    positions = torch.arange(numel, device=device, dtype=torch.int64)
    ranks = positions - starts[sorted_ids.clamp(0, num_logical - 1)]
    q0 = quota[sorted_ids.clamp(0, num_logical - 1).int(), 0].long()
    p0 = cand_phys[sorted_ids.clamp(0, num_logical - 1).int(), 0].long()
    p1 = cand_phys[sorted_ids.clamp(0, num_logical - 1).int(), 1].long()
    phys = torch.where(ranks < q0, p0, p1)
    out = torch.empty(numel, dtype=torch.int32, device=device)
    out[sort_idx] = phys.int()
    return out


# ---- persistent scratch (module-level, lazy init per device) ----

_wf_scratch = {}


def _get_wf_scratch(num_logical: int, num_gpus: int, device: torch.device):
    key = (num_logical, num_gpus, str(device))
    if key not in _wf_scratch:
        _wf_scratch[key] = {
            "counts": torch.zeros(num_logical, dtype=torch.float32, device=device),
            "quota": torch.empty((num_logical, 2), dtype=torch.int32, device=device),
            "alloc": torch.empty((num_logical, 2), dtype=torch.float32, device=device),
            "rank_counter": torch.zeros(num_logical, dtype=torch.int32, device=device),
        }
    return _wf_scratch[key]


def wf_dispatch(
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    logic_expert_cnt: torch.Tensor,
    num_gpus: int,
    num_physical: int,
    policy: str = "iter",
) -> torch.Tensor:
    """GPU-side water-filling dispatch (per-forward, per-layer).

    Replaces logical_to_physical_experts with quota-based replica selection.
    log2phy / logic_expert_cnt remain pure placement data (unmodified).

    Args:
        topk_ids: [num_tokens, top_k] logical expert ids.
        log2phy: [E, pad_k] int32, physical replica ids (-1 padded).
        logic_expert_cnt: [E] int32, actual replica counts.
        num_gpus: EP size (G).
        num_physical: total physical experts (phy_exp_num).
        policy: "wf" (1 pass) or "wf_iter" (8 passes coordinate descent).

    Returns:
        [num_tokens, top_k] physical expert ids.
    """
    global _wf_dispatch_use_triton

    device = topk_ids.device
    E = log2phy.shape[0]
    experts_per_gpu = num_physical // num_gpus if num_gpus > 0 else 1
    # cand_phys: [E, 2] first two replicas (-1 pad)
    cand_phys = log2phy[:, :2].contiguous()
    # cand_gpu: [E, 2] GPU id per replica (phys // experts_per_gpu)
    cand_gpu = torch.where(
        cand_phys >= 0,
        cand_phys // experts_per_gpu,
        torch.full_like(cand_phys, -1),
    ).to(torch.int32)
    rep_count = logic_expert_cnt.clamp(max=2).to(torch.int32)

    flat = topk_ids.reshape(-1).to(torch.int32)
    numel = flat.numel()
    if numel == 0:
        return topk_ids

    scratch = _get_wf_scratch(E, num_gpus, device)
    counts = scratch["counts"]
    quota = scratch["quota"]
    alloc = scratch["alloc"]
    rank_counter = scratch["rank_counter"]
    max_passes = 1 if policy == "wf" else 8

    if _wf_dispatch_use_triton:
        try:
            counts.zero_()
            BLOCK_SIZE = min(1024, triton.next_power_of_2(numel))
            grid = (ceil_div(numel, BLOCK_SIZE),)
            _wf_count_kernel[grid](
                flat,
                counts,
                E,
                numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            hot_order = torch.argsort(counts, descending=True, stable=True)
            block_g = max(1, triton.next_power_of_2(num_gpus))
            _wf_quota_kernel[(1,)](
                counts,
                hot_order,
                cand_gpu,
                rep_count,
                quota,
                alloc,
                E,
                MAX_PASSES=max_passes,
                BLOCK_G=block_g,
            )
            rank_counter.zero_()
            out_flat = torch.empty(numel, dtype=torch.int32, device=device)
            _wf_scatter_kernel[grid](
                flat,
                quota,
                cand_phys,
                rank_counter,
                out_flat,
                E,
                numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out_flat.view_as(topk_ids)
        except Exception:
            _wf_dispatch_use_triton = False
            logger.warning(
                "triton WF dispatch unavailable, falling back to torch",
                exc_info=True,
            )

    # torch fallback
    out_flat = _wf_dispatch_torch(
        flat,
        counts,
        cand_gpu,
        cand_phys,
        rep_count,
        quota,
        rank_counter,
        E,
        numel,
        max_passes,
    )
    return out_flat.view_as(topk_ids)
