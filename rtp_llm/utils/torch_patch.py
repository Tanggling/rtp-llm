import logging
from typing import List

import torch

# 1. 保存原始 concat
original_concat = torch.concat


# 2. 定义自定义 concat
def custom_concat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    try:
        return original_concat(tensors, dim=dim).contiguous()
    except RuntimeError as e:
        if "cat_cuda" in str(e) and "Float8_e4m3fn" in str(e):
            logging.info("Caught Float8_e4m3fn concat error. Falling back to CPU.")
            original_device = tensors[0].device
            tensors_cpu = [t.cpu() for t in tensors]
            result_cpu = original_concat(tensors_cpu, dim=dim).contiguous()
            return result_cpu.to(original_device)
        else:
            raise


# 3. 劫持 torch.concat
torch.concat = custom_concat


# NCCL does not accept UE8M0 directly. Checkpoint scales are byte-sized, so
# broadcast an aliasing uint8 view and preserve the original tensor dtype.
if hasattr(torch, "float8_e8m0fnu"):
    import torch.distributed as _dist

    _ue8m0_dtype = torch.float8_e8m0fnu
    _original_broadcast = _dist.broadcast

    def _ue8m0_broadcast(tensor, *args, **kwargs):
        if tensor.dtype is _ue8m0_dtype:
            return _original_broadcast(tensor.view(torch.uint8), *args, **kwargs)
        return _original_broadcast(tensor, *args, **kwargs)

    _dist.broadcast = _ue8m0_broadcast
