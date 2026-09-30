"""Per-forward expert activation probe (EXPERT_STATS_PROBE=1).

The engine already owns an ``ExpertStatsBuffer`` for EPLB, but it only keeps
``log_stats[layer, logical_expert]`` and ``gpu_loads[layer, ep_rank]`` -- the
per-*physical*-expert resolution is destroyed inside the recording kernel by the
``physical_id // experts_per_rank`` bucketing. This probe owns its own buffers so
it can keep all three distributions at once:

    log_stats[layer, logical_expert]    logical expert activation
    phy_stats[layer, physical_expert]   physical expert activation (replica level)
    gpu_loads[layer, ep_rank]           derived by block-summing phy_stats

``gpu_loads`` is a pure function of ``phy_stats`` because the EPLB config asserts
``phy_exp_num % ep_size == 0``, so the GPU view is self-consistent with the
physical view by construction -- and cross-checkable against the C++ export.

The probe is fully separate from the EPLB path: it never touches
``ExpertStatsBuffer`` or ``ep_kernels.record_expert_stats``, so with the env flag
off it costs one ``is None`` check per MoE layer.

Every prefill forward exports one NPZ and then zeroes the buffers, which is what
lets the offline collector attribute forwards to client-side batch windows.

Env:
    EXPERT_STATS_PROBE=1            enable
    EXPERT_STATS_PROBE_DIR=<dir>    export directory (required to export)
    EXPERT_STATS_PROBE_RANKS=0      comma-separated ranks that record+export
    EXPERT_STATS_PROBE_MAX_EXPORTS  stop exporting after N files (0 = unlimited)
"""

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.math import ceil_div

_LOGGER = logging.getLogger(__name__)

SCHEMA = "rtp_llm.expert_stats_probe/v1"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


def _env_ranks(name: str, default: str = "0") -> Optional[frozenset]:
    """Parse a rank filter. ``all`` (or empty) means every rank."""
    raw = os.environ.get(name, default).strip().lower()
    if raw in ("all", "*"):
        return None
    return frozenset(int(part) for part in raw.split(",") if part.strip())


EXPERT_STATS_PROBE_ENABLED: bool = _env_flag("EXPERT_STATS_PROBE")
_EXPORT_DIR: str = os.environ.get("EXPERT_STATS_PROBE_DIR", "")
_EXPORT_RANKS: Optional[frozenset] = _env_ranks("EXPERT_STATS_PROBE_RANKS")
_MAX_EXPORTS: int = int(os.environ.get("EXPERT_STATS_PROBE_MAX_EXPORTS", "0"))
# Recorded into every NPZ so the collector can label an arm without a side file.
_PLACEMENT: str = os.environ.get("EPLB_PLACEMENT", "eplb")
_DISPATCH_POLICY: str = os.environ.get("EPLB_DISPATCH_POLICY", "rr")


@triton.jit
def _record_probe_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    log_stats_ptr,
    phy_stats_ptr,
    total_ids,
    layer_idx,
    log_exp_num,
    phy_exp_num,
    log_stride,
    phy_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Same shape as ep_kernels._record_expert_stats_kernel, but the second
    buffer is indexed by physical expert id instead of by EP rank."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_ids

    logical_ids = tl.load(logical_topk_ids_ptr + offsets, mask=mask, other=-1).to(
        tl.int64
    )
    physical_ids = tl.load(physical_topk_ids_ptr + offsets, mask=mask, other=-1).to(
        tl.int64
    )

    # Padded / invalid ids are steered into the trailing trash column instead of
    # being dropped, so the fallback and the kernel agree bit for bit.
    valid_logical = mask & (logical_ids >= 0) & (logical_ids < log_exp_num)
    logical_slot = tl.where(valid_logical, logical_ids, log_exp_num)
    tl.atomic_add(log_stats_ptr + layer_idx * log_stride + logical_slot, 1, mask=mask)

    valid_physical = mask & (physical_ids >= 0) & (physical_ids < phy_exp_num)
    physical_slot = tl.where(valid_physical, physical_ids, phy_exp_num)
    tl.atomic_add(phy_stats_ptr + layer_idx * phy_stride + physical_slot, 1, mask=mask)


def _record_probe_torch(
    logical_topk_ids: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    log_stats_buf: torch.Tensor,
    phy_stats_buf: torch.Tensor,
    layer_idx: int,
    log_exp_num: int,
    phy_exp_num: int,
) -> None:
    """Pure-torch fallback (PPU and any device without a usable triton backend).

    Boolean indexing would force a device->host sync, so invalid ids are routed
    to the trash column with ``torch.where`` instead of being filtered out.
    """
    for ids, buf, limit in (
        (logical_topk_ids, log_stats_buf, log_exp_num),
        (physical_topk_ids, phy_stats_buf, phy_exp_num),
    ):
        flat = ids.reshape(-1).long()
        valid = (flat >= 0) & (flat < limit)
        slots = torch.where(valid, flat, torch.full_like(flat, limit))
        buf[layer_idx].scatter_add_(0, slots, torch.ones_like(slots, dtype=buf.dtype))


class ExpertStatsProbe:
    """Owns the probe buffers and the per-forward NPZ export."""

    def __init__(self) -> None:
        self.layer_num: int = 0
        self.log_exp_num: int = 0
        self.phy_exp_num: int = 0
        self.ep_size: int = 0
        self.experts_per_rank: int = 0
        self.rank: int = 0
        self.records_enabled: bool = False

        self._log_stats: Optional[torch.Tensor] = None
        self._phy_stats: Optional[torch.Tensor] = None
        # Only a prefill forward records; decode counters are dropped so every
        # exported NPZ describes exactly one prefill forward.
        self._recording: bool = False
        self._input_tokens: int = 0
        # Host-side bookkeeping; the offline runner asserts one gate call per
        # layer per batch and the online probe keeps the same contract.
        self._layer_calls: Optional[np.ndarray] = None
        self._num_tokens: int = 0
        self._export_seq: int = 0
        self._exports_written: int = 0
        self._use_triton: bool = True
        self._complaints: Dict[str, int] = {}

    # ---------------------------------------------------------------- setup

    def configure(
        self,
        layer_num: int,
        log_exp_num: int,
        phy_exp_num: int,
        ep_size: int,
        rank: int,
    ) -> None:
        """Called once from the model descriptor after weights are bound."""
        self.layer_num = int(layer_num)
        self.log_exp_num = int(log_exp_num)
        self.phy_exp_num = int(phy_exp_num)
        self.ep_size = max(int(ep_size), 1)
        self.experts_per_rank = ceil_div(self.phy_exp_num, self.ep_size)
        self.rank = int(rank)
        self.records_enabled = _EXPORT_RANKS is None or self.rank in _EXPORT_RANKS
        if self.phy_exp_num % self.ep_size != 0:
            self._complain(
                "uneven_ep",
                "phy_exp_num=%d is not divisible by ep_size=%d; derived gpu_loads "
                "will attribute the tail experts to the last rank",
                self.phy_exp_num,
                self.ep_size,
            )
        _LOGGER.info(
            "[EXPERT_STATS_PROBE] rank=%d layers=%d log_exp=%d phy_exp=%d ep=%d "
            "recording=%s dir=%s",
            self.rank,
            self.layer_num,
            self.log_exp_num,
            self.phy_exp_num,
            self.ep_size,
            self.records_enabled,
            _EXPORT_DIR or "<unset>",
        )

    def _ensure_buffers(self, device: torch.device) -> None:
        if self._log_stats is not None:
            return
        # One trailing trash column absorbs padded/invalid expert ids.
        self._log_stats = torch.zeros(
            (self.layer_num, self.log_exp_num + 1), dtype=torch.int32, device=device
        )
        self._phy_stats = torch.zeros(
            (self.layer_num, self.phy_exp_num + 1), dtype=torch.int32, device=device
        )
        self._layer_calls = np.zeros(self.layer_num, dtype=np.int64)

    # --------------------------------------------------------------- record

    def begin_forward(self, is_prefill: bool, input_tokens: int) -> None:
        """Arm recording for a prefill forward; disarm for decode."""
        self._recording = bool(is_prefill) and self.records_enabled
        if self._recording:
            self._input_tokens = int(input_tokens)

    def record(
        self,
        layer_idx: int,
        logical_topk_ids: torch.Tensor,
        physical_topk_ids: torch.Tensor,
    ) -> None:
        """Accumulate one MoE layer's routing decision for the current forward."""
        if not self._recording or self.layer_num == 0:
            return
        if layer_idx < 0 or layer_idx >= self.layer_num:
            self._complain(
                "layer_range", "layer_idx=%d outside [0,%d)", layer_idx, self.layer_num
            )
            return
        total_ids = logical_topk_ids.numel()
        if total_ids == 0:
            return
        try:
            self._ensure_buffers(logical_topk_ids.device)
            self._record_ids(layer_idx, logical_topk_ids, physical_topk_ids, total_ids)
        except Exception:  # profiling must never take down the server
            self._complain("record", "expert stats probe record failed")
            return
        self._layer_calls[layer_idx] += 1
        if self._num_tokens == 0:
            self._num_tokens = int(logical_topk_ids.shape[0])

    def _record_ids(
        self,
        layer_idx: int,
        logical_topk_ids: torch.Tensor,
        physical_topk_ids: torch.Tensor,
        total_ids: int,
    ) -> None:
        if self._use_triton:
            BLOCK_SIZE = min(1024, triton.next_power_of_2(total_ids))
            grid = (ceil_div(total_ids, BLOCK_SIZE),)
            try:
                _record_probe_kernel[grid](
                    logical_topk_ids.contiguous(),
                    physical_topk_ids.contiguous(),
                    self._log_stats,
                    self._phy_stats,
                    total_ids,
                    layer_idx,
                    self.log_exp_num,
                    self.phy_exp_num,
                    self._log_stats.shape[1],
                    self._phy_stats.shape[1],
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return
            except Exception:
                self._use_triton = False
                _LOGGER.warning(
                    "[EXPERT_STATS_PROBE] triton kernel unavailable, falling back "
                    "to torch",
                    exc_info=True,
                )
        _record_probe_torch(
            logical_topk_ids,
            physical_topk_ids,
            self._log_stats,
            self._phy_stats,
            layer_idx,
            self.log_exp_num,
            self.phy_exp_num,
        )

    # ---------------------------------------------------------------- flush

    def end_forward(self, success: bool) -> None:
        """Close the forward: export when it was a successful prefill, else drop.

        A forward that raised is discarded whole rather than half-exported, and a
        decode forward is discarded because the probe reports prefill only.
        """
        recording = self._recording
        self._recording = False
        if recording and success:
            self.flush()
        else:
            self._reset()

    def flush(self) -> None:
        """Export the current forward's counters, then zero them.

        A forward that routed no tokens (or a rank that is not recording) is a
        no-op.
        """
        if not self.records_enabled or self._log_stats is None:
            return
        if self._num_tokens == 0:
            return
        try:
            self._export()
        except Exception:
            self._complain("export", "expert stats probe export failed")
        finally:
            self._reset()

    def _reset(self) -> None:
        if self._log_stats is not None:
            self._log_stats.zero_()
            self._phy_stats.zero_()
        if self._layer_calls is not None:
            self._layer_calls.fill(0)
        self._num_tokens = 0
        self._input_tokens = 0

    def _derive_gpu_loads(self, phy_stats: np.ndarray) -> np.ndarray:
        """Block-sum physical experts into their owning EP rank.

        Padded so an uneven phy_exp_num/ep_size split still yields [L, ep_size]
        (the tail lands on the last rank, matching the kernel's clamp).
        """
        layer_num = phy_stats.shape[0]
        padded = self.experts_per_rank * self.ep_size
        if padded > self.phy_exp_num:
            phy_stats = np.pad(phy_stats, ((0, 0), (0, padded - self.phy_exp_num)))
        blocks = phy_stats.reshape(layer_num, self.ep_size, self.experts_per_rank)
        return blocks.sum(axis=2).astype(np.int64)

    def _export(self) -> None:
        if not _EXPORT_DIR:
            self._complain("no_dir", "EXPERT_STATS_PROBE_DIR is unset; nothing exported")
            return
        if 0 < _MAX_EXPORTS <= self._exports_written:
            return

        timestamp_ms = int(time.time() * 1000)
        # One D2H copy per buffer per forward; the trash column is split off here.
        log_full = self._log_stats.cpu().numpy()
        phy_full = self._phy_stats.cpu().numpy()
        log_stats = log_full[:, : self.log_exp_num]
        phy_stats = phy_full[:, : self.phy_exp_num]
        dropped = int(log_full[:, self.log_exp_num].sum()) + int(
            phy_full[:, self.phy_exp_num].sum()
        )
        gpu_loads = self._derive_gpu_loads(phy_stats.astype(np.int64))

        os.makedirs(_EXPORT_DIR, exist_ok=True)
        path = os.path.join(
            _EXPORT_DIR,
            f"expert_stats_rank{self.rank:03d}_seq{self._export_seq:06d}.npz",
        )
        np.savez_compressed(
            path,
            schema=SCHEMA,
            timestamp_ms=timestamp_ms,
            rank=self.rank,
            export_seq=self._export_seq,
            layer_num=self.layer_num,
            log_exp_num=self.log_exp_num,
            phy_exp_num=self.phy_exp_num,
            ep_size=self.ep_size,
            experts_per_rank=self.experts_per_rank,
            num_tokens=self._num_tokens,
            input_tokens=self._input_tokens,
            dropped_ids=dropped,
            layer_calls=self._layer_calls.copy(),
            placement=_PLACEMENT,
            dispatch=_DISPATCH_POLICY,
            log_stats=log_stats,
            phy_stats=phy_stats,
            gpu_loads=gpu_loads,
        )
        self._export_seq += 1
        self._exports_written += 1

    # ---------------------------------------------------------------- misc

    def _complain(self, key: str, message: str, *fmt_args) -> None:
        """Log each distinct problem once; the probe must stay silent afterwards."""
        seen = self._complaints.get(key, 0)
        self._complaints[key] = seen + 1
        if seen == 0:
            _LOGGER.warning("[EXPERT_STATS_PROBE] " + message, *fmt_args, exc_info=True)


class _DisabledProbe:
    """No-op stand-in so call sites stay branch-free when the flag is off."""

    records_enabled = False

    def configure(self, *_args, **_kwargs) -> None:
        return

    def begin_forward(self, *_args, **_kwargs) -> None:
        return

    def record(self, *_args, **_kwargs) -> None:
        return

    def end_forward(self, *_args, **_kwargs) -> None:
        return

    def flush(self, *_args, **_kwargs) -> None:
        return


_PROBE = None


def get_expert_stats_probe():
    """Lazy singleton; returns a no-op probe unless EXPERT_STATS_PROBE=1."""
    global _PROBE
    if _PROBE is None:
        _PROBE = ExpertStatsProbe() if EXPERT_STATS_PROBE_ENABLED else _DisabledProbe()
    return _PROBE
