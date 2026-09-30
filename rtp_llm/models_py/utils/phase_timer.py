"""Phase-staged CUDA-event timing for the models_py MoE forward path.

Ported from the MoE-Bench vLLM expander (``distributed/expander/phase_timer.py``)
and adapted to RTP-LLM's router/executor split.

Checkpoint taxonomy, per MoE layer::

    T0  moe_start      GenericMoeLayer.forward entry
    T1  route_done     gate + topk + EPLB logical->physical remap done
    T2  dispatch_done  FusedMoe: router.prepare() returned (all-to-all done)
    T3  gemm_done      FusedMoe: fused_experts.execute() returned
    T4  combine_done   FusedMoe: router.finalize() returned

    gate+route = T1 - T0      dispatch  = T2 - T1
    gemm       = T3 - T2      combine   = T4 - T3
    moe_total  = T4 - T0

The reference fuses T0..T2 into a single ``gate+dispatch`` phase. Here ``route``
is split out because the EPLB dispatch policies under comparison (``rr`` / ``wf``
/ ``wf_iter`` / ``wf_k``) live exactly in that window; ``gate_dispatch = T2 - T0``
is still reported so numbers stay comparable with the reference.

``gate+route`` still bundles the gate GEMM, top-k and the expert-load stats
around the policy call, which is not enough when the arms under comparison vary
placement *and* policy together (eplb+wf_k vs circulant+wf) -- differencing
``gate+route`` across arms then leaves the two confounded. Setting
``PHASE_TIMER_ROUTE_SPLIT=1`` adds two sub-checkpoints inside T0..T1 that isolate
the policy call itself::

    route_gate_topk  moe_start  -> topk_done     gate GEMM + top-k
    route_policy     topk_done  -> policy_done   rr / wf / wf_iter / wf_k
    route_tail       policy_done -> route_done   record_expert_stats + debug dumps

Design constraints carried over from the reference:

* Only ``torch.cuda.Event`` records sit on the critical path -- no per-layer
  host sync. Exactly one sync happens per report, on the last event recorded.
* Every checkpoint is also reported as an offset from a per-forward ``epoch``
  event, so a post-processing step can do cross-rank barrier correction (the
  straggler GEMM window is ``max_r(T3_r) - max_r(T2_r)``, not ``mean_r(T3-T2)``).
* Recording is gated: it is armed by ``PHASE_TIMER=1``, only ever active on
  prefill, and skipped for forwards much smaller than the first one seen (so
  a tail chunk does not pollute the per-layer averages).
* A forward that raises is rolled back wholesale, never half-recorded.

Environment variables::

    PHASE_TIMER=1                  arm recording (default off, zero overhead)
    PHASE_TIMER_REPORT_STEP=N      report every N recorded forwards (default 1)
    PHASE_TIMER_EXPORT_DIR=<dir>   also dump per-forward JSON for offline
                                   cross-rank barrier correction
    PHASE_TIMER_DISPATCH_SPLIT=1   record the dispatch sub-checkpoint channel
                                   (quant / layout / payload) -- extra events
    PHASE_TIMER_ROUTE_SPLIT=1      record the route sub-checkpoint channel
                                   (gate_topk / policy / tail) -- extra events
    PHASE_TIMER_GEMM_SPLIT=1       record the gemm sub-checkpoint channel
                                   (align / kernels / tail) -- extra events;
                                   `gemm_kernels` is the part that scales with
                                   this rank's expert load
    DEEPEP_TRAFFIC_STATS=1         attach DeepEP send/receive and local-assignment
                                   counters to each recorded layer; tensors stay
                                   on device until the existing flush sync
    PHASE_TIMER_MIN_TOKENS=N       override the token gate (default: half the
                                   token count of the first recorded forward)
    PHASE_TIMER_STRICT=1           raise on out-of-order checkpoints instead of
                                   dropping them with a one-shot warning
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

_LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


MOE_CHECKPOINTS = (
    "moe_start",
    "route_done",
    "dispatch_done",
    "gemm_done",
    "combine_done",
)
MOE_PHASE_NAMES = ("gate+route", "dispatch", "gemm", "combine")

# Sub-checkpoints inside router.prepare(). Order follows the RTP-LLM DeepEP
# normal router, which quantizes before computing the dispatch layout (the
# reference vLLM router does the two in the opposite order).
DISPATCH_SUB_CHECKPOINTS = ("quant_done", "layout_done")
DISPATCH_SUB_PHASE_NAMES = ("dispatch_quant", "dispatch_layout", "dispatch_payload")

# Sub-checkpoints inside T0..T1, isolating the EPLB dispatch policy call from
# the gate GEMM / top-k before it and the expert-load stats after it.
ROUTE_SUB_CHECKPOINTS = ("topk_done", "policy_done")
ROUTE_SUB_PHASE_NAMES = ("route_gate_topk", "route_policy", "route_tail")

# Sub-checkpoints inside T2..T3, isolating the expert GEMMs from the alignment
# before them and the activation / top-k reduction after them. Without this the
# `gemm` phase is the whole executor call, whose load-independent tail dilutes
# any expert-imbalance signal the GEMMs themselves carry.
GEMM_SUB_CHECKPOINTS = ("align_done", "kernels_done")
GEMM_SUB_PHASE_NAMES = ("gemm_align", "gemm_kernels", "gemm_tail")

ATTN_CHECKPOINTS = ("attn_start", "attn_end")
LAYER_CHECKPOINTS = ("layer_start", "layer_end")
MOE_BLOCK_CHECKPOINTS = ("moe_block_start", "moe_block_end")

# Read once at import so the disabled path costs a single attribute load.
PHASE_TIMER_ENABLED: bool = _env_flag("PHASE_TIMER")
_DISPATCH_SPLIT: bool = _env_flag("PHASE_TIMER_DISPATCH_SPLIT")
_ROUTE_SPLIT: bool = _env_flag("PHASE_TIMER_ROUTE_SPLIT")
_GEMM_SPLIT: bool = _env_flag("PHASE_TIMER_GEMM_SPLIT")
_DEEPEP_TRAFFIC_STATS: bool = _env_flag("DEEPEP_TRAFFIC_STATS")
_STRICT: bool = _env_flag("PHASE_TIMER_STRICT")
_REPORT_STEP: int = int(os.environ.get("PHASE_TIMER_REPORT_STEP", "1") or "1")
_EXPORT_DIR: str = os.environ.get("PHASE_TIMER_EXPORT_DIR", "")
_MIN_TOKENS_ENV: str = os.environ.get("PHASE_TIMER_MIN_TOKENS", "")

# Arm identity, echoed into the log line and the export so a reduced table is
# self-describing (which placement / policy produced these numbers).
_PLACEMENT: str = os.environ.get("EPLB_PLACEMENT", "eplb")
_DISPATCH_POLICY: str = os.environ.get("EPLB_DISPATCH_POLICY", "rr")


@dataclass
class _MoeRecord:
    """One MoE layer's ordered checkpoint events."""

    layer_idx: int
    events: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    sub: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    route: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    gemm: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    traffic: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return len(self.events) == len(MOE_CHECKPOINTS)


@dataclass
class _PairRecord:
    """A start/end event pair scoped to one decoder layer."""

    layer_idx: int
    start: torch.cuda.Event
    end: Optional[torch.cuda.Event] = None


@dataclass
class _ForwardRecord:
    """All events recorded during a single model forward."""

    num_tokens: int
    epoch: torch.cuda.Event
    moe: List[_MoeRecord] = field(default_factory=list)
    moe_block: List[_PairRecord] = field(default_factory=list)
    attn: List[_PairRecord] = field(default_factory=list)
    layer: List[_PairRecord] = field(default_factory=list)


class PhaseTimer:
    """Accumulates CUDA-event checkpoints and derives phase breakdowns.

    Not thread-safe by design: the models_py forward path is single-threaded per
    device, and adding a lock would put a host-side mutex on the critical path.
    """

    def __init__(self) -> None:
        self.capture_armed: bool = PHASE_TIMER_ENABLED
        self.dispatch_split: bool = _DISPATCH_SPLIT
        self.route_split: bool = _ROUTE_SPLIT
        self.gemm_split: bool = _GEMM_SPLIT
        self._forwards: List[_ForwardRecord] = []
        self._cur: Optional[_ForwardRecord] = None
        self._prefill_tokens: Optional[int] = (
            int(_MIN_TOKENS_ENV) * 2 if _MIN_TOKENS_ENV else None
        )
        self._last_event: Optional[torch.cuda.Event] = None
        self._flush_cnt: int = 0
        self._export_seq: int = 0
        self._warned: set = set()

    # ------------------------------------------------------------------ arming

    def arm(
        self,
        *,
        dispatch_split: Optional[bool] = None,
        route_split: Optional[bool] = None,
        gemm_split: Optional[bool] = None,
    ) -> None:
        self.capture_armed = True
        if dispatch_split is not None:
            self.dispatch_split = dispatch_split
        if route_split is not None:
            self.route_split = route_split
        if gemm_split is not None:
            self.gemm_split = gemm_split

    def disarm(self) -> None:
        self.capture_armed = False
        self._cur = None

    @property
    def recording(self) -> bool:
        return self._cur is not None

    # ---------------------------------------------------------- forward window

    def begin_forward(self, num_tokens: int) -> None:
        """Open a recording window. No-op unless armed and the token gate passes."""
        self._cur = None
        if not self.capture_armed:
            return
        num_tokens = int(num_tokens)
        if self._prefill_tokens is None:
            self._prefill_tokens = num_tokens
        elif num_tokens < self._prefill_tokens // 2:
            # A short tail chunk would drag every per-layer average down.
            return
        epoch = self._event()
        epoch.record()
        self._last_event = epoch
        self._cur = _ForwardRecord(num_tokens=num_tokens, epoch=epoch)

    def end_forward(self, *, success: bool) -> None:
        """Close the window. A failed forward is discarded, never half-kept."""
        cur = self._cur
        self._cur = None
        if cur is None:
            return
        if success:
            self._forwards.append(cur)

    # ------------------------------------------------------------- checkpoints

    def record_moe_checkpoint(self, name: str, layer_idx: int = -1) -> None:
        """Record one of ``MOE_CHECKPOINTS`` in strict order."""
        cur = self._cur
        if cur is None:
            return
        if name not in MOE_CHECKPOINTS:
            raise ValueError(f"Unknown MoE checkpoint: {name}")
        if name == MOE_CHECKPOINTS[0]:
            if cur.moe and not cur.moe[-1].complete:
                # Drop the stale partial record, otherwise this layer's later
                # checkpoints would be absorbed into the previous layer's.
                self._complain(
                    f"previous MoE checkpoint sequence for layer "
                    f"{cur.moe[-1].layer_idx} is incomplete"
                )
                cur.moe.pop()
            cur.moe.append(_MoeRecord(layer_idx=layer_idx))
        elif not cur.moe:
            # FusedMoe reached from a model_desc that does not record moe_start.
            self._complain(f"MoE checkpoint {name} has no moe_start")
            return
        record = cur.moe[-1]
        expected_idx = len(record.events)
        expected = (
            MOE_CHECKPOINTS[expected_idx]
            if expected_idx < len(MOE_CHECKPOINTS)
            else None
        )
        if name != expected:
            # Leave the record incomplete so per_layer() discards it wholesale
            # rather than reporting a phase measured against the wrong bound.
            self._complain(
                f"out-of-order MoE checkpoint: expected={expected}, got={name}"
            )
            return
        self._record_into(record.events, name)

    def record_dispatch_split(self, name: str) -> None:
        """Record a sub-checkpoint inside ``router.prepare()``."""
        if name not in DISPATCH_SUB_CHECKPOINTS:
            raise ValueError(f"Unknown dispatch sub-checkpoint: {name}")
        cur = self._cur
        if cur is None or not self.dispatch_split:
            return
        if not cur.moe:
            self._complain(f"dispatch sub-checkpoint {name} has no moe_start")
            return
        self._record_into(cur.moe[-1].sub, name)

    def record_route_split(self, name: str) -> None:
        """Record a sub-checkpoint inside the gate+route window (T0..T1)."""
        if name not in ROUTE_SUB_CHECKPOINTS:
            raise ValueError(f"Unknown route sub-checkpoint: {name}")
        cur = self._cur
        if cur is None or not self.route_split:
            return
        if not cur.moe:
            self._complain(f"route sub-checkpoint {name} has no moe_start")
            return
        self._record_into(cur.moe[-1].route, name)

    def record_gemm_split(self, name: str) -> None:
        """Record a sub-checkpoint inside the expert-GEMM window (T2..T3)."""
        if name not in GEMM_SUB_CHECKPOINTS:
            raise ValueError(f"Unknown gemm sub-checkpoint: {name}")
        cur = self._cur
        if cur is None or not self.gemm_split:
            return
        if not cur.moe:
            self._complain(f"gemm sub-checkpoint {name} has no moe_start")
            return
        self._record_into(cur.moe[-1].gemm, name)

    def record_deepep_traffic(
        self,
        *,
        source_tokens: int,
        source_assignments: torch.Tensor,
        dispatch_send_by_rank: torch.Tensor,
        dispatch_recv_rows: int,
        local_assignments: torch.Tensor,
    ) -> None:
        """Attach device-resident DeepEP traffic counters to the current layer."""
        if not _DEEPEP_TRAFFIC_STATS:
            return
        cur = self._cur
        if cur is None:
            return
        if not cur.moe:
            self._complain("DeepEP traffic record has no moe_start")
            return
        record = cur.moe[-1]
        if record.traffic:
            self._complain(
                f"duplicate DeepEP traffic record for layer {record.layer_idx}"
            )
            return
        record.traffic = {
            "source_tokens": int(source_tokens),
            "source_assignments": source_assignments,
            "dispatch_send_by_rank": dispatch_send_by_rank,
            "dispatch_recv_rows": int(dispatch_recv_rows),
            "local_assignments": local_assignments,
        }

    def record_moe_scheduled_rows(self, scheduled_rows: torch.Tensor) -> None:
        """Attach the executor's useful GEMM-row count to the traffic record."""
        if not _DEEPEP_TRAFFIC_STATS:
            return
        cur = self._cur
        if cur is None:
            return
        if not cur.moe or not cur.moe[-1].traffic:
            self._complain("MoE scheduled-row record has no DeepEP traffic record")
            return
        traffic = cur.moe[-1].traffic
        if "scheduled_rows" in traffic:
            self._complain(
                f"duplicate MoE scheduled-row record for layer {cur.moe[-1].layer_idx}"
            )
            return
        traffic["scheduled_rows"] = scheduled_rows

    def moe_block_begin(self, layer_idx: int) -> None:
        self._pair_begin("moe_block", layer_idx)

    def moe_block_end(self, layer_idx: int) -> None:
        self._pair_end("moe_block", layer_idx)

    def attn_begin(self, layer_idx: int) -> None:
        self._pair_begin("attn", layer_idx)

    def attn_end(self, layer_idx: int) -> None:
        self._pair_end("attn", layer_idx)

    def layer_begin(self, layer_idx: int) -> None:
        self._pair_begin("layer", layer_idx)

    def layer_end(self, layer_idx: int) -> None:
        self._pair_end("layer", layer_idx)

    # ----------------------------------------------------------------- derived

    def per_layer(self, forward: int = -1) -> Dict[str, Any]:
        """Per-layer phase breakdown in ms for one recorded forward.

        Returns parallel lists indexed by position in ``layer_index``, plus a
        ``checkpoints`` list of ``{name: ms_from_epoch}`` dicts for offline
        cross-rank correction.
        """
        if not self._forwards:
            return {}
        self.synchronize()
        rec = self._forwards[forward]
        epoch = rec.epoch

        moe_by_layer = {r.layer_idx: r for r in rec.moe if r.complete}
        block_by_layer = {r.layer_idx: r for r in rec.moe_block if r.end is not None}
        attn_by_layer = {r.layer_idx: r for r in rec.attn if r.end is not None}
        layer_by_layer = {r.layer_idx: r for r in rec.layer if r.end is not None}

        indices = sorted(
            set(layer_by_layer) | set(moe_by_layer) | set(attn_by_layer)
        )
        keys = (
            "gate_route",
            "dispatch",
            "gemm",
            "combine",
            "gate_dispatch",
            "moe_total",
            "moe_block",
            "attn",
            "layer",
            "router",
            "overhead",
        )
        out: Dict[str, Any] = {k: [] for k in keys}
        out["layer_index"] = indices
        out["checkpoints"] = []
        out["dispatch_split"] = []
        out["route_split"] = []
        out["gemm_split"] = []
        if _DEEPEP_TRAFFIC_STATS:
            out["deepep_traffic"] = []
        out["num_tokens"] = rec.num_tokens

        for idx in indices:
            moe = moe_by_layer.get(idx)
            if moe is not None:
                ev = moe.events
                gate_route = ev["moe_start"].elapsed_time(ev["route_done"])
                dispatch = ev["route_done"].elapsed_time(ev["dispatch_done"])
                gemm = ev["dispatch_done"].elapsed_time(ev["gemm_done"])
                combine = ev["gemm_done"].elapsed_time(ev["combine_done"])
                gate_dispatch = ev["moe_start"].elapsed_time(ev["dispatch_done"])
                moe_total = ev["moe_start"].elapsed_time(ev["combine_done"])
                out["checkpoints"].append(
                    {n: epoch.elapsed_time(e) for n, e in ev.items()}
                )
                out["dispatch_split"].append(self._split_phases(moe, epoch))
                out["route_split"].append(self._split_route(moe, epoch))
                out["gemm_split"].append(self._split_gemm(moe, epoch))
                if _DEEPEP_TRAFFIC_STATS:
                    out["deepep_traffic"].append(self._traffic_values(moe))
            else:
                gate_route = dispatch = gemm = combine = 0.0
                gate_dispatch = moe_total = 0.0
                out["checkpoints"].append({})
                out["dispatch_split"].append({})
                out["route_split"].append({})
                out["gemm_split"].append({})
                if _DEEPEP_TRAFFIC_STATS:
                    out["deepep_traffic"].append({})

            block = block_by_layer.get(idx)
            moe_block = (
                block.start.elapsed_time(block.end) if block is not None else moe_total
            )
            attn_rec = attn_by_layer.get(idx)
            attn = attn_rec.start.elapsed_time(attn_rec.end) if attn_rec else 0.0
            layer_rec = layer_by_layer.get(idx)
            layer = layer_rec.start.elapsed_time(layer_rec.end) if layer_rec else 0.0

            out["gate_route"].append(gate_route)
            out["dispatch"].append(dispatch)
            out["gemm"].append(gemm)
            out["combine"].append(combine)
            out["gate_dispatch"].append(gate_dispatch)
            out["moe_total"].append(moe_total)
            out["moe_block"].append(moe_block)
            out["attn"].append(attn)
            out["layer"].append(layer)
            # Everything inside the MoE block but outside T0..T4: shared expert,
            # its gate, and the allreduce on the shared branch.
            out["router"].append(max(0.0, moe_block - moe_total))
            out["overhead"].append(max(0.0, layer - attn - moe_block))

        return out

    def report(self) -> Dict[str, Any]:
        """Per-rank sums across every recorded forward."""
        if not self._forwards:
            return {}
        totals: Dict[str, float] = {}
        layers = 0
        moe_layers = 0
        tokens = 0
        split_totals: Dict[str, float] = {}
        for i in range(len(self._forwards)):
            pl = self.per_layer(i)
            if not pl:
                continue
            layers += len(pl["layer_index"])
            moe_layers += sum(1 for c in pl["checkpoints"] if c)
            tokens += pl["num_tokens"]
            for key in (
                "gate_route",
                "dispatch",
                "gemm",
                "combine",
                "gate_dispatch",
                "moe_total",
                "moe_block",
                "attn",
                "layer",
                "router",
                "overhead",
            ):
                totals[key] = totals.get(key, 0.0) + sum(pl[key])
            for channel in ("dispatch_split", "route_split", "gemm_split"):
                for split in pl[channel]:
                    for name, val in split.items():
                        # *_from_epoch entries are absolute offsets, not
                        # durations: only meaningful per layer, in the export.
                        if name.endswith("_from_epoch"):
                            continue
                        split_totals[name] = split_totals.get(name, 0.0) + val
        out: Dict[str, Any] = {f"{k}_ms": v for k, v in totals.items()}
        out.update({f"{k}_ms": v for k, v in split_totals.items()})
        out["forwards"] = len(self._forwards)
        out["layers"] = layers
        out["moe_layers"] = moe_layers
        out["tokens"] = tokens
        return out

    # ---------------------------------------------------------------- flushing

    def flush(self, *, rank: int = 0, tag: str = "") -> None:
        """Sync, log a one-line summary, optionally dump JSON, then reset.

        Accumulates silently until ``PHASE_TIMER_REPORT_STEP`` forwards are in.
        """
        if not self._forwards:
            return
        self._flush_cnt += 1
        if _REPORT_STEP > 1 and self._flush_cnt % _REPORT_STEP != 0:
            return
        rep = self.report()
        if not rep:
            self.reset()
            return
        _LOGGER.warning(
            "[PHASE_TIMER] rank=%d%s fwd=%d tokens=%d layers=%d/%d "
            "moe_total=%.3f (gate+route=%.3f dispatch=%.3f gemm=%.3f combine=%.3f) "
            "gate_dispatch=%.3f attn=%.3f layer=%.3f router=%.3f overhead=%.3f ms",
            rank,
            f" tag={tag}" if tag else "",
            rep["forwards"],
            rep["tokens"],
            rep["moe_layers"],
            rep["layers"],
            rep["moe_total_ms"],
            rep["gate_route_ms"],
            rep["dispatch_ms"],
            rep["gemm_ms"],
            rep["combine_ms"],
            rep["gate_dispatch_ms"],
            rep["attn_ms"],
            rep["layer_ms"],
            rep["router_ms"],
            rep["overhead_ms"],
        )
        if self.route_split:
            _LOGGER.warning(
                "[PHASE_TIMER] rank=%d policy=%s placement=%s route_split "
                "gate_topk=%.3f policy=%.3f tail=%.3f ms",
                rank,
                _DISPATCH_POLICY,
                _PLACEMENT,
                rep.get("route_gate_topk_ms", 0.0),
                rep.get("route_policy_ms", 0.0),
                rep.get("route_tail_ms", 0.0),
            )
        if self.dispatch_split:
            _LOGGER.warning(
                "[PHASE_TIMER] rank=%d dispatch_split quant=%.3f layout=%.3f "
                "payload=%.3f ms",
                rank,
                rep.get("dispatch_quant_ms", 0.0),
                rep.get("dispatch_layout_ms", 0.0),
                rep.get("dispatch_payload_ms", 0.0),
            )
        if self.gemm_split:
            _LOGGER.warning(
                "[PHASE_TIMER] rank=%d gemm_split align=%.3f kernels=%.3f "
                "tail=%.3f ms",
                rank,
                rep.get("gemm_align_ms", 0.0),
                rep.get("gemm_kernels_ms", 0.0),
                rep.get("gemm_tail_ms", 0.0),
            )
        if _EXPORT_DIR:
            self._export(rank=rank, tag=tag, report=rep)
        self.reset()

    def reset(self) -> None:
        self._forwards.clear()
        self._cur = None
        self._last_event = None

    def synchronize(self) -> None:
        """Sync exactly once, on the last event recorded.

        Events on a single stream complete in order, so waiting on the last one
        guarantees every earlier one is ready for ``elapsed_time``.
        """
        if self._last_event is not None:
            self._last_event.synchronize()

    # ----------------------------------------------------------------- private

    @staticmethod
    def _event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def _record_into(self, store: Dict[str, torch.cuda.Event], name: str) -> None:
        e = self._event()
        e.record()
        store[name] = e
        self._last_event = e

    def _pair_begin(self, channel: str, layer_idx: int) -> None:
        cur = self._cur
        if cur is None:
            return
        e = self._event()
        e.record()
        self._last_event = e
        getattr(cur, channel).append(_PairRecord(layer_idx=layer_idx, start=e))

    def _pair_end(self, channel: str, layer_idx: int) -> None:
        cur = self._cur
        if cur is None:
            return
        records = getattr(cur, channel)
        if not records or records[-1].end is not None:
            self._complain(f"{channel} end without a matching begin")
            return
        e = self._event()
        e.record()
        self._last_event = e
        records[-1].end = e

    @staticmethod
    def _sub_phases(
        moe: _MoeRecord,
        epoch: torch.cuda.Event,
        sub: Dict[str, torch.cuda.Event],
        checkpoints: tuple,
        phase_names: tuple,
        lo: str,
        hi: str,
    ) -> Dict[str, float]:
        """Subdivide the ``lo``..``hi`` phase by a sub-checkpoint channel.

        Returns durations for ``phase_names`` plus each sub-checkpoint's offset
        from the forward epoch. Empty if the channel was not fully recorded.
        """
        if len(sub) != len(checkpoints):
            return {}
        ev = moe.events
        bounds = [ev[lo]] + [sub[n] for n in checkpoints] + [ev[hi]]
        out = {
            name: bounds[i].elapsed_time(bounds[i + 1])
            for i, name in enumerate(phase_names)
        }
        out.update({f"{n}_from_epoch": epoch.elapsed_time(sub[n]) for n in sub})
        return out

    @classmethod
    def _split_phases(
        cls, moe: _MoeRecord, epoch: torch.cuda.Event
    ) -> Dict[str, float]:
        """Decompose router.prepare() into quant / layout / payload."""
        return cls._sub_phases(
            moe,
            epoch,
            moe.sub,
            DISPATCH_SUB_CHECKPOINTS,
            DISPATCH_SUB_PHASE_NAMES,
            "route_done",
            "dispatch_done",
        )

    @classmethod
    def _split_gemm(cls, moe: _MoeRecord, epoch: torch.cuda.Event) -> Dict[str, float]:
        """Decompose the expert-GEMM window into align / kernels / tail."""
        return cls._sub_phases(
            moe,
            epoch,
            moe.gemm,
            GEMM_SUB_CHECKPOINTS,
            GEMM_SUB_PHASE_NAMES,
            "dispatch_done",
            "gemm_done",
        )

    @staticmethod
    def _traffic_values(moe: _MoeRecord) -> Dict[str, Any]:
        """Materialize traffic counters after the existing phase-timer sync."""
        traffic = moe.traffic
        if not traffic:
            return {}
        if "scheduled_rows" not in traffic:
            raise RuntimeError(
                f"DeepEP traffic for layer {moe.layer_idx} has no scheduled-row count"
            )
        send = traffic["dispatch_send_by_rank"]
        assignments = traffic["local_assignments"]
        return {
            "source_tokens": int(traffic["source_tokens"]),
            "source_assignments": int(traffic["source_assignments"].item()),
            "dispatch_send_by_rank": [int(value) for value in send.tolist()],
            "dispatch_recv_rows": int(traffic["dispatch_recv_rows"]),
            "local_assignments": int(assignments.item()),
            "scheduled_rows": int(traffic["scheduled_rows"].item()),
        }

    @classmethod
    def _split_route(
        cls, moe: _MoeRecord, epoch: torch.cuda.Event
    ) -> Dict[str, float]:
        """Decompose gate+route into gate_topk / policy / tail."""
        return cls._sub_phases(
            moe,
            epoch,
            moe.route,
            ROUTE_SUB_CHECKPOINTS,
            ROUTE_SUB_PHASE_NAMES,
            "moe_start",
            "route_done",
        )

    def _export(self, *, rank: int, tag: str, report: Dict[str, Any]) -> None:
        try:
            os.makedirs(_EXPORT_DIR, exist_ok=True)
            payload = {
                "schema": "rtp_llm.phase_timer/v2",
                "rank": rank,
                "tag": tag,
                "seq": self._export_seq,
                "timestamp_ms": int(time.time() * 1000),
                "meta": {
                    "placement": _PLACEMENT,
                    "dispatch_policy": _DISPATCH_POLICY,
                    "route_split": self.route_split,
                    "dispatch_split": self.dispatch_split,
                    "gemm_split": self.gemm_split,
                    "deepep_traffic_stats": _DEEPEP_TRAFFIC_STATS,
                },
                "report": report,
                "forwards": [self.per_layer(i) for i in range(len(self._forwards))],
            }
            path = os.path.join(
                _EXPORT_DIR,
                f"phase_timer_rank{rank:03d}_seq{self._export_seq:06d}.json",
            )
            with open(path, "w") as f:
                json.dump(payload, f)
            self._export_seq += 1
        except OSError as exc:  # profiling must never take down the server
            self._complain(f"phase timer export failed: {exc}")

    def _complain(self, msg: str) -> None:
        """Report a protocol violation.

        Raises under PHASE_TIMER_STRICT, otherwise warns once and lets the
        caller recover.
        """
        if _STRICT:
            raise RuntimeError(msg)
        if msg not in self._warned:
            self._warned.add(msg)
            _LOGGER.warning("[PHASE_TIMER] %s (dropping, warned once)", msg)


_TIMER: Optional[PhaseTimer] = None


def get_phase_timer() -> PhaseTimer:
    global _TIMER
    if _TIMER is None:
        _TIMER = PhaseTimer()
    return _TIMER


__all__ = [
    "ATTN_CHECKPOINTS",
    "DISPATCH_SUB_CHECKPOINTS",
    "DISPATCH_SUB_PHASE_NAMES",
    "GEMM_SUB_CHECKPOINTS",
    "GEMM_SUB_PHASE_NAMES",
    "LAYER_CHECKPOINTS",
    "MOE_BLOCK_CHECKPOINTS",
    "MOE_CHECKPOINTS",
    "MOE_PHASE_NAMES",
    "PHASE_TIMER_ENABLED",
    "ROUTE_SUB_CHECKPOINTS",
    "ROUTE_SUB_PHASE_NAMES",
    "PhaseTimer",
    "get_phase_timer",
]
