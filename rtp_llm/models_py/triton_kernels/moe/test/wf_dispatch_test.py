"""Tests for the water-filling dispatch (wf_dispatch) in ep_kernels.py.

Covers: per-expert demand conservation, replica validity, tile-snap (_snap2)
conservation and sub-tile agreement, and triton/torch-fallback agreement.
"""

import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.moe import ep_kernels

E = 8  # logical experts
G = 4  # GPUs
NUM_PHYSICAL = 12  # 3 physical experts per GPU

# experts 0..3 have 2 replicas, 4..7 have 1
LOG2PHY = torch.tensor(
    [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [8, -1],
        [9, -1],
        [10, -1],
        [11, -1],
    ],
    dtype=torch.int32,
)
EXPERT_CNT = torch.tensor([2, 2, 2, 2, 1, 1, 1, 1], dtype=torch.int32)


def _make_topk(num_tokens: int, topk: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, E, (num_tokens, topk), generator=g, device="cuda")
    # sprinkle sentinel rows
    ids[::37, 0] = -1
    return ids


def _demand(topk_ids: torch.Tensor) -> torch.Tensor:
    flat = topk_ids.reshape(-1)
    flat = flat[(flat >= 0) & (flat < E)]
    return torch.bincount(flat.long(), minlength=E)


def _replica_histogram(out: torch.Tensor, topk_ids: torch.Tensor):
    """Per logical expert: [count on replica0, count on replica1]."""
    hist = torch.zeros(E, 2, dtype=torch.int64)
    flat_in = topk_ids.reshape(-1).long()
    flat_out = out.reshape(-1).long()
    for e in range(E):
        sel = (flat_in == e).nonzero(as_tuple=True)[0]
        outs = flat_out[sel]
        p0, p1 = int(LOG2PHY[e, 0]), int(LOG2PHY[e, 1])
        hist[e, 0] = int((outs == p0).sum())
        hist[e, 1] = int((outs == p1).sum()) if p1 >= 0 else 0
        # every token must land on one of this expert's replicas
        assert int(((outs == p0) | (outs == p1)).sum()) == outs.numel()
    return hist


class WfDispatchTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        ep_kernels._wf_dispatch_use_triton = True

    def _run(self, topk_ids, policy, tile_m):
        return ep_kernels.wf_dispatch(
            topk_ids,
            LOG2PHY.cuda(),
            EXPERT_CNT.cuda(),
            G,
            NUM_PHYSICAL,
            policy=policy,
            tile_m=tile_m,
        )

    def test_demand_conservation_and_validity(self):
        for policy in ("wf", "wf_iter"):
            for tile_m in (0, 128):
                topk_ids = _make_topk(2048, 8, seed=1)
                out = self._run(topk_ids, policy, tile_m)
                self.assertEqual(out.shape, topk_ids.shape)
                demand = _demand(topk_ids)
                hist = _replica_histogram(out, topk_ids)
                for e in range(E):
                    self.assertEqual(
                        int(hist[e, 0] + hist[e, 1]),
                        int(demand[e]),
                        f"expert {e} demand not conserved "
                        f"(policy={policy}, tile={tile_m})",
                    )

    def test_sentinels_preserved(self):
        topk_ids = _make_topk(512, 4, seed=2)
        out = self._run(topk_ids, "wf_iter", 128)
        sentinel = topk_ids.reshape(-1) == -1
        self.assertTrue(torch.all(out.reshape(-1)[sentinel] == -1))

    def test_snap_subtile_agreement(self):
        # With tile=128 and per-expert demand < 128, snap must push every
        # 2-replica expert entirely onto ONE replica (split rounds to 0 or c).
        # 512 slots / 8 experts ~= 64 each, comfortably under one tile even
        # with sampling variance (256x4 sits exactly at 128 and overshoots).
        topk_ids = _make_topk(128, 4, seed=3)
        out = self._run(topk_ids, "wf_iter", 128)
        hist = _replica_histogram(out, topk_ids)
        demand = _demand(topk_ids)
        for e in range(4):  # 2-replica experts only
            self.assertLess(int(demand[e]), 128)
            on_one_replica = int(hist[e, 0]) == 0 or int(hist[e, 1]) == 0
            self.assertTrue(
                on_one_replica,
                f"expert {e} split across replicas despite sub-tile snap: "
                f"{int(hist[e,0])}/{int(hist[e,1])}",
            )

    def test_returns_caller_dtype(self):
        # generic_moe allocates topk_ids in the executor's topk_ids_dtype (int64
        # for the deepgemm hybrid path, int32 for the triton one) and reuses that
        # buffer downstream, so dispatch must hand back the same dtype.
        for dtype in (torch.int32, torch.int64):
            topk_ids = _make_topk(256, 4, seed=5).to(dtype)
            for use_triton in (True, False):
                ep_kernels._wf_dispatch_use_triton = use_triton
                try:
                    out = self._run(topk_ids, "wf_iter", 128)
                finally:
                    ep_kernels._wf_dispatch_use_triton = True
                self.assertEqual(out.dtype, dtype, f"triton={use_triton}")

    def test_single_column_log2phy(self):
        # pad_k = num_replicas - num_experts + 1, so REDUNDANT_EXPERT=0 gives a
        # [E, 1] log2phy. There is no second replica to read: every token must
        # land on its expert's only physical slot, sentinels stay -1.
        log2phy = torch.arange(E, dtype=torch.int32).reshape(E, 1).cuda()
        cnt = torch.ones(E, dtype=torch.int32).cuda()
        topk_ids = _make_topk(512, 4, seed=6)
        for use_triton in (True, False):
            ep_kernels._wf_dispatch_use_triton = use_triton
            try:
                out = ep_kernels.wf_dispatch(
                    topk_ids, log2phy, cnt, G, E, policy="wf_iter", tile_m=128
                )
            finally:
                ep_kernels._wf_dispatch_use_triton = True
            flat_in, flat_out = topk_ids.reshape(-1), out.reshape(-1)
            valid = flat_in >= 0
            self.assertTrue(
                torch.equal(flat_out[valid], flat_in[valid]), f"triton={use_triton}"
            )
            self.assertTrue(torch.all(flat_out[~valid] == -1), f"triton={use_triton}")

    def test_static_matches_live_and_prepares_once(self):
        topk_ids = _make_topk(1024, 8, seed=7)
        log2phy = LOG2PHY.cuda()
        cnt = EXPERT_CNT.cuda()
        live = ep_kernels.wf_dispatch(
            topk_ids, log2phy, cnt, G, NUM_PHYSICAL, policy="wf_iter", tile_m=128
        )
        disp = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        original = ep_kernels._wf_prepare_candidates_triton
        with mock.patch.object(
            ep_kernels, "_wf_prepare_candidates_triton", wraps=original
        ) as prep:
            static1 = ep_kernels.wf_dispatch(
                topk_ids,
                log2phy,
                cnt,
                G,
                NUM_PHYSICAL,
                policy="wf_iter",
                tile_m=128,
                dispatcher=disp,
                static_placement=True,
            )
            static2 = ep_kernels.wf_dispatch(
                topk_ids,
                log2phy,
                cnt,
                G,
                NUM_PHYSICAL,
                policy="wf_iter",
                tile_m=128,
                dispatcher=disp,
                static_placement=True,
            )
        self.assertEqual(prep.call_count, 1)
        live_hist = _replica_histogram(live, topk_ids)
        self.assertTrue(torch.equal(live_hist, _replica_histogram(static1, topk_ids)))
        self.assertTrue(torch.equal(live_hist, _replica_histogram(static2, topk_ids)))

    def test_preinitialized_static_skips_first_forward_prep(self):
        topk_ids = _make_topk(1024, 8, seed=12)
        log2phy = LOG2PHY.cuda()
        cnt = EXPERT_CNT.cuda()
        disp = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        disp.prepare_candidates(log2phy, cnt, NUM_PHYSICAL // G)

        original = ep_kernels._wf_prepare_candidates_triton
        with mock.patch.object(
            ep_kernels, "_wf_prepare_candidates_triton", wraps=original
        ) as prep:
            static = ep_kernels.wf_dispatch(
                topk_ids,
                log2phy,
                cnt,
                G,
                NUM_PHYSICAL,
                policy="wf_iter",
                dispatcher=disp,
                static_placement=True,
            )
        self.assertEqual(prep.call_count, 0)
        live = ep_kernels.wf_dispatch(
            topk_ids, log2phy, cnt, G, NUM_PHYSICAL, policy="wf_iter"
        )
        self.assertTrue(
            torch.equal(
                _replica_histogram(live, topk_ids),
                _replica_histogram(static, topk_ids),
            )
        )

    def test_preinitialized_static_accepts_inference_tensors(self):
        # Loader weights are materialised under torch.inference_mode(), and
        # inference tensors raise on ._version, so the candidate cache key must
        # not read it unguarded -- that crashed model construction on startup.
        topk_ids = _make_topk(1024, 8, seed=13)
        with torch.inference_mode():
            log2phy = LOG2PHY.cuda()
            cnt = EXPERT_CNT.cuda()
        self.assertTrue(log2phy.is_inference())
        disp = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        disp.prepare_candidates(log2phy, cnt, NUM_PHYSICAL // G)

        original = ep_kernels._wf_prepare_candidates_triton
        with mock.patch.object(
            ep_kernels, "_wf_prepare_candidates_triton", wraps=original
        ) as prep:
            static = ep_kernels.wf_dispatch(
                topk_ids,
                log2phy,
                cnt,
                G,
                NUM_PHYSICAL,
                policy="wf_iter",
                dispatcher=disp,
                static_placement=True,
            )
        self.assertEqual(prep.call_count, 0)
        live = ep_kernels.wf_dispatch(
            topk_ids, log2phy, cnt, G, NUM_PHYSICAL, policy="wf_iter"
        )
        self.assertTrue(
            torch.equal(
                _replica_histogram(live, topk_ids),
                _replica_histogram(static, topk_ids),
            )
        )

    def test_static_refreshes_after_inplace_placement_update(self):
        topk_ids = _make_topk(1024, 8, seed=8)
        log2phy = LOG2PHY.cuda()
        cnt = EXPERT_CNT.cuda()
        disp = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        ep_kernels.wf_dispatch(
            topk_ids,
            log2phy,
            cnt,
            G,
            NUM_PHYSICAL,
            dispatcher=disp,
            static_placement=True,
        )
        previous = disp.cand_phys.clone()
        # C++ ExpertBalancer applies its one-time circulant plan with copy_, which
        # increments Tensor._version and must invalidate a mapping cached during
        # the startup forward.
        log2phy.copy_(torch.roll(log2phy, shifts=1, dims=0))
        original = ep_kernels._wf_prepare_candidates_triton
        with mock.patch.object(
            ep_kernels, "_wf_prepare_candidates_triton", wraps=original
        ) as prep:
            ep_kernels.wf_dispatch(
                topk_ids,
                log2phy,
                cnt,
                G,
                NUM_PHYSICAL,
                dispatcher=disp,
                static_placement=True,
            )
        self.assertEqual(prep.call_count, 1)
        self.assertFalse(torch.equal(previous, disp.cand_phys))
        self.assertTrue(torch.equal(disp.cand_phys[:, 0], log2phy[:, 0]))

    def test_dynamic_refreshes_every_call(self):
        topk_ids = _make_topk(512, 4, seed=9)
        log2phy = LOG2PHY.cuda()
        cnt = EXPERT_CNT.cuda()
        disp = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        original = ep_kernels._wf_prepare_candidates_triton
        with mock.patch.object(
            ep_kernels, "_wf_prepare_candidates_triton", wraps=original
        ) as prep:
            ep_kernels.wf_dispatch(
                topk_ids, log2phy, cnt, G, NUM_PHYSICAL, dispatcher=disp
            )
            log2phy.copy_(torch.roll(log2phy, shifts=1, dims=0))
            ep_kernels.wf_dispatch(
                topk_ids, log2phy, cnt, G, NUM_PHYSICAL, dispatcher=disp
            )
        self.assertEqual(prep.call_count, 2)
        self.assertTrue(torch.equal(disp.cand_phys[:, 0], log2phy[:, 0]))

    def test_static_dispatchers_do_not_share_candidates(self):
        topk_ids = _make_topk(512, 4, seed=10)
        log2phy1 = LOG2PHY.cuda()
        log2phy2 = torch.roll(LOG2PHY, shifts=1, dims=0).cuda()
        cnt = EXPERT_CNT.cuda()
        disp1 = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        disp2 = ep_kernels.WfDispatcher(E, G, topk_ids.device)
        for disp, placement in ((disp1, log2phy1), (disp2, log2phy2)):
            ep_kernels.wf_dispatch(
                topk_ids,
                placement,
                cnt,
                G,
                NUM_PHYSICAL,
                dispatcher=disp,
                static_placement=True,
            )
        self.assertNotEqual(disp1.cand_phys.data_ptr(), disp2.cand_phys.data_ptr())
        self.assertFalse(torch.equal(disp1.cand_phys, disp2.cand_phys))

    def test_static_requires_per_layer_dispatcher(self):
        topk_ids = _make_topk(16, 2, seed=11)
        with self.assertRaisesRegex(ValueError, "per-layer dispatcher"):
            ep_kernels.wf_dispatch(
                topk_ids,
                LOG2PHY.cuda(),
                EXPERT_CNT.cuda(),
                G,
                NUM_PHYSICAL,
                static_placement=True,
            )

    def test_triton_matches_torch_fallback(self):
        topk_ids = _make_topk(1024, 8, seed=4)
        out_triton = self._run(topk_ids, "wf_iter", 0)
        ep_kernels._wf_dispatch_use_triton = False
        try:
            out_torch = self._run(topk_ids, "wf_iter", 0)
        finally:
            ep_kernels._wf_dispatch_use_triton = True
        # Token-to-replica assignment order can differ between the atomic
        # triton scatter and the deterministic sort-based fallback, but the
        # per-replica quotas (and hence histograms) are identical.
        h1 = _replica_histogram(out_triton, topk_ids)
        h2 = _replica_histogram(out_torch, topk_ids)
        self.assertTrue(
            torch.equal(h1, h2),
            f"triton vs torch replica histograms differ:\n{h1}\nvs\n{h2}",
        )


if __name__ == "__main__":
    unittest.main()
