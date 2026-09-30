#!/usr/bin/env python3
"""Isolated per-impl measurement separating GPU time (CUDA graph replay, zero
host overhead) from host time (CPU wall per call, no sync).

argv: <impl:target|cur|opt> <tokens> <tile> <passes> <dtype:i32|i64>"""
import sys, os, types, importlib.util, time
import numpy as np, torch, triton
import triton.language as tl

impl, tokens, tile, passes, dtn = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                   int(sys.argv[4]), sys.argv[5])
for k, v in (("VLLM_EXPANDER_DISPATCH_TILE_M", tile), ("VLLM_EXPANDER_ITER_PASSES", passes),
             ("EPLB_WF_TILE_M", tile), ("EPLB_WF_ITER_PASSES", passes)):
    os.environ[k] = str(v)

REPO = "/home/admin/workspace/aop_lab/app_source/tangling/code/RTP-LLM/github-opensource"
TARGET = "/home/admin/workspace/aop_lab/app_source/tangling/code/MoE-Bench-Experiments"
NPZ = ("/home/admin/workspace/aop_lab/app_source/tangling/code/moe_experiments/"
       "online_experiments/qwen3.5-122b-a10b-fp8_dsv4-flash-online_bs4_20260813/"
       "armA_r128_bs4/expert_stats/expert_stats_rank000_seq000002.npz")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("vllm"); pkg.__path__ = []; sys.modules["vllm"] = pkg
mm = types.ModuleType("vllm.triton_utils"); mm.triton, mm.tl = triton, tl
sys.modules["vllm.triton_utils"] = mm

dev = "cuda"
E, G, NP, PAD, CAP = 256, 8, 384, 129, 48
per, red = E // G, CAP - E // G
lp = torch.full((E, PAD), -1, dtype=torch.int32); cn = torch.ones(E, dtype=torch.int32)
for l in range(E):
    r = l // per; lp[l, 0] = r * CAP + l % per
    if l % per < red:
        lp[l, 1] = ((r - 1) % G) * CAP + per + l % per; cn[l] = 2
lp = lp.to(dev).contiguous(); cn = cn.to(dev)
pr = np.load(NPZ)["log_stats"][24].astype(np.float64); pr /= pr.sum()
numel = tokens * 8
topk = torch.from_numpy(np.random.default_rng(0).choice(E, size=numel, p=pr)).to(
    dev).to(torch.int32 if dtn == "i32" else torch.int64).view(-1, 8)

if impl == "target":
    spec = importlib.util.spec_from_file_location(
        "_tgt", TARGET + "/vllm/distributed/expander/dispatch.py")
    tgt = importlib.util.module_from_spec(spec); spec.loader.exec_module(tgt)
    d = tgt.ExpanderDispatcher(lp[:, :2].to(torch.long), G, NP)
    assert d._tile_m == tile and d._iter_passes == passes
    load = torch.zeros(NP, dtype=torch.float32, device=dev)
    rec = torch.zeros((), dtype=torch.int32, device=dev)
    pol = "iter" if passes > 1 else "wf"
    fn = lambda: d.select(topk, pol, load, rec)
elif impl == "cur":
    import rtp_llm.models_py.triton_kernels.moe.ep_kernels as ep
    assert ep._WF_TILE_M == tile and ep._WF_ITER_PASSES == passes
    pol = "wf_iter" if passes > 1 else "wf"
    fn = lambda: ep.wf_dispatch(topk, lp, cn, G, NP, policy=pol)
else:
    import optmod
    fn = lambda: optmod.opt_dispatch(topk, lp, cn, G, NP, passes=passes, tile=tile)

# ---- warm up (JIT + autotune + allocator) ----
for _ in range(30):
    fn()
torch.cuda.synchronize()

# ---- host time: CPU wall per call, no sync (launch-side cost) ----
t0 = time.perf_counter()
for _ in range(200):
    fn()
host = (time.perf_counter() - t0) * 1000 / 200
torch.cuda.synchronize()

# ---- GPU time: CUDA graph replay removes all host overhead ----
gpu = float("nan")
try:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    best = []
    for _ in range(3):
        a.record()
        for _ in range(50):
            g.replay()
        b.record(); torch.cuda.synchronize()
        best.append(a.elapsed_time(b) / 50)
    gpu = min(best)
except Exception as ex:
    print(f"# graph capture failed: {type(ex).__name__}: {ex}", file=sys.stderr)

# ---- stream time: back-to-back async launches (what timeit measured) ----
a, b = torch.cuda.Event(True), torch.cuda.Event(True)
st = []
for _ in range(3):
    a.record()
    for _ in range(60):
        fn()
    b.record(); torch.cuda.synchronize()
    st.append(a.elapsed_time(b) / 60)
print(f"{gpu:.4f} {host:.4f} {min(st):.4f}")
