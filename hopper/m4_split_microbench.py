"""M4 split-KV (flash-decoding) head-512: fix low-concurrency + q=5 cudagraph (C2).

FA3 num_splits>1 == the blueprint's 16-segment 3D split: one decode's KV is chopped into
num_splits chunks computed by separate CTAs, then flash_fwd_combine reduces them. This is
the lever that fills the GPU at b=1 (M1 showed b=1=8.5% because 4 KV heads => too few CTAs).

Gates (blueprint M4):
  1. CORRECTNESS: split output == fp32 block-table oracle (the split reduction is exact).
  2. LOW-CONCURRENCY WIN: b=1 BW climbs as num_splits rises (the actual TTFS-at-c1 lever).
  3. FULL-CUDAGRAPH at q=5 (the literal C2 proof): capture+replay the MTP verify shape.

Run: CUDA_VISIBLE_DEVICES=1 python m4_split_microbench.py
"""
import argparse
import sys
import torch

PEAK_BW_GBPS = 3350.0
RTOL, ATOL = 2e-2, 2e-2
PAGE = 16  # prod (OQ1: page=16 cp.async)


def make_paged(b, kv, h, hk, d, sq, seed, dev="cuda", scaleval=0.1):
    g = torch.Generator(device=dev).manual_seed(seed)
    bps = (kv + PAGE - 1) // PAGE
    num_blocks = bps * b * 2 + 1
    kc = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * scaleval
    vc = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * scaleval
    perm = torch.randperm(num_blocks - 1, generator=g, device=dev)[: b * bps] + 1
    pt = perm.to(torch.int32).reshape(b, bps)
    cs = torch.full((b,), kv, device=dev, dtype=torch.int32)
    q = torch.randn(b, sq, h, d, generator=g, device=dev, dtype=torch.bfloat16) * scaleval
    return q, kc, vc, pt, cs


def reference_q1(q, kc, vc, pt, cs, scale):
    b, _, h, d = q.shape
    hk = kc.shape[2]; gpr = h // hk
    out = torch.empty(b, 1, h, d, device=q.device, dtype=q.dtype)
    for i in range(b):
        L = int(cs[i]); blk = pt[i].long()
        kg = kc[blk].reshape(-1, hk, d)[:L].float()
        vg = vc[blk].reshape(-1, hk, d)[:L].float()
        qi = q[i, 0].float().view(hk, gpr, d)
        sc = torch.einsum("kgd,lkd->kgl", qi, kg) * scale
        out[i, 0] = torch.einsum("kgl,lkd->kgd", sc.softmax(-1), vg).reshape(h, d).to(q.dtype)
    return out


def run(fa, q, kc, vc, pt, cs, scale, num_splits):
    return fa.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cs, page_table=pt,
                                      causal=True, softmax_scale=scale, num_splits=num_splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nheads", type=int, default=32)
    ap.add_argument("--nheads_k", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=512)
    args = ap.parse_args()
    import flash_attn_interface as fa
    h, hk, d = args.nheads, args.nheads_k, args.headdim
    scale = 1.0 / (d ** 0.5)
    print(f"module: {fa.flash_attn_3_cuda.__file__}")

    print("== correctness (split reduction exact, q=1 paged, fp32 oracle) ==")
    ok_all = True
    for ns in (2, 4, 16):
        for (b, kv) in [(1, 9216), (2, 4096)]:
            q, kc, vc, pt, cs = make_paged(b, kv, h, hk, d, 1, seed=0)
            try:
                out = run(fa, q, kc, vc, pt, cs, scale, ns)
                out = out[0] if isinstance(out, tuple) else out
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  splits={ns} b={b} kv={kv}: FAILED {type(e).__name__}: {e}"); ok_all = False; continue
            ref = reference_q1(q, kc, vc, pt, cs, scale)
            err = (out.float() - ref.float()).abs().max().item()
            ok = torch.isfinite(out).all().item() and torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)
            print(f"  splits={ns:>2} b={b} kv={kv:>5} | max_abs_err={err:.3e} | {'OK' if ok else 'MISMATCH'}")
            ok_all = ok_all and ok
    if not ok_all:
        print("M4_CORRECTNESS_FAIL"); sys.exit(3)
    print("M4_CORRECTNESS_PASS")

    print("== low-concurrency win: b=1 BW vs num_splits (M1 baseline was ~8.5% at splits=1) ==")
    def bw(b, kv, ns, iters=50, warmup=15):
        q, kc, vc, pt, cs = make_paged(b, kv, h, hk, d, 1, seed=0)
        for _ in range(warmup): run(fa, q, kc, vc, pt, cs, scale, ns)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
        for _ in range(iters): run(fa, q, kc, vc, pt, cs, scale, ns)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / iters
        by = 2 * b * kv * hk * d * 2 + 2 * b * h * d * 2
        return by / (ms * 1e-3) / 1e9
    for kv in (9216, 32768):
        row = []
        for ns in (1, 4, 8, 16, 32):
            g = bw(1, kv, ns); row.append(f"ns={ns}:{100*g/PEAK_BW_GBPS:4.1f}%")
        print(f"  b=1 kv={kv:>5} | " + "  ".join(row))

    print("== full cudagraph capture at q=5 (MTP verify shape, C2 proof) ==")
    b, kv, sq, ns = 4, 4096, 5, 16
    q, kc, vc, pt, cs = make_paged(b, kv, h, hk, d, sq, seed=1, scaleval=1.0)  # O(1) outputs
    eager = run(fa, q, kc, vc, pt, cs, scale, ns).clone(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): run(fa, q, kc, vc, pt, cs, scale, ns)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run(fa, q, kc, vc, pt, cs, scale, ns)
    except Exception as e:
        print(f"  CAPTURE FAILED: {type(e).__name__}: {e}"); print("M4_CUDAGRAPH_FAIL"); sys.exit(4)
    graph.replay(); torch.cuda.synchronize(); rep0 = captured.clone()
    q.copy_(torch.randn_like(q)); graph.replay(); torch.cuda.synchronize(); rep1 = captured.clone()
    eager2 = run(fa, q, kc, vc, pt, cs, scale, ns); torch.cuda.synchronize()
    same = torch.allclose(rep0.float(), eager.float(), rtol=RTOL, atol=ATOL)
    reflects = torch.allclose(rep1.float(), eager2.float(), rtol=RTOL, atol=ATOL)
    delta = (rep0.float() - rep1.float()).abs().max().item()
    print(f"  q={sq} splits={ns} | replay==eager:{same} reflects_input:{reflects} delta:{delta:.3e} changed:{delta>10*ATOL}")
    if same and reflects and delta > 10 * ATOL:
        print("M4_CUDAGRAPH_PASS"); sys.exit(0)
    print("M4_CUDAGRAPH_FAIL"); sys.exit(4)


if __name__ == "__main__":
    main()
