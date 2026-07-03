"""M1 dense head-512 microbench + correctness gate (Gemma4 global-attention decode).

Two gates, both required for an M1 GO:
  1. CORRECTNESS: kernel output matches an fp32 grouped reference (softmax(qk^T)v) at
     q=1 decode across several (batch, seqlen) shapes. A fast-but-wrong kernel FAILS here.
  2. PERF: achieved HBM bandwidth vs H100 peak on a filled-GPU shape (>45% gate).

Non-paged, contiguous KV, q=1 decode, causal, bf16, GQA (h/hk groups).
Note: q=1 causal is bottom-right aligned => the single query attends ALL keys (no mask),
so the reference is a plain full-attention softmax.

Run on the box with the overlay venv + LD_LIBRARY_PATH and a FREE GPU:
    CUDA_VISIBLE_DEVICES=1 python m1_dense_microbench.py --batch 32 --seqlen 9216
"""
import argparse
import sys
import torch

PEAK_BW_GBPS = 3350.0  # H100 80GB HBM3 board spec ~3.35 TB/s
RTOL, ATOL = 2e-2, 2e-2  # bf16 output rounding + FMA-order (blueprint 5.5 bar)


def reference_attention(q, k, v, scale):
    """fp32 grouped reference for q=1. q[b,1,h,d], k/v[b,S,hk,d] -> [b,1,h,d].

    Memory-safe: no head expansion (view h as (hk, g) and einsum grouped).
    """
    b, sq, h, d = q.shape
    S, hk = k.shape[1], k.shape[2]
    g = h // hk
    q5 = q.float().view(b, sq, hk, g, d)          # [b,1,hk,g,d]
    kf = k.float()                                 # [b,S,hk,d]
    vf = v.float()                                 # [b,S,hk,d]
    scores = torch.einsum("bokgd,bskd->bkgos", q5, kf) * scale   # [b,hk,g,1,S]
    probs = scores.softmax(dim=-1)
    out = torch.einsum("bkgos,bskd->bokgd", probs, vf)           # [b,1,hk,g,d]
    return out.reshape(b, sq, h, d).to(q.dtype)


def run_kernel(fa, q, k, v, scale):
    out = fa.flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=1)
    return out[0] if isinstance(out, tuple) else out


def check_correctness(fa, shapes, h, hk, d):
    """Return True iff kernel matches the fp32 reference on every shape."""
    dev, dt = "cuda", torch.bfloat16
    scale = 1.0 / (d ** 0.5)
    all_ok = True
    for (b, sk) in shapes:
        torch.manual_seed(0)
        q = torch.randn(b, 1, h, d, device=dev, dtype=dt) * 0.1
        k = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
        v = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
        try:
            out = run_kernel(fa, q, k, v, scale)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  check b={b} S={sk}: RUN FAILED {type(e).__name__}: {e}")
            all_ok = False
            continue
        ref = reference_attention(q, k, v, scale)
        finite = torch.isfinite(out).all().item()
        max_err = (out.float() - ref.float()).abs().max().item()
        ok = finite and torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)
        print(f"  check b={b:>3} S={sk:>6} | finite={finite} max_abs_err={max_err:.3e} | "
              f"{'OK' if ok else 'MISMATCH'}")
        all_ok = all_ok and ok
    return all_ok


def bench(fa, b, sk, h, hk, d, iters, warmup):
    dev, dt = "cuda", torch.bfloat16
    scale = 1.0 / (d ** 0.5)
    torch.manual_seed(0)
    q = torch.randn(b, 1, h, d, device=dev, dtype=dt) * 0.1
    k = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
    v = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
    kv_bytes = 2 * b * sk * hk * d * q.element_size()
    total_bytes = kv_bytes + 2 * b * 1 * h * d * q.element_size()  # + Q read + O write
    for _ in range(warmup):
        run_kernel(fa, q, k, v, scale)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        run_kernel(fa, q, k, v, scale)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    bw = total_bytes / (ms * 1e-3) / 1e9
    pct = 100.0 * bw / PEAK_BW_GBPS
    print(f"perf b={b} S={sk} h={h} hk={hk} d={d} | {ms*1e3:.1f} us/iter | "
          f"KV={kv_bytes/1e6:.1f}MB | BW={bw:.0f} GB/s = {pct:.1f}% of {PEAK_BW_GBPS:.0f} peak")
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=9216)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--nheads", type=int, default=32)
    ap.add_argument("--nheads_k", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--gate-pct", type=float, default=45.0)
    args = ap.parse_args()

    import flash_attn_interface as fa
    h, hk, d = args.nheads, args.nheads_k, args.headdim

    print(f"module: {fa.flash_attn_3_cuda.__file__}")
    print("== correctness (fp32 grouped reference, q=1 decode) ==")
    shapes = [(1, 256), (1, 2048), (2, 512), (4, 2048), (8, 9216), (16, 4096)]
    correct = check_correctness(fa, shapes, h, hk, d)
    if not correct:
        print("M1_CORRECTNESS_FAIL")
        sys.exit(3)
    print("M1_CORRECTNESS_PASS")

    print("== perf ==")
    pct = bench(fa, args.batch, args.seqlen, h, hk, d, args.iters, args.warmup)
    ok = pct > args.gate_pct
    print(f"M1_GATE_{'PASS' if ok else 'FAIL'} ({pct:.1f}% vs {args.gate_pct:.0f}%)")
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
