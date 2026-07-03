"""M1 dense head-512 microbench (Gemma4 global-attention decode shape).

Proves the square head-512 fwd instantiation runs and measures achieved HBM
bandwidth vs H100 peak. Non-paged, contiguous KV, q=1 decode, causal, bf16.
Gate (blueprint M1): >45% of peak BW on this microbench.

Run on the box with the overlay venv + LD_LIBRARY_PATH set and a FREE GPU:
    CUDA_VISIBLE_DEVICES=1 python m1_dense_microbench.py --seqlen 9216
"""
import argparse
import sys
import torch

# H100 80GB HBM3 peak (GB/s). Board spec ~3.35 TB/s.
PEAK_BW_GBPS = 3350.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=9216)      # ~prod avg KV
    ap.add_argument("--nheads", type=int, default=32)         # Gemma4: 4 kv x 8
    ap.add_argument("--nheads_k", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=512)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    import flash_attn_interface as fa

    dev = "cuda"
    dt = torch.bfloat16
    b, sq, sk, h, hk, d = 1, 1, args.seqlen, args.nheads, args.nheads_k, args.headdim
    torch.manual_seed(0)
    q = torch.randn(b, sq, h, d, device=dev, dtype=dt) * 0.1
    k = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
    v = torch.randn(b, sk, hk, d, device=dev, dtype=dt) * 0.1
    scale = 1.0 / (d ** 0.5)

    try:
        out = fa.flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=1, pack_gqa=False)
        if isinstance(out, tuple):
            out = out[0]
        torch.cuda.synchronize()
    except Exception as e:
        print(f"RUN FAILED: {type(e).__name__}: {e}")
        sys.exit(2)

    finite = torch.isfinite(out).all().item()
    print(f"shape={tuple(out.shape)} dtype={out.dtype} finite={finite} "
          f"mean={out.float().mean().item():.4e} std={out.float().std().item():.4e}")
    if not finite:
        print("OUTPUT NOT FINITE"); sys.exit(3)

    # Bytes moved: dominant term is the KV read (q=1 decode reads all KV once).
    kv_bytes = 2 * sk * hk * d * q.element_size()   # K + V
    q_bytes = b * sq * h * d * q.element_size()
    o_bytes = b * sq * h * d * q.element_size()
    total_bytes = kv_bytes + q_bytes + o_bytes

    for _ in range(args.warmup):
        fa.flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=1, pack_gqa=False)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        fa.flash_attn_func(q, k, v, softmax_scale=scale, causal=True, num_splits=1, pack_gqa=False)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    bw = total_bytes / (ms * 1e-3) / 1e9
    pct = 100.0 * bw / PEAK_BW_GBPS
    print(f"seqlen={sk} h={h} hk={hk} d={d} | {ms*1e3:.1f} us/iter | "
          f"KV={kv_bytes/1e6:.1f}MB total={total_bytes/1e6:.1f}MB | "
          f"BW={bw:.0f} GB/s = {pct:.1f}% of {PEAK_BW_GBPS:.0f} peak")
    print(f"M1_GATE_{'PASS' if pct > 45 else 'FAIL'} ({pct:.1f}% vs 45%)")


if __name__ == "__main__":
    main()
