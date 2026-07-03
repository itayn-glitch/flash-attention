"""M2 paged head-512 correctness + perf (Option C: cp.async paged, block_size=16).

Proves the head-512 kernel follows the paged block-table indirection correctly and
still streams KV efficiently. block_size=16 (prod) => page_size % kBlockN(64) != 0 =>
FA3 auto-selects the cp.async paged path (pagedkv_tma=false), i.e. blueprint Option C.

Correctness gate: kernel output matches an fp32 reference that GATHERS each sequence's
KV THROUGH the (randomized) block table. A kernel that reads contiguous physical memory
instead of following the table FAILS. Ragged cache_seqlens exercise boundary handling.

Run on the box (overlay venv + LD_LIBRARY_PATH, free GPU), after a paged build:
    CUDA_VISIBLE_DEVICES=1 python m2_paged_microbench.py --batch 32 --seqlen 9216
"""
import argparse
import sys
import torch

PEAK_BW_GBPS = 3350.0
RTOL, ATOL = 2e-2, 2e-2
PAGE = 16  # prod kv_block_size


def make_paged(b, kv, h, hk, d, ragged, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    bps = (kv + PAGE - 1) // PAGE
    num_blocks = bps * b * 2 + 1                      # over-allocate for a random table
    kc = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    vc = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    perm = torch.randperm(num_blocks - 1, generator=g, device=dev)[: b * bps] + 1
    page_table = perm.to(torch.int32).reshape(b, bps)  # randomized physical blocks
    if ragged and b > 1:
        lens = torch.randint(max(1, kv // 2), kv + 1, (b,), generator=g, device=dev)
        lens[0] = kv
    else:
        lens = torch.full((b,), kv, device=dev, dtype=torch.long)
    cache_seqlens = lens.to(torch.int32)
    q = torch.randn(b, 1, h, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    return q, kc, vc, page_table, cache_seqlens


def reference(q, kc, vc, page_table, cache_seqlens, scale):
    """fp32 reference: gather each seq's KV THROUGH the block table, full-attn softmax."""
    b, _, h, d = q.shape
    hk = kc.shape[2]
    gpr = h // hk
    out = torch.empty(b, 1, h, d, device=q.device, dtype=q.dtype)
    for i in range(b):
        L = int(cache_seqlens[i])
        blk = page_table[i].long()
        kg = kc[blk].reshape(-1, hk, d)[:L].float()          # [L,hk,d]
        vg = vc[blk].reshape(-1, hk, d)[:L].float()
        qi = q[i, 0].float().view(hk, gpr, d)                 # [hk,g,d]
        scores = torch.einsum("kgd,lkd->kgl", qi, kg) * scale
        probs = scores.softmax(dim=-1)
        oi = torch.einsum("kgl,lkd->kgd", probs, vg).reshape(h, d)
        out[i, 0] = oi.to(q.dtype)
    return out


def run_kernel(fa, q, kc, vc, page_table, cache_seqlens, scale):
    return fa.flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=cache_seqlens, page_table=page_table,
        causal=True, softmax_scale=scale, num_splits=1)


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
    scale = 1.0 / (d ** 0.5)
    print(f"module: {fa.flash_attn_3_cuda.__file__}")

    print("== correctness (fp32 gathered-through-block-table reference) ==")
    ok_all = True
    for (b, kv) in [(1, 256), (1, 2048), (2, 512), (4, 2048), (8, 9216), (16, 4096)]:
        q, kc, vc, pt, cs = make_paged(b, kv, h, hk, d, ragged=True, seed=0)
        try:
            out = run_kernel(fa, q, kc, vc, pt, cs, scale)
            out = out[0] if isinstance(out, tuple) else out
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  b={b} kv={kv}: RUN FAILED {type(e).__name__}: {e}"); ok_all = False; continue
        ref = reference(q, kc, vc, pt, cs, scale)
        finite = torch.isfinite(out).all().item()
        err = (out.float() - ref.float()).abs().max().item()
        ok = finite and torch.allclose(out.float(), ref.float(), rtol=RTOL, atol=ATOL)
        print(f"  b={b:>3} kv={kv:>6} ragged | finite={finite} max_abs_err={err:.3e} | {'OK' if ok else 'MISMATCH'}")
        ok_all = ok_all and ok
    if not ok_all:
        print("M2_CORRECTNESS_FAIL"); sys.exit(3)
    print("M2_CORRECTNESS_PASS")

    print("== perf ==")
    b, kv = args.batch, args.seqlen
    q, kc, vc, pt, cs = make_paged(b, kv, h, hk, d, ragged=False, seed=0)  # equal lens for a clean BW number
    for _ in range(args.warmup):
        run_kernel(fa, q, kc, vc, pt, cs, scale)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(args.iters):
        run_kernel(fa, q, kc, vc, pt, cs, scale)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / args.iters
    kv_bytes = 2 * b * kv * hk * d * 2
    bw = (kv_bytes + 2 * b * h * d * 2) / (ms * 1e-3) / 1e9
    pct = 100.0 * bw / PEAK_BW_GBPS
    print(f"perf b={b} kv={kv} | {ms*1e3:.1f} us/iter | BW={bw:.0f} GB/s = {pct:.1f}% of peak")
    print(f"M2_PERF_{'PASS' if pct > args.gate_pct else 'NOTE'} ({pct:.1f}% vs {args.gate_pct:.0f}%)")
    sys.exit(0)


if __name__ == "__main__":
    main()
