"""M5 dequant-on-load correctness: bf16 query + fp8(e4m3) paged KV cache -> dequant->bf16 wgmma.

Validates the Path-B producer-convert kernel (kv_is_fp8 dispatch, PagedKVManager<ElementKV>,
cp.async fp8->staging, convert->bf16 smem with k/v_descale folded in). This is a CORRECTNESS
gate on the scalar-convert scaffold; NO perf claim here (scalar convert is the known-losing path).

Oracles (P1/P2 review #2):
 - PRIMARY = dequant-to-BF16 then attention: kernel rounds fp8->float*descale->bf16 before the
   wgmma, and prod Triton does the same (mma f32.bf16.bf16). So the oracle must round KV to bf16.
 - DIAGNOSTIC = dequant-to-FP32 then attention (no bf16 rounding). If BF16 passes & FP32 differs
   -> expected rounding; if BF16 FAILS -> kernel/layout bug.

Barrier stress (P2): q=5 x num_splits=16 x multi-row exercises the shared ProducerConvert barrier
under staggered producer progress (not just a q=1 toy).

Run (box overlay venv, free GPU), after a dequant build:
  FLASH_PRINT_SMEM=1 CUDA_VISIBLE_DEVICES=1 python m5_dequant_microbench.py
"""
import argparse, sys, torch

PAGE = 16
RTOL, ATOL = 3e-2, 3e-2          # fp8 quant error is IN BOTH kernel and oracle (same bytes) -> tight-ish
E4M3_MAX = 448.0


def quantize_fp8(x_bf16):
    """Per-tensor symmetric quant bf16 -> e4m3. Returns (fp8 tensor, scalar descale)."""
    amax = x_bf16.abs().amax().clamp(min=1e-4)
    scale = (amax / E4M3_MAX).float()
    q = (x_bf16.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def make_paged(b, kv, h, hk, d, qlen, ragged, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    bps = (kv + PAGE - 1) // PAGE
    num_blocks = bps * b * 2 + 1
    kc_bf16 = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    vc_bf16 = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    kc, ks = quantize_fp8(kc_bf16)
    vc, vs = quantize_fp8(vc_bf16)
    perm = torch.randperm(num_blocks - 1, generator=g, device=dev)[: b * bps] + 1
    page_table = perm.to(torch.int32).reshape(b, bps)
    if ragged and b > 1:
        lens = torch.randint(max(1, kv // 2), kv + 1, (b,), generator=g, device=dev); lens[0] = kv
    else:
        lens = torch.full((b,), kv, device=dev, dtype=torch.long)
    cache_seqlens = lens.to(torch.int32)
    q = torch.randn(b, qlen, h, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    k_descale = torch.full((b, hk), ks.item(), device=dev, dtype=torch.float32)
    v_descale = torch.full((b, hk), vs.item(), device=dev, dtype=torch.float32)
    return q, kc, vc, page_table, cache_seqlens, k_descale, v_descale


def reference(q, kc, vc, page_table, cache_seqlens, scale, ks, vs, causal, round_bf16):
    """Gather fp8 KV through the block table, dequant (optionally round to bf16), attention."""
    b, qlen, h, d = q.shape
    hk = kc.shape[2]; gpr = h // hk
    out = torch.empty(b, qlen, h, d, device=q.device, dtype=q.dtype)
    for i in range(b):
        L = int(cache_seqlens[i])
        blk = page_table[i].long()
        kg = kc[blk].reshape(-1, hk, d)[:L].float() * ks
        vg = vc[blk].reshape(-1, hk, d)[:L].float() * vs
        if round_bf16:                                   # match kernel: KV rounded to bf16 pre-MMA
            kg = kg.to(torch.bfloat16).float(); vg = vg.to(torch.bfloat16).float()
        for j in range(qlen):
            qj = q[i, j].float().view(hk, gpr, d)
            s = torch.einsum("kgd,lkd->kgl", qj, kg) * scale
            if causal:                                   # query j = position L-qlen+j, attends [0, L-qlen+j]
                s[:, :, (L - qlen + j + 1):] = float("-inf")
            p = s.softmax(dim=-1)
            out[i, j] = torch.einsum("kgl,lkd->kgd", p, vg).reshape(h, d).to(q.dtype)
    return out


def run_kernel(fa, q, kc, vc, pt, cs, scale, ks, vs, splits, causal):
    return fa.flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=cs, page_table=pt, causal=causal,
        softmax_scale=scale, num_splits=splits, k_descale=ks, v_descale=vs)


def check(fa, tag, b, kv, h, hk, d, qlen, scale, splits, causal, seed=0):
    q, kc, vc, pt, cs, ks, vs = make_paged(b, kv, h, hk, d, qlen, ragged=(qlen == 1), seed=seed)
    try:
        out = run_kernel(fa, q, kc, vc, pt, cs, scale, ks, vs, splits, causal)
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  {tag}: RUN FAILED {type(e).__name__}: {e}"); return False
    finite = torch.isfinite(out).all().item()
    ref_bf16 = reference(q, kc, vc, pt, cs, scale, ks.view(-1)[0].item(), vs.view(-1)[0].item(), causal, True)
    ref_fp32 = reference(q, kc, vc, pt, cs, scale, ks.view(-1)[0].item(), vs.view(-1)[0].item(), causal, False)
    e_bf16 = (out.float() - ref_bf16.float()).abs().max().item()
    e_fp32 = (out.float() - ref_fp32.float()).abs().max().item()
    ok = finite and torch.allclose(out.float(), ref_bf16.float(), rtol=RTOL, atol=ATOL)
    print(f"  {tag:<34} finite={finite} err_bf16={e_bf16:.3e} (diag err_fp32={e_fp32:.3e}) | {'OK' if ok else 'MISMATCH'}")
    return ok


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

    print("== q=1 causal correctness (primary bf16 oracle; fp32 diagnostic) ==")
    ok = True
    for (b, kv) in [(1, 256), (1, 2048), (2, 512), (4, 2048), (8, 9216), (16, 4096)]:
        ok &= check(fa, f"b={b} kv={kv} q=1 split=1", b, kv, h, hk, d, 1, scale, 1, True)

    print("== q=5 x num_splits=16 x multi-row (ProducerConvert barrier stress, non-causal clean oracle) ==")
    for (b, kv) in [(8, 9216), (16, 4096)]:
        ok &= check(fa, f"b={b} kv={kv} q=5 split=16", b, kv, h, hk, d, 5, scale, 16, False)

    print("M5_DEQUANT_CORRECTNESS_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
