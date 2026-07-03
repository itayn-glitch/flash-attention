"""M5 dequant-on-load correctness: bf16 query + fp8(e4m3) paged KV cache -> dequant->bf16 wgmma.

Validates the Path-B producer-convert kernel (kv_is_fp8 dispatch, PagedKVManager<ElementKV>,
cp.async fp8->staging, convert->bf16 smem with k/v_descale folded in). CORRECTNESS gate on the
scalar-convert scaffold; NO perf claim (scalar convert is the known-losing path).

Hardening (P1/P2 review #5):
 - PER-(batch,kv-head) DISTINCT descales, cache quantized consistently per (b,kh): catches wrong
   bidb/bidh_kv indexing, swapped strides, head-0 broadcast -- not just "descale entirely missing".
 - q=5 CAUSAL split=16 (promoted MTP verify shape + mask x barrier interaction), plus q=5 ragged.
 - NODESCALE discriminator kept.
Oracles: PRIMARY = dequant-to-BF16 then attention (kernel rounds fp8->float*descale->bf16 pre-wgmma,
matching prod mma f32.bf16.bf16). DIAGNOSTIC = dequant-to-FP32 (attribution only).

Scope of a PASS here: fp8-KV dequant path correct for NO-softcap, BF16 output, vs BF16-dequant oracle.
Softcap / fp8-output / Triton parity are separate later gates.

Run (box overlay venv, free GPU), after a dequant build:
  FLASH_PRINT_SMEM=1 CUDA_VISIBLE_DEVICES=1 python m5_dequant_microbench.py
"""
import argparse, sys, torch

PAGE = 16
RTOL, ATOL = 3e-2, 3e-2
E4M3_MAX = 448.0


def make_paged(b, kv, h, hk, d, qlen, ragged, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    bps = (kv + PAGE - 1) // PAGE
    num_blocks = bps * b * 2 + 1
    kc_bf16 = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    vc_bf16 = torch.randn(num_blocks, PAGE, hk, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    perm = torch.randperm(num_blocks - 1, generator=g, device=dev)[: b * bps] + 1
    page_table = perm.to(torch.int32).reshape(b, bps)
    # Per-(batch, kv-head) quantization with DISTINCT data-derived scales. Each batch's physical
    # blocks are disjoint (permutation), so per-(b,kh) quant is self-consistent.
    kc = torch.zeros(num_blocks, PAGE, hk, d, device=dev, dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    ks = torch.ones(b, hk, device=dev, dtype=torch.float32)
    vs = torch.ones(b, hk, device=dev, dtype=torch.float32)
    for i in range(b):
        blk = page_table[i].long()
        for kh in range(hk):
            sk = (kc_bf16[blk, :, kh, :].abs().amax() / E4M3_MAX).clamp(min=1e-6).item()
            sv = (vc_bf16[blk, :, kh, :].abs().amax() / E4M3_MAX).clamp(min=1e-6).item()
            ks[i, kh], vs[i, kh] = sk, sv
            kc[blk, :, kh, :] = (kc_bf16[blk, :, kh, :].float() / sk).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            vc[blk, :, kh, :] = (vc_bf16[blk, :, kh, :].float() / sv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    if ragged and b > 1:
        lens = torch.randint(max(1, kv // 2), kv + 1, (b,), generator=g, device=dev); lens[0] = kv
    else:
        lens = torch.full((b,), kv, device=dev, dtype=torch.long)
    cache_seqlens = lens.to(torch.int32)
    q = torch.randn(b, qlen, h, d, generator=g, device=dev, dtype=torch.bfloat16) * 0.1
    return q, kc, vc, page_table, cache_seqlens, ks, vs


def reference(q, kc, vc, page_table, cache_seqlens, scale, ks, vs, causal, round_bf16):
    """Gather fp8 KV through the block table, dequant with PER-(b,kh) scale, attention.
    ks/vs: (b, hk) descale tensors (or None -> descale=1 for NODESCALE discriminator)."""
    b, qlen, h, d = q.shape
    hk = kc.shape[2]; gpr = h // hk
    out = torch.empty(b, qlen, h, d, device=q.device, dtype=q.dtype)
    for i in range(b):
        L = int(cache_seqlens[i])
        blk = page_table[i].long()
        kg = kc[blk].reshape(-1, hk, d)[:L].float()
        vg = vc[blk].reshape(-1, hk, d)[:L].float()
        if ks is not None:                                # per-head descale: (hk,) broadcast over l,d
            kg = kg * ks[i].view(1, hk, 1); vg = vg * vs[i].view(1, hk, 1)
        if round_bf16:
            kg = kg.to(torch.bfloat16).float(); vg = vg.to(torch.bfloat16).float()
        for j in range(qlen):
            qj = q[i, j].float().view(hk, gpr, d)
            s = torch.einsum("kgd,lkd->kgl", qj, kg) * scale
            if causal:
                s[:, :, (L - qlen + j + 1):] = float("-inf")
            p = s.softmax(dim=-1)
            out[i, j] = torch.einsum("kgl,lkd->kgd", p, vg).reshape(h, d).to(q.dtype)
    return out


def run_kernel(fa, q, kc, vc, pt, cs, scale, ks, vs, splits, causal):
    return fa.flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=cs, page_table=pt, causal=causal,
        softmax_scale=scale, num_splits=splits, k_descale=ks, v_descale=vs)


def check(fa, tag, b, kv, h, hk, d, qlen, scale, splits, causal, ragged, seed=0):
    q, kc, vc, pt, cs, ks, vs = make_paged(b, kv, h, hk, d, qlen, ragged, seed)
    try:
        out = run_kernel(fa, q, kc, vc, pt, cs, scale, ks, vs, splits, causal)
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  {tag:<40} RUN FAILED {type(e).__name__}: {e}"); return False
    finite = torch.isfinite(out).all().item()
    ref_bf16 = reference(q, kc, vc, pt, cs, scale, ks, vs, causal, True)
    ref_nods = reference(q, kc, vc, pt, cs, scale, None, None, causal, True)   # descale-not-applied discriminator
    e_bf16 = (out.float() - ref_bf16.float()).abs().max().item()
    e_nods = (out.float() - ref_nods.float()).abs().max().item()
    ok = finite and torch.allclose(out.float(), ref_bf16.float(), rtol=RTOL, atol=ATOL)
    print(f"  {tag:<40} finite={finite} err_bf16={e_bf16:.2e} err_NODESCALE={e_nods:.2e} | {'OK' if ok else 'MISMATCH'}")
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
    ok = True

    print("== q=1 causal, PER-(b,kh) distinct descales (indexing/stride check) ==")
    for (b, kv) in [(1, 256), (1, 2048), (2, 512), (4, 2048), (8, 9216), (16, 4096)]:
        ok &= check(fa, f"b={b} kv={kv} q=1 split=1 ragged", b, kv, h, hk, d, 1, scale, 1, True, ragged=True)

    print("== q=5 split=16 multi-row -- causal (MTP verify) + non-causal + ragged ==")
    ok &= check(fa, "b=8 kv=9216 q=5 split=16 causal",   8, 9216, h, hk, d, 5, scale, 16, True,  ragged=False)
    ok &= check(fa, "b=16 kv=4096 q=5 split=16 causal",  16, 4096, h, hk, d, 5, scale, 16, True,  ragged=False)
    ok &= check(fa, "b=8 kv=9216 q=5 split=16 noncausal", 8, 9216, h, hk, d, 5, scale, 16, False, ragged=False)
    ok &= check(fa, "b=16 kv=4096 q=5 split=16 causal RAGGED", 16, 4096, h, hk, d, 5, scale, 16, True, ragged=True)

    print("M5_DEQUANT_CORRECTNESS_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
