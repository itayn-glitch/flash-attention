"""M5 step 1a: native fp8 head-512 (all-fp8 wgmma) baseline -- accuracy + speed.

FA3 fp8 requires q,k,v all e4m3 (flash_api.cpp:775 k/v dtype == q dtype) with per-tensor
descales -> fp8 wgmma. This is NOT prod's scheme (prod = bf16 q + fp8 KV -> bf16 mma);
that's option 2 (dequant-on-load), built later for the token-exact comparison.

Two accuracies, to quantify whether RHT/block-quant is needed:
  A. KERNEL correctness: fp8 kernel vs fp32 attention on the SAME dequantized fp8 values.
     (isolates the fp8 wgmma/softmax math from quantization noise; should be tight.)
  B. QUANT error: fp8 kernel vs fp32 attention on the ORIGINAL bf16 values.
     (this is the fp8 accuracy loss RHT + block quant would recover.)
Plus the fp8 traffic/speed win (KV is 1 byte/elem, vs bf16 2 bytes).

Run: CUDA_VISIBLE_DEVICES=1 python m5_fp8_microbench.py --softcap 50
"""
import argparse
import sys
import torch

PEAK_BW_GBPS = 3350.0
E4M3_MAX = 448.0


def per_tensor_fp8(x):
    """Quantize bf16 tensor -> (e4m3 tensor, scale) with true ~= fp8 * scale."""
    amax = x.abs().max().clamp_min(1e-6)
    scale = (amax / E4M3_MAX).float()
    xq = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale


def ref_fp32(q, k, v, scale, softcap):
    """fp32 attention (q=1, full-attn) on whatever q,k,v (already dequantized) are given."""
    b, _, h, d = q.shape
    hk = k.shape[2]; gpr = h // hk
    out = torch.empty(b, 1, h, d, device=q.device, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for i in range(b):
        qi = qf[i, 0].view(hk, gpr, d)
        sc = torch.einsum("kgd,skd->kgs", qi, kf[i]) * scale
        if softcap > 0:
            sc = softcap * torch.tanh(sc / softcap)
        out[i, 0] = torch.einsum("kgs,skd->kgd", sc.softmax(-1), vf[i]).reshape(h, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=9216)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--nheads", type=int, default=32)
    ap.add_argument("--nheads_k", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=512)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    import flash_attn_interface as fa
    dev = "cuda"
    h, hk, d = args.nheads, args.nheads_k, args.headdim
    scale = 1.0 / (d ** 0.5)
    print(f"module: {fa.flash_attn_3_cuda.__file__}  softcap={args.softcap}")

    def build(b, kv, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        q = torch.randn(b, 1, h, d, generator=g, device=dev, dtype=torch.bfloat16)
        k = torch.randn(b, kv, hk, d, generator=g, device=dev, dtype=torch.bfloat16)
        v = torch.randn(b, kv, hk, d, generator=g, device=dev, dtype=torch.bfloat16)
        qq, sq = per_tensor_fp8(q); kq, sk = per_tensor_fp8(k); vq, sv = per_tensor_fp8(v)
        qd = torch.full((b, hk), sq.item(), device=dev, dtype=torch.float32)
        kd = torch.full((b, hk), sk.item(), device=dev, dtype=torch.float32)
        vd = torch.full((b, hk), sv.item(), device=dev, dtype=torch.float32)
        return q, k, v, qq, kq, vq, qd, kd, vd

    def run(qq, kq, vq, qd, kd, vd):
        o = fa.flash_attn_func(qq, kq, vq, softmax_scale=scale, causal=True,
                               q_descale=qd, k_descale=kd, v_descale=vd, softcap=args.softcap)
        return o[0] if isinstance(o, tuple) else o

    print("== accuracy (A: kernel vs fp32-on-fp8 | B: quant err vs fp32-on-bf16) ==")
    ok = True
    for (b, kv) in [(1, 2048), (2, 512), (4, 2048), (8, 9216)]:
        q, k, v, qq, kq, vq, qd, kd, vd = build(b, kv, 0)
        try:
            out = run(qq, kq, vq, qd, kd, vd); torch.cuda.synchronize()
        except Exception as e:
            print(f"  b={b} kv={kv}: FAILED {type(e).__name__}: {e}"); ok = False; continue
        # A: reference on the exact dequantized fp8 values the kernel saw.
        refA = ref_fp32(qq.float() * qd[0, 0], kq.float() * kd[0, 0], vq.float() * vd[0, 0], scale, args.softcap)
        # B: reference on original bf16.
        refB = ref_fp32(q, k, v, scale, args.softcap)
        errA = (out.float() - refA).abs().max().item()
        errB = (out.float() - refB).abs().max().item()
        relB = ((out.float() - refB).abs().mean() / refB.abs().mean().clamp_min(1e-6)).item()
        okA = torch.allclose(out.float(), refA, rtol=5e-2, atol=5e-2)
        print(f"  b={b:>2} kv={kv:>5} | A(kernel) max={errA:.3e} {'OK' if okA else 'HI'} | "
              f"B(quant) max={errB:.3e} rel={relB:.3f}")
        ok = ok and okA
    print("M5_FP8_KERNEL_OK" if ok else "M5_FP8_KERNEL_SUSPECT")

    print("== perf (fp8 KV = 1 byte/elem; note the traffic halving vs bf16) ==")
    b, kv = args.batch, args.seqlen
    q, k, v, qq, kq, vq, qd, kd, vd = build(b, kv, 1)
    for _ in range(15): run(qq, kq, vq, qd, kd, vd)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
    for _ in range(args.iters): run(qq, kq, vq, qd, kd, vd)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / args.iters
    kv_bytes = 2 * b * kv * hk * d * 1  # fp8 = 1 byte
    bw = (kv_bytes + 2 * b * h * d * 1) / (ms * 1e-3) / 1e9
    print(f"perf b={b} kv={kv} | {ms*1e3:.1f} us/iter | fp8 KV={kv_bytes/1e6:.1f}MB | "
          f"BW={bw:.0f} GB/s = {100*bw/PEAK_BW_GBPS:.1f}% of peak")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
