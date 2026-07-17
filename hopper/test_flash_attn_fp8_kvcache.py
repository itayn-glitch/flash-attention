import math
import os

import pytest
import torch

from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
        reason="FP8 KV dequantization requires Hopper",
    ),
    pytest.mark.skipif(
        os.getenv("FLASH_ATTENTION_DISABLE_DEQUANTKV", "FALSE") == "TRUE",
        reason="FP8 KV dequantization is disabled",
    ),
]

PAGE_SIZE = 16
HEAD_DIM = 512


def _make_paged_fp8_kv(seqlen_q, seqlen_k, ragged):
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    batch_size, num_heads, num_heads_kv = 2, 8, 2
    blocks_per_sequence = math.ceil(seqlen_k / PAGE_SIZE)
    num_blocks = batch_size * blocks_per_sequence + 1
    page_table = (
        torch.randperm(num_blocks - 1, generator=generator, device=device)[: batch_size * blocks_per_sequence]
        .add(1)
        .to(torch.int32)
        .reshape(batch_size, blocks_per_sequence)
    )
    k_cache = (
        torch.randn(
            num_blocks,
            PAGE_SIZE,
            num_heads_kv,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        .mul(8)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn(
            num_blocks,
            PAGE_SIZE,
            num_heads_kv,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        .mul(8)
        .to(torch.float8_e4m3fn)
    )
    q = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        HEAD_DIM,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.1)
    cache_seqlens = torch.full((batch_size,), seqlen_k, device=device, dtype=torch.int32)
    if ragged:
        cache_seqlens[-1] -= PAGE_SIZE + 3

    scale_index = torch.arange(batch_size * num_heads_kv, device=device, dtype=torch.float32).reshape(
        batch_size, num_heads_kv
    )
    k_descale = 0.015 + scale_index * 0.007
    v_descale = 0.021 + scale_index.flip((0, 1)) * 0.009
    return q, k_cache, v_cache, page_table, cache_seqlens, k_descale, v_descale


def _attention_ref(
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    k_descale,
    v_descale,
    causal,
):
    batch_size, seqlen_q, num_heads, head_dim = q.shape
    num_heads_kv = k_cache.shape[2]
    q_heads_per_kv_head = num_heads // num_heads_kv
    softmax_scale = head_dim**-0.5
    out = torch.empty_like(q)
    for batch_idx in range(batch_size):
        seqlen = int(cache_seqlens[batch_idx])
        blocks = page_table[batch_idx].long()
        k = k_cache[blocks].reshape(-1, num_heads_kv, head_dim)[:seqlen].to(torch.bfloat16).float()
        v = v_cache[blocks].reshape(-1, num_heads_kv, head_dim)[:seqlen].to(torch.bfloat16).float()
        for query_idx in range(seqlen_q):
            q_row = q[batch_idx, query_idx].float().reshape(num_heads_kv, q_heads_per_kv_head, head_dim)
            scores = torch.einsum("kgd,lkd->kgl", q_row, k) * softmax_scale
            if k_descale is not None:
                scores *= k_descale[batch_idx].reshape(num_heads_kv, 1, 1)
            if causal:
                scores[:, :, seqlen - seqlen_q + query_idx + 1 :] = -torch.inf
            probabilities = torch.softmax(scores, dim=-1)
            output = torch.einsum("kgl,lkd->kgd", probabilities, v)
            if v_descale is not None:
                output *= v_descale[batch_idx].reshape(num_heads_kv, 1, 1)
            out[batch_idx, query_idx] = output.reshape(num_heads, head_dim).to(torch.bfloat16)
    return out


@pytest.mark.parametrize(
    "seqlen_q,num_splits,causal,ragged,precompute_metadata",
    [
        pytest.param(1, 1, True, True, False, id="decode-ragged"),
        pytest.param(5, 16, True, False, True, id="split-causal-metadata"),
        pytest.param(5, 16, False, False, False, id="split-noncausal"),
        pytest.param(5, 16, True, True, True, id="split-causal-ragged-metadata"),
    ],
)
def test_paged_fp8_kv_dequant(seqlen_q, num_splits, causal, ragged, precompute_metadata):
    q, k_cache, v_cache, page_table, cache_seqlens, k_descale, v_descale = _make_paged_fp8_kv(
        seqlen_q, 383, ragged
    )
    scheduler_metadata = None
    if precompute_metadata:
        scheduler_metadata = get_scheduler_metadata(
            q.shape[0],
            seqlen_q,
            383,
            q.shape[2],
            k_cache.shape[2],
            HEAD_DIM,
            cache_seqlens,
            qkv_dtype=torch.bfloat16,
            page_size=PAGE_SIZE,
            causal=causal,
            num_splits=num_splits,
        )
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
    )
    ref = _attention_ref(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        k_descale,
        v_descale,
        causal,
    )
    no_scale_ref = _attention_ref(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        None,
        None,
        causal,
    )

    assert (ref.float() - no_scale_ref.float()).abs().max().item() > 0.1
    assert (out.float() - ref.float()).abs().max() < (out.float() - no_scale_ref.float()).abs().max()
    torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)
