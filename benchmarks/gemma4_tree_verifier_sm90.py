#!/usr/bin/env python3
"""SM90 CuTe gate for Gemma 4 q<=31 tree-verifier attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.tree_mask import MAX_TREE_DEPTH, tree_bitset_mask, tree_causal_mask


PAGE_SIZE = 16
Q_ANCHORS = (2, 5, 9, 13, 17, 21, 25, 31)
GLOBAL_LAYERS = 12
SWA_LAYERS = 48


@dataclass(frozen=True)
class LayerSpec:
    name: str
    head_dim: int
    window: int | None
    tp1_kv_heads: int

    def local_heads(self, tp: int) -> tuple[int, int]:
        return 32 // tp, self.tp1_kv_heads // tp


LAYERS = (
    LayerSpec("global", 512, None, 4),
    LayerSpec("swa", 256, 1024, 16),
)


def parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def capacity_for_q(q: int) -> int:
    if 2 <= q <= 16:
        return 16
    if q <= 31:
        return 32
    raise ValueError("q must be in 2..31")


def metadata_capacity_for_q(q_capacity: int, layer: LayerSpec, tp: int) -> int:
    q_heads, kv_heads = layer.local_heads(tp)
    gqa = q_heads // kv_heads
    tile_m = 64 if layer.head_dim > 256 else 128
    return max(q_capacity, math.ceil(tile_m / gqa))


def active_lengths(q: int, batch: int) -> tuple[int, ...]:
    choices = (q, max(2, q - 1), max(2, (q + 1) // 2), max(2, q - 3))
    return tuple(choices[index % len(choices)] for index in range(batch))


def context_lengths(context: int, q_lens: tuple[int, ...]) -> tuple[int, ...]:
    offsets = (0, 113, 337, 701)
    return tuple(
        max(q_len + PAGE_SIZE, context - offsets[index % len(offsets)])
        for index, q_len in enumerate(q_lens)
    )


def tree_for_length(length: int, topology: int) -> tuple[list[int], list[int]]:
    parents = list(range(length))
    depths = [0] * length
    for node in range(1, length):
        if topology % 3 == 0:
            parent = (node - 1) // 2
        elif topology % 3 == 1:
            parent = 0
        else:
            parent = (node - 1) // 3
        parents[node] = parent
        depths[node] = depths[parent] + 1
    if max(depths, default=0) > MAX_TREE_DEPTH:
        raise ValueError("tree exceeds the compiled parent-walk depth")
    return parents, depths


def ancestors(parents: list[int], node: int) -> set[int]:
    result = {node}
    while parents[node] != node:
        node = parents[node]
        result.add(node)
    return result


def make_case(
    q_anchor: int,
    batch: int,
    context: int,
    layer: LayerSpec,
    tp: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    capacity = capacity_for_q(q_anchor)
    q_lens = active_lengths(q_anchor, batch)
    k_lens = context_lengths(context, q_lens)
    q_heads, kv_heads = layer.local_heads(tp)
    metadata_capacity = metadata_capacity_for_q(capacity, layer, tp)
    max_pages = math.ceil(max(k_lens) / PAGE_SIZE)
    pages_per_seq = max_pages + 3
    physical_pages = batch * pages_per_seq
    generator = torch.Generator(device=device).manual_seed(seed)
    q_scale, k_scale, v_scale = 0.75, 0.5, 0.25

    total_q = batch * capacity
    q_real = torch.randn(
        total_q,
        q_heads,
        layer.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).clamp_(-2, 2)
    q_fp8 = (q_real / q_scale).to(fp8)
    q = q_fp8.to(torch.bfloat16).mul_(q_scale)

    cache_shape = (physical_pages, PAGE_SIZE, kv_heads, layer.head_dim)
    k_real = torch.randn(
        cache_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).clamp_(-2, 2)
    v_real = torch.randn(
        cache_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).clamp_(-2, 2)
    k = (k_real / k_scale).to(fp8)
    v = (v_real / v_scale).to(fp8)

    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    for seq_idx, seq_len in enumerate(k_lens):
        page_count = math.ceil(seq_len / PAGE_SIZE)
        physical = torch.arange(
            seq_idx * pages_per_seq,
            seq_idx * pages_per_seq + page_count,
            dtype=torch.int32,
            device=device,
        )
        order = torch.randperm(page_count, device=device, generator=generator)
        page_table[seq_idx, :page_count] = physical[order]

    # SM90 evaluates mask_mod for a complete WGMMA M tile. CuTe predicates the
    # result but may still issue auxiliary loads for inactive lanes, so compact
    # metadata is padded to the logical M-tile width rather than active q.
    parents = torch.arange(metadata_capacity, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    depths = torch.zeros(batch, metadata_capacity, dtype=torch.int32, device=device)
    ancestor_bits = torch.zeros(batch, metadata_capacity, dtype=torch.int32, device=device)
    trees = []
    for seq_idx, q_len in enumerate(q_lens):
        parent_values, depth_values = tree_for_length(q_len, seq_idx)
        parents[seq_idx, :q_len] = torch.tensor(parent_values, device=device)
        depths[seq_idx, :q_len] = torch.tensor(depth_values, device=device)
        ancestor_bits[seq_idx, :q_len] = torch.tensor(
            [sum(1 << ancestor for ancestor in ancestors(parent_values, row)) for row in range(q_len)],
            dtype=torch.int32,
            device=device,
        )
        trees.append(parent_values)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.empty_like(q),
        "cu_q": torch.arange(0, total_q + 1, capacity, dtype=torch.int32, device=device),
        "q_lens_tensor": torch.tensor(q_lens, dtype=torch.int32, device=device),
        "k_lens_tensor": torch.tensor(k_lens, dtype=torch.int32, device=device),
        "page_table": page_table,
        "parents": parents,
        "depths": depths,
        "ancestor_bits": ancestor_bits,
        "trees": trees,
        "capacity": capacity,
        "metadata_capacity": metadata_capacity,
        "q_lens": q_lens,
        "k_lens": k_lens,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": layer.head_dim,
        "window": layer.window,
        "scales": (q_scale, k_scale, v_scale),
    }


def trusted_reference(case: dict[str, Any]) -> torch.Tensor:
    _, k_scale, v_scale = case["scales"]
    q_heads, kv_heads = case["q_heads"], case["kv_heads"]
    group = q_heads // kv_heads
    result = torch.zeros_like(case["q"], dtype=torch.float32)
    scale = 1.0 / math.sqrt(case["head_dim"])

    for seq_idx, (q_len, k_len) in enumerate(zip(case["q_lens"], case["k_lens"], strict=True)):
        page_count = math.ceil(k_len / PAGE_SIZE)
        pages = case["page_table"][seq_idx, :page_count].long()
        keys = case["k"][pages].reshape(-1, kv_heads, case["head_dim"])[:k_len]
        values = case["v"][pages].reshape(-1, kv_heads, case["head_dim"])[:k_len]
        keys = keys.to(torch.bfloat16).mul(k_scale).float()
        values = values.to(torch.bfloat16).mul(v_scale).float()
        q_start = seq_idx * case["capacity"]
        queries = case["q"][q_start : q_start + q_len]
        queries = queries.reshape(q_len, kv_heads, group, case["head_dim"])
        scores = torch.einsum("qhgd,khd->hgqk", queries.float(), keys).mul_(scale)

        prefix_len = k_len - q_len
        visible = torch.zeros(q_len, k_len, dtype=torch.bool, device=queries.device)
        for row in range(q_len):
            visible[row, :prefix_len] = True
            for tree_row in ancestors(case["trees"][seq_idx], row):
                visible[row, prefix_len + tree_row] = True
            if case["window"] is not None:
                window_start = max(0, prefix_len + row + 1 - case["window"])
                visible[row, :window_start] = False
        scores.masked_fill_(~visible[None, None], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum("hgqk,khd->qhgd", probabilities, values)
        result[q_start : q_start + q_len] = output.reshape(
            q_len, q_heads, case["head_dim"]
        )
    return result


def run_candidate(case: dict[str, Any], num_splits: int) -> None:
    _, k_scale, v_scale = case["scales"]
    window_left = None if case["window"] is None else case["window"] - 1
    window_right = None if case["window"] is None else 0
    _flash_attn_fwd(
        case["q"],
        case["k"],
        case["v"],
        out=case["out"],
        cu_seqlens_q=case["cu_q"],
        seqused_q=case["q_lens_tensor"],
        seqused_k=case["k_lens_tensor"],
        max_seqlen_q=case["capacity"],
        max_seqlen_k=max(case["k_lens"]),
        page_table=case["page_table"],
        softmax_scale=k_scale / math.sqrt(case["head_dim"]),
        causal=True,
        window_size_left=window_left,
        window_size_right=window_right,
        num_splits=num_splits,
        pack_gqa=True,
        mask_mod=case["mask_mod"],
        aux_tensors=[case["parents"], case["depths"], case["ancestor_bits"]],
    )
    case["out"].mul_(v_scale)


def parity(case: dict[str, Any], reference: torch.Tensor) -> dict[str, float | int]:
    active = torch.zeros(case["q"].shape[0], dtype=torch.bool, device=case["q"].device)
    for seq_idx, q_len in enumerate(case["q_lens"]):
        start = seq_idx * case["capacity"]
        active[start : start + q_len] = True
    active_values = active[:, None, None].expand_as(case["out"])
    delta = case["out"].float()[active_values] - reference[active_values]
    inactive = case["out"].float()[~active_values]
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(reference[active_values])
        ).item(),
        "inactive_max_abs": inactive.abs().max().item() if inactive.numel() else 0.0,
        "nan": torch.isnan(case["out"]).sum().item(),
        "inf": torch.isinf(case["out"]).sum().item(),
    }


def measure(case: dict[str, Any], reference: torch.Tensor, num_splits: int, warmups: int, replays: int) -> dict[str, Any]:
    from flash_attn.cute.interface import _flash_attn_fwd_combine

    case["out"].zero_()
    for _ in range(warmups):
        run_candidate(case, num_splits)
    torch.cuda.synchronize()
    compile_sizes = (
        len(_flash_attn_fwd.compile_cache.cache),
        len(_flash_attn_fwd_combine.compile_cache.cache),
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_candidate(case, num_splits)
    graph.replay()
    torch.cuda.synchronize()
    capture_parity = parity(case, reference)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    replay_parity = parity(case, reference)
    compile_sizes_after = (
        len(_flash_attn_fwd.compile_cache.cache),
        len(_flash_attn_fwd_combine.compile_cache.cache),
    )
    elapsed_us = start.elapsed_time(end) * 1000.0 / replays
    del graph
    return {
        "latency_us": elapsed_us,
        "capture_parity": capture_parity,
        "replay_parity": replay_parity,
        "parity_pass": capture_parity["max_abs"] <= 2e-2
        and capture_parity["relative_l2"] <= 1e-2
        and capture_parity["nan"] == 0
        and capture_parity["inf"] == 0
        and replay_parity["max_abs"] <= 2e-2
        and replay_parity["relative_l2"] <= 1e-2,
        "graph_replays": replays,
        "compile_cache_before": compile_sizes,
        "compile_cache_after": compile_sizes_after,
        "post_ready_jit_count": sum(
            after - before for before, after in zip(compile_sizes, compile_sizes_after, strict=True)
        ),
        "graph_stable": compile_sizes == compile_sizes_after,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=parse_ints, default=Q_ANCHORS)
    parser.add_argument("--batches", type=parse_ints, default=(1, 4))
    parser.add_argument("--contexts", type=parse_ints, default=(1024, 9600, 17400))
    parser.add_argument("--tp", type=parse_ints, default=(1, 2))
    parser.add_argument("--splits", type=parse_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument("--layers", default="global,swa")
    parser.add_argument("--swa-window", type=int, default=1024)
    parser.add_argument("--tree-mask", choices=("parent", "bitset"), default="bitset")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 H100 is required")
    if any(q < 2 or q > 31 for q in args.q):
        raise ValueError("q must be in 2..31")
    selected_layers = tuple(
        LayerSpec(
            layer.name,
            layer.head_dim,
            None if layer.name == "swa" and args.swa_window == 0 else layer.window,
            layer.tp1_kv_heads,
        )
        for layer in LAYERS
        if layer.name in set(args.layers.split(","))
    )
    if not selected_layers:
        raise ValueError("--layers must select global and/or swa")
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "/tmp/cute-tree-q31-cache")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.time(),
        "device": torch.cuda.get_device_name(),
        "rows": [],
        "failures": [],
    }
    atomic_write(args.output, payload)

    cell = 0
    for tp in args.tp:
        for context in args.contexts:
            for batch in args.batches:
                for q in args.q:
                    for layer in selected_layers:
                        case = make_case(q, batch, context, layer, tp, args.seed + cell)
                        case["mask_mod"] = (
                            tree_causal_mask if args.tree_mask == "parent" else tree_bitset_mask
                        )
                        reference = trusted_reference(case)
                        gqa = case["q_heads"] // case["kv_heads"]
                        packed_m = case["capacity"] * gqa
                        for num_splits in args.splits:
                            measured = measure(
                                case, reference, num_splits, args.warmups, args.replays
                            )
                            row = {
                                "tp_degree": tp,
                                "context": context,
                                "batch": batch,
                                "q": q,
                                "capacity": case["capacity"],
                                "metadata_capacity": case["metadata_capacity"],
                                "tree_mask": args.tree_mask,
                                "query_lengths": case["q_lens"],
                                "layer_type": layer.name,
                                "head_dim": layer.head_dim,
                                "local_q_heads": case["q_heads"],
                                "local_kv_heads": case["kv_heads"],
                                "gqa_ratio": gqa,
                                "packed_m": packed_m,
                                "m64_tiles_per_kv_head": math.ceil(packed_m / 64),
                                "num_splits": num_splits,
                                "estimated_ctas": batch
                                * case["kv_heads"]
                                * math.ceil(packed_m / 64)
                                * num_splits,
                                **measured,
                            }
                            payload["rows"].append(row)
                            if not row["parity_pass"] or not row["graph_stable"]:
                                payload["failures"].append(row)
                            atomic_write(args.output, payload)
                            print(json.dumps(row), flush=True)
                        del case, reference
                        torch.cuda.empty_cache()
                        cell += 1

    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in payload["rows"]:
        key = (
            row["tp_degree"], row["context"], row["batch"], row["q"], row["layer_type"]
        )
        if row["parity_pass"] and row["graph_stable"]:
            if key not in best or row["latency_us"] < best[key]["latency_us"]:
                best[key] = row
    payload["best_rows"] = list(best.values())
    payload["status"] = "complete" if not payload["failures"] else "failed"
    payload["completed_at"] = time.time()
    atomic_write(args.output, payload)
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
