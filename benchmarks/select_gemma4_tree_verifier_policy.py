#!/usr/bin/env python3
"""Reduce exhaustive tree-verifier measurements into capacity-bucket split policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RUNTIME_KEYS = ("tp_degree", "context", "batch", "layer_type", "capacity")


def valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("parity_pass")
        and row.get("graph_stable")
        and row.get("post_ready_jit_count") == 0
    )


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in RUNTIME_KEYS)


def reduce_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if valid(row):
            grouped[key(row)][row["num_splits"]].append(row)

    policy = []
    for runtime_key, by_split in sorted(grouped.items()):
        expected_q = {
            row["q"] for split_rows in by_split.values() for row in split_rows
        }
        candidates = []
        for split, split_rows in by_split.items():
            q_values = {row["q"] for row in split_rows}
            if q_values != expected_q:
                continue
            latencies = [row["latency_us"] for row in split_rows]
            candidates.append(
                {
                    "num_splits": split,
                    "mean_us": sum(latencies) / len(latencies),
                    "worst_us": max(latencies),
                    "rows": len(split_rows),
                    "q_values": sorted(q_values),
                }
            )
        if not candidates:
            raise RuntimeError(f"no complete valid split candidates for {runtime_key}")
        selected = min(candidates, key=lambda item: (item["mean_us"], item["worst_us"]))
        policy.append(
            {
                **dict(zip(RUNTIME_KEYS, runtime_key, strict=True)),
                "selected": selected,
                "candidates": sorted(candidates, key=lambda item: item["num_splits"]),
            }
        )
    return {
        "schema_version": 1,
        "policy_keys": RUNTIME_KEYS,
        "dispatch_contract": "capacity bucket only; exact q is never a policy key",
        "entries": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload.get("status") != "complete" or payload.get("failures"):
        raise RuntimeError("policy input must be a completed, failure-free gate")
    result = reduce_policy(payload["rows"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"entries": len(result["entries"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
