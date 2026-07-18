#!/usr/bin/env python3
"""Compare CuTe q<=31 verifier rows with the measured Triton control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GLOBAL_LAYERS = 12
SWA_LAYERS = 48
CELL_KEYS = ("tp_degree", "context", "batch", "q", "layer_type")


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[key] for key in CELL_KEYS)


def valid(row: dict[str, Any]) -> bool:
    return bool(row.get("parity_pass") and row.get("graph_stable"))


def fastest(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not valid(row):
            continue
        key = cell_key(row)
        if key not in selected or row["latency_us"] < selected[key]["latency_us"]:
            selected[key] = row
    return selected


def compare(cute_payload: dict[str, Any], triton_payload: dict[str, Any]) -> dict[str, Any]:
    cute = fastest(cute_payload["rows"])
    triton = fastest(triton_payload["rows"])
    comparisons = []
    for key in sorted(cute.keys() & triton.keys()):
        cute_row, triton_row = cute[key], triton[key]
        delta_us = cute_row["latency_us"] - triton_row["latency_us"]
        comparisons.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "cute_us": cute_row["latency_us"],
                "cute_num_splits": cute_row["num_splits"],
                "triton_us": triton_row["latency_us"],
                "triton_arm": triton_row["arm"],
                "delta_us": delta_us,
                "speedup": triton_row["latency_us"] / cute_row["latency_us"],
            }
        )

    by_pass: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in comparisons:
        key = (row["tp_degree"], row["context"], row["batch"], row["q"])
        by_pass.setdefault(key, {})[row["layer_type"]] = row
    pass_rows = []
    for key, layers in sorted(by_pass.items()):
        if set(layers) != {"global", "swa"}:
            continue
        cute_us = (
            GLOBAL_LAYERS * layers["global"]["cute_us"]
            + SWA_LAYERS * layers["swa"]["cute_us"]
        )
        triton_us = (
            GLOBAL_LAYERS * layers["global"]["triton_us"]
            + SWA_LAYERS * layers["swa"]["triton_us"]
        )
        pass_rows.append(
            {
                **dict(zip(("tp_degree", "context", "batch", "q"), key, strict=True)),
                "cute_attention_pass_ms": cute_us / 1000.0,
                "triton_attention_pass_ms": triton_us / 1000.0,
                "delta_ms": (cute_us - triton_us) / 1000.0,
                "speedup": triton_us / cute_us,
                "global_splits": layers["global"]["cute_num_splits"],
                "swa_splits": layers["swa"]["cute_num_splits"],
            }
        )
    return {
        "schema_version": 1,
        "matched_cells": len(comparisons),
        "missing_in_cute": [list(key) for key in sorted(triton.keys() - cute.keys())],
        "missing_in_triton": [list(key) for key in sorted(cute.keys() - triton.keys())],
        "cells": comparisons,
        "attention_passes": pass_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cute", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        json.loads(args.cute.read_text()), json.loads(args.triton.read_text())
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"matched_cells": payload["matched_cells"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
