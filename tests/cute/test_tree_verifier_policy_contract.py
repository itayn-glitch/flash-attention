from benchmarks.select_gemma4_tree_verifier_policy import reduce_policy


def row(q, split, latency):
    return {
        "tp_degree": 1,
        "context": 9600,
        "batch": 4,
        "layer_type": "global",
        "capacity": 16,
        "q": q,
        "num_splits": split,
        "latency_us": latency,
        "parity_pass": True,
        "graph_stable": True,
        "post_ready_jit_count": 0,
    }


def test_policy_uses_capacity_bucket_not_exact_q():
    result = reduce_policy(
        [
            row(2, 4, 10),
            row(5, 4, 12),
            row(13, 4, 14),
            row(2, 8, 11),
            row(5, 8, 11),
            row(13, 8, 11),
        ]
    )
    entry = result["entries"][0]
    assert result["dispatch_contract"] == "capacity bucket only; exact q is never a policy key"
    assert entry["capacity"] == 16
    assert entry["selected"]["num_splits"] == 8
    assert entry["selected"]["q_values"] == [2, 5, 13]
