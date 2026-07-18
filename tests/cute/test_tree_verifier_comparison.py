from benchmarks.compare_gemma4_tree_verifier import compare


def row(layer, latency, *, arm=None, splits=None):
    value = {
        "tp_degree": 1,
        "context": 9600,
        "batch": 4,
        "q": 13,
        "layer_type": layer,
        "latency_us": latency,
        "parity_pass": True,
        "graph_stable": True,
    }
    if arm is not None:
        value["arm"] = arm
    if splits is not None:
        value["num_splits"] = splits
    return value


def test_compare_selects_fastest_arms_and_weights_model_layers():
    cute = {"rows": [row("global", 80, splits=2), row("global", 70, splits=4), row("swa", 20, splits=1)]}
    triton = {"rows": [row("global", 100, arm="triton_2d"), row("global", 90, arm="triton_3d"), row("swa", 25, arm="triton_3d")]}
    result = compare(cute, triton)
    assert result["matched_cells"] == 2
    assert result["attention_passes"][0]["cute_attention_pass_ms"] == 1.8
    assert result["attention_passes"][0]["triton_attention_pass_ms"] == 2.28
    assert result["attention_passes"][0]["global_splits"] == 4
