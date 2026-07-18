import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "gemma4_tree_verifier_sm90.py"
KERNEL = ROOT / "flash_attn" / "cute" / "flash_fwd_sm90.py"


def test_harness_covers_generic_capacity_and_tp_geometry():
    source = HARNESS.read_text()
    ast.parse(source)
    assert "Q_ANCHORS = (2, 5, 9, 13, 17, 21, 25, 31)" in source
    assert "return 16" in source and "return 32" in source
    assert 'LayerSpec("global", 512, None, 4)' in source
    assert 'LayerSpec("swa", 256, 1024, 16)' in source
    assert "pack_gqa=True" in source
    assert 'choices=("parent", "bitset")' in source
    assert 'default="bitset"' in source
    assert 'case["mask_mod"]' in source
    assert 'case["ancestor_bits"]' in source
    assert "metadata_capacity_for_q" in source
    assert "math.ceil(tile_m / gqa)" in source


def test_harness_hoists_scales_out_of_graph_capture():
    source = HARNESS.read_text()
    run_candidate = source.split("def run_candidate", 1)[1].split("def parity", 1)[0]
    assert ".item()" not in run_candidate
    assert "softmax_scale=k_scale" in run_candidate
    assert 'case["out"].mul_(v_scale)' in run_candidate
    assert "torch.cuda.CUDAGraph" in source


def test_multistage_k_convert_keeps_conversion_type_contract():
    source = KERNEL.read_text()
    check_type = source.split("def _check_type", 1)[1].split("@cute.jit", 1)[0]
    assert "if const_expr(self.k_convert):" in check_type
    assert "self.k_convert and self.alias_convert_smem" not in check_type
