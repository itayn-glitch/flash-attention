import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TREE_MASK = ROOT / "flash_attn" / "cute" / "tree_mask.py"
MASK = ROOT / "flash_attn" / "cute" / "mask.py"
INTERFACE = ROOT / "flash_attn" / "cute" / "interface.py"


def test_tree_mask_is_bounded_and_uses_parent_depth_metadata():
    source = TREE_MASK.read_text()
    tree = ast.parse(source)
    assert "MAX_TREE_DEPTH = 8" in source
    assert "parents, depths = aux_tensors" in source
    assert "n_idx < prefix_len_ssa" in source
    assert "spec_idx == current" in source
    assert any(isinstance(node, ast.For) for node in ast.walk(tree))
    assert "logical query rows in one WGMMA M tile" in source
    assert "def tree_bitset_mask" in source
    assert "utils.shl_u32" in source
    assert "ancestor_bits[batch[0], m_idx[0]]" in source
    assert "cutlass.Uint32(spec_idx[0])" in source


def test_sm90_composes_custom_mask_with_causal_and_local_masks():
    source = MASK.read_text()
    interface_source = INTERFACE.read_text()
    resolver = interface_source.split("def _resolve_causal_local_window", 1)[1].split(
        "def _flash_attn_fwd", 1
    )[0]
    assert "if mask_mod is not None:" not in resolver
    assert source.count("self.apply_mask_mod_sm90_scalar(") == 2
    assert "if const_expr(mask_mod is not None):" in source
    causal_branch = source.split("else:  # Causal or local", 1)[1].split(
        "@cute.jit\n    def apply_mask_mod_sm90_scalar", 1
    )[0]
    custom_call = causal_branch.split("self.apply_mask_mod_sm90_scalar", 1)[1]
    assert "inactive M-tile lanes" in custom_call
    assert "True," in custom_call
    scalar_mask = source.split("def apply_mask_mod_sm90_scalar", 1)[1].split(
        "def apply_mask_mod_sm100_scalar", 1
    )[0]
    assert scalar_mask.index("if out_of_bounds:") < scalar_mask.index("mask_value = mask_mod(")


def test_multistage_convert_does_not_alias_fp8_and_bf16_buffers():
    source = (ROOT / "flash_attn" / "cute" / "flash_fwd_sm90.py").read_text()
    assert "self.alias_convert_smem = self.num_stages == 1" in source
    assert "else storage.sK8" in source
    assert "else storage.sV8" in source
