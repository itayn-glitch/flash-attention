import cutlass
import cutlass.cute as cute

from flash_attn.cute import utils


MAX_TREE_DEPTH = 8


@cute.jit
def tree_causal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Keep committed prefix tokens and ancestors of each speculative node.

    Parent/depth metadata must cover the logical query rows in one WGMMA M tile;
    inactive rows use identity parents and depth zero.
    """
    del head
    parents, depths = aux_tensors[0], aux_tensors[1]
    prefix_len = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    prefix_len_ssa = utils.scalar_to_ssa(prefix_len, cutlass.Int32)
    spec_idx = n_idx - prefix_len_ssa
    current = m_idx
    query_depth = utils.scalar_to_ssa(depths[batch[0], m_idx[0]], cutlass.Int32)
    visible = n_idx < prefix_len_ssa

    for step in cutlass.range_constexpr(MAX_TREE_DEPTH + 1):
        step_ssa = utils.scalar_to_ssa(cutlass.Int32(step), cutlass.Int32)
        visible = visible | ((step_ssa <= query_depth) & (spec_idx == current))
        current = utils.scalar_to_ssa(parents[batch[0], current[0]], cutlass.Int32)

    return visible


@cute.jit
def tree_bitset_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    m_idx: cute.TensorSSA,
    n_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Apply the same tree mask from one compact ancestor bitset per query row."""
    del head
    ancestor_bits = aux_tensors[2]
    prefix_len = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    prefix_len_ssa = utils.scalar_to_ssa(prefix_len, cutlass.Int32)
    spec_idx = n_idx - prefix_len_ssa
    row_bits = cutlass.Uint32(ancestor_bits[batch[0], m_idx[0]])
    spec_bit = utils.shl_u32(cutlass.Uint32(1), cutlass.Uint32(spec_idx[0]))
    return (n_idx < prefix_len_ssa) | cutlass.Boolean(row_bits & spec_bit)
