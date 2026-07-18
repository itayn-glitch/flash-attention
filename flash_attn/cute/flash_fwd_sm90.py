# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

from types import SimpleNamespace
from typing import Callable, Literal, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    produce_block_sparse_loads,
    consume_block_sparse_loads,
)
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from cutlass.cute import FastDivmodDivisor

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        dtype_k=None,
        **kwargs,
    ):
        # M10b: K-convert-on-load. K (and/or V) may be stored fp8 in HBM while the
        # QK/PV compute dtype (kwargs["dtype"], e.g. bf16) differs -- mirrors the
        # existing V-convert (Arm C) split, applied to K. The base ctor only allows
        # dtype_v != dtype when dtype == Float8E4M3FN (the fp8-QK hybrid arms), so for
        # a bf16-QK K/V-convert arm we withhold dtype_v from the base call and patch
        # self.dtype_v/self.dtype_pv/self.v_convert in ourselves afterward -- this
        # keeps the existing fp8-QK Arm C path (dtype==fp8) byte-for-byte unchanged.
        dtype_arg = kwargs.get("dtype", args[0] if args else None)
        dtype_v_arg = kwargs.get("dtype_v", None)
        defer_dtype_v = (
            dtype_v_arg is not None
            and dtype_arg != cutlass.Float8E4M3FN
            and dtype_v_arg != dtype_arg
        )
        if defer_dtype_v:
            kwargs = dict(kwargs)
            kwargs.pop("dtype_v")
        super().__init__(*args, **kwargs)
        if defer_dtype_v:
            self.dtype_v = dtype_v_arg
            self.dtype_pv = self.dtype
            self.v_convert = self.dtype_pv != self.dtype_v
        # M10b: K storage dtype (fp8, paged) vs self.dtype (QK compute, bf16).
        self.dtype_k = dtype_k if dtype_k is not None else self.dtype
        self.k_convert = self.dtype_k != self.dtype
        assert self.output_quant_key is None, (
            f"Fused quant output not implemented for {type(self).__name__}"
        )
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.is_split_kv = is_split_kv
        self.alias_convert_smem = self.num_stages == 1
        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    def _check_type(
        self,
        mQ_type,
        mK_type,
        mV_type,
        mO_type,
        mLSE_type,
        mCuSeqlensQ_type,
        mCuSeqlensK_type,
        mSeqUsedQ_type,
        mSeqUsedK_type,
        is_split_kv: bool = False,
    ):
        # M10b: bf16(-or-f16)-QK with fp8-paged K (+ optionally V) convert-on-load.
        # Q is handed already-dequantized (real-valued) bf16/f16; K is raw fp8 codes
        # matching self.dtype_k; V may be fp8 (convert-on-load) or match self.dtype.
        if const_expr(self.k_convert):
            if const_expr(mQ_type not in [cutlass.BFloat16, cutlass.Float16]):
                raise TypeError("M10b K-convert path: Q must be bf16 or f16")
            if const_expr(mK_type != self.dtype_k):
                raise TypeError(f"M10b K-convert path: K must be stored as {self.dtype_k}")
            if const_expr(
                mV_type not in [cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16]
            ):
                raise TypeError("M10b K-convert path: V must be fp8_e4m3/f16/bf16")
            if const_expr(mO_type not in [cutlass.BFloat16, cutlass.Float16, Float32]):
                raise TypeError("M10b K-convert path: O must be bf16/f16/f32")
            if const_expr(mLSE_type not in [None, Float32]):
                raise TypeError("LSE tensor must be Float32")
            if const_expr(mCuSeqlensQ_type not in [None, Int32]):
                raise TypeError("cu_seqlens_q tensor must be Int32")
            if const_expr(mCuSeqlensK_type not in [None, Int32]):
                raise TypeError("cu_seqlens_k tensor must be Int32")
            if const_expr(mSeqUsedQ_type not in [None, Int32]):
                raise TypeError("seqused_q tensor must be Int32")
            if const_expr(mSeqUsedK_type not in [None, Int32]):
                raise TypeError("seqused_k tensor must be Int32")
            assert mQ_type == self.dtype
            return
        return super()._check_type(
            mQ_type,
            mK_type,
            mV_type,
            mO_type,
            mLSE_type,
            mCuSeqlensQ_type,
            mCuSeqlensK_type,
            mSeqUsedQ_type,
            mSeqUsedK_type,
            is_split_kv=is_split_kv,
        )

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        # M10b: separate raw-fp8 K staging atom (paged cp.async load target), mirrors sV8.
        self.sK8_layout_atom = None
        if self.k_convert:
            self.sK8_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype_k, self.tile_hdim
                ),
                self.dtype_k,
            )
        # fa4hybrid: sV is the PV-GEMM operand buffer, typed by the PV compute dtype
        # (bf16/fp16 in hybrid arms), NOT the QK dtype.
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype_pv, self.tile_hdimv
            ),
            self.dtype_pv,
        )
        # fa4hybrid Arm C: separate raw-fp8 V staging atom (TMA/cp.async load target)
        self.sV8_layout_atom = None
        if self.v_convert:
            self.sV8_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype_v, self.tile_hdimv
                ),
                self.dtype_v,
            )
        # fa4hybrid: O epilogue buffer typed by the output dtype (bf16)
        sO_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype_o, self.tile_hdimv
            ),
            self.dtype_o,
        )
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype_pv, self.tile_n
                ),
                self.dtype_pv,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        atom_layout_n = 2 if self.tile_hdim > 256 or self.tile_hdimv > 256 else 1
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, atom_layout_n, 1),
            tiler_mn=(64, self.tile_n),
        )
        # fa4hybrid: the PV wgmma runs in dtype_pv (fp16/bf16 for hybrid arms). fp16/bf16
        # wgmma accepts an MN-major B operand, so V keeps its natural (tile_n, head_dim_v)
        # smem layout with a transpose_view -- NO physical V transpose (this is the whole
        # point of the hybrid design; fp8 wgmma would force a K-major V).
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype_pv,
            self.dtype_pv,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                atom_layout_n,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, min(256, self.tile_hdimv)),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        # fa4hybrid: each smem buffer is typed by its own dtype:
        #   sQ/sK: self.dtype (fp8 in hybrid), sV: dtype_pv (fp16/bf16), sV8: dtype_v (fp8
        #   staging, Arm C only). The O epilogue overlays the sQ buffer but O is bf16 while
        #   Q may be fp8, so the union buffer is sized max(bytes(sQ), bytes(sO)) in units
        #   of self.dtype.
        sV8_cosize = (
            cute.cosize(self.sV8_layout) if const_expr(self.sV8_layout is not None) else 0
        )
        sK8_cosize = (
            cute.cosize(self.sK8_layout) if const_expr(self.sK8_layout is not None) else 0
        )
        # M10 tile64: the fp8 K staging (sK8) is ALIASED onto sK's bf16 region instead of
        # getting its own buffer -- sK is 2x the bytes/elem so it holds the fp8 staging
        # in-place. This reclaims one ~32KB buffer, the exact overflow that forced
        # tile_n 64->32. Size sK's MemRange to cover BOTH the bf16 compute layout and the
        # (smaller) fp8 staging layout (fp8 bytes rounded up to whole bf16 elements).
        cosize_sK = cute.cosize(self.sK_layout)
        if const_expr(self.k_convert):
            cosize_sK = max(
                cosize_sK,
                (sK8_cosize * self.dtype_k.width + self.dtype.width - 1) // self.dtype.width,
            )
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sK],
            self.buffer_align_bytes,
        ]
        # M10 tile64: the fp8 V staging (sV8) is ALIASED onto sV's bf16 region (same
        # technique + safety argument as sK8/sK), reclaiming a second ~32KB buffer.
        cosize_sV = cute.cosize(self.sV_layout)
        if const_expr(self.v_convert and self.alias_convert_smem):
            cosize_sV = max(
                cosize_sV,
                (sV8_cosize * self.dtype_v.width + self.dtype_pv.width - 1) // self.dtype_pv.width,
            )
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype_pv, cosize_sV],
            self.buffer_align_bytes,
        ]
        sK8_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype_k, sK8_cosize],
            self.buffer_align_bytes,
        ]
        sV8_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype_v, sV8_cosize],
            self.buffer_align_bytes,
        ]
        cosize_sQO = max(
            cute.cosize(self.sQ_layout),
            (cute.cosize(self.sO_layout) * self.dtype_o.width + self.dtype.width - 1)
            // self.dtype.width,
        )
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQO], self.buffer_align_bytes
        ]
        cosize_sQV = max(cosize_sQO, cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype_pv, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        if const_expr(self.v_convert and not self.k_convert):
            if const_expr(self.alias_convert_smem):
                @cute.struct
                class SharedStorageQKVV8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sP: sP_struct
            else:
                @cute.struct
                class SharedStorageQKVV8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sV8: sV8_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sP: sP_struct

            assert not self.Q_in_regs, "fa4hybrid Arm C does not support Q_in_regs"
            return SharedStorageQKVV8

        if const_expr(self.k_convert and not self.v_convert):
            if const_expr(self.alias_convert_smem):
                @cute.struct
                class SharedStorageQKVK8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sP: sP_struct
            else:
                @cute.struct
                class SharedStorageQKVK8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sK8: sK8_struct
                    sP: sP_struct

            assert not self.Q_in_regs, "M10b K-convert does not support Q_in_regs"
            return SharedStorageQKVK8

        if const_expr(self.k_convert and self.v_convert):
            if const_expr(self.alias_convert_smem):
                @cute.struct
                class SharedStorageQKVK8V8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sP: sP_struct
            else:
                @cute.struct
                class SharedStorageQKVK8V8:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_struct
                    mbar_ptr_V: mbar_ptr_V_struct
                    sV: sV_struct
                    sV8: sV8_struct
                    sQ: sQ_struct
                    sK: sK_struct
                    sK8: sK8_struct
                    sP: sP_struct

            assert not self.Q_in_regs, "M10b K-convert does not support Q_in_regs"
            return SharedStorageQKVK8V8

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mDynamicCausal: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        output_scale: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            ),
            is_split_kv=self.is_split_kv,
        )

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        num_splits = Int32(1)
        if const_expr(not self.is_split_kv):
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        else:
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        mO = layout_utils.select(mO, O_layout_transpose)
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        # fa4hybrid: V keeps its natural (s_k, dv, ...) layout in gmem for ALL arms.
        # (The old fa4build all-fp8 gmem V-transpose is gone -- TMA cannot transpose, and
        # the hybrid PV wgmma in fp16/bf16 takes MN-major V directly.)
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        # fa4hybrid: per-buffer dtypes. sV is the PV compute buffer (dtype_pv); Arm C adds
        # a raw-fp8 staging buffer sV8 (dtype_v) that TMA/cp.async lands into.
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(dt, LayoutEnum.ROW_MAJOR, shape, stage)
            for dt, shape, stage in [
                (mQ.element_type, (self.tile_m, self.tile_hdim), None),
                # M10b: sK is the QK-GEMM compute buffer, typed self.dtype (bf16 when
                # k_convert), NOT mK.element_type (the fp8 storage dtype) -- mirrors sV.
                (self.dtype, (self.tile_n, self.tile_hdim), self.num_stages),
                (self.dtype_pv, (self.tile_n, self.tile_hdimv), self.num_stages),
                # sO layout dtype possibly different from mO dtype when using splitkv (fp32)
                (self.dtype_o if const_expr(not self.is_split_kv) else mQ.element_type,
                 (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sV8_layout = None
        if const_expr(self.v_convert):
            self.sV8_layout = sm90_utils.make_smem_layout(
                mV.element_type,
                LayoutEnum.ROW_MAJOR,
                (self.tile_n, self.tile_hdimv),
                self.num_stages,
            )
        # M10b: raw fp8 K staging smem layout (paged cp.async load target). Mirrors sV8.
        self.sK8_layout = None
        if const_expr(self.k_convert):
            self.sK8_layout = sm90_utils.make_smem_layout(
                mK.element_type,
                LayoutEnum.ROW_MAJOR,
                (self.tile_n, self.tile_hdim),
                self.num_stages,
            )
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                self.dtype_pv, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        # fa4hybrid: V TMA lands into the staging buffer (fp8) for Arm C, else into sV.
        sV_tma_layout = self.sV8_layout if const_expr(self.v_convert) else self.sV_layout
        sK_tma_layout = self.sK8_layout if const_expr(self.k_convert) else self.sK_layout
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, sK_tma_layout),
                ("V", mV, sV_tma_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(sK_tma_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(sV_tma_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mDynamicCausal,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sK8_layout,
            self.sV_layout,
            self.sV8_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            output_scale,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mDynamicCausal: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sK8_layout: cute.ComposedLayout | None,
        sV_layout: cute.ComposedLayout,
        sV8_layout: cute.ComposedLayout | None,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        num_splits: Int32 = Int32(1),
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
        output_scale: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # M10b: raw fp8 K staging buffer (paged cp.async target); consumer upconverts
        # sK8 -> sK (bf16) in the MAIN LOOP before the QK wgmma. Mirrors sV8/v_convert.
        # M10 tile64: sK8 ALIASES sK's smem region (reinterpreted as fp8) rather than a
        # separate buffer -- reclaims ~32KB so tile_n=64 fits the SM90 228KB budget.
        # Safe: pipeline_k gates the producer's next fp8 load on consumer_release, which
        # only fires after the QK gemm reading sK has drained (warpgroup.wait_group), so
        # the fp8 staging write never overlaps a live bf16 sK read; k_convert_fn adds a
        # read-complete barrier so the cooperative in-place fp8->bf16 convert is race-free.
        sK8 = None
        if const_expr(self.k_convert):
            sK8_storage = storage.sK if const_expr(self.alias_convert_smem) else storage.sK8
            sK8 = sK8_storage.get_tensor(
                sK8_layout.outer, swizzle=sK8_layout.inner, dtype=self.dtype_k
            )
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=self.dtype_pv
            )
        # fa4hybrid Arm C: raw fp8 V staging buffer (TMA target); consumer upconverts
        # sV8 -> sV (fp16) before the PV wgmma.
        # M10 tile64: sV8 ALIASES sV's smem region (reinterpreted as fp8), mirroring
        # sK8/sK. Safe by the same argument -- pipeline_v gates the producer's next fp8
        # V load on consumer_release, which fires only after the PV gemm reading sV has
        # drained; v_convert_fn adds a read-complete barrier for the in-place convert.
        sV8 = None
        if const_expr(self.v_convert):
            sV8_storage = storage.sV if const_expr(self.alias_convert_smem) else storage.sV8
            sV8 = sV8_storage.get_tensor(
                sV8_layout.outer, swizzle=sV8_layout.inner, dtype=self.dtype_v
            )
        # V is presented to the PV mma as (head_dim_v, tile_n) via a transpose_view of the
        # natural MN-major (tile_n, head_dim_v) buffer -- valid for fp16/bf16 wgmma.
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator; O is stored in dtype_o (bf16 for hybrid arms)
        sO = storage.sQ.get_tensor(
            sO_layout.outer,
            swizzle=sO_layout.inner,
            dtype=self.dtype_o if const_expr(not self.is_split_kv) else self.dtype,
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        self._mDynamicCausal = mDynamicCausal
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                # M10b: K loads (TMA or paged cp.async) land in the fp8 staging buffer
                sK8 if const_expr(self.k_convert) else sK,
                # Arm C: V loads (TMA or paged cp.async) land in the fp8 staging buffer
                sV8 if const_expr(self.v_convert) else sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sK8,
                sVt,
                sV,
                sV8,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
                num_splits,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block, split_idx, num_splits
                    )
                    if const_expr(self._mDynamicCausal is not None):
                        psc_producer = self._mDynamicCausal[batch_idx]
                        if not psc_producer:
                            # Mirror the consumer's bidirectional split range so the
                            # producer loads exactly the K/V blocks the consumer
                            # processes. Any divergence here deadlocks the pipeline.
                            n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                            if const_expr(self.is_split_kv):
                                num_n_blocks_per_split = cute.ceil_div(n_block_max_full, num_splits)
                                n_block_min = split_idx * num_n_blocks_per_split
                                n_block_max = cutlass.min(
                                    n_block_min + num_n_blocks_per_split, n_block_max_full
                                )
                            else:
                                n_block_min = Int32(0)
                                n_block_max = n_block_max_full
                    # Clamp n_block to 0 when n_block_max == 0 (can happen with causal
                    # + pack_gqa when seqlen_k < tile_n). TMA handles n_block=-1
                    # gracefully (fills zeros), but cp.async would crash on
                    # out-of-bounds page table access.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    if is_kv_load_warp:
                        if const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(n_block)
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        else:
                            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                page_idx_prev = (
                                    mPageTable[batch_idx, n_block_prev]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                    page_idx=page_idx_prev,
                                )
                            n_block = n_block_min
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block, producer_state=kv_producer_state, page_idx=page_idx
                            )
                            kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(block, sX[None, None, producer_state.index], K_or_V)
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sK8: Optional[cute.Tensor],
        sVt: cute.Tensor,
        sV: cute.Tensor,
        sV8: Optional[cute.Tensor],
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_tensors: Optional[list],
        fastdiv_mods=None,
        num_splits: Int32 = Int32(1),
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype_pv
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        # ///////////////////////////////////////////////////////////////////////////////
        # fa4hybrid Arm C: cooperative fp8 -> fp16 V upconversion (staging sV8 -> sV).
        # All num_mma_threads participate; each converts a disjoint slice of the
        # (tile_n, tile_hdimv) stage, then a named barrier + async fence makes the fp16
        # tile visible to the PV wgmma of both warpgroups.
        # ///////////////////////////////////////////////////////////////////////////////
        v_convert_fn = None
        if const_expr(self.v_convert):
            v_cvt_elems = 128 // self.dtype_pv.width  # 8 fp16 -> 16B stores (swizzle-atomic)
            v_cvt_thr_cols = self.tile_hdimv // v_cvt_elems
            v_cvt_thr_rows = self.num_mma_threads // v_cvt_thr_cols
            assert self.num_mma_threads % v_cvt_thr_cols == 0
            assert self.tile_n % v_cvt_thr_rows == 0
            v_cvt_thr_layout = cute.make_ordered_layout(
                (v_cvt_thr_rows, v_cvt_thr_cols), order=(1, 0)
            )
            v_cvt_val_layout = cute.make_layout((1, v_cvt_elems))
            v_cvt_atom_src = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype_v,
                num_bits_per_copy=v_cvt_elems * self.dtype_v.width,
            )
            v_cvt_atom_dst = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype_pv,
                num_bits_per_copy=v_cvt_elems * self.dtype_pv.width,
            )
            v_cvt_copy_src = cute.make_tiled_copy_tv(
                v_cvt_atom_src, v_cvt_thr_layout, v_cvt_val_layout
            ).get_slice(tidx)
            v_cvt_copy_dst = cute.make_tiled_copy_tv(
                v_cvt_atom_dst, v_cvt_thr_layout, v_cvt_val_layout
            ).get_slice(tidx)
            tVcvt_S = v_cvt_copy_src.partition_S(sV8)  # (CPY, rest_m, rest_n, stage)
            tVcvt_D = v_cvt_copy_dst.partition_D(sV)

            def v_convert_fn(stage):
                src = tVcvt_S[None, None, None, stage]
                dst = tVcvt_D[None, None, None, stage]
                frag8 = cute.make_fragment_like(src)
                cute.autovec_copy(src, frag8)
                frag16 = cute.make_fragment_like(frag8, self.dtype_pv)
                if const_expr(self.dtype_pv == cutlass.BFloat16):
                    # M10b: no direct packed fp8->bf16 cvt lowering; hop through fp16
                    # (exact for e4m3 source: e4m3 has <= 3 mantissa bits, well within
                    # both fp16's 10 and bf16's 7, so no extra rounding is introduced).
                    frag_mid = cute.make_fragment_like(frag8, cutlass.Float16)
                    frag_mid.store(frag8.load().to(cutlass.Float16))
                    frag16.store(frag_mid.load().to(self.dtype_pv))
                else:
                    frag16.store(frag8.load().to(self.dtype_pv))
                # M10 tile64: sV8 aliases sV's region. Rendezvous ALL mma threads after the
                # fp8 read (frag8) and before any bf16 write, so a store can't clobber
                # another thread's not-yet-read fp8 source.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.VConvert),
                    number_of_threads=self.num_mma_threads,
                )
                cute.autovec_copy(frag16, dst)
                # Make the fp16 stores visible to the async proxy (wgmma), then rendezvous
                # both mma warpgroups so no PV gemm starts on a partially converted tile.
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.VConvert),
                    number_of_threads=self.num_mma_threads,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # M10b: cooperative fp8 -> bf16 K upconversion (staging sK8 -> sK). Structurally
        # different from v_convert_fn: K feeds the QK wgmma in the MAIN LOOP, so this
        # runs right after pipeline_k.consumer_wait and BEFORE mma_qk_fn (V converts in
        # the PV epilogue instead, after pipeline_v.consumer_wait and before mma_pv_fn).
        # Reuses the VConvert named barrier id -- safe because K-convert and V-convert
        # rendezvous sequentially (never concurrently) within one n-block iteration,
        # both across the same self.num_mma_threads.
        # ///////////////////////////////////////////////////////////////////////////////
        k_convert_fn = None
        if const_expr(self.k_convert):
            k_cvt_elems = 128 // self.dtype.width
            k_cvt_thr_cols = self.tile_hdim // k_cvt_elems
            k_cvt_thr_rows = self.num_mma_threads // k_cvt_thr_cols
            assert self.num_mma_threads % k_cvt_thr_cols == 0
            assert self.tile_n % k_cvt_thr_rows == 0
            k_cvt_thr_layout = cute.make_ordered_layout(
                (k_cvt_thr_rows, k_cvt_thr_cols), order=(1, 0)
            )
            k_cvt_val_layout = cute.make_layout((1, k_cvt_elems))
            k_cvt_atom_src = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype_k,
                num_bits_per_copy=k_cvt_elems * self.dtype_k.width,
            )
            k_cvt_atom_dst = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=k_cvt_elems * self.dtype.width,
            )
            k_cvt_copy_src = cute.make_tiled_copy_tv(
                k_cvt_atom_src, k_cvt_thr_layout, k_cvt_val_layout
            ).get_slice(tidx)
            k_cvt_copy_dst = cute.make_tiled_copy_tv(
                k_cvt_atom_dst, k_cvt_thr_layout, k_cvt_val_layout
            ).get_slice(tidx)
            tKcvt_S = k_cvt_copy_src.partition_S(sK8)  # (CPY, rest_m, rest_n, stage)
            tKcvt_D = k_cvt_copy_dst.partition_D(sK)

            def k_convert_fn(stage):
                src = tKcvt_S[None, None, None, stage]
                dst = tKcvt_D[None, None, None, stage]
                frag8 = cute.make_fragment_like(src)
                cute.autovec_copy(src, frag8)
                frag16 = cute.make_fragment_like(frag8, self.dtype)
                if const_expr(self.dtype == cutlass.BFloat16):
                    # M10b: no direct packed fp8->bf16 cvt lowering; hop through fp16
                    # (exact for e4m3 source: e4m3 has <= 3 mantissa bits, well within
                    # both fp16's 10 and bf16's 7, so no extra rounding is introduced).
                    frag_mid = cute.make_fragment_like(frag8, cutlass.Float16)
                    frag_mid.store(frag8.load().to(cutlass.Float16))
                    frag16.store(frag_mid.load().to(self.dtype))
                else:
                    frag16.store(frag8.load().to(self.dtype))
                # M10 tile64: sK8 aliases sK's region. Every thread has now read its fp8
                # slice into registers (frag8); rendezvous ALL mma threads BEFORE any
                # thread writes the wider bf16 result back into the same bytes, else a
                # bf16 store would clobber another thread's not-yet-read fp8 source.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.VConvert),
                    number_of_threads=self.num_mma_threads,
                )
                cute.autovec_copy(frag16, dst)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.VConvert),
                    number_of_threads=self.num_mma_threads,
                )

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
            v_convert_fn=v_convert_fn,
            k_convert_fn=k_convert_fn,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
            k_convert_fn=k_convert_fn,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
            v_convert_fn=v_convert_fn,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            psc = (
                self._mDynamicCausal[batch_idx]
                if const_expr(self._mDynamicCausal is not None)
                else None
            )
            mask = AttentionMaskCls(seqlen, dynamic_causal=psc)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all, seqlen=seqlen, softmax=softmax, score_mod_fn=score_mod_fn
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(self._mDynamicCausal is not None):
                # Per-sequence causal: psc == 0 means this sequence is processed
                # bidirectionally. get_n_block_min_max may have applied a causal
                # upper bound (when the kernel is compiled causal) and, for
                # split-KV, partitioned that (possibly causal) range. For a
                # bidirectional sequence each split must instead own a DISJOINT
                # slice of the FULL key range. Recompute [n_block_min, n_block_max)
                # over the full range here, and IDENTICALLY on the producer side
                # (see the K/V load loop), so the pipeline block counts agree -- a
                # producer/consumer mismatch deadlocks the kernel (GPU spins).
                # The previous code only reset n_block_max to the global max while
                # leaving n_block_min at its split offset, so splits overlapped and
                # keys were double-counted -> corrupted softmax (rel_err ~0.33).
                if not psc:
                    n_block_max_full = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
                    if const_expr(self.is_split_kv):
                        num_n_blocks_per_split = cute.ceil_div(n_block_max_full, num_splits)
                        n_block_min = split_idx * num_n_blocks_per_split
                        n_block_max = cutlass.min(
                            n_block_min + num_n_blocks_per_split, n_block_max_full
                        )
                    else:
                        n_block_min = Int32(0)
                        n_block_max = n_block_max_full
            n_block_max_orig = n_block_max
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                        score_mod_fn=score_mod_fn,
                        is_first_block=True,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                    O_should_accumulate = True
                # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    if const_expr(self._mDynamicCausal is not None):
                        if not psc:
                            n_block_min_causal_local_mask = n_block_min
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # The remaining iterations have no masking
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                    )
                    O_should_accumulate = True
                # Separate iterations with local masking on the left
                if const_expr(self.is_local and block_info.window_size_left is not None):
                    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                        )
                        O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    mask_fn,
                    score_mod_fn,
                    O_should_accumulate,
                    self.mask_mod,
                    fastdiv_mods,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
                if const_expr(self.is_split_kv):
                    if split_idx > 0:
                        if const_expr(not self.pack_gqa):
                            sink_val = -Float32.inf
                        else:
                            sink_val.fill(-Float32.inf)

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            # Override empty splits so combine kernel gives zero weight
            if const_expr(self.is_split_kv):
                if n_block_min >= n_block_max_orig:
                    acc_O.fill(Float32(0.0))
                    softmax.row_sum.fill(-Float32.inf)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        k_convert_fn: Optional[Callable] = None,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        if const_expr(k_convert_fn is not None):
            k_convert_fn(kv_consumer_state.index)
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype_pv)
        )
        if const_expr(self.dtype_pv == cutlass.Float8E4M3FN):
            tOrP_cur.store(tOrP_acc.load().to(self.dtype_pv))
        else:
            utils.cvt_f16(tOrP_acc, tOrP_cur)

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
        v_convert_fn: Optional[Callable] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        if const_expr(v_convert_fn is not None):
            v_convert_fn(kv_consumer_state.index)
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        v_convert_fn: Optional[Callable] = None,
        k_convert_fn: Optional[Callable] = None,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        if const_expr(k_convert_fn is not None):
            k_convert_fn(smem_pipe_read.index)
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype_pv)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        if const_expr(self.dtype_pv == cutlass.Float8E4M3FN):
            # all-fp8 arm only: generic convert (cvt_f16 is a bf16/f16-only fast path)
            tOrP_cur.store(tOrP_acc.load().to(self.dtype_pv))
        else:
            # fa4hybrid: P is downcast fp32 -> dtype_pv (fp16/bf16) for the PV wgmma
            utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        if const_expr(v_convert_fn is not None):
            v_convert_fn(smem_pipe_read.index)
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        v_convert_fn: Optional[Callable] = None,
        k_convert_fn: Optional[Callable] = None,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        if const_expr(k_convert_fn is not None):
            k_convert_fn(smem_pipe_read.index)
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        if const_expr(v_convert_fn is not None):
            v_convert_fn(smem_pipe_read_v.index)
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype_pv)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        if const_expr(self.dtype_pv == cutlass.Float8E4M3FN):
            # all-fp8 arm only: generic convert (cvt_f16 is a bf16/f16-only fast path)
            tOrP_cur.store(tOrP_acc.load().to(self.dtype_pv))
        else:
            # fa4hybrid: P is downcast fp32 -> dtype_pv (fp16/bf16) for the PV wgmma
            utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
