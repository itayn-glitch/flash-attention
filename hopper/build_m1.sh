#!/usr/bin/env bash
# Build the gated bf16 square-head-512 FA3 kernel on the H100 box and smoke-test it.
# Only compiles flash_api.cpp + the hdim512 bf16 (packgqa) instantiation for fast M1 iteration.
# Usage:  bash build_m1.sh [seqlen]   (run from the hopper/ dir; GPU 1-7 must be free)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OV=/opt/dlami/nvme/gemma-latency-lab/fork-endpoint-runner/run_20260701T170841Z_grouped_overlay
PY="$OV/venv/bin/python"
SP="$OV/venv/lib/python3.10/site-packages"
CU13=/opt/dlami/nvme/text-classification-gpu5/gemma-dflash/venv/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')$CU13:${LD_LIBRARY_PATH:-}"
export HOME="${HOME:-/opt/dlami/nvme/gemma-hopper-build/home}"; mkdir -p "$HOME"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_USE_SYSTEM_CTK=1

# Gate the build to exactly bf16 square-512 (fwd, packgqa, non-split/paged/softcap/varlen/local).
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS="${NVCC_THREADS:-4}"

"$PY" generate_kernels.py -o instantiations >/dev/null 2>&1
echo "== building =="
"$PY" setup.py build_ext --inplace 2>&1 | grep -iE "error:|undefined|fatal|Traceback|RuntimeError|\.so ->" | grep -viE "deprecated|warning" | tail -20
ls -la flash_attn_3_cuda*.so 2>/dev/null | tail -1

echo "== smoke =="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
"$PY" m1_dense_microbench.py --seqlen "${1:-2048}" --iters 50 --warmup 10
