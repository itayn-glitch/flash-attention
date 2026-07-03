#!/usr/bin/env bash
# Build the gated bf16 square-head-512 FA3 kernel on the H100 box and run the M1 gate
# (correctness + perf). Only compiles flash_api.cpp + the hdim512 bf16 (packgqa)
# instantiation. Fails CLOSED: any build error aborts before benchmarking, and a stale
# .so can never be benchmarked (it is deleted first).
#   Usage:  bash build_m1.sh [seqlen] [batch]   (run from hopper/; GPU 1-7 must be free)
set -euo pipefail
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

# Explicitly enable square-512 (default-disabled in setup.py so broad builds don't
# silently instantiate unvalidated M2+ 512 variants), gate everything else off.
export FLASH_ATTENTION_DISABLE_HDIM512=FALSE
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS="${NVCC_THREADS:-4}"

SO=(flash_attn_3_cuda*.so)
echo "== deleting any stale .so (fail-closed) =="
rm -f flash_attn_3_cuda*.so

"$PY" generate_kernels.py -o instantiations >/dev/null

echo "== building (errors abort) =="
# pipefail makes the pipeline return setup.py's status; tee preserves the full log.
"$PY" setup.py build_ext --inplace 2>&1 | tee /tmp/m1_build.log | \
    grep -iE "error:|undefined|fatal|Traceback|RuntimeError|\.so ->" | grep -viE "deprecated|warning" || true

test -f flash_attn_3_cuda*.so || { echo "BUILD FAILED: no .so produced"; exit 1; }
echo "== built module =="
"$PY" -c "import flash_attn_3_cuda as m; print('import OK:', m.__file__)"

echo "== M1 gate (correctness + perf) =="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
"$PY" m1_dense_microbench.py --seqlen "${1:-9216}" --batch "${2:-32}"
