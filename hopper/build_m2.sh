#!/usr/bin/env bash
# M2: build bf16 square-head-512 with PAGED KV enabled (cp.async, Option C) and run the
# M2 gate (paged correctness through a randomized block table + perf). block_size=16 =>
# page%kBlockN(64)!=0 => FA3 auto-selects the cp.async paged path (pagedkv_tma=false).
# Fails CLOSED. Usage: bash build_m2.sh [seqlen] [batch]   (from hopper/; free GPU 1-7)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

OV=/opt/dlami/nvme/gemma-latency-lab/fork-endpoint-runner/run_20260701T170841Z_grouped_overlay
PY="$OV/venv/bin/python"
SP="$OV/venv/lib/python3.10/site-packages"
CU13=/opt/dlami/nvme/text-classification-gpu5/gemma-dflash/venv/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')$CU13:${LD_LIBRARY_PATH:-}"
export HOME="${HOME:-/opt/dlami/nvme/gemma-hopper-build/home}"; mkdir -p "$HOME"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_USE_SYSTEM_CTK=1

# Enable square-512 + PAGED (cp.async). packgqa stays on (paged forces it). Everything
# else off. NOTE: DISABLE_PAGEDKV is NOT set here (that's the M2 delta vs build_m1.sh).
export FLASH_ATTENTION_DISABLE_HDIM512=FALSE
export FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export MAX_JOBS="${MAX_JOBS:-16}" NVCC_THREADS="${NVCC_THREADS:-4}"

echo "== deleting stale .so (fail-closed) =="
rm -f flash_attn_3_cuda*.so
"$PY" generate_kernels.py -o instantiations >/dev/null
echo "== building (paged enabled; errors abort) =="
"$PY" setup.py build_ext --inplace 2>&1 | tee /tmp/m2_build.log | \
    grep -iE "error:|undefined|fatal|Traceback|RuntimeError|\.so ->" | grep -viE "deprecated|warning" || true
test -f flash_attn_3_cuda*.so || { echo "BUILD FAILED: no .so produced"; exit 1; }
"$PY" -c "import torch; import flash_attn_3_cuda as m; print('import OK:', m.__file__)"

echo "== M2 gate (paged correctness + perf) =="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
"$PY" m2_paged_microbench.py --seqlen "${1:-9216}" --batch "${2:-32}"
