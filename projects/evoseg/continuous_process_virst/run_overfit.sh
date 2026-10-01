#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <gpu> <long-root> <groundmore-root> <output-dir> <seed>" >&2
    exit 64
fi

GPU_INDEX="$1"
LONG_ROOT="$2"
GROUNDMORE_ROOT="$3"
OUTPUT_DIR="$4"
SEED="$5"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-continuous-process-virst}"
VIRST_REPO="${VIRST_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/VIRST}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"
GROUNDMORE_SOURCE="${GROUNDMORE_SOURCE:-/9950backfile/chenjiahui/evo_artifacts/datasets/GroundMoRe-official}"

mkdir -p "${OUTPUT_DIR}"
cd "${VIRST_REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVOSEG_ROOT}:${VIRST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

"${VIRST_ENV}/bin/python" "${EVOSEG_ROOT}/projects/evoseg/continuous_process_virst/train_overfit.py" \
    --long-root "${LONG_ROOT}" \
    --groundmore-root "${GROUNDMORE_ROOT}" \
    --groundmore-source "${GROUNDMORE_SOURCE}" \
    --groundmore-metadata "${GROUNDMORE_SOURCE}/trainval_v2.json" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --warmup-steps "${CPG_WARMUP_STEPS:-64}" \
    --joint-steps "${CPG_JOINT_STEPS:-512}" \
    --frames 8 \
    --learning-rate 1e-5 \
    --checkpoint "${VIRST_REPO}/checkpoints/virst_checkpoint.pt" \
    --sam2-checkpoint "${VIRST_REPO}/checkpoints/sam2.1_hiera_large.pt" \
    --videochat-checkpoint "${VIRST_REPO}/checkpoints/videochat" \
    --tokenizer "${VIRST_REPO}/model/videochat" 2>&1 | tee "${OUTPUT_DIR}/train.log"
