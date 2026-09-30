#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <gpu> <dataset-root> <output-dir>" >&2
    exit 64
fi

GPU_INDEX="$1"
DATASET_ROOT="$2"
OUTPUT_DIR="$3"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
VIRST_REPO="${VIRST_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/VIRST}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"

mkdir -p "${OUTPUT_DIR}"
cd "${VIRST_REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVOSEG_ROOT}:${VIRST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

"${VIRST_ENV}/bin/python" -m projects.evoseg.process_virst.train_sft \
    --dataset-root "${DATASET_ROOT}" --output-dir "${OUTPUT_DIR}" \
    --seed 11 --warmup-steps 2 --ordered-steps 4 --frames 8 \
    --learning-rate 1e-5 \
    --checkpoint "${VIRST_REPO}/checkpoints/virst_checkpoint.pt" \
    --sam2-checkpoint "${VIRST_REPO}/checkpoints/sam2.1_hiera_large.pt" \
    --videochat-checkpoint "${VIRST_REPO}/checkpoints/videochat" \
    --tokenizer "${VIRST_REPO}/model/videochat" 2>&1 | tee "${OUTPUT_DIR}/train.log"
