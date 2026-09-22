#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <gpu> <run-dir>" >&2
    exit 64
fi

GPU_INDEX="$1"
RUN_DIR="$2"

EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-temporal-compiler-sam31}"
INSTRUCTSEG_REPO="${INSTRUCTSEG_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/InstructSeg}"
INSTRUCTSEG_ENV="${INSTRUCTSEG_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/instructseg}"
MODEL_PATH="${MODEL_PATH:-/9950backfile/chenjiahui/evo_artifacts/models/InstructSeg-official}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/9950backfile/chenjiahui/evo_artifacts/datasets/long_rvos/valid/JPEGImages}"
VISION_TOWER="${VISION_TOWER:-/9950backfile/chenjiahui/evo_artifacts/models/siglip-so400m-patch14-384-processor}"
MASK_CONFIG="${MASK_CONFIG:-${INSTRUCTSEG_REPO}/instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

INPUT_JSON="${RUN_DIR}/refyoutube_input.json"
MAPPING_JSON="${RUN_DIR}/expression_mapping.json"
OUTPUT_ROOT="${RUN_DIR}/output"
ANNOTATION_ROOT="${OUTPUT_ROOT}/Annotations"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
LOG_DIR="${RUN_DIR}/logs"

for required in \
    "${INPUT_JSON}" \
    "${MAPPING_JSON}" \
    "${MODEL_PATH}" \
    "${IMAGE_FOLDER}" \
    "${VISION_TOWER}" \
    "${MASK_CONFIG}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required InstructSeg input is missing: ${required}" >&2
        exit 66
    fi
done

mkdir -p "${LOG_DIR}"
cd "${INSTRUCTSEG_REPO}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${EVOSEG_ROOT}:${INSTRUCTSEG_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

"${INSTRUCTSEG_ENV}/bin/python" \
    instructseg/eval/seg/eval_rvos.py \
    --model_path "${MODEL_PATH}" \
    --json_path "${INPUT_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --save_path "${OUTPUT_ROOT}" \
    --vision_tower "${VISION_TOWER}" \
    --mask_config "${MASK_CONFIG}" \
    --dataset_name RefYoutube \
    --use_soft False \
    --use_temporal_query True \
    --use_vmtf True \
    --reference_frame_num 4 \
    --dataloader_num_workers 2 \
    > "${LOG_DIR}/inference.log" 2>&1

"${INSTRUCTSEG_ENV}/bin/python" \
    -m projects.evoseg.temporal_compiler.instructseg_long_rvos_adapter evaluate \
    --mapping-json "${MAPPING_JSON}" \
    --annotation-root "${ANNOTATION_ROOT}" \
    --output-predictions "${PREDICTIONS}" \
    > "${LOG_DIR}/evaluation.log" 2>&1
