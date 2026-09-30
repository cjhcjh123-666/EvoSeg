#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <gpu> <dataset-root> <run-dir> <run-name> <master-port> <process-checkpoint>" >&2
    exit 64
fi

GPU_INDEX="$1"
DATASET_ROOT="$2"
RUN_DIR="$3"
RUN_NAME="$4"
MASTER_PORT="$5"
PROCESS_CHECKPOINT="$6"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
VIRST_REPO="${VIRST_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/VIRST}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"
MAPPING_JSON="${DATASET_ROOT}/expression_mapping.json"
FRAME_AUDIT="${RUN_DIR}/frame_audit.jsonl"
OUTPUT_BASE="${RUN_DIR}/output"
COMPLETED_ROOT="${OUTPUT_BASE}/mevis_test/eval_mevis_test_${RUN_NAME}"

for required in \
    "${MAPPING_JSON}" \
    "${DATASET_ROOT}/mevis/valid/meta_expressions.json" \
    "${PROCESS_CHECKPOINT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required ProcessVIRST input is missing: ${required}" >&2
        exit 66
    fi
done

mkdir -p "${RUN_DIR}"
cd "${VIRST_REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVOSEG_ROOT}:${VIRST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PROCESS_VIRST_CHECKPOINT="${PROCESS_CHECKPOINT}"
export PROCESS_VIRST_DIAGNOSTICS="${RUN_DIR}/process_diagnostics.jsonl"
export PROCESS_VIRST_PERMUTATIONS=reverse,block_swap
export PROCESS_VIRST_OFFICIAL_EVAL_PATH="${EVOSEG_ROOT}/projects/evoseg/temporal_compiler/virst_instrumented_eval.py"
export VIRST_OFFICIAL_EVAL_PATH="${VIRST_REPO}/eval.py"
export VIRST_FRAME_AUDIT_PATH="${FRAME_AUDIT}"
export VIRST_COMPLETED_OUTPUT_ROOT="${COMPLETED_ROOT}"

"${VIRST_ENV}/bin/deepspeed" --master_port "${MASTER_PORT}" \
    "${EVOSEG_ROOT}/projects/evoseg/process_virst/virst_process_eval.py" \
    --tokenizer "${VIRST_REPO}/model/videochat" \
    --videochat_checkpoint "${VIRST_REPO}/checkpoints/videochat" \
    --sam2_checkpoint "${VIRST_REPO}/checkpoints/sam2.1_hiera_large.pt" \
    --dataset mevis_test --eval_output_root "${OUTPUT_BASE}" \
    --eval_log_root "${OUTPUT_BASE}" --version qwen_2 \
    --output_dir "${RUN_DIR}/trainer_output" \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --group_by_modality_length True --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing --bf16 True \
    --local_num_frames 4 --mm_local_num_frames 4 \
    --vision_encode_type video_image --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --num_train_epochs 1 --learning_rate 1e-5 --weight_decay 0. \
    --batch_size_per_device 1 --bce_loss_weight 1 --dice_loss_weight 1 \
    --steps_per_epoch 2 --num_classes_per_sample 1 \
    --seg_image_length 32 --num_seg_keyframes 3 --seg_image_size 1024 \
    --logging_steps 1 --wandb False --wandb_train_name "${RUN_NAME}" \
    --keyframe_scheme uniform \
    --model_checkpoint "${VIRST_REPO}/checkpoints/virst_checkpoint.pt" \
    --rvos_root "${DATASET_ROOT}" 2>&1 | tee "${RUN_DIR}/eval.log"

"${VIRST_ENV}/bin/python" \
    -m projects.evoseg.temporal_compiler.virst_long_rvos_adapter evaluate \
    --mapping-json "${MAPPING_JSON}" \
    --output-root "${COMPLETED_ROOT}" \
    --frame-audit-jsonl "${FRAME_AUDIT}" \
    --output-predictions "${RUN_DIR}/predictions.jsonl"
