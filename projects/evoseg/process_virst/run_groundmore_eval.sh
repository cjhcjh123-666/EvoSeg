#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <gpu> <dataset-root> <run-dir> <run-name> <port> <baseline|process>" >&2
    exit 64
fi

GPU_INDEX="$1"
DATASET_ROOT="$2"
RUN_DIR="$3"
RUN_NAME="$4"
MASTER_PORT="$5"
MODE="$6"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
VIRST_REPO="${VIRST_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/VIRST}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"

if [[ "${MODE}" == "process" ]]; then
    RUNNER="${EVOSEG_ROOT}/projects/evoseg/process_virst/virst_process_eval.py"
    export PROCESS_VIRST_GROUNDMORE_EXACT20=1
    export PROCESS_VIRST_DIAGNOSTICS="${RUN_DIR}/process_diagnostics.jsonl"
    export PROCESS_VIRST_PERMUTATION=reverse
elif [[ "${MODE}" == "baseline" ]]; then
    RUNNER="${EVOSEG_ROOT}/projects/evoseg/process_virst/groundmore_virst_eval.py"
else
    echo "Mode must be baseline or process" >&2
    exit 64
fi

mkdir -p "${RUN_DIR}"
cd "${VIRST_REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVOSEG_ROOT}:${VIRST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export VIRST_OFFICIAL_EVAL_PATH="${VIRST_REPO}/eval.py"

"${VIRST_ENV}/bin/deepspeed" --master_port "${MASTER_PORT}" "${RUNNER}" \
    --tokenizer "${VIRST_REPO}/model/videochat" \
    --videochat_checkpoint "${VIRST_REPO}/checkpoints/videochat" \
    --sam2_checkpoint "${VIRST_REPO}/checkpoints/sam2.1_hiera_large.pt" \
    --dataset mevis_test --eval_output_root "${RUN_DIR}/output" \
    --eval_log_root "${RUN_DIR}/output" --version qwen_2 \
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
    --seg_image_length 20 --num_seg_keyframes 3 --seg_image_size 1024 \
    --logging_steps 1 --wandb False --wandb_train_name "${RUN_NAME}" \
    --keyframe_scheme uniform \
    --model_checkpoint "${VIRST_REPO}/checkpoints/virst_checkpoint.pt" \
    --rvos_root "${DATASET_ROOT}" 2>&1 | tee "${RUN_DIR}/eval.log"
