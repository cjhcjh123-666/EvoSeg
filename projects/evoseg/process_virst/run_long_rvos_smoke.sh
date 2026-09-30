#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <gpu> <run-dir> <master-port>" >&2
    exit 64
fi

GPU_INDEX="$1"
RUN_DIR="$2"
MASTER_PORT_VALUE="$3"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
VIRST_REPO="${VIRST_REPO:-/9950backfile/chenjiahui/evo_artifacts/external/VIRST}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/9950backfile/chenjiahui/evo_artifacts/results/temporal_seg/latest/manifest.json}"
DATASET_ROOT="${RUN_DIR}/dataset_root"
SMOKE_MANIFEST="${RUN_DIR}/smoke_manifest.json"
RUN_NAME="process_virst_smoke"

mkdir -p "${RUN_DIR}/logs"
"${VIRST_ENV}/bin/python" -m projects.evoseg.process_virst.prepare_smoke_manifest \
    --manifest "${SOURCE_MANIFEST}" --output "${SMOKE_MANIFEST}" --videos 2
"${VIRST_ENV}/bin/python" -m projects.evoseg.temporal_compiler.virst_long_rvos_adapter prepare \
    --manifest "${SMOKE_MANIFEST}" --dataset-root "${DATASET_ROOT}" \
    --max-expressions-per-object 1

cd "${VIRST_REPO}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export DS_ZERO_STAGE=3
export DS_OFFLOAD_OPTIMIZER_DEVICE=none
export DS_OFFLOAD_PARAM_DEVICE=none
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${EVOSEG_ROOT}:${VIRST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export VIRST_OFFICIAL_EVAL_PATH="${VIRST_REPO}/eval.py"
export PROCESS_VIRST_DIAGNOSTICS="${RUN_DIR}/process_diagnostics.jsonl"
export PROCESS_VIRST_PERMUTATION=reverse

"${VIRST_ENV}/bin/deepspeed" --master_port "${MASTER_PORT_VALUE}" \
    "${EVOSEG_ROOT}/projects/evoseg/process_virst/virst_process_eval.py" \
    --tokenizer "${VIRST_REPO}/model/videochat" \
    --videochat_checkpoint "${VIRST_REPO}/checkpoints/videochat" \
    --sam2_checkpoint "${VIRST_REPO}/checkpoints/sam2.1_hiera_large.pt" \
    --dataset mevis_test \
    --eval_output_root "${RUN_DIR}/output" \
    --eval_log_root "${RUN_DIR}/output" \
    --version qwen_2 \
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
    --rvos_root "${DATASET_ROOT}" 2>&1 | tee "${RUN_DIR}/logs/smoke.log"
