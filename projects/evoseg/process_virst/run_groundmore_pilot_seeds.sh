#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/9950backfile/chenjiahui/evo_artifacts/results/process_virst/20260930_process_virst_sft}"
DATASET_ROOT="${2:-${RUN_ROOT}/groundmore_pilot16_dataset}"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
SOURCE_ROOT="${GROUNDMORE_SOURCE_ROOT:-/9950backfile/chenjiahui/evo_artifacts/datasets/GroundMoRe-official}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"
SEEDS=(11 23 42)
GPUS=(1 2 4)
PORTS=(29711 29723 29742)

for seed in "${SEEDS[@]}"; do
    session="processvirst-pilot-seed${seed}"
    while tmux has-session -t "${session}" 2>/dev/null; do
        sleep 10
    done
    checkpoint="${RUN_ROOT}/pilot_fp32_seed${seed}/process_virst_sft.pt"
    summary="${RUN_ROOT}/pilot_fp32_seed${seed}/summary.json"
    if [[ ! -f "${checkpoint}" || ! -f "${summary}" ]]; then
        echo "Missing successful training outputs for seed ${seed}" >&2
        exit 66
    fi
done

pids=()
for index in 0 1 2; do
    seed="${SEEDS[$index]}"
    run_name="process_groundmore_pilot16_seed${seed}"
    run_dir="${RUN_ROOT}/groundmore_pilot16_seed${seed}"
    PROCESS_VIRST_CHECKPOINT="${RUN_ROOT}/pilot_fp32_seed${seed}/process_virst_sft.pt" \
        "${EVOSEG_ROOT}/projects/evoseg/process_virst/run_groundmore_eval.sh" \
        "${GPUS[$index]}" "${DATASET_ROOT}" "${run_dir}" "${run_name}" \
        "${PORTS[$index]}" process &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "${pid}"
done

for seed in "${SEEDS[@]}"; do
    run_name="process_groundmore_pilot16_seed${seed}"
    run_dir="${RUN_ROOT}/groundmore_pilot16_seed${seed}"
    "${VIRST_ENV}/bin/python" \
        -m projects.evoseg.process_virst.groundmore_adapter evaluate \
        --mapping "${DATASET_ROOT}/groundmore_mapping.json" \
        --output-root "${run_dir}/output/mevis_test/eval_mevis_test_${run_name}" \
        --source-root "${SOURCE_ROOT}" \
        --output-csv "${run_dir}/per_expression.csv"
    "${VIRST_ENV}/bin/python" \
        -m projects.evoseg.process_virst.summarize_order_diagnostics \
        --diagnostics "${run_dir}/process_diagnostics.jsonl" \
        --mapping "${DATASET_ROOT}/groundmore_mapping.json" \
        --output-csv "${run_dir}/order_diagnostics.csv" \
        --output-json "${run_dir}/order_summary.json" \
        --bootstrap-iterations 2000 --seed 42
done
