#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/9950backfile/chenjiahui/evo_artifacts/results/process_virst/20260930_process_virst_sft}"
DATASET_ROOT="${2:-${RUN_ROOT}/groundmore_pilot16_dataset}"
EVOSEG_ROOT="${EVOSEG_ROOT:-/tmp/EvoSeg-process-aware-virst}"
SOURCE_ROOT="${GROUNDMORE_SOURCE_ROOT:-/9950backfile/chenjiahui/evo_artifacts/datasets/GroundMoRe-official}"
VIRST_ENV="${VIRST_ENV:-/9950backfile/chenjiahui/evo_artifacts/envs/virst}"

names=(control_noorder_seed11 control_global_seed11)
training_dirs=(pilot_control_noorder_seed11 pilot_control_global_seed11)
gpus=(6 7)
ports=(29761 29771)
pids=()

for index in 0 1; do
    name="${names[$index]}"
    checkpoint="${RUN_ROOT}/${training_dirs[$index]}/process_virst_sft.pt"
    summary="${RUN_ROOT}/${training_dirs[$index]}/summary.json"
    if [[ ! -f "${checkpoint}" || ! -f "${summary}" ]]; then
        echo "Missing successful control outputs: ${training_dirs[$index]}" >&2
        exit 66
    fi
    run_dir="${RUN_ROOT}/groundmore_pilot16_${name}"
    run_name="process_groundmore_pilot16_${name}"
    PROCESS_VIRST_CHECKPOINT="${checkpoint}" \
        "${EVOSEG_ROOT}/projects/evoseg/process_virst/run_groundmore_eval.sh" \
        "${gpus[$index]}" "${DATASET_ROOT}" "${run_dir}" "${run_name}" \
        "${ports[$index]}" process &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

for name in "${names[@]}"; do
    run_dir="${RUN_ROOT}/groundmore_pilot16_${name}"
    run_name="process_groundmore_pilot16_${name}"
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
