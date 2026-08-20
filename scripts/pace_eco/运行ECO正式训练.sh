#!/usr/bin/env bash
# 从随机初始化运行 PACE-ECO 的 4096 环境、3000 次更新统一比较训练。

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_训练命令公共.sh"

if [[ "$#" -ne 3 ]]; then
    pace_usage_error "bash ${BASH_SOURCE[0]} <GPU编号：0至4> <随机种子> <每回合能耗预算_J>"
fi

gpu_id="$1"
seed="$2"
energy_budget_j="$3"
pace_validate_gpu_id "${gpu_id}"
pace_validate_seed "${seed}"
pace_validate_positive_number "${energy_budget_j}" "每回合能耗预算"

budget_tag="${energy_budget_j//./p}"
pace_run_training \
    "${gpu_id}" \
    "Isaac-PACE-ECO-Flat-Anymal-D-v0" \
    4096 \
    3000 \
    "${seed}" \
    "formal_cu126_eco_budget${budget_tag}J_seed${seed}" \
    "gpt_GPU${gpu_id}_cu126_ECO_预算${budget_tag}J_3000次更新_seed${seed}" \
    --energy_budget_j "${energy_budget_j}"
