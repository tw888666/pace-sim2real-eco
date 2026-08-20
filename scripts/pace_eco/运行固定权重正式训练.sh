#!/usr/bin/env bash
# 从随机初始化运行固定权重 PPO 的 4096 环境、3000 次更新统一比较训练。

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_训练命令公共.sh"

if [[ "$#" -ne 2 ]]; then
    pace_usage_error "bash ${BASH_SOURCE[0]} <GPU编号：0至4> <随机种子>"
fi

gpu_id="$1"
seed="$2"
pace_validate_gpu_id "${gpu_id}"
pace_validate_seed "${seed}"

pace_run_training \
    "${gpu_id}" \
    "Isaac-PACE-FixedWeight-Flat-Anymal-D-v0" \
    4096 \
    3000 \
    "${seed}" \
    "formal_cu126_fixed_weight_seed${seed}" \
    "gpt_GPU${gpu_id}_cu126固定权重PPO_3000次更新_seed${seed}"
