#!/usr/bin/env bash
# 用正式4096环境做2次更新容量冒烟；生成的权重禁止用于正式评估。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"

if [[ "$#" -ne 3 ]]; then
    pace_multi_fail "用法：$0 GPU编号 stage1|stage2 地形"
fi
gpu="$1"
stage="$2"
terrain="$3"
pace_multi_validate_gpu "${gpu}"
pace_multi_require_pace_data
if [[ "${stage}" == "stage1" ]]; then
    [[ "${terrain}" != "mixed" ]] || pace_multi_fail "阶段一容量冒烟禁止 mixed。"
    ppo_seed=900
elif [[ "${stage}" == "stage2" ]]; then
    [[ "${terrain}" == "mixed" ]] || pace_multi_fail "阶段二容量冒烟只允许 mixed。"
    ppo_seed=901
else
    pace_multi_fail "阶段只能是 stage1 或 stage2。"
fi
protocol_role="${stage}_capacity_smoke_train"
base="$(pace_multi_seed_base "${protocol_role}")"
offset="$(pace_multi_terrain_offset "${terrain}")"
terrain_seed="$((base + offset + ppo_seed))"
task="$(pace_multi_task_id "${terrain}" task_only)"
launch_root="${PACE_MULTI_LOG_ROOT}/capacity_smoke/launch"
rsl_root="${PACE_MULTI_LOG_ROOT}/capacity_smoke/rsl_rl"
mkdir -p "${launch_root}" "${rsl_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_path="${launch_root}/gpt_GPU${gpu}_${stage}_${terrain}_capacity_smoke_${timestamp}.log"
run_name="gpt_multi_terrain_v1_4_${stage}_smoke_capacity_${terrain}_ppo_seed${ppo_seed}_terrain_seed${terrain_seed}"
model_count_before="$(find "${rsl_root}" -type f -name 'model_1.pt' | wc -l)"
cd "${PACE_MULTI_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_MULTI_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_MULTI_ROOT}" TERM=xterm "${PACE_MULTI_CONDA}" run -n "${PACE_MULTI_ENV}" --no-capture-output "${PACE_MULTI_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_MULTI_TRAIN}" --task "${task}" --num_envs 4096 --max_iterations 2 --seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --protocol_role "${protocol_role}" --log_root "${rsl_root}" --headless --device cuda:0 --run_name "${run_name}" 2>&1 | tee "${log_path}"
model_count_after="$(find "${rsl_root}" -type f -name 'model_1.pt' | wc -l)"
if [[ "${model_count_after}" -ne "$((model_count_before + 1))" ]]; then
    pace_multi_fail "容量冒烟未新增且仅新增一个 model_1.pt；拒绝把无产物运行记为通过。日志：${log_path}"
fi
