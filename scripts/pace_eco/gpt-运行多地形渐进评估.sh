#!/usr/bin/env bash
# 用法：GPU 阶段 地形 方法 PPO_seed checkpoint B_ref_JSON 渐进holdout授权

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"

if [[ "$#" -ne 8 ]]; then
    pace_multi_fail "用法：$0 GPU编号 stage1|stage2 地形 方法 PPO_seed checkpoint B_ref_JSON 渐进holdout授权"
fi
gpu="$1"
stage="$2"
terrain="$3"
method="$4"
ppo_seed="$5"
checkpoint="$6"
reference_json="$7"
authorization="$8"
pace_multi_validate_gpu "${gpu}"
pace_multi_require_pace_data
pace_multi_validate_ppo_seed "${stage}_formal" "${ppo_seed}"
[[ -f "${checkpoint}" ]] || pace_multi_fail "检查点不存在：${checkpoint}"
[[ -f "${reference_json}" ]] || pace_multi_fail "B_ref JSON 不存在：${reference_json}"
[[ -f "${authorization}" ]] || pace_multi_fail "渐进 holdout 授权不存在：${authorization}"
[[ "${stage}" == "stage1" || "${stage}" == "stage2" ]] || pace_multi_fail "阶段只能是 stage1 或 stage2。"
[[ "${stage}" != "stage1" || "${terrain}" != "mixed" ]] || pace_multi_fail "阶段一禁止 mixed。"
[[ "${stage}" != "stage2" || "${terrain}" == "mixed" ]] || pace_multi_fail "阶段二只允许 mixed。"

role="${stage}_holdout"
base="$(pace_multi_seed_base "${role}")"
offset="$(pace_multi_terrain_offset "${terrain}")"
terrain_seed="$((base + offset))"
task="$(pace_multi_task_id "${terrain}" "${method}")"
launch_root="${PACE_MULTI_LOG_ROOT}/launch/${stage}"
mkdir -p "${launch_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
batch_group="${timestamp}"
log_path="${launch_root}/gpt_GPU${gpu}_${stage}_progressive_holdout_${terrain}_${method}_seed${ppo_seed}_${timestamp}.log"
cd "${PACE_MULTI_ROOT}"
for batch_index in 0 1 2 3; do
    echo "[PACE] 启动渐进式 Terrain20sWide holdout 批次 $((batch_index + 1))/4；批次组=${batch_group}。" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_MULTI_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_MULTI_ROOT}" TERM=xterm timeout --signal=TERM --kill-after=60s 30m "${PACE_MULTI_CONDA}" run -n "${PACE_MULTI_ENV}" --no-capture-output "${PACE_MULTI_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_MULTI_ROOT}/scripts/pace_eco/gpt_progressive_multi_terrain_eval.py" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --batch_index "${batch_index}" --batch_group "${batch_group}" --stage "${stage}" --split holdout --output_root "${PACE_MULTI_RESULT_ROOT}" --headless --device cuda:0 --energy_reference_json "${reference_json}" --holdout_authorization "${authorization}" 2>&1 | tee -a "${log_path}"
done
