#!/usr/bin/env bash
# 用法：GPU 阶段 split 地形 方法 PPO_seed checkpoint B_ref_JSON [holdout授权]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"

if [[ "$#" -lt 8 || "$#" -gt 9 ]]; then
    pace_multi_fail "用法：$0 GPU编号 stage1|stage2 calibration|holdout 地形 方法 PPO_seed checkpoint B_ref_JSON [holdout授权]"
fi
gpu="$1"
stage="$2"
split="$3"
terrain="$4"
method="$5"
ppo_seed="$6"
checkpoint="$7"
reference_json="$8"
authorization="${9:-}"
pace_multi_validate_gpu "${gpu}"
pace_multi_require_pace_data
[[ -f "${checkpoint}" ]] || pace_multi_fail "检查点不存在：${checkpoint}"
if [[ "${split}" == "calibration" && "${method}" == "task_only" ]]; then
    pace_multi_validate_ppo_seed "${stage}_budget" "${ppo_seed}"
elif [[ "${split}" == "calibration" ]]; then
    pace_multi_fail "v1.4 calibration 只评估任务型 PPO 以冻结 B_ref。"
else
    pace_multi_validate_ppo_seed "${stage}_formal" "${ppo_seed}"
fi
if [[ "${split}" == "calibration" && "${method}" == "task_only" ]]; then
    [[ "${reference_json}" == "-" ]] || pace_multi_fail "任务型 B_ref calibration 的 B_ref 参数必须写 -。"
else
    [[ -f "${reference_json}" ]] || pace_multi_fail "B_ref JSON 不存在：${reference_json}"
fi
[[ "${split}" == "calibration" || "${split}" == "holdout" ]] || pace_multi_fail "split 只能是 calibration 或 holdout。"
if [[ "${split}" == "holdout" ]]; then
    [[ -f "${authorization}" ]] || pace_multi_fail "holdout 授权不存在。"
elif [[ -n "${authorization}" ]]; then
    pace_multi_fail "calibration 禁止提供 holdout 授权。"
fi
role="${stage}_${split}"
base="$(pace_multi_seed_base "${role}")"
offset="$(pace_multi_terrain_offset "${terrain}")"
terrain_seed="$((base + offset))"
task="$(pace_multi_task_id "${terrain}" "${method}")"
launch_root="${PACE_MULTI_LOG_ROOT}/launch/${stage}"
mkdir -p "${launch_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
batch_group="${timestamp}"
log_path="${launch_root}/gpt_GPU${gpu}_${stage}_${split}_${terrain}_${method}_seed${ppo_seed}_${timestamp}.log"
additional=()
if [[ "${reference_json}" != "-" ]]; then
    additional+=(--energy_reference_json "${reference_json}")
fi
if [[ "${split}" == "holdout" ]]; then
    additional+=(--holdout_authorization "${authorization}")
fi
cd "${PACE_MULTI_ROOT}"
for batch_index in 0 1 2 3; do
    echo "[PACE] 启动 Terrain20sWide 评估批次 $((batch_index + 1))/4；批次组=${batch_group}。" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_MULTI_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_MULTI_ROOT}" TERM=xterm timeout --signal=TERM --kill-after=60s 30m "${PACE_MULTI_CONDA}" run -n "${PACE_MULTI_ENV}" --no-capture-output "${PACE_MULTI_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_MULTI_EVAL}" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --batch_index "${batch_index}" --batch_group "${batch_group}" --stage "${stage}" --split "${split}" --output_root "${PACE_MULTI_RESULT_ROOT}" --headless --device cuda:0 "${additional[@]}" 2>&1 | tee -a "${log_path}"
done
