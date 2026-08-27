#!/usr/bin/env bash
# 用法：GPU stage1 holdout directional 地形 方法 PPO_seed checkpoint v2.1_B_ref holdout授权

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.2公共.sh"

if [[ "$#" -ne 10 ]]; then
    pace_direction_fail "用法：$0 GPU stage1 holdout directional 地形 task_only|fixed_weight|eco PPO_seed checkpoint v2.1_B_ref_JSON holdout授权"
fi
gpu="$1"
stage="$2"
split="$3"
variant="$4"
terrain="$5"
method="$6"
ppo_seed="$7"
checkpoint="$8"
v2_reference="$9"
authorization="${10}"
pace_direction_validate_gpu "${gpu}"
pace_direction_require_pace_data
[[ "${stage}" == "stage1" && "${split}" == "holdout" ]] || \
    pace_direction_fail "v2.2 当前只允许 stage1 holdout。"
[[ "${variant}" == "directional" && "${terrain}" != "mixed" ]] || \
    pace_direction_fail "v2.2 主矩阵只允许 E2 单一地形。"
[[ -f "${checkpoint}" && "${checkpoint}" == */model_2999.pt ]] || \
    pace_direction_fail "正式评估只接受存在的 model_2999.pt。"
[[ -f "${v2_reference}" ]] || pace_direction_fail "必须提供已冻结的 v2.1 B_ref。"
[[ -f "${authorization}" ]] || pace_direction_fail "v2.2 holdout 授权不存在。"
case "${terrain}:${method}:${ppo_seed}" in
    flat:task_only:[1-3]|flat:fixed_weight:[1-3]|flat:eco:[1-3]|rough:task_only:[1-3]|rough:fixed_weight:[1-3]|rough:eco:[1-3]|stairs:task_only:[1-3]|stairs:fixed_weight:[1-3]|stairs:eco:[1-3]|boxes:task_only:[1-3]|boxes:fixed_weight:[1-3]|boxes:eco:[1-3]|slope:task_only:[1-3]|slope:fixed_weight:[1-3]|slope:eco:[1-3]) ;;
    *) pace_direction_fail "模型不属于 v2.2 冻结45模型评估矩阵；seed4/5禁止评估。" ;;
esac

role="stage1_holdout"
base="$(pace_direction_seed_base "${role}")"
offset="$(pace_direction_terrain_offset "${terrain}")"
terrain_seed="$((base + offset))"
task="$(pace_direction_v2_2_task_id "${variant}" "${terrain}" "${method}")"
launch_root="${PACE_DIRECTION_LOG_ROOT}/launch/stage1"
mkdir -p "${launch_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
batch_group="${timestamp}"
log_path="${launch_root}/gpt_GPU${gpu}_stage1_holdout_directional_${terrain}_${method}_seed${ppo_seed}_${timestamp}.log"
cd "${PACE_DIRECTION_ROOT}"
for batch_index in 0 1 2 3; do
    printf '[PACE-v2.2] 启动评估批次 %s/4；批次组=%s。\n' \
        "$((batch_index + 1))" "${batch_group}" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_DIRECTION_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_DIRECTION_ROOT}" TERM=xterm timeout --signal=TERM --kill-after=60s 30m "${PACE_DIRECTION_CONDA}" run -n "${PACE_DIRECTION_ENV}" --no-capture-output "${PACE_DIRECTION_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_DIRECTION_EVAL}" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --batch_index "${batch_index}" --batch_group "${batch_group}" --stage stage1 --split holdout --energy_reference_json "${v2_reference}" --holdout_authorization "${authorization}" --output_root "${PACE_DIRECTION_RESULT_ROOT}" --direction_protocol_version v2.2 --headless --device cuda:0 2>&1 | tee -a "${log_path}"
done
summary_path="${PACE_DIRECTION_RESULT_ROOT}/stage1/holdout/gpt_方向评估_directional_${terrain}_${method}_seed${ppo_seed}_${batch_group}/gpt-方向条件评估摘要.json"
[[ -f "${summary_path}" ]] || pace_direction_fail \
    "四批评估进程结束但正式摘要不存在：${summary_path}"
printf '[PACE-v2.2] 评估产物校验通过：%s\n' "${summary_path}"
