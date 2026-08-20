#!/usr/bin/env bash
# 用法：GPU 阶段 split 变体 地形 方法 PPO_seed checkpoint v2_B_ref|- [holdout授权] [E0_v1_B_ref]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件公共.sh"

if [[ "$#" -lt 9 || "$#" -gt 11 ]]; then
    pace_direction_fail "用法：$0 GPU stage1|stage2 calibration|holdout directional|observation_control|legacy_v1 地形 task_only|eco PPO_seed checkpoint v2_B_ref_JSON|- [holdout授权] [E0_v1_B_ref]"
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
authorization="${10:-}"
legacy_reference="${11:-}"
pace_direction_validate_gpu "${gpu}"
pace_direction_require_pace_data
[[ -f "${checkpoint}" ]] || pace_direction_fail "检查点不存在：${checkpoint}"
[[ "${stage}" == "stage1" || "${stage}" == "stage2" ]] || pace_direction_fail "阶段非法。"
[[ "${split}" == "calibration" || "${split}" == "holdout" ]] || pace_direction_fail "split 只能是 calibration 或 holdout。"
if [[ "${stage}" == "stage1" && "${terrain}" == "mixed" ]]; then pace_direction_fail "阶段一禁止 mixed。"; fi
if [[ "${stage}" == "stage2" && "${terrain}" != "mixed" ]]; then pace_direction_fail "阶段二只允许 mixed。"; fi
if [[ "${split}" == "calibration" ]]; then
    [[ "${variant}" == "directional" && "${method}" == "task_only" ]] || pace_direction_fail "v2 calibration 只允许 E2 task_only。"
    pace_direction_validate_ppo_seed "${stage}_budget" "${ppo_seed}"
    [[ "${v2_reference}" == "-" ]] || pace_direction_fail "calibration 的 v2 B_ref 参数必须写 -。"
    [[ -z "${authorization}" && -z "${legacy_reference}" ]] || pace_direction_fail "calibration 禁止授权和旧 B_ref。"
else
    pace_direction_validate_ppo_seed "${stage}_formal" "${ppo_seed}"
    [[ -f "${v2_reference}" ]] || pace_direction_fail "holdout 必须提供 v2 B_ref。"
    [[ -f "${authorization}" ]] || pace_direction_fail "holdout 授权不存在。"
    if [[ "${variant}" == "legacy_v1" ]]; then
        [[ -f "${legacy_reference}" ]] || pace_direction_fail "E0 交叉评估必须提供历史 v1 B_ref。"
    elif [[ -n "${legacy_reference}" ]]; then
        pace_direction_fail "E1/E2 禁止提供 E0 v1 B_ref。"
    fi
fi
role="${stage}_${split}"
base="$(pace_direction_seed_base "${role}")"
offset="$(pace_direction_terrain_offset "${terrain}")"
terrain_seed="$((base + offset))"
task="$(pace_direction_task_id "${variant}" "${terrain}" "${method}")"
launch_root="${PACE_DIRECTION_LOG_ROOT}/launch/${stage}"
mkdir -p "${launch_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
batch_group="${timestamp}"
log_path="${launch_root}/gpt_GPU${gpu}_${stage}_${split}_${variant}_${terrain}_${method}_seed${ppo_seed}_${timestamp}.log"
additional=()
if [[ "${v2_reference}" != "-" ]]; then additional+=(--energy_reference_json "${v2_reference}"); fi
if [[ "${split}" == "holdout" ]]; then additional+=(--holdout_authorization "${authorization}"); fi
if [[ "${variant}" == "legacy_v1" ]]; then additional+=(--training_energy_reference_json "${legacy_reference}"); fi
cd "${PACE_DIRECTION_ROOT}"
for batch_index in 0 1 2 3; do
    printf '[PACE-v2] 启动评估批次 %s/4；批次组=%s。\n' "$((batch_index + 1))" "${batch_group}" | tee -a "${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_DIRECTION_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_DIRECTION_ROOT}" TERM=xterm timeout --signal=TERM --kill-after=60s 30m "${PACE_DIRECTION_CONDA}" run -n "${PACE_DIRECTION_ENV}" --no-capture-output "${PACE_DIRECTION_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_DIRECTION_EVAL}" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --batch_index "${batch_index}" --batch_group "${batch_group}" --stage "${stage}" --split "${split}" --output_root "${PACE_DIRECTION_RESULT_ROOT}" --headless --device cuda:0 "${additional[@]}" 2>&1 | tee -a "${log_path}"
done
summary_path="${PACE_DIRECTION_RESULT_ROOT}/${stage}/${split}/gpt_方向评估_${variant}_${terrain}_${method}_seed${ppo_seed}_${batch_group}/gpt-方向条件评估摘要.json"
[[ -f "${summary_path}" ]] || pace_direction_fail \
    "四批评估进程结束但正式摘要不存在，拒绝当作成功：${summary_path}"
printf '[PACE-v2] 评估产物校验通过：%s\n' "${summary_path}"
