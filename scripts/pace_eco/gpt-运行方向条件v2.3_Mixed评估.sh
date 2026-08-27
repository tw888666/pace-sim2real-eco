#!/usr/bin/env bash
# 用法：GPU calibration|holdout 方法 PPO_seed checkpoint [预算JSON|-] [holdout授权|-]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"

[[ "$#" -ge 5 && "$#" -le 7 ]] || pace_v23_fail "用法：$0 GPU calibration|holdout 方法 PPO_seed checkpoint [预算JSON|-] [holdout授权|-]"
gpu="$1"; split="$2"; method="$3"; ppo_seed="$4"; checkpoint="$5"; budget="${6:--}"; authorization="${7:--}"
pace_v23_gpu "${gpu}"
pace_v23_require_data
[[ -f "${checkpoint}" && "${checkpoint}" == */model_2999.pt ]] || pace_v23_fail "只接受model_2999.pt。"
case "${split}:${method}:${ppo_seed}" in
    calibration:task_only:0) stage=stage3; manifest="${PACE_V23_CALIBRATION_MANIFEST}" ;;
    holdout:task_only:[123]|holdout:fixed_weight:[123]|holdout:eco:[123]) stage=stage6; manifest="${PACE_V23_HOLDOUT_MANIFEST}" ;;
    *) pace_v23_fail "评估角色/方法/seed不属于冻结矩阵。" ;;
esac
[[ -f "${manifest}" ]] || pace_v23_fail "冻结manifest不存在：${manifest}"
additional=(--direction_protocol_version v2.3 --manifest "${manifest}")
if [[ "${split}" == "holdout" ]]; then
    [[ -f "${budget}" && -f "${authorization}" ]] || pace_v23_fail "holdout必须提供预算冻结JSON和授权文件。"
    additional+=(--energy_reference_json "${budget}" --holdout_authorization "${authorization}")
elif [[ "${budget}" != "-" || "${authorization}" != "-" ]]; then
    pace_v23_fail "calibration禁止读取预算或holdout授权。"
fi
base_seed="$(pace_v23_seed "${split}" 0)"
task="$(pace_v23_task "${method}")"
group="$(date '+%Y%m%d_%H%M%S')"
launch="${PACE_V23_LOG_ROOT}/launch/${stage}/${split}"
mkdir -p "${launch}"
for batch in 0 1 2 3; do
    log="${launch}/gpt_GPU${gpu}_${split}_${method}_seed${ppo_seed}_batch${batch}_${group}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_V23_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_V23_ROOT}" TERM=xterm timeout --signal=TERM --kill-after=60s 30m "${PACE_V23_CONDA}" run -n "${PACE_V23_ENV}" --no-capture-output "${PACE_V23_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_V23_EVAL}" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed "${ppo_seed}" --terrain_seed "${base_seed}" --batch_index "${batch}" --batch_group "${group}" --stage "${stage}" --split "${split}" --output_root "${PACE_V23_RESULT_ROOT}" --headless --device cuda:0 "${additional[@]}" 2>&1 | tee "${log}"
done
