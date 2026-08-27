#!/usr/bin/env bash
# 用法：GPU smoke_train|budget_train|formal_train 方法 PPO_seed [预算JSON|-] [续训checkpoint]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"

[[ "$#" -ge 4 && "$#" -le 6 ]] || pace_v23_fail "用法：$0 GPU smoke_train|budget_train|formal_train task_only|fixed_weight|eco PPO_seed [预算JSON|-] [续训checkpoint]"
gpu="$1"; role="$2"; method="$3"; ppo_seed="$4"; budget="${5:--}"; resume="${6:-}"
pace_v23_gpu "${gpu}"
pace_v23_require_data
[[ "${ppo_seed}" =~ ^[0-9]+$ ]] || pace_v23_fail "PPO seed必须是非负整数。"
case "${role}:${method}:${ppo_seed}" in
    smoke_train:fixed_weight:903) updates=2 ;;
    budget_train:task_only:0) updates=3000; pace_v23_require_tmux ;;
    formal_train:task_only:[123]|formal_train:fixed_weight:[123]|formal_train:eco:[123]) updates=3000; pace_v23_require_tmux ;;
    *) pace_v23_fail "角色/方法/seed不属于v2.3冻结矩阵。" ;;
esac
if [[ "${updates}" -eq 3000 && -z "${resume}" && -n "$(git -C "${PACE_V23_ROOT}" status --porcelain)" ]]; then
    pace_v23_fail "v2.3正式/预算训练要求Git工作树干净；请先冻结阶段配置。"
fi
[[ -z "${resume}" || "${role}" != "smoke_train" ]] || pace_v23_fail "smoke禁止恢复。"
additional=(--direction_protocol_version v2.3)
if [[ "${method}" == "eco" ]]; then
    [[ "${role}" == "formal_train" && -f "${budget}" ]] || pace_v23_fail "ECO正式训练必须提供冻结的Mixed B80 JSON。"
    additional+=(--energy_reference_json "${budget}")
elif [[ "${budget}" != "-" ]]; then
    pace_v23_fail "Task-only/Fixed-weight不读取预算，占位符应为-。"
fi
if [[ -n "${resume}" ]]; then
    [[ -f "${resume}" && "${resume}" == *.pt ]] || pace_v23_fail "恢复checkpoint不存在或不是.pt。"
    additional+=(--resume_from "${resume}")
fi
terrain_seed="$(pace_v23_seed "${role}" "${ppo_seed}")"
task="$(pace_v23_task "${method}")"
launch_root="${PACE_V23_LOG_ROOT}/launch/${role}"
rsl_root="${PACE_V23_LOG_ROOT}/rsl_rl/${role}"
mkdir -p "${launch_root}" "${rsl_root}"
run_name="gpt_direction_v2_3_mixed_${role}_${method}_ppo_seed${ppo_seed}_terrain_seed${terrain_seed}"
if [[ -z "${resume}" ]]; then
    existing="$(find "${rsl_root}" -type f -name gpt_复现信息.json -path "*_${run_name}/gpt_复现信息.json" | wc -l)"
    [[ "${existing}" -eq 0 ]] || pace_v23_fail "已存在同方法/seed运行，拒绝训练候选模型。"
fi
log="${launch_root}/gpt_GPU${gpu}_${role}_${method}_seed${ppo_seed}_$(date '+%Y%m%d_%H%M%S').log"
printf 'v2.3 Mixed：任务=%s PPO_seed=%s terrain_seed=%s log=%s\n' "${task}" "${ppo_seed}" "${terrain_seed}" "${log}"
cd "${PACE_V23_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_V23_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_V23_ROOT}" TERM=xterm "${PACE_V23_CONDA}" run -n "${PACE_V23_ENV}" --no-capture-output "${PACE_V23_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_V23_TRAIN}" --task "${task}" --num_envs 4096 --max_iterations "${updates}" --seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --protocol_role "${role}" --log_root "${rsl_root}" --headless --device cuda:0 --run_name "${run_name}" "${additional[@]}" 2>&1 | tee "${log}"
if [[ "${updates}" -eq 2 ]]; then final=model_1.pt; else final=model_2999.pt; fi
mapfile -t checkpoints < <(find "${rsl_root}" -type f -name "${final}" -path "*_${run_name}/${final}" | sort)
[[ "${#checkpoints[@]}" -eq 1 ]] || pace_v23_fail "训练结束但未找到唯一${final}。"
printf '[PACE-v2.3] 训练产物校验通过：%s\n' "${checkpoints[0]}"
