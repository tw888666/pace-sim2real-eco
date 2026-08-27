#!/usr/bin/env bash
# 用法：GPU stage1 smoke_train|formal_train directional 地形 方法 PPO_seed [v2.1_B_ref|-] [续训检查点]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.2公共.sh"

if [[ "$#" -lt 7 || "$#" -gt 9 ]]; then
    pace_direction_fail "用法：$0 GPU stage1 smoke_train|formal_train directional 地形 task_only|fixed_weight|eco PPO_seed [v2.1_B_ref_JSON|-] [续训检查点]"
fi
gpu="$1"
stage="$2"
role_suffix="$3"
variant="$4"
terrain="$5"
method="$6"
ppo_seed="$7"
extra="${8:-}"
resume_checkpoint="${9:-}"
pace_direction_validate_gpu "${gpu}"
pace_direction_require_pace_data
[[ "${stage}" == "stage1" ]] || pace_direction_fail "v2.2 当前只允许 stage1。"
[[ "${variant}" == "directional" ]] || pace_direction_fail "v2.2 主矩阵只允许 E2 directional。"
[[ "${terrain}" != "mixed" ]] || pace_direction_fail "v2.2 当前不包含 mixed。"
[[ "${ppo_seed}" =~ ^[0-9]+$ ]] || pace_direction_fail "PPO seed 必须是非负整数。"
case "${role_suffix}" in
    smoke_train)
        [[ "${terrain}" == "flat" && "${method}" == "fixed_weight" && "${ppo_seed}" == "902" ]] || \
            pace_direction_fail "v2.2 smoke 只允许 flat fixed_weight seed902。"
        [[ -z "${resume_checkpoint}" ]] || pace_direction_fail "smoke 禁止续训。"
        max_iterations=2
        num_envs=16
        ;;
    formal_train)
        pace_direction_v2_2_require_new_target "${terrain}" "${method}" "${ppo_seed}"
        pace_direction_require_tmux
        if [[ -z "${resume_checkpoint}" ]]; then pace_direction_require_clean_worktree; fi
        max_iterations=3000
        num_envs=4096
        ;;
    *) pace_direction_fail "v2.2 只允许 smoke_train 或 formal_train。" ;;
esac

protocol_role="stage1_${role_suffix}"
base="$(pace_direction_seed_base "${protocol_role}")"
offset="$(pace_direction_terrain_offset "${terrain}")"
terrain_seed="$((base + offset + ppo_seed))"
task="$(pace_direction_v2_2_task_id "${variant}" "${terrain}" "${method}")"
additional=(--direction_protocol_version v2.2)
if [[ "${method}" == "eco" ]]; then
    [[ -f "${extra}" ]] || pace_direction_fail "v2.2 ECO 必须提供已冻结的 v2.1 B_ref JSON。"
    additional+=(--energy_reference_json "${extra}")
elif [[ -n "${extra}" && "${extra}" != "-" ]]; then
    pace_direction_fail "v2.2 task_only/fixed_weight 不接受 B_ref；占位符请写 -。"
fi
if [[ -n "${resume_checkpoint}" ]]; then
    [[ -f "${resume_checkpoint}" && "${resume_checkpoint}" == *.pt ]] || \
        pace_direction_fail "续训检查点不存在或不是 .pt。"
    additional+=(--resume_from "${resume_checkpoint}")
fi

launch_root="${PACE_DIRECTION_LOG_ROOT}/launch/stage1"
rsl_root="${PACE_DIRECTION_LOG_ROOT}/rsl_rl/stage1"
mkdir -p "${launch_root}" "${rsl_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_path="${launch_root}/gpt_GPU${gpu}_stage1_${role_suffix}_directional_${terrain}_${method}_seed${ppo_seed}_${timestamp}.log"
run_name="gpt_direction_v2_2_stage1_${role_suffix}_directional_${terrain}_${method}_ppo_seed${ppo_seed}_terrain_seed${terrain_seed}"
if [[ -z "${resume_checkpoint}" ]]; then
    existing_count="$(find "${rsl_root}" -type f -name 'gpt_复现信息.json' -path "*_${run_name}/gpt_复现信息.json" | wc -l)"
    [[ "${existing_count}" -eq 0 ]] || pace_direction_fail \
        "已存在 ${existing_count} 个同任务/方法/seed 运行，拒绝创建重复模型。"
fi
printf '协议=v2.2 任务=%s PPO_seed=%s 地形seed=%s 日志=%s\n' \
    "${task}" "${ppo_seed}" "${terrain_seed}" "${log_path}"
cd "${PACE_DIRECTION_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_DIRECTION_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_DIRECTION_ROOT}" TERM=xterm "${PACE_DIRECTION_CONDA}" run -n "${PACE_DIRECTION_ENV}" --no-capture-output "${PACE_DIRECTION_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_DIRECTION_TRAIN}" --task "${task}" --num_envs "${num_envs}" --max_iterations "${max_iterations}" --seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --protocol_role "${protocol_role}" --log_root "${rsl_root}" --headless --device cuda:0 --run_name "${run_name}" "${additional[@]}" 2>&1 | tee "${log_path}"
if [[ "${max_iterations}" -eq 2 ]]; then final_checkpoint="model_1.pt"; else final_checkpoint="model_2999.pt"; fi
mapfile -t completed_models < <(find "${rsl_root}" -type f -name "${final_checkpoint}" -path "*_${run_name}/${final_checkpoint}" | sort)
[[ "${#completed_models[@]}" -eq 1 ]] || pace_direction_fail \
    "训练进程结束但未找到唯一 ${final_checkpoint}（实际 ${#completed_models[@]} 个），拒绝当作成功。"
printf '[PACE-v2.2] 训练产物校验通过：%s\n' "${completed_models[0]}"
