#!/usr/bin/env bash
# 用法：GPU 阶段 角色 地形 方法 PPO_seed [B_ref冻结JSON或续训占位符-] [续训检查点]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"

if [[ "$#" -lt 6 || "$#" -gt 8 ]]; then
    pace_multi_fail "用法：$0 GPU编号 stage1|stage2 budget_train|formal_train 地形 方法 PPO_seed [PACE-ECO的B_ref JSON；其他方法续训写-] [续训检查点]"
fi
gpu="$1"
stage="$2"
role_suffix="$3"
terrain="$4"
method="$5"
ppo_seed="$6"
extra="${7:-}"
resume_checkpoint="${8:-}"
pace_multi_validate_gpu "${gpu}"
pace_multi_require_tmux
pace_multi_require_clean_worktree
pace_multi_require_pace_data
[[ "${ppo_seed}" =~ ^[0-9]+$ ]] || pace_multi_fail "PPO seed 必须是非负整数。"
[[ "${stage}" == "stage1" || "${stage}" == "stage2" ]] || pace_multi_fail "阶段只能是 stage1 或 stage2。"
if [[ "${stage}" == "stage1" && "${terrain}" == "mixed" ]]; then
    pace_multi_fail "阶段一禁止 mixed。"
fi
if [[ "${stage}" == "stage2" && "${terrain}" != "mixed" ]]; then
    pace_multi_fail "阶段二只允许 mixed。"
fi
protocol_role="${stage}_${role_suffix}"
base="$(pace_multi_seed_base "${protocol_role}")"
offset="$(pace_multi_terrain_offset "${terrain}")"
terrain_seed="$((base + offset + ppo_seed))"
task="$(pace_multi_task_id "${terrain}" "${method}")"
case "${role_suffix}" in
    budget_train) [[ "${method}" == "task_only" ]] || pace_multi_fail "budget_train 只允许 task_only。"; pace_multi_validate_ppo_seed "${stage}_budget" "${ppo_seed}" ;;
    formal_train) pace_multi_validate_ppo_seed "${stage}_formal" "${ppo_seed}" ;;
    *) pace_multi_fail "未知训练角色：${role_suffix}" ;;
esac
additional=()
tag="${method}"
if [[ "${method}" == "fixed_weight" ]]; then
    [[ -z "${extra}" || "${extra}" == "-" ]] || pace_multi_fail "固定权重 PPO 已冻结为 PACE 论文 W100=-0.00016，不接受选择文件或系数参数。"
    tag="${method}_pace_paper_W100"
elif [[ "${method}" == "eco" ]]; then
    [[ -f "${extra}" ]] || pace_multi_fail "PACE-ECO 必须提供存在的 B_ref 冻结 JSON。"
    additional+=(--energy_reference_json "${extra}")
elif [[ -n "${extra}" && "${extra}" != "-" ]]; then
    pace_multi_fail "任务型 PPO 不接受第七个参数；续训占位符请写 -。"
fi
if [[ -n "${resume_checkpoint}" ]]; then
    [[ -f "${resume_checkpoint}" && "${resume_checkpoint}" == *.pt ]] || pace_multi_fail "续训检查点不存在或不是 .pt：${resume_checkpoint}"
    additional+=(--resume_from "${resume_checkpoint}")
fi
launch_root="${PACE_MULTI_LOG_ROOT}/launch/${stage}"
rsl_root="${PACE_MULTI_LOG_ROOT}/rsl_rl/${stage}"
mkdir -p "${launch_root}" "${rsl_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_path="${launch_root}/gpt_GPU${gpu}_${stage}_${terrain}_${tag}_seed${ppo_seed}_${timestamp}.log"
run_name="gpt_multi_terrain_v1_4_${stage}_formal_${role_suffix}_${terrain}_${tag}_ppo_seed${ppo_seed}_terrain_seed${terrain_seed}"
printf '任务=%s PPO_seed=%s 地形seed=%s 日志=%s\n' "${task}" "${ppo_seed}" "${terrain_seed}" "${log_path}"
cd "${PACE_MULTI_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_MULTI_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_MULTI_ROOT}" TERM=xterm "${PACE_MULTI_CONDA}" run -n "${PACE_MULTI_ENV}" --no-capture-output "${PACE_MULTI_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_MULTI_TRAIN}" --task "${task}" --num_envs 4096 --max_iterations 3000 --seed "${ppo_seed}" --terrain_seed "${terrain_seed}" --protocol_role "${protocol_role}" --log_root "${rsl_root}" --headless --device cuda:0 --run_name "${run_name}" "${additional[@]}" 2>&1 | tee "${log_path}"
