#!/usr/bin/env bash
# 方向条件 v2.1 独立训练/评估公共函数。

set -Eeuo pipefail

PACE_DIRECTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACE_DIRECTION_ISAACLAB_ROOT="/home/xy.chen/tw/isaaclab_ws/IsaacLab"
PACE_DIRECTION_CONDA="/home/xy.chen/miniconda3/bin/conda"
PACE_DIRECTION_ENV="env_isaaclab_v100"
PACE_DIRECTION_TRAIN="${PACE_DIRECTION_ROOT}/scripts/pace_eco/train.py"
PACE_DIRECTION_EVAL="${PACE_DIRECTION_ROOT}/scripts/pace_eco/direction_conditioned_eval.py"
PACE_DIRECTION_LOG_ROOT="${PACE_DIRECTION_ROOT}/logs/supplementary/direction_conditioned_v2_1"
PACE_DIRECTION_RESULT_ROOT="${PACE_DIRECTION_ROOT}/results/supplementary/direction_conditioned_v2_1"
PACE_DIRECTION_DATA_ROOT="${PACE_ECO_DATA_ROOT:-${PACE_DIRECTION_ROOT}/pace_data}"


pace_direction_fail() {
    printf '%s\n' "$1" >&2
    exit 2
}


pace_direction_require_pace_data() {
    local fitting="${PACE_DIRECTION_DATA_ROOT}/1_in_air/anymal/fitting.npy"
    local replay="${PACE_DIRECTION_DATA_ROOT}/1_in_air/anymal/data.npy"
    [[ -f "${fitting}" && -f "${replay}" ]] || pace_direction_fail \
        "缺少 ANYmal D PACE 标定数据；请在单行命令前设置 PACE_ECO_DATA_ROOT=/绝对路径/pace_data。"
}


pace_direction_validate_gpu() {
    case "$1" in
        0|1|2|3|4|5) ;;
        *) pace_direction_fail "GPU 编号只能是 0--5。" ;;
    esac
}


pace_direction_require_tmux() {
    if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
        pace_direction_fail "拒绝启动 3000 次训练：必须先进入 tmux 或 GNU Screen。"
    fi
}


pace_direction_require_clean_worktree() {
    if [[ -n "$(git -C "${PACE_DIRECTION_ROOT}" status --porcelain)" ]]; then
        printf '%s\n' "[警告] 当前 Git 工作树包含未提交修改；按 nullptr 的要求继续训练，运行记录会保留实际配置和 Git 状态。" >&2
    fi
}


pace_direction_terrain_token() {
    case "$1" in
        flat) printf 'Flat' ;;
        rough) printf 'Rough' ;;
        stairs) printf 'Stairs' ;;
        boxes) printf 'Boxes' ;;
        slope) printf 'Slope' ;;
        mixed) printf 'MixedTerrain' ;;
        *) pace_direction_fail "未知地形：$1" ;;
    esac
}


pace_direction_task_id() {
    local variant="$1"
    local terrain="$2"
    local method="$3"
    local variant_token method_token terrain_token
    terrain_token="$(pace_direction_terrain_token "${terrain}")"
    case "${variant}" in
        observation_control) variant_token="DirectionObsControl" ;;
        directional) variant_token="DirectionConditioned" ;;
        legacy_v1)
            case "${method}" in
                task_only) method_token="TaskOnly" ;;
                eco) method_token="ECO" ;;
                *) pace_direction_fail "E0 只允许 task_only 或 eco。" ;;
            esac
            printf 'Isaac-PACE-%s-%s-Terrain20sWide-Anymal-D-v0' "${method_token}" "${terrain_token}"
            return
            ;;
        *) pace_direction_fail "未知实验变体：${variant}" ;;
    esac
    case "${method}" in
        task_only) method_token="TaskOnly" ;;
        eco) method_token="ECO" ;;
        *) pace_direction_fail "v2.1 只包含 task_only 和 eco。" ;;
    esac
    printf 'Isaac-PACE-%s-%s-%s-Terrain20sWide-Anymal-D-v0' \
        "${variant_token}" "${method_token}" "${terrain_token}"
}


pace_direction_terrain_offset() {
    case "$1" in
        flat) printf '0' ;;
        rough) printf '1000' ;;
        stairs) printf '2000' ;;
        boxes) printf '3000' ;;
        slope) printf '4000' ;;
        mixed) printf '5000' ;;
        *) pace_direction_fail "未知地形：$1" ;;
    esac
}


pace_direction_seed_base() {
    case "$1" in
        stage1_smoke_train|stage1_capacity_smoke_train) printf '950000' ;;
        stage1_budget_train) printf '510000' ;;
        stage1_formal_train) printf '530000' ;;
        stage1_calibration) printf '540000' ;;
        stage1_holdout) printf '550000' ;;
        stage2_budget_train) printf '610000' ;;
        stage2_formal_train) printf '630000' ;;
        stage2_calibration) printf '640000' ;;
        stage2_holdout) printf '650000' ;;
        stage2_smoke_train|stage2_capacity_smoke_train) printf '960000' ;;
        *) pace_direction_fail "未知 v2 协议角色：$1" ;;
    esac
}


pace_direction_validate_ppo_seed() {
    local group="$1"
    local seed="$2"
    local allowed
    case "${group}" in
        stage1_smoke) allowed="900" ;;
        stage1_budget) allowed="0" ;;
        stage1_formal) allowed="1 2 3 4 5" ;;
        stage2_smoke) allowed="901" ;;
        stage2_budget) allowed="0" ;;
        stage2_formal) allowed="1 2 3 4 5" ;;
        *) pace_direction_fail "未知 PPO seed 组：${group}" ;;
    esac
    [[ " ${allowed} " == *" ${seed} "* ]] || pace_direction_fail \
        "PPO seed ${seed} 不属于 ${group}：${allowed}。"
}
