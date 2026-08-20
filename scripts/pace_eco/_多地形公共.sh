#!/usr/bin/env bash
# 多地形补充实验公共启动函数；不修改现有 Flat 脚本。

set -Eeuo pipefail

PACE_MULTI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACE_MULTI_ISAACLAB_ROOT="/home/xy.chen/tw/isaaclab_ws/IsaacLab"
PACE_MULTI_CONDA="/home/xy.chen/miniconda3/bin/conda"
PACE_MULTI_ENV="env_isaaclab_v100"
PACE_MULTI_TRAIN="${PACE_MULTI_ROOT}/scripts/pace_eco/train.py"
PACE_MULTI_EVAL="${PACE_MULTI_ROOT}/scripts/pace_eco/multi_terrain_eval.py"
PACE_MULTI_LOG_ROOT="${PACE_MULTI_ROOT}/logs/supplementary/multi_terrain_v1_4"
PACE_MULTI_RESULT_ROOT="${PACE_MULTI_ROOT}/results/supplementary/multi_terrain_v1_4"
PACE_MULTI_DATA_ROOT="${PACE_ECO_DATA_ROOT:-${PACE_MULTI_ROOT}/pace_data}"


pace_multi_fail() {
    printf '%s\n' "$1" >&2
    exit 2
}


pace_multi_require_pace_data() {
    local fitting="${PACE_MULTI_DATA_ROOT}/1_in_air/anymal/fitting.npy"
    local replay="${PACE_MULTI_DATA_ROOT}/1_in_air/anymal/data.npy"
    [[ -f "${fitting}" && -f "${replay}" ]] || pace_multi_fail \
        "缺少ANYmal D的PACE标定数据。请在单行命令中设置 PACE_ECO_DATA_ROOT=/绝对路径/pace_data；不会回退到其他工作树。"
}


pace_multi_validate_gpu() {
    case "$1" in
        0|1|2|3|4|5) ;;
        *) pace_multi_fail "GPU 编号只能是 0、1、2、3、4 或 5。" ;;
    esac
}


pace_multi_require_tmux() {
    if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
        pace_multi_fail "拒绝启动 3000 次训练：必须先进入 tmux 或 GNU Screen。"
    fi
}


pace_multi_require_clean_worktree() {
    if [[ -n "$(git -C "${PACE_MULTI_ROOT}" status --porcelain)" ]]; then
        pace_multi_fail "拒绝启动 3000 次实验：多地形工作树必须先由 nullptr 审阅并提交为干净状态。"
    fi
}


pace_multi_task_id() {
    local terrain="$1"
    local method="$2"
    local terrain_token method_token
    case "${terrain}" in
        flat) terrain_token="Flat" ;;
        rough) terrain_token="Rough" ;;
        stairs) terrain_token="Stairs" ;;
        boxes) terrain_token="Boxes" ;;
        slope) terrain_token="Slope" ;;
        mixed) terrain_token="MixedTerrain" ;;
        *) pace_multi_fail "未知地形：${terrain}" ;;
    esac
    case "${method}" in
        task_only) method_token="TaskOnly" ;;
        fixed_weight) method_token="FixedWeight" ;;
        eco) method_token="ECO" ;;
        *) pace_multi_fail "未知方法：${method}" ;;
    esac
    printf 'Isaac-PACE-%s-%s-Terrain20sWide-Anymal-D-v0' "${method_token}" "${terrain_token}"
}


pace_multi_terrain_offset() {
    case "$1" in
        flat) printf '0' ;;
        rough) printf '1000' ;;
        stairs) printf '2000' ;;
        boxes) printf '3000' ;;
        slope) printf '4000' ;;
        mixed) printf '5000' ;;
        *) pace_multi_fail "未知地形：$1" ;;
    esac
}


pace_multi_seed_base() {
    case "$1" in
        stage1_smoke_train|stage1_capacity_smoke_train) printf '930000' ;;
        stage1_budget_train) printf '310000' ;;
        stage1_formal_train) printf '330000' ;;
        stage1_calibration) printf '340000' ;;
        stage1_holdout) printf '350000' ;;
        stage2_budget_train) printf '410000' ;;
        stage2_formal_train) printf '430000' ;;
        stage2_calibration) printf '440000' ;;
        stage2_holdout) printf '450000' ;;
        stage2_smoke_train|stage2_capacity_smoke_train) printf '940000' ;;
        *) pace_multi_fail "未知协议角色：$1" ;;
    esac
}


pace_multi_validate_ppo_seed() {
    local group="$1"
    local seed="$2"
    local allowed=""
    case "${group}" in
        stage1_smoke) allowed="900" ;;
        stage1_budget) allowed="0" ;;
        stage1_formal) allowed="1 2 3 4 5" ;;
        stage2_smoke) allowed="901" ;;
        stage2_budget) allowed="0" ;;
        stage2_formal) allowed="1 2 3 4 5" ;;
        *) pace_multi_fail "未知 PPO seed 组：${group}" ;;
    esac
    [[ " ${allowed} " == *" ${seed} "* ]] || pace_multi_fail "PPO seed ${seed} 不属于 ${group}：${allowed}。"
}
