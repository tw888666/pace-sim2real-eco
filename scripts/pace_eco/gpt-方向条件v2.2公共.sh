#!/usr/bin/env bash
# 方向条件 v2.2 的独立路径和三方法守卫；复用 v2.1 已冻结的通用函数。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件公共.sh"

PACE_DIRECTION_LOG_ROOT="${PACE_DIRECTION_ROOT}/logs/supplementary/direction_conditioned_v2_2"
PACE_DIRECTION_RESULT_ROOT="${PACE_DIRECTION_ROOT}/results/supplementary/direction_conditioned_v2_2"
PACE_DIRECTION_V2_1_B_REF="${PACE_DIRECTION_ROOT}/results/supplementary/direction_conditioned_v2_1/stage1/gpt-阶段一v2_B_ref冻结文件.json"


pace_direction_v2_2_task_id() {
    local variant="$1"
    local terrain="$2"
    local method="$3"
    local method_token terrain_token
    [[ "${variant}" == "directional" ]] || pace_direction_fail "v2.2 主矩阵只允许 directional。"
    terrain_token="$(pace_direction_terrain_token "${terrain}")"
    case "${method}" in
        task_only) method_token="TaskOnly" ;;
        fixed_weight) method_token="FixedWeight" ;;
        eco) method_token="ECO" ;;
        *) pace_direction_fail "v2.2 只包含 task_only、fixed_weight 和 eco。" ;;
    esac
    printf 'Isaac-PACE-DirectionConditioned-%s-%s-Terrain20sWide-Anymal-D-v0' \
        "${method_token}" "${terrain_token}"
}


pace_direction_v2_2_require_new_target() {
    local terrain="$1"
    local method="$2"
    local seed="$3"
    [[ " ${seed} " == *" 1 "* || " ${seed} " == *" 2 "* || " ${seed} " == *" 3 "* ]] || \
        pace_direction_fail "v2.2 新训练只允许 PPO seed1--3。"
    case "${method}:${terrain}:${seed}" in
        fixed_weight:flat:*|fixed_weight:rough:*|fixed_weight:stairs:*|fixed_weight:boxes:*|fixed_weight:slope:*) ;;
        task_only:stairs:2|task_only:stairs:3|eco:stairs:2|eco:stairs:3) ;;
        task_only:boxes:*|eco:boxes:*|task_only:slope:*|eco:slope:*) ;;
        *) pace_direction_fail \
            "${method}/${terrain}/seed${seed} 不属于 v2.2 新训练清单；已迁移模型禁止重复训练。" ;;
    esac
}
