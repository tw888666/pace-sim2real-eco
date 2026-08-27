#!/usr/bin/env bash

set -Eeuo pipefail

PACE_V23_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACE_V23_ISAACLAB_ROOT="/home/xy.chen/tw/isaaclab_ws/IsaacLab"
PACE_V23_CONDA="/home/xy.chen/miniconda3/bin/conda"
PACE_V23_ENV="env_isaaclab_v100"
PACE_V23_TRAIN="${PACE_V23_ROOT}/scripts/pace_eco/train.py"
PACE_V23_EVAL="${PACE_V23_ROOT}/scripts/pace_eco/direction_conditioned_eval.py"
PACE_V23_LOG_ROOT="${PACE_V23_ROOT}/logs/supplementary/direction_conditioned_v2_3_mixed"
PACE_V23_RESULT_ROOT="${PACE_V23_ROOT}/results/supplementary/direction_conditioned_v2_3_mixed"
PACE_V23_DATA_ROOT="${PACE_ECO_DATA_ROOT:-/home/xy.chen/tw/dataset/pace_data}"
PACE_V23_MANIFEST_ROOT="${PACE_V23_RESULT_ROOT}/manifests"
PACE_V23_CALIBRATION_MANIFEST="${PACE_V23_MANIFEST_ROOT}/calibration_manifest.json"
PACE_V23_HOLDOUT_MANIFEST="${PACE_V23_MANIFEST_ROOT}/holdout_manifest.json"
PACE_V23_BUDGET="${PACE_V23_RESULT_ROOT}/stage3/gpt-v2.3-Mixed-B_ref-B80冻结.json"

pace_v23_fail() { printf '%s\n' "$1" >&2; exit 2; }

pace_v23_gpu() {
    case "$1" in 0|1|2|3|4|5) ;; *) pace_v23_fail "GPU编号只能是0--5。" ;; esac
}

pace_v23_require_data() {
    [[ -f "${PACE_V23_DATA_ROOT}/1_in_air/anymal/fitting.npy" && -f "${PACE_V23_DATA_ROOT}/1_in_air/anymal/data.npy" ]] || \
        pace_v23_fail "缺少 ANYmal D PACE 数据；请设置 PACE_ECO_DATA_ROOT。"
}

pace_v23_require_tmux() {
    [[ -n "${TMUX:-}" || -n "${STY:-}" ]] || pace_v23_fail "3000次训练必须在tmux或GNU Screen中启动。"
}

pace_v23_task() {
    case "$1" in
        task_only) printf 'Isaac-PACE-DirectionConditioned-V23Mixed-TaskOnly-Terrain20sWide-Anymal-D-v0' ;;
        fixed_weight) printf 'Isaac-PACE-DirectionConditioned-V23Mixed-FixedWeight-Terrain20sWide-Anymal-D-v0' ;;
        eco) printf 'Isaac-PACE-DirectionConditioned-V23Mixed-ECO-Terrain20sWide-Anymal-D-v0' ;;
        *) pace_v23_fail "方法只能是task_only、fixed_weight或eco。" ;;
    esac
}

pace_v23_seed() {
    local role="$1" ppo_seed="$2" base
    case "${role}" in
        smoke_train) base=970000 ;;
        budget_train) base=710000 ;;
        formal_train) base=730000 ;;
        calibration) base=740000 ;;
        holdout) base=750000 ;;
        *) pace_v23_fail "未知v2.3角色：${role}" ;;
    esac
    printf '%s' "$((base + ppo_seed))"
}
