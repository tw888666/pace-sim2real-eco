#!/usr/bin/env bash
# 用实际碰撞网格审计 Terrain20sWide 宽场地形与历史终止项；不加载或生成模型权重。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"
trap 'status=$?; printf "[PACE] 状态审计 Shell 退出码：%s\\n" "${status}"' EXIT

if [[ "$#" -ne 2 ]]; then
    pace_multi_fail "用法：$0 GPU编号 flat|rough|stairs|boxes|slope"
fi
gpu="$1"
terrain="$2"
pace_multi_validate_gpu "${gpu}"
pace_multi_require_pace_data
case "${terrain}" in
    flat|rough|stairs|boxes|slope) ;;
    *) pace_multi_fail "状态审计地形必须是 flat、rough、stairs、boxes 或 slope。" ;;
esac
base="$(pace_multi_seed_base stage1_smoke_train)"
offset="$(pace_multi_terrain_offset "${terrain}")"
terrain_seed="$((base + offset + 900))"
audit_root="${PACE_MULTI_LOG_ROOT}/state_audit"
mkdir -p "${audit_root}/launch" "${audit_root}/results"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_path="${audit_root}/launch/gpt_GPU${gpu}_${terrain}_state_audit_${timestamp}.log"
result_path="${audit_root}/results/gpt-${terrain}-仿真状态审计-${timestamp}.json"
cd "${PACE_MULTI_ROOT}"
CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_MULTI_DATA_ROOT}" PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_MULTI_ROOT}" TERM=xterm "${PACE_MULTI_CONDA}" run -n "${PACE_MULTI_ENV}" --no-capture-output "${PACE_MULTI_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_MULTI_ROOT}/scripts/pace_eco/audit_multi_terrain_sim_state.py" --terrain "${terrain}" --terrain_seed "${terrain_seed}" --output "${result_path}" --headless --device cuda:0 2>&1 | tee "${log_path}"
