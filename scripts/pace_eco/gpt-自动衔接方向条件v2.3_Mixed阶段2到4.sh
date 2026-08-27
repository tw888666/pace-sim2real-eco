#!/usr/bin/env bash
# 在独立tmux中运行；只做阶段依赖衔接，不改变训练或评估参数。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"
pace_v23_require_tmux

budget_log="${1:-}"
[[ -f "${budget_log}" ]] || pace_v23_fail "必须提供正在执行的阶段2启动日志。"

while ! grep -Eq '\[PACE-v2\.3\] 训练产物校验通过：.*/model_2999\.pt' "${budget_log}"; do
    if grep -Eqi 'Traceback|CUDA out of memory|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)|(^|[^[:alnum:]_])NaN([^[:alnum:]_]|$)|Segmentation fault|Aborted|阶段2会话退出码=[1-9]' "${budget_log}"; then
        pace_v23_fail "阶段2日志出现失败标记，自动链条停止：${budget_log}"
    fi
    sleep 60
done

mapfile -t budget_checkpoints < <(find "${PACE_V23_LOG_ROOT}/rsl_rl/budget_train" -type f -name model_2999.pt | sort)
[[ "${#budget_checkpoints[@]}" -eq 1 ]] || pace_v23_fail "阶段2应有唯一model_2999.pt，实际${#budget_checkpoints[@]}。"
budget_checkpoint="${budget_checkpoints[0]}"

bash "${PACE_V23_ROOT}/scripts/pace_eco/gpt-运行方向条件v2.3_Mixed评估.sh" 3 calibration task_only 0 "${budget_checkpoint}" - -

PYTHONPATH="${PACE_V23_ROOT}" "${PACE_V23_CONDA}" run -n "${PACE_V23_ENV}" --no-capture-output python "${PACE_V23_ROOT}/scripts/pace_eco/freeze_direction_conditioned_v2_3_mixed_budget.py" --calibration_root "${PACE_V23_RESULT_ROOT}/stage3/calibration" --manifest "${PACE_V23_CALIBRATION_MANIFEST}" --checkpoint "${budget_checkpoint}" --output "${PACE_V23_BUDGET}"
[[ -f "${PACE_V23_BUDGET}" ]] || pace_v23_fail "阶段3没有生成预算冻结文件。"

while true; do
    busy=()
    for gpu in 2 3 4 5; do
        processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
        [[ -z "${processes}" ]] || busy+=("GPU${gpu}:${processes//$'\n'/,}")
    done
    [[ "${#busy[@]}" -eq 0 ]] && break
    printf '[PACE-v2.3] 等待阶段4 GPU2--5空闲：%s\n' "${busy[*]}"
    sleep 60
done

bash "${PACE_V23_ROOT}/scripts/pace_eco/gpt-启动方向条件v2.3_Mixed阶段4队列.sh"
