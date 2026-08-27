#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"
pace_v23_require_tmux
[[ -f "${PACE_V23_BUDGET}" ]] || pace_v23_fail "阶段3预算尚未冻结：${PACE_V23_BUDGET}"
runner="${PACE_V23_ROOT}/scripts/pace_eco/gpt-运行方向条件v2.3_Mixed训练.sh"

worker() {
    local gpu="$1"; shift
    while [[ "$#" -gt 0 ]]; do
        local method="$1" seed="$2"; shift 2
        if [[ "${method}" == eco ]]; then extra="${PACE_V23_BUDGET}"; else extra=-; fi
        bash "${runner}" "${gpu}" formal_train "${method}" "${seed}" "${extra}"
    done
}

worker 2 task_only 1 fixed_weight 1 eco 1 & p2=$!
worker 3 task_only 2 fixed_weight 2 & p3=$!
worker 4 task_only 3 fixed_weight 3 & p4=$!
worker 5 eco 2 eco 3 & p5=$!
wait "${p2}" "${p3}" "${p4}" "${p5}"
printf '%s\n' '[PACE-v2.3] 阶段4固定队列全部完成。'
