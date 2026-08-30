#!/usr/bin/env bash
# 用法：GPU PPO_seed；按冻结授权串行评估该 seed 的三种方法。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"

[[ "$#" -eq 2 ]] || pace_v23_fail "用法：$0 GPU编号 PPO_seed"
gpu="$1"
ppo_seed="$2"
pace_v23_gpu "${gpu}"
case "${ppo_seed}" in 1|2|3) ;; *) pace_v23_fail "阶段6 PPO seed只能是1、2、3。" ;; esac

authorization="${PACE_V23_RESULT_ROOT}/stage5/gpt-v2.3-Mixed九模型holdout授权.json"
if [[ ! -f "${PACE_V23_BUDGET}" || ! -f "${authorization}" || ! -f "${PACE_V23_HOLDOUT_MANIFEST}" ]]; then
    pace_v23_fail "阶段6缺少预算、holdout授权或冻结manifest。"
fi

checkpoint_for() {
    python3 -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); matches=[m["检查点"] for m in data["模型"] if m["方法"] == sys.argv[2] and int(m["PPO_seed"]) == int(sys.argv[3])]; len(matches) == 1 or sys.exit("授权中的方法/seed检查点不唯一"); print(matches[0])' "${authorization}" "$1" "${ppo_seed}"
}

for method in task_only fixed_weight eco; do
    checkpoint="$(checkpoint_for "${method}")"
    command=(bash "${PACE_V23_ROOT}/scripts/pace_eco/gpt-运行方向条件v2.3_Mixed评估.sh" "${gpu}" holdout "${method}" "${ppo_seed}" "${checkpoint}" "${PACE_V23_BUDGET}" "${authorization}")
    "${command[@]}"
done

printf '[PACE-v2.3] 阶段6 seed%s/GPU%s队列全部完成。\n' "${ppo_seed}" "${gpu}"
