#!/usr/bin/env bash
# 用法：GPU 阶段 B_ref_JSON 渐进holdout授权 地形:方法:PPO_seed [...]

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_多地形公共.sh"

if [[ "$#" -lt 5 ]]; then
    pace_multi_fail "用法：$0 GPU编号 stage1|stage2 B_ref_JSON 渐进holdout授权 地形:方法:PPO_seed [...]"
fi
gpu="$1"
stage="$2"
reference_json="$3"
authorization="$4"
shift 4
pace_multi_validate_gpu "${gpu}"
[[ -f "${reference_json}" ]] || pace_multi_fail "B_ref JSON 不存在：${reference_json}"
[[ -f "${authorization}" ]] || pace_multi_fail "渐进 holdout 授权不存在：${authorization}"

for specification in "$@"; do
    IFS=: read -r terrain method ppo_seed extra <<<"${specification}"
    [[ -n "${terrain}" && -n "${method}" && -n "${ppo_seed}" && -z "${extra:-}" ]] \
        || pace_multi_fail "评估项格式必须为 地形:方法:PPO_seed：${specification}"
    task="$(pace_multi_task_id "${terrain}" "${method}")"
    checkpoint="$(python - "${authorization}" "${task}" "${ppo_seed}" <<'PY'
import json
import sys

authorization, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(authorization, encoding="utf-8"))
matches = [
    item["检查点"]
    for item in data.get("模型", [])
    if item.get("任务") == task and int(item.get("PPO_seed", -1)) == seed
]
if len(matches) != 1:
    raise SystemExit(f"授权清单中模型匹配数量错误：{task} seed{seed} -> {len(matches)}")
print(matches[0])
PY
)"
    printf '[PACE] GPU%s 渐进评估队列开始：%s/%s/seed%s\n' "${gpu}" "${terrain}" "${method}" "${ppo_seed}"
    bash "${PACE_MULTI_ROOT}/scripts/pace_eco/gpt-运行多地形渐进评估.sh" \
        "${gpu}" "${stage}" "${terrain}" "${method}" "${ppo_seed}" \
        "${checkpoint}" "${reference_json}" "${authorization}"
    printf '[PACE] GPU%s 渐进评估队列完成：%s/%s/seed%s\n' "${gpu}" "${terrain}" "${method}" "${ppo_seed}"
done
