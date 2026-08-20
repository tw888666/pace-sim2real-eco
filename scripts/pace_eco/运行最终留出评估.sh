#!/usr/bin/env bash
# 使用 holdout_v1 独立状态集执行最终 200 回合评估。

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -ne 2 ]]; then
    printf '用法错误：bash %s <GPU编号：0至4> <model_2999.pt>\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

exec bash "${script_dir}/运行正式评估.sh" "$1" "$2" holdout_v1
