#!/usr/bin/env bash
# 对新协议 model_2999.pt 执行 200 回合确定性正式评估。

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_训练命令公共.sh"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    pace_usage_error \
        "bash ${BASH_SOURCE[0]} <GPU编号：0至4> <model_2999.pt> [calibration_v1|holdout_v1]"
fi

gpu_id="$1"
checkpoint_input="$2"
state_set="${3:-calibration_v1}"
pace_validate_gpu_id "${gpu_id}"
pace_require_evaluation_runtime_files

case "${state_set}" in
    calibration_v1|holdout_v1)
        ;;
    *)
        pace_usage_error "评估状态集必须是 calibration_v1 或 holdout_v1，实际为：${state_set}"
        ;;
esac

if ! checkpoint="$(realpath -e -- "${checkpoint_input}")"; then
    pace_usage_error "检查点不存在：${checkpoint_input}"
fi
checkpoint_name="$(basename -- "${checkpoint}")"
if [[ "${checkpoint_name}" != "model_2999.pt" ]]; then
    pace_usage_error "正式评估只接受新协议最终检查点 model_2999.pt，实际为：${checkpoint_name}"
fi

run_dir="$(dirname -- "${checkpoint}")"
mapfile -t metadata < <(
    "${PACE_BASE_PYTHON}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
record_path = run_dir / "gpt_复现信息.json"
agent_path = run_dir / "gpt_算法配置.json"
for path in (record_path, agent_path):
    if not path.is_file():
        raise SystemExit(f"评估目录缺少记录文件：{path}")

record = json.loads(record_path.read_text(encoding="utf-8"))
agent_cfg = json.loads(agent_path.read_text(encoding="utf-8"))
print(record["task"])
print(record["seed"])
print(agent_cfg["algorithm"].get("energy_budget_j", ""))
PY
)

if [[ "${#metadata[@]}" -ne 3 ]]; then
    printf '无法从检查点目录读取完整评估配置：%s\n' "${run_dir}" >&2
    exit 1
fi

task="${metadata[0]}"
seed="${metadata[1]}"
energy_budget_j="${metadata[2]}"
if [[ "${state_set}" == "holdout_v1" ]]; then
    case "${seed}" in
        1|2|3|4|5)
            ;;
        *)
            printf '最终留出评估只接受冻结的正式训练种子 1至5，实际为：%s\n' "${seed}" >&2
            exit 2
            ;;
    esac
fi
extra_args=(--state_set "${state_set}")
case "${task}" in
    Isaac-PACE-TaskOnly-Flat-Anymal-D-v0)
        if [[ -n "${energy_budget_j}" ]]; then
            printf '任务型 PPO 运行不应记录能耗预算：%s\n' "${energy_budget_j}" >&2
            exit 2
        fi
        task_tag="任务型PPO"
        ;;
    Isaac-PACE-FixedWeight-Flat-Anymal-D-v0)
        if [[ -n "${energy_budget_j}" ]]; then
            printf '固定权重 PPO 运行不应记录能耗预算：%s\n' "${energy_budget_j}" >&2
            exit 2
        fi
        task_tag="固定权重PPO"
        ;;
    Isaac-PACE-ECO-Flat-Anymal-D-v0)
        pace_validate_positive_number "${energy_budget_j}" "记录中的每回合能耗预算"
        extra_args+=(--energy_budget_j "${energy_budget_j}")
        task_tag="ECO"
        ;;
    *)
        printf '拒绝评估未知任务：%s\n' "${task}" >&2
        exit 2
        ;;
esac

pace_run_evaluation \
    "${gpu_id}" \
    "${task}" \
    "${checkpoint}" \
    32 \
    200 \
    "${seed}" \
    "gpt_GPU${gpu_id}_cu126_${task_tag}_model2999_200回合_${state_set}_seed${seed}" \
    "${extra_args[@]}"
