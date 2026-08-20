#!/usr/bin/env bash
# 从同一次运行的检查点继续训练，统一完成 3000 次更新。

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_训练命令公共.sh"

if [[ "$#" -ne 2 ]]; then
    pace_usage_error "bash ${BASH_SOURCE[0]} <GPU编号：0至4> <model_检查点.pt>"
fi

gpu_id="$1"
checkpoint_input="$2"
pace_validate_gpu_id "${gpu_id}"
pace_require_runtime_files

if ! checkpoint="$(realpath -e -- "${checkpoint_input}")"; then
    pace_usage_error "检查点不存在：${checkpoint_input}"
fi
checkpoint_name="$(basename -- "${checkpoint}")"
if [[ ! "${checkpoint_name}" =~ ^model_([0-9]+)[.]pt$ ]]; then
    pace_usage_error "检查点文件名必须形如 model_2500.pt，实际为：${checkpoint_name}"
fi
checkpoint_index="${BASH_REMATCH[1]}"
if (( checkpoint_index >= 2999 )); then
    pace_usage_error "${checkpoint_name} 已达到或超过 3000 次更新的统一终点 model_2999.pt，无需续训。"
fi

run_dir="$(dirname -- "${checkpoint}")"
mapfile -t run_metadata < <(
    "${PACE_BASE_PYTHON}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
record_path = run_dir / "gpt_复现信息.json"
env_path = run_dir / "gpt_环境配置.json"
agent_path = run_dir / "gpt_算法配置.json"
for path in (record_path, env_path, agent_path):
    if not path.is_file():
        raise SystemExit(f"续训目录缺少记录文件：{path}")

record = json.loads(record_path.read_text(encoding="utf-8"))
env_cfg = json.loads(env_path.read_text(encoding="utf-8"))
agent_cfg = json.loads(agent_path.read_text(encoding="utf-8"))
print(record["task"])
print(record["seed"])
print(record["run_name"])
print(env_cfg["scene"]["num_envs"])
print(agent_cfg["algorithm"].get("energy_budget_j", ""))
PY
)

if [[ "${#run_metadata[@]}" -ne 5 ]]; then
    printf '无法从检查点目录读取完整续训配置：%s\n' "${run_dir}" >&2
    exit 1
fi

task="${run_metadata[0]}"
seed="${run_metadata[1]}"
run_name="${run_metadata[2]}"
num_envs="${run_metadata[3]}"
energy_budget_j="${run_metadata[4]}"

if [[ "${num_envs}" != "4096" ]]; then
    printf '拒绝续训：统一比较只允许 4096 环境，原运行环境数为 %s。\n' "${num_envs}" >&2
    exit 2
fi

extra_args=(--resume_from "${checkpoint}")
case "${task}" in
    Isaac-PACE-TaskOnly-Flat-Anymal-D-v0|Isaac-PACE-FixedWeight-Flat-Anymal-D-v0)
        if [[ -n "${energy_budget_j}" ]]; then
            printf '非 ECO 运行不应记录能耗预算：%s\n' "${energy_budget_j}" >&2
            exit 2
        fi
        ;;
    Isaac-PACE-ECO-Flat-Anymal-D-v0)
        pace_validate_positive_number "${energy_budget_j}" "记录中的每回合能耗预算"
        extra_args+=(--energy_budget_j "${energy_budget_j}")
        ;;
    *)
        printf '拒绝续训未知任务：%s\n' "${task}" >&2
        exit 2
        ;;
esac

pace_run_training \
    "${gpu_id}" \
    "${task}" \
    4096 \
    3000 \
    "${seed}" \
    "${run_name}" \
    "gpt_GPU${gpu_id}_cu126续训至3000次更新_seed${seed}" \
    "${extra_args[@]}"
