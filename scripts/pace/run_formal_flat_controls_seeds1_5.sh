#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "用法：$0 <物理GPU编号> <训练种子> [训练种子 ...]" >&2
    exit 2
fi

gpu_id="$1"
shift
training_seeds=("$@")

if [[ ! "$gpu_id" =~ ^[0-5]$ ]]; then
    echo "物理 GPU 编号必须是 0 至 5；启动前仍需检查占用和错误状态。" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
workspace_root="$(dirname -- "$repo_root")"
formal_source_root="${PACE_ECO_FORMAL_SOURCE_ROOT:-$workspace_root/PACE-ECO-multi-terrain}"
python_bin="${PACE_ECO_PYTHON_BIN:-/home/xy.chen/miniconda3/envs/env_isaaclab_v100/bin/python}"
evaluator="$repo_root/scripts/pace/evaluate_formal_flat_terrains.py"
result_root="$repo_root/results/terrain_evaluation/formal_flat_controls_seeds1_5_n200"
episode_name="gpt-正式平地模型地形评估-逐回合.csv"
summary_name="gpt-正式平地模型地形评估-汇总.csv"
pairing_name="gpt-正式平地模型地形评估-配对信息.json"
methods=(task_only fixed_weight eco)

[[ -d "$formal_source_root" ]] || { echo "多地形代码目录不存在：$formal_source_root" >&2; exit 1; }
[[ -x "$python_bin" ]] || { echo "Python 解释器不可执行：$python_bin" >&2; exit 1; }
[[ -f "$evaluator" ]] || { echo "评估脚本不存在：$evaluator" >&2; exit 1; }

checkpoint_dir_for_method() {
    case "$1" in
        task_only) echo "pace_task_only_flat_anymal_d" ;;
        fixed_weight) echo "pace_fixed_weight_flat_anymal_d" ;;
        eco) echo "pace_eco_flat_anymal_d" ;;
        *) echo "未知方法：$1" >&2; return 2 ;;
    esac
}

task_for_method() {
    case "$1" in
        task_only) echo "Isaac-PACE-TaskOnly-Flat-Anymal-D-v0" ;;
        fixed_weight) echo "Isaac-PACE-FixedWeight-Flat-Anymal-D-v0" ;;
        eco) echo "Isaac-PACE-ECO-Flat-Anymal-D-v0" ;;
        *) echo "未知方法：$1" >&2; return 2 ;;
    esac
}

checkpoint_for() {
    local method="$1"
    local seed="$2"
    local checkpoint_dir
    checkpoint_dir="$(checkpoint_dir_for_method "$method")"
    mapfile -t matches < <(find -H "$repo_root/logs/rsl_rl/$checkpoint_dir" -mindepth 2 -maxdepth 2 -type f -path "*_seed${seed}/model_2999.pt" | sort)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "方法 $method、训练种子 $seed 应有且仅有一个正式 model_2999.pt，实际找到 ${#matches[@]} 个。" >&2
        return 1
    fi
    echo "${matches[0]}"
}

result_is_complete() {
    local output_dir="$1"
    [[ -f "$output_dir/$episode_name" && -f "$output_dir/$summary_name" && -f "$output_dir/$pairing_name" ]]
}

run_one() {
    local method="$1"
    local seed="$2"
    local checkpoint="$3"
    local task="$4"
    local output_dir="$result_root/${method}-seed${seed}-flat"
    local log_path="$output_dir/gpt-正式平地控制评估运行日志.log"
    local reference_path="$result_root/task_only-seed${seed}-flat/$episode_name"

    if result_is_complete "$output_dir"; then
        echo "[跳过] GPU=$gpu_id 方法=$method 种子=$seed 平地控制：结果已完整。"
        return 0
    fi
    if [[ -e "$output_dir/$episode_name" || -e "$output_dir/$summary_name" || -e "$output_dir/$pairing_name" ]]; then
        echo "[停止] $output_dir 存在残缺正式结果，拒绝覆盖。" >&2
        return 1
    fi
    if [[ "$method" != "task_only" && ! -f "$reference_path" ]]; then
        echo "[停止] 缺少任务型配对参考：$reference_path" >&2
        return 1
    fi

    mkdir -p "$output_dir"
    cmd=(env
        "PACE_ECO_DATA_ROOT=$repo_root/pace_data"
        "PYTHONPATH=$formal_source_root:$repo_root/source/pace_sim2real"
        "CUDA_VISIBLE_DEVICES=$gpu_id"
        "$python_bin"
        "$evaluator"
        --checkpoint "$checkpoint"
        --agent_task "$task"
        --terrain flat
        --difficulty 0.5
        --num_envs 200
        --goal_distance 3.0
        --max_time 8.0
        --terrain_seed 12345
        --env_seed 24680
        --device cuda:0
        --headless
        --output_dir "$output_dir"
    )
    if [[ "$method" != "task_only" ]]; then
        cmd+=(--paired_reference "$reference_path")
    fi

    echo "[开始] GPU=$gpu_id 方法=$method 种子=$seed 地形=flat"
    if ! "${cmd[@]}" >"$log_path" 2>&1; then
        echo "[失败] GPU=$gpu_id 方法=$method 种子=$seed 地形=flat；日志末尾如下：" >&2
        tail -n 80 "$log_path" >&2
        return 1
    fi
    if ! result_is_complete "$output_dir" || [[ "$(wc -l < "$output_dir/$episode_name")" -ne 201 ]]; then
        echo "[失败] 平地控制结果不完整或不是 200 条：$output_dir" >&2
        return 1
    fi
    echo "[完成] GPU=$gpu_id 方法=$method 种子=$seed 地形=flat"
}

mkdir -p "$result_root"
cd "$repo_root"
for seed in "${training_seeds[@]}"; do
    if [[ ! "$seed" =~ ^[1-5]$ ]]; then
        echo "本批处理只接受训练种子 1 至 5，实际为：$seed" >&2
        exit 2
    fi
    for method in "${methods[@]}"; do
        checkpoint="$(checkpoint_for "$method" "$seed")"
        task="$(task_for_method "$method")"
        run_one "$method" "$seed" "$checkpoint" "$task"
    done
done
echo "[队列完成] GPU=$gpu_id 平地控制训练种子=${training_seeds[*]}"
