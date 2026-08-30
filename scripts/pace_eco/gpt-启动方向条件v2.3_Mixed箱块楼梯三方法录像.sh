#!/usr/bin/env bash
# 用法：GPU编号 输出根目录；固定录制seed1、holdout batch0的箱块与上行楼梯中等难度实例。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpt-方向条件v2.3_Mixed公共.sh"

[[ "$#" -eq 2 ]] || pace_v23_fail "用法：$0 GPU编号 输出根目录"
gpu="$1"
output_root="$2"
pace_v23_gpu "${gpu}"
[[ ! -e "${output_root}" ]] || pace_v23_fail "拒绝覆盖录像目录：${output_root}"

authorization="${PACE_V23_RESULT_ROOT}/stage5/gpt-v2.3-Mixed九模型holdout授权.json"
[[ -f "${authorization}" && -f "${PACE_V23_BUDGET}" ]] || pace_v23_fail "缺少v2.3预算或holdout授权。"
mkdir -p "${output_root}"

checkpoint_for() {
    python3 -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); matches=[m["检查点"] for m in data["模型"] if m["方法"] == sys.argv[2] and int(m["PPO_seed"]) == 1]; len(matches) == 1 or sys.exit("授权中的方法/seed1检查点不唯一"); print(matches[0])' "${authorization}" "$1"
}

for scene_spec in "boxes:32" "stairs_up:22"; do
    scene="${scene_spec%%:*}"
    env_index="${scene_spec##*:}"
    for method in task_only fixed_weight eco; do
        checkpoint="$(checkpoint_for "${method}")"
        task="$(pace_v23_task "${method}")"
        method_output="${output_root}/${scene}/${method}"
        log="${output_root}/gpt_GPU${gpu}_${scene}_${method}_seed1录像.log"
        command=("${PACE_V23_CONDA}" run -n "${PACE_V23_ENV}" --no-capture-output "${PACE_V23_ISAACLAB_ROOT}/isaaclab.sh" -p "${PACE_V23_ROOT}/scripts/pace_eco/gpt_录制方向条件v2.3_Mixed三方法.py" --task "${task}" --checkpoint "${checkpoint}" --ppo_seed 1 --batch_index 0 --env_index "${env_index}" --scene "${scene}" --energy_reference_json "${PACE_V23_BUDGET}" --holdout_authorization "${authorization}" --duration_s 20 --video_fps 25 --resolution 1280 720 --output_dir "${method_output}" --headless --device cuda:0)
        CUDA_VISIBLE_DEVICES="${gpu}" PACE_ECO_DATA_ROOT="${PACE_V23_DATA_ROOT}" PYTHONUNBUFFERED=1 PYTHONPATH="${PACE_V23_ROOT}" TERM=xterm "${command[@]}" 2>&1 | tee "${log}"
    done
done

video_count="$(find "${output_root}" -type f -name '*.mp4' | wc -l)"
[[ "${video_count}" -eq 6 ]] || pace_v23_fail "箱块与楼梯三方法录像数量错误：${video_count}"
printf '[PACE-v2.3] GPU%s箱块与楼梯三方法录像全部完成：%s\n' "${gpu}" "${output_root}"
