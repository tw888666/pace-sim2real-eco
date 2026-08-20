#!/usr/bin/env bash
# PACE-ECO 训练和评估脚本的公共参数校验与启动函数；请勿直接执行本文件。

set -Eeuo pipefail

PACE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACE_ISAACLAB_ROOT="/home/xy.chen/tw/isaaclab_ws/IsaacLab"
PACE_CONDA="/home/xy.chen/miniconda3/bin/conda"
PACE_BASE_PYTHON="/home/xy.chen/miniconda3/bin/python"
PACE_CONDA_ENV="env_isaaclab_v100"
PACE_TRAIN_SCRIPT="${PACE_PROJECT_ROOT}/scripts/pace_eco/train.py"
PACE_EVALUATION_SCRIPT="${PACE_PROJECT_ROOT}/scripts/pace_eco/eval.py"
PACE_LOG_ROOT="${PACE_PROJECT_ROOT}/logs/launch"
PACE_DATA_ROOT="${PACE_ECO_DATA_ROOT:-${PACE_PROJECT_ROOT}/pace_data}"


pace_usage_error() {
    printf '用法错误：%s\n' "$1" >&2
    exit 2
}


pace_validate_gpu_id() {
    local gpu_id="$1"
    case "${gpu_id}" in
        0|1|2|3|4|5)
            ;;
        *)
            printf 'GPU 编号必须是 0、1、2、3、4 或 5，实际为：%s\n' "${gpu_id}" >&2
            exit 2
            ;;
    esac
}


pace_validate_seed() {
    local seed="$1"
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        printf '随机种子必须是非负整数，实际为：%s\n' "${seed}" >&2
        exit 2
    fi
}


pace_validate_positive_number() {
    local value="$1"
    local label="$2"
    if [[ ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        printf '%s必须是正数，实际为：%s\n' "${label}" "${value}" >&2
        exit 2
    fi
    if ! awk -v number="${value}" 'BEGIN { exit !(number > 0) }'; then
        printf '%s必须大于 0，实际为：%s\n' "${label}" "${value}" >&2
        exit 2
    fi
}


pace_require_persistent_session() {
    local workload="$1"
    local threshold="$2"
    if [[ -n "${TMUX:-}" ]] || [[ -n "${STY:-}" ]]; then
        return
    fi
    printf '拒绝启动%s：超过 %s 的长任务必须在 tmux 或 GNU Screen 会话中运行。\n' \
        "${workload}" "${threshold}" >&2
    printf '请先执行：tmux new-session -s <会话名>，再在会话内运行原命令。\n' >&2
    exit 2
}


pace_require_runtime_files() {
    if [[ ! -x "${PACE_CONDA}" ]]; then
        printf '找不到可执行的 Conda：%s\n' "${PACE_CONDA}" >&2
        exit 1
    fi
    if [[ ! -x "${PACE_BASE_PYTHON}" ]]; then
        printf '找不到用于读取续训记录的 Python：%s\n' "${PACE_BASE_PYTHON}" >&2
        exit 1
    fi
    if [[ ! -d "${PACE_ISAACLAB_ROOT}" ]]; then
        printf '找不到 Isaac Lab 目录：%s\n' "${PACE_ISAACLAB_ROOT}" >&2
        exit 1
    fi
    if [[ ! -f "${PACE_TRAIN_SCRIPT}" ]]; then
        printf '找不到训练入口：%s\n' "${PACE_TRAIN_SCRIPT}" >&2
        exit 1
    fi
    if [[ ! -f "${PACE_DATA_ROOT}/1_in_air/anymal/fitting.npy" ]]; then
        printf '缺少ANYmal D的PACE标定数据，请设置 PACE_ECO_DATA_ROOT：%s\n' \
            "${PACE_DATA_ROOT}" >&2
        exit 1
    fi
}


pace_require_evaluation_runtime_files() {
    pace_require_runtime_files
    if [[ ! -f "${PACE_EVALUATION_SCRIPT}" ]]; then
        printf '找不到评估入口：%s\n' "${PACE_EVALUATION_SCRIPT}" >&2
        exit 1
    fi
}


pace_run_training() {
    if [[ "$#" -lt 7 ]]; then
        pace_usage_error "pace_run_training 内部参数不足。"
    fi

    local gpu_id="$1"
    local task="$2"
    local num_envs="$3"
    local max_iterations="$4"
    local seed="$5"
    local run_name="$6"
    local log_stem="$7"
    shift 7

    pace_validate_gpu_id "${gpu_id}"
    pace_validate_seed "${seed}"
    if [[ ! "${max_iterations}" =~ ^[0-9]+$ ]] || (( max_iterations <= 0 )); then
        pace_usage_error "训练更新次数必须是正整数，实际为：${max_iterations}"
    fi
    if (( max_iterations > 1000 )); then
        pace_require_persistent_session "训练" 1000
    fi
    pace_require_runtime_files

    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    local log_path="${PACE_LOG_ROOT}/${log_stem}_${timestamp}.log"
    mkdir -p "${PACE_LOG_ROOT}"

    printf '将使用物理 GPU%s 启动训练。\n' "${gpu_id}"
    printf '终端日志：%s\n' "${log_path}"

    # 训练入口仍由 Isaac Lab 包装器启动，但以项目根目录作为当前目录。
    # 因此 logs/rsl_rl 会写入 PACE-ECO，而不是 Isaac Lab 官方仓库。
    cd "${PACE_PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PACE_ECO_DATA_ROOT="${PACE_DATA_ROOT}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PACE_PROJECT_ROOT}" \
    TERM=xterm \
    "${PACE_CONDA}" run \
    -n "${PACE_CONDA_ENV}" \
    --no-capture-output \
    "${PACE_ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${PACE_TRAIN_SCRIPT}" \
    --task "${task}" \
    --num_envs "${num_envs}" \
    --max_iterations "${max_iterations}" \
    --seed "${seed}" \
    --headless \
    --device cuda:0 \
    --run_name "${run_name}" \
    "$@" \
    2>&1 | tee "${log_path}"
}


pace_run_evaluation() {
    if [[ "$#" -lt 7 ]]; then
        pace_usage_error "pace_run_evaluation 内部参数不足。"
    fi

    local gpu_id="$1"
    local task="$2"
    local checkpoint="$3"
    local num_envs="$4"
    local episodes="$5"
    local seed="$6"
    local log_stem="$7"
    shift 7

    pace_validate_gpu_id "${gpu_id}"
    pace_validate_seed "${seed}"
    if [[ ! "${num_envs}" =~ ^[0-9]+$ ]] || (( num_envs <= 0 )); then
        pace_usage_error "评估环境数必须是正整数，实际为：${num_envs}"
    fi
    if [[ ! "${episodes}" =~ ^[0-9]+$ ]] || (( episodes < 200 )); then
        pace_usage_error "正式评估回合数必须是至少 200 的整数，实际为：${episodes}"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        pace_usage_error "评估检查点不存在：${checkpoint}"
    fi
    pace_require_evaluation_runtime_files

    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    local log_path="${PACE_LOG_ROOT}/${log_stem}_${timestamp}.log"
    mkdir -p "${PACE_LOG_ROOT}"

    printf '将使用物理 GPU%s 评估策略，不进行训练或权重更新。\n' "${gpu_id}"
    printf '终端日志：%s\n' "${log_path}"

    cd "${PACE_PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PACE_ECO_DATA_ROOT="${PACE_DATA_ROOT}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PACE_PROJECT_ROOT}" \
    TERM=xterm \
    "${PACE_CONDA}" run \
    -n "${PACE_CONDA_ENV}" \
    --no-capture-output \
    "${PACE_ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${PACE_EVALUATION_SCRIPT}" \
    --task "${task}" \
    --checkpoint "${checkpoint}" \
    --num_envs "${num_envs}" \
    --episodes "${episodes}" \
    --seed "${seed}" \
    --headless \
    --device cuda:0 \
    "$@" \
    2>&1 | tee "${log_path}"
}
