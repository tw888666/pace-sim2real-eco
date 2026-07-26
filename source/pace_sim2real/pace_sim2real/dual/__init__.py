"""Immutable checkpoint and idempotent dual-update protocol."""

from .protocol import (
    DualEvaluationResult,
    DualState,
    apply_evaluation_result,
    atomic_write_json,
    file_sha256,
    freeze_checkpoint,
    load_dual_state,
)
from .metrics import (
    RegressionMetrics,
    evaluate_trajectory_predictions,
    regression_metrics,
    undiscounted_return_to_go,
    validate_trajectory_metrics,
)
from .dataset import (
    CRITIC_DATASET_SCHEMA_VERSION,
    atomic_torch_save,
    load_critic_dataset,
    validate_critic_dataset,
)
from .comparison import (
    OfflineCostCritic,
    fit_time_only_linear_baseline,
    module_sha256,
    predict_in_batches,
    predict_time_only,
    state_dict_sha256,
    train_offline_cost_critic,
)

__all__ = [
    "DualEvaluationResult",
    "DualState",
    "apply_evaluation_result",
    "atomic_write_json",
    "file_sha256",
    "freeze_checkpoint",
    "load_dual_state",
    "RegressionMetrics",
    "evaluate_trajectory_predictions",
    "regression_metrics",
    "undiscounted_return_to_go",
    "validate_trajectory_metrics",
    "CRITIC_DATASET_SCHEMA_VERSION",
    "atomic_torch_save",
    "load_critic_dataset",
    "validate_critic_dataset",
    "OfflineCostCritic",
    "fit_time_only_linear_baseline",
    "module_sha256",
    "predict_in_batches",
    "predict_time_only",
    "state_dict_sha256",
    "train_offline_cost_critic",
]
