"""PACE 三类算法的 RSL-RL 5.0.1 配置。"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PacePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "pace_eco_lab.rl.ppo:PacePPO"
    entropy_initial: float = 0.002
    entropy_final: float = 0.0005
    entropy_turnover_iteration: float = 2_000.0
    entropy_slope: float = 2.5e-3


@configclass
class PaceLagrangianAlgorithmCfg(PacePpoAlgorithmCfg):
    class_name: str = "pace_eco_lab.rl.ppo_lagrangian:PPOLagrangian"
    cost_gamma: float = 0.998
    cost_lam: float = 0.81
    energy_budget_j: float = 0.0
    lagrange_initial: float = 0.0
    lagrange_learning_rate: float = 1.0e-3


def _actor_cfg() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.5),
    )


def _critic_cfg() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )


def _ppo_cfg() -> PacePpoAlgorithmCfg:
    return PacePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=10,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


def _lagrangian_cfg() -> PaceLagrangianAlgorithmCfg:
    return PaceLagrangianAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=10,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class PaceTaskOnlyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 统一执行 3000 次更新；RSL-RL 从 0 编号，最终检查点为 model_2999.pt。
    max_iterations = 3_000
    save_interval = 500
    experiment_name = "pace_task_only_flat_anymal_d"
    empirical_normalization = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None
    actor = _actor_cfg()
    critic = _critic_cfg()
    algorithm = _ppo_cfg()


@configclass
class PaceFixedWeightPPORunnerCfg(PaceTaskOnlyPPORunnerCfg):
    experiment_name = "pace_fixed_weight_flat_anymal_d"


@configclass
class PaceEcoPPORunnerCfg(PaceTaskOnlyPPORunnerCfg):
    experiment_name = "pace_eco_flat_anymal_d"
    algorithm = _lagrangian_cfg()
