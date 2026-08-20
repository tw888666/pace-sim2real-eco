"""RSL-RL 5.0.1 的 PACE 熵调度 PPO。"""

from __future__ import annotations

from typing import ClassVar

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from pace_eco_lab.rl.schedules import entropy_coefficient


class PacePPO(PPO):
    """仅扩展熵调度与可恢复迭代状态，不改变 PPO 其余行为。"""

    storage_class: ClassVar[type[RolloutStorage]] = RolloutStorage

    def __init__(
        self,
        *args,
        entropy_initial: float = 0.002,
        entropy_final: float = 0.0005,
        entropy_turnover_iteration: float = 2_000.0,
        entropy_slope: float = 2.5e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.entropy_initial = entropy_initial
        self.entropy_final = entropy_final
        self.entropy_turnover_iteration = entropy_turnover_iteration
        self.entropy_slope = entropy_slope
        self.pace_iteration = 0
        self._pace_env = None

    @classmethod
    def construct_algorithm(cls, obs, env, cfg: dict, device: str):
        """按 RSL-RL 5.0.1 官方构造路径创建模型和可替换存储。"""

        cfg["algorithm"].pop("class_name")
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if cfg["algorithm"].get("rnd_cfg") is not None or cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("PACE 第一阶段不启用 RND 或 symmetry 扩展。")
        cfg["algorithm"].pop("share_cnn_encoders", None)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Actor Model: {actor}")
        print(f"Critic Model: {critic}")
        storage = cls.storage_class("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = cls(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm._pace_env = env.unwrapped
        algorithm._sync_environment_iteration()
        return algorithm

    def _sync_environment_iteration(self) -> None:
        if self._pace_env is not None:
            self._pace_env.pace_learning_iteration = self.pace_iteration

    def update(self) -> dict[str, float]:
        self.entropy_coef = entropy_coefficient(
            self.pace_iteration,
            initial=self.entropy_initial,
            final=self.entropy_final,
            turnover=self.entropy_turnover_iteration,
            slope=self.entropy_slope,
        )
        losses = super().update()
        losses["entropy_coefficient"] = self.entropy_coef
        self.pace_iteration += 1
        self._sync_environment_iteration()
        return losses

    def save(self) -> dict:
        payload = super().save()
        payload["pace_iteration"] = self.pace_iteration
        payload["pace_entropy_coefficient"] = self.entropy_coef
        return payload

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or bool(load_cfg.get("iteration", False)):
            self.pace_iteration = int(loaded_dict.get("pace_iteration", loaded_dict.get("iter", 0)))
            self.entropy_coef = float(
                loaded_dict.get(
                    "pace_entropy_coefficient",
                    entropy_coefficient(
                        self.pace_iteration,
                        initial=self.entropy_initial,
                        final=self.entropy_final,
                        turnover=self.entropy_turnover_iteration,
                        slope=self.entropy_slope,
                    ),
                )
            )
            self._sync_environment_iteration()
        return load_iteration
