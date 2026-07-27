# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Pace-Sim2real-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_sim2real_env_cfg:PaceSim2realEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Pace-Eco-Anymal-D-Flat-v0",
    entry_point=f"{__name__}.pace_energy_env:PaceEnergyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_pace_eco_env_cfg:AnymalDPaceEcoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDPaceEcoPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Pace-Eco-Anymal-D-Flat-Play-v0",
    entry_point=f"{__name__}.pace_energy_env:PaceEnergyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_pace_eco_env_cfg:AnymalDPaceEcoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDPaceEcoPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Pace-Eco-Anymal-D-Flat-Unconstrained-v0",
    entry_point=f"{__name__}.pace_energy_env:PaceEnergyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_pace_eco_env_cfg:AnymalDPaceEcoEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDPaceEcoUnconstrainedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-Pace-Eco-Anymal-D-Flat-Unconstrained-Play-v0",
    entry_point=f"{__name__}.pace_energy_env:PaceEnergyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_pace_eco_env_cfg:AnymalDPaceEcoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDPaceEcoUnconstrainedPPORunnerCfg"
        ),
    },
)
