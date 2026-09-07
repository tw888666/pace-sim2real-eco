"""PACE-ECO 第一阶段的固定常量。"""

from __future__ import annotations

import os
from pathlib import Path

from pace_eco_lab.multi_terrain_protocol import MULTI_TERRAIN_TASK_IDS
from pace_eco_lab.direction_conditioned_protocol import DIRECTION_CONDITIONED_TASK_IDS
from pace_eco_lab.direction_conditioned_v2_2_protocol import DIRECTION_CONDITIONED_V2_2_TASK_IDS
from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACE_DATA_ROOT = Path(os.environ.get("PACE_ECO_DATA_ROOT", PROJECT_ROOT / "pace_data")).expanduser().resolve()
PACE_FITTING_PATH = PACE_DATA_ROOT / "1_in_air/anymal/fitting.npy"
PACE_REPLAY_PATH = PACE_DATA_ROOT / "1_in_air/anymal/data.npy"

PACE_FITTING_SHA256 = "4436941fa5e9a5e8e1ef93d55956fcffdb4c4c4526b8ef3e145e6ea619fbe1c8"
PACE_REPLAY_SHA256 = "edf2e7648602802c87bbe52bb5c098c41553e8c683b2d9013d6f3f4df8bca621"

PACE_JOINT_NAMES: tuple[str, ...] = (
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
)

TASK_ONLY_ID = "Isaac-PACE-TaskOnly-Flat-Anymal-D-v0"
FIXED_WEIGHT_ID = "Isaac-PACE-FixedWeight-Flat-Anymal-D-v0"
ECO_ID = "Isaac-PACE-ECO-Flat-Anymal-D-v0"
REGISTERED_TASKS = (
    TASK_ONLY_ID,
    FIXED_WEIGHT_ID,
    ECO_ID,
    *MULTI_TERRAIN_TASK_IDS,
    *DIRECTION_CONDITIONED_TASK_IDS,
    *(
        task_id
        for task_id in DIRECTION_CONDITIONED_V2_2_TASK_IDS
        if task_id not in DIRECTION_CONDITIONED_TASK_IDS
    ),
    *DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS,
)

ACTOR_OBSERVATION_DIM = 48
CRITIC_OBSERVATION_DIM = 353
ACTION_DIM = 12

PHYSICS_DT_S = 0.0025
DECIMATION = 8
POLICY_DT_S = PHYSICS_DT_S * DECIMATION
POLICY_FREQUENCY_HZ = 1.0 / POLICY_DT_S
EPISODE_LENGTH_S = 20.0

PD_STIFFNESS = 85.0
PD_DAMPING = 0.6
# 论文 v2 Table 2/3 的 ANYmal 执行器限制。
EFFORT_LIMIT_NM = 89.0
SATURATION_EFFORT_NM = 140.0
VELOCITY_LIMIT_RAD_S = 8.5
GLOBAL_DELAY_STEPS = 3
ACTION_SCALE = 0.5

ELECTRICAL_COEFFICIENT = 0.0192
GRAVITY_M_S2 = 9.81

GROUND_STATIC_FRICTION = 0.8
GROUND_DYNAMIC_FRICTION = 0.6
# 论文只规定随机化 ground friction（地面摩擦），未另给附录范围；采用其
# “standard practice”对应的常用 locomotion（运动）范围。
GROUND_FRICTION_RANDOMIZATION_RANGE = (0.5, 1.25)
PUSH_INTERVAL_RANGE_S = (10.0, 15.0)
PUSH_VELOCITY_RANGE_M_S = (-0.5, 0.5)

ROLLOUT_STEPS = 24
PENALTY_HALF_LIFE_ITERATIONS = 500.0
ENTROPY_INITIAL = 0.002
ENTROPY_FINAL = 0.0005
ENTROPY_TURNOVER_ITERATION = 2_000.0
ENTROPY_SLOPE = 2.5e-3
