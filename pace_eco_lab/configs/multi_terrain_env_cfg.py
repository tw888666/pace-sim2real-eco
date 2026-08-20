"""仅替换地面的固定20秒多地形配置；历史 Flat 奖励、终止和观测全部继承。"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from pace_eco_lab.configs.env_cfg import PaceTaskOnlyEnvCfg, configure_evaluation
from pace_eco_lab.multi_terrain_protocol import (
    EVAL_NUM_ENVS,
    EVAL_BATCHES,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    PACE_PAPER_FIXED_WEIGHT,
    TERRAIN_DISTRIBUTIONS,
    TERRAIN_LENGTH_M,
    TERRAIN_WIDTH_M,
    TRAIN_TERRAIN_COLS,
    TRAIN_TERRAIN_ROWS,
    evaluation_batch_offset,
    terrain_column_metadata,
)
from pace_eco_lab.terrains import LongTerrainCfg


def _sub_terrains(category: str) -> dict[str, LongTerrainCfg]:
    try:
        specs = TERRAIN_DISTRIBUTIONS[category]
    except KeyError as error:
        raise ValueError(f"未知多地形类别：{category}") from error
    return {
        terrain if direction == "level" else f"{terrain}_{direction}": LongTerrainCfg(
            category=terrain,
            direction=direction,
            proportion=proportion,
        )
        for terrain, direction, proportion in specs
    }


def make_long_terrain_importer(category: str, terrain_seed: int) -> TerrainImporterCfg:
    """生成分层难度地形；环境不注册性能驱动课程项。"""

    generator = TerrainGeneratorCfg(
        seed=terrain_seed,
        # 这里只控制几何按行分层，训练时不会按表现升降级。
        curriculum=True,
        size=(TERRAIN_LENGTH_M, TERRAIN_WIDTH_M),
        border_width=1.0,
        border_height=0.5,
        num_rows=TRAIN_TERRAIN_ROWS,
        num_cols=TRAIN_TERRAIN_COLS,
        color_scheme="height",
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains=_sub_terrains(category),
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        collision_group=-1,
        # None 使训练环境从全部难度层均匀初始化；之后不执行升降级。
        max_init_terrain_level=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


@configclass
class PaceTerrain20sTaskOnlyEnvCfg(PaceTaskOnlyEnvCfg):
    """与历史 Flat 相同的20秒任务，只把 plane 替换为生成式长地形。"""

    pace_terrain_category: str = "flat"
    pace_terrain_seed: int = 110_000
    pace_terrain_column_metadata: tuple[str, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = make_long_terrain_importer(
            self.pace_terrain_category,
            self.pace_terrain_seed,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.pace_terrain_column_metadata = terrain_column_metadata(
            self.pace_terrain_category,
            TRAIN_TERRAIN_COLS,
        )


@configclass
class PaceFlatTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "flat"


@configclass
class PaceRoughTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "rough"


@configclass
class PaceStairsTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "stairs"


@configclass
class PaceBoxesTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "boxes"


@configclass
class PaceSlopeTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "slope"


@configclass
class PaceMixedTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    pace_terrain_category: str = "mixed"


def _make_fixed(base_cls, name: str):
    @configclass
    class Fixed(base_cls):
        def __post_init__(self):
            super().__post_init__()
            self.rewards.energy.weight = PACE_PAPER_FIXED_WEIGHT

    Fixed.__name__ = name
    Fixed.__qualname__ = name
    return Fixed


def _make_eco(base_cls, name: str):
    @configclass
    class Eco(base_cls):
        pass

    Eco.__name__ = name
    Eco.__qualname__ = name
    return Eco


PaceFlatTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceFlatTerrain20sTaskOnlyEnvCfg, "PaceFlatTerrain20sFixedWeightEnvCfg"
)
PaceRoughTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceRoughTerrain20sTaskOnlyEnvCfg, "PaceRoughTerrain20sFixedWeightEnvCfg"
)
PaceStairsTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceStairsTerrain20sTaskOnlyEnvCfg, "PaceStairsTerrain20sFixedWeightEnvCfg"
)
PaceBoxesTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceBoxesTerrain20sTaskOnlyEnvCfg, "PaceBoxesTerrain20sFixedWeightEnvCfg"
)
PaceSlopeTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceSlopeTerrain20sTaskOnlyEnvCfg, "PaceSlopeTerrain20sFixedWeightEnvCfg"
)
PaceMixedTerrain20sFixedWeightEnvCfg = _make_fixed(
    PaceMixedTerrain20sTaskOnlyEnvCfg, "PaceMixedTerrain20sFixedWeightEnvCfg"
)

PaceFlatTerrain20sEcoEnvCfg = _make_eco(
    PaceFlatTerrain20sTaskOnlyEnvCfg, "PaceFlatTerrain20sEcoEnvCfg"
)
PaceRoughTerrain20sEcoEnvCfg = _make_eco(
    PaceRoughTerrain20sTaskOnlyEnvCfg, "PaceRoughTerrain20sEcoEnvCfg"
)
PaceStairsTerrain20sEcoEnvCfg = _make_eco(
    PaceStairsTerrain20sTaskOnlyEnvCfg, "PaceStairsTerrain20sEcoEnvCfg"
)
PaceBoxesTerrain20sEcoEnvCfg = _make_eco(
    PaceBoxesTerrain20sTaskOnlyEnvCfg, "PaceBoxesTerrain20sEcoEnvCfg"
)
PaceSlopeTerrain20sEcoEnvCfg = _make_eco(
    PaceSlopeTerrain20sTaskOnlyEnvCfg, "PaceSlopeTerrain20sEcoEnvCfg"
)
PaceMixedTerrain20sEcoEnvCfg = _make_eco(
    PaceMixedTerrain20sTaskOnlyEnvCfg, "PaceMixedTerrain20sEcoEnvCfg"
)


def configure_terrain20s_evaluation(
    env_cfg: PaceTerrain20sTaskOnlyEnvCfg,
    terrain_seed: int,
    state_set: str,
    batch_index: int,
):
    """恢复历史评估状态逻辑，并配置200实例中的一个独立批次。"""

    if not 0 <= int(batch_index) < EVAL_BATCHES:
        raise ValueError("评估批次索引越界。")
    configure_evaluation(
        env_cfg,
        state_set=state_set,
        state_index_offset=evaluation_batch_offset(batch_index),
    )
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise RuntimeError("Terrain20s 评估要求 terrain_generator。")
    generator.seed = int(terrain_seed)
    generator.num_rows = EVAL_TERRAIN_ROWS
    generator.num_cols = EVAL_TERRAIN_COLS
    generator.difficulty_range = (0.10, 0.90)
    env_cfg.scene.num_envs = EVAL_NUM_ENVS
    env_cfg.scene.terrain.max_init_terrain_level = EVAL_TERRAIN_ROWS - 1
    env_cfg.pace_terrain_seed = int(terrain_seed)
    env_cfg.pace_terrain_column_metadata = terrain_column_metadata(
        env_cfg.pace_terrain_category,
        EVAL_TERRAIN_COLS,
    )
    return env_cfg


__all__ = [name for name in globals() if name.startswith("Pace") or name.startswith("configure_")]
