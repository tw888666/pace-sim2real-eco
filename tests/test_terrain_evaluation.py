from types import SimpleNamespace

import torch

from pace_sim2real.evaluation import (
    TERRAIN_VARIANTS,
    TerrainEpisodeAccumulator,
    TerrainEvaluationProtocol,
    assert_pairing_matches,
    configure_eval_protocol,
    configure_eval_terrain,
    pairing_signature,
    summarize_episode_rows,
)


def _fake_env_cfg():
    command_ranges = SimpleNamespace(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-3.14, 3.14),
    )
    command = SimpleNamespace(
        heading_command=True,
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        ranges=command_ranges,
    )
    reset_base = SimpleNamespace(
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5)},
        }
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            num_envs=32,
            terrain=SimpleNamespace(
                terrain_type="plane",
                terrain_generator=None,
                max_init_terrain_level=5,
            ),
            height_scanner=object(),
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(enable_corruption=True, height_scan=object()),
        ),
        commands=SimpleNamespace(base_velocity=command),
        events=SimpleNamespace(
            add_base_mass=object(),
            base_com=object(),
            base_external_force_torque=object(),
            push_robot=object(),
            reset_base=reset_base,
        ),
        curriculum=SimpleNamespace(terrain_levels=object()),
        pace_energy=SimpleNamespace(publish_eval_state=False),
        sim=SimpleNamespace(dt=0.0025),
        decimation=8,
        episode_length_s=20.0,
        seed=None,
    )


def _fake_rough_terrain_cfg():
    return SimpleNamespace(
        seed=None,
        curriculum=True,
        num_rows=1,
        num_cols=1,
        difficulty_range=(0.0, 1.0),
        sub_terrains={key: SimpleNamespace(proportion=0.25) for key in TERRAIN_VARIANTS.values()},
    )


def test_evaluation_config_changes_only_terrain_and_deterministic_protocol() -> None:
    env_cfg = _fake_env_cfg()
    rough_cfg = _fake_rough_terrain_cfg()
    protocol = TerrainEvaluationProtocol(terrain="stairs", num_envs=8)

    terrain_cfg = configure_eval_terrain(env_cfg, protocol, rough_terrains_cfg=rough_cfg)
    step_dt = configure_eval_protocol(env_cfg, protocol)

    assert env_cfg.scene.terrain.terrain_type == "generator"
    assert env_cfg.scene.terrain.terrain_generator is terrain_cfg
    assert env_cfg.scene.terrain.max_init_terrain_level is None
    assert set(terrain_cfg.sub_terrains) == {"pyramid_stairs_inv"}
    assert terrain_cfg.sub_terrains["pyramid_stairs_inv"].proportion == 1.0
    assert terrain_cfg.difficulty_range == (0.5, 0.5)
    assert terrain_cfg.seed == 12_345
    assert env_cfg.scene.height_scanner is None
    assert env_cfg.observations.policy.height_scan is None
    assert env_cfg.observations.policy.enable_corruption is False
    assert env_cfg.scene.num_envs == 8
    assert env_cfg.seed == 24_680
    assert env_cfg.commands.base_velocity.ranges.lin_vel_x == (1.0, 1.0)
    assert env_cfg.events.add_base_mass is None
    assert env_cfg.events.base_com is None
    assert env_cfg.events.base_external_force_torque is None
    assert env_cfg.events.push_robot is None
    assert env_cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["velocity_range"]["pitch"] == (0.0, 0.0)
    assert env_cfg.pace_energy.publish_eval_state is True
    assert step_dt == 0.02
    assert env_cfg.episode_length_s == 8.02

    # The shared Isaac Lab template must not be mutated by one evaluation.
    assert rough_cfg.sub_terrains["pyramid_stairs_inv"].proportion == 0.25
    assert rough_cfg.difficulty_range == (0.0, 1.0)


def test_flat_control_keeps_plane_and_disables_height_scan() -> None:
    env_cfg = _fake_env_cfg()
    protocol = TerrainEvaluationProtocol(terrain="flat", num_envs=8)

    terrain_cfg = configure_eval_terrain(env_cfg, protocol, rough_terrains_cfg=_fake_rough_terrain_cfg())

    assert terrain_cfg is None
    assert protocol.terrain_variant == "plane"
    assert env_cfg.scene.terrain.terrain_type == "plane"
    assert env_cfg.scene.terrain.terrain_generator is None
    assert env_cfg.scene.terrain.max_init_terrain_level is None
    assert env_cfg.scene.height_scanner is None
    assert env_cfg.observations.policy.height_scan is None
    assert env_cfg.curriculum.terrain_levels is None


def test_accumulator_freezes_terminal_state_and_explicit_timeout() -> None:
    accumulator = TerrainEpisodeAccumulator(
        3,
        device="cpu",
        step_dt=0.02,
        goal_distance_m=3.0,
        command_x_mps=1.0,
    )

    accumulator.update(
        step_index=1,
        progress_m=torch.tensor([1.0, 0.5, 0.4]),
        forward_velocity_mps=torch.tensor([1.0, 0.0, 1.0]),
        step_energy_j=torch.tensor([10.0, 5.0, 1.0]),
        step_electrical_j=torch.tensor([4.0, 2.0, 0.4]),
        step_mechanical_j=torch.tensor([5.0, 2.0, 0.5]),
        step_potential_j=torch.tensor([1.0, 1.0, 0.1]),
        terminated=torch.tensor([False, True, False]),
        truncated=torch.zeros(3, dtype=torch.bool),
    )
    accumulator.update(
        step_index=2,
        # env 1 imitates an auto-reset position and huge post-reset energy.
        progress_m=torch.tensor([3.1, 999.0, 1.0]),
        forward_velocity_mps=torch.ones(3),
        step_energy_j=torch.tensor([20.0, 1000.0, 2.0]),
        step_electrical_j=torch.tensor([8.0, 400.0, 0.8]),
        step_mechanical_j=torch.tensor([10.0, 500.0, 1.0]),
        step_potential_j=torch.tensor([2.0, 100.0, 0.2]),
        terminated=torch.zeros(3, dtype=torch.bool),
        truncated=torch.zeros(3, dtype=torch.bool),
    )
    accumulator.update(
        step_index=3,
        progress_m=torch.tensor([-100.0, -100.0, 1.5]),
        forward_velocity_mps=torch.ones(3),
        step_energy_j=torch.tensor([1000.0, 1000.0, 3.0]),
        step_electrical_j=torch.tensor([400.0, 400.0, 1.2]),
        step_mechanical_j=torch.tensor([500.0, 500.0, 1.5]),
        step_potential_j=torch.tensor([100.0, 100.0, 0.3]),
        terminated=torch.zeros(3, dtype=torch.bool),
        truncated=torch.zeros(3, dtype=torch.bool),
    )
    accumulator.finalize_timeouts(max_steps=3)

    assert accumulator.status.tolist() == [1, 2, 3]
    assert accumulator.finish_steps.tolist() == [2, 1, 3]
    torch.testing.assert_close(accumulator.final_progress_m, torch.tensor([3.1, 0.5, 1.5]))
    torch.testing.assert_close(accumulator.final_energy_j, torch.tensor([30.0, 5.0, 6.0]))

    rows = accumulator.episode_rows({"algorithm": "test"})
    assert rows[0]["time_to_goal_s"] == 0.04
    assert rows[0]["energy_per_meter_j"] == 30.0 / float(torch.tensor(3.1))
    assert rows[1]["progress_m"] == 0.5
    assert rows[1]["energy_j"] == 5.0
    assert rows[1]["energy_per_meter_j"] is None
    assert rows[2]["status"] == "timeout"
    assert rows[2]["finish_steps"] == 3
    assert rows[2]["duration_s"] == 0.06


def test_summary_never_rewards_failed_low_energy_episodes() -> None:
    rows = [
        {
            "status": "success",
            "progress_m": 3.0,
            "energy_j": 600.0,
            "electrical_energy_j": 300.0,
            "mechanical_energy_j": 200.0,
            "potential_energy_j": 100.0,
            "energy_per_meter_j": 200.0,
            "mean_power_w": 200.0,
            "time_to_goal_s": 3.0,
            "tracking_rmse": 0.1,
        },
        {
            "status": "fall",
            "progress_m": 0.5,
            "energy_j": 50.0,
            "electrical_energy_j": 25.0,
            "mechanical_energy_j": 20.0,
            "potential_energy_j": 5.0,
            "energy_per_meter_j": None,
            "mean_power_w": 20.0,
            "time_to_goal_s": None,
            "tracking_rmse": 0.4,
        },
    ]

    summary = summarize_episode_rows(rows)

    assert summary["success_rate"] == 0.5
    assert summary["fall_rate"] == 0.5
    assert summary["successful_mean_energy_j"] == 600.0
    assert summary["successful_mean_energy_per_meter_j"] == 200.0
    assert summary["successful_mean_tracking_rmse"] == 0.1
    assert summary["efficiency_population"] == "successful_episodes_only"


def test_pairing_signature_and_reference_check_ignore_algorithm_only() -> None:
    base = [
        {
            "algorithm": "ppo",
            "env_id": env_id,
            "terrain": "box",
            "difficulty": 0.5,
            "terrain_seed": 12345,
            "env_seed": 24680,
            "terrain_rows": 10,
            "terrain_cols": 20,
            "terrain_level": level,
            "terrain_type": terrain_type,
            "start_x_w": float(env_id),
            "start_y_w": 0.0,
            "start_z_w": 0.5,
        }
        for env_id, (level, terrain_type) in enumerate(((2, 0), (7, 1)))
    ]
    same_pairing = [{**row, "algorithm": "ppo_lagrangian"} for row in base]

    assert pairing_signature(base) == pairing_signature(same_pairing)
    assert_pairing_matches(base, same_pairing)

    wrong_level = [dict(row) for row in same_pairing]
    wrong_level[1]["terrain_level"] = 8
    try:
        assert_pairing_matches(base, wrong_level)
    except ValueError as exc:
        assert "paired evaluation mismatch" in str(exc)
    else:
        raise AssertionError("different terrain levels must fail paired evaluation")

    wrong_start = [dict(row) for row in same_pairing]
    wrong_start[0]["start_x_w"] = 0.01
    try:
        assert_pairing_matches(base, wrong_start)
    except ValueError as exc:
        assert "start state differs" in str(exc)
    else:
        raise AssertionError("different paired initial states must be rejected")
