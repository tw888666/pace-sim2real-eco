from pace_eco_lab.training_log_audit import audit_training_log


def test_empty_completed_episode_telemetry_is_recorded_but_not_fatal():
    audit = audit_training_log(
        """Learning iteration 7/3000
Mean value loss: 1.2345
Mean normalized_episode_cost loss: nan
Mean constraint_violation loss: nan
Learning iteration 2999/3000
Mean value loss: 0.1234
"""
    )
    assert audit["状态"] == "通过"
    assert audit["空回合遥测非有限值"]["数量"] == 2
    assert audit["空回合遥测非有限值"]["首次迭代"] == 7
    assert audit["空回合遥测非有限值"]["末次迭代"] == 7


def test_optimizer_nonfinite_is_fatal():
    audit = audit_training_log("Learning iteration 8/3000\nMean surrogate loss: nan\n")
    assert audit["状态"] == "失败"
    assert len(audit["意外非有限值"]) == 1


def test_runtime_failure_is_fatal():
    audit = audit_training_log("Learning iteration 8/3000\nCUDA out of memory\n")
    assert audit["状态"] == "失败"
    assert len(audit["致命错误"]) == 1
