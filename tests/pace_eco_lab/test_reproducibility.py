from __future__ import annotations

from dataclasses import dataclass

from pace_eco_lab.reproducibility import configuration_fingerprint


@dataclass
class FakeConfig:
    value: int

    def to_dict(self):
        return {"value": self.value}


def test_fingerprint_is_stable_and_contains_seed():
    first, payload = configuration_fingerprint("task", 3, FakeConfig(1), FakeConfig(2))
    second, _ = configuration_fingerprint("task", 3, FakeConfig(1), FakeConfig(2))
    changed, _ = configuration_fingerprint("task", 4, FakeConfig(1), FakeConfig(2))
    assert first == second
    assert first != changed
    assert payload["seed"] == 3


def test_target_iterations_and_run_name_do_not_break_resume():
    class FakeAgent:
        def __init__(self, max_iterations, run_name):
            self.max_iterations = max_iterations
            self.run_name = run_name

        def to_dict(self):
            return {
                "max_iterations": self.max_iterations,
                "run_name": self.run_name,
                "resume": False,
                "load_run": ".*",
                "load_checkpoint": "model_.*.pt",
                "learning_rate": 1.0e-3,
            }

    first, _ = configuration_fingerprint("task", 0, FakeConfig(1), FakeAgent(300, "gpt_dev"))
    resumed, _ = configuration_fingerprint("task", 0, FakeConfig(1), FakeAgent(30_000, "gpt_dev"))
    assert first == resumed
