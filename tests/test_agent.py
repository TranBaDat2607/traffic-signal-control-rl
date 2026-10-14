"""Tests for the DQN update, replay memory, settings and seeding. No SUMO needed."""

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from agent import Agent, Memory, Sample
from seeding import seed_everything
from settings import TrainingSettings, load_testing_settings, load_training_settings

ROOT = Path(__file__).resolve().parent.parent


def small_settings(**overrides: object) -> TrainingSettings:
    values = load_training_settings(ROOT / "settings/training_settings.yaml").model_dump()
    values.update(num_layers=1, width_layers=32, learning_rate=0.01, **overrides)
    return TrainingSettings.model_validate(values)


class ConstantQ(nn.Module):
    """A frozen target network that predicts the same Q-value everywhere."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 4), self.value)


def fit_single_transition(*, terminated: bool, reward: float, gamma: float) -> float:
    """Replay one transition many times against a constant target; return Q(s, a)."""
    seed_everything(0)
    agent = Agent(settings=small_settings())
    agent.model.target_model = ConstantQ(10.0)
    agent.model.update_target_network = lambda _tau: None  # keep the target fixed

    state = np.full(240, 0.5, dtype=np.float32)
    memory = Memory(size_max=10, size_min=1)
    memory.add_sample(Sample(state, 2, reward, state, terminated))
    for _ in range(400):
        agent.replay(memory, gamma=gamma, batch_size=1)
    return float(agent.model.predict_one(state)[0, 2])


def test_replay_bootstraps_from_target_network() -> None:
    q = fit_single_transition(terminated=False, reward=1.0, gamma=0.5)
    assert q == pytest.approx(1.0 + 0.5 * 10.0, abs=0.05)


def test_replay_does_not_bootstrap_from_terminal_state() -> None:
    q = fit_single_transition(terminated=True, reward=1.0, gamma=0.5)
    assert q == pytest.approx(1.0, abs=0.05)


def test_replay_waits_for_memory_warmup() -> None:
    agent = Agent(settings=small_settings())
    memory = Memory(size_max=10, size_min=5)
    state = np.zeros(240, dtype=np.float32)
    memory.add_sample(Sample(state, 0, 0.0, state))
    assert agent.replay(memory, gamma=0.75, batch_size=4) is None


def test_memory_is_a_ring_buffer() -> None:
    memory = Memory(size_max=3, size_min=1)
    state = np.zeros(1, dtype=np.float32)
    for i in range(5):
        memory.add_sample(Sample(state, i, 0.0, state))
    assert len(memory) == 3
    assert sorted(s.action for s in memory.get_samples(10)) == [2, 3, 4]


def test_greedy_agent_ignores_info() -> None:
    agent = Agent(settings=small_settings(), epsilon=0.0)
    state = np.random.default_rng(0).random(240, dtype=np.float32)
    expected = int(np.argmax(agent.model.predict_one(state)))
    assert agent.choose_action(state, {"lane_queues": {"x": 99}}) == expected


def test_same_seed_gives_same_network() -> None:
    seed_everything(3)
    a = Agent(settings=small_settings())
    seed_everything(3)
    b = Agent(settings=small_settings())
    for pa, pb in zip(a.model.model.parameters(), b.model.model.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_settings_reject_unknown_keys() -> None:
    values = small_settings().model_dump()
    values["learning_rat"] = 0.1
    with pytest.raises(ValidationError, match="learning_rat"):
        TrainingSettings.model_validate(values)


def test_settings_reject_memory_min_above_max() -> None:
    with pytest.raises(ValidationError, match="memory_size_min"):
        small_settings(memory_size_min=100, memory_size_max=10)


def test_repository_settings_files_load() -> None:
    load_training_settings(ROOT / "settings/training_settings.yaml")
    testing = load_testing_settings(ROOT / "settings/testing_settings.yaml")
    assert testing.episode_seed >= testing.total_episodes  # held out from training seeds
