"""Tests for the shared episode loop, against a fake environment (no SUMO).

The fake also documents the contract ``run_episode`` relies on: Gymnasium's
``reset``/``step`` plus per-second ``EnvStats`` in ``info["stats"]``.
"""

from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from environment import EnvStats
from episode import run_episode
from policy import FixedTimeController

if TYPE_CHECKING:
    from agent import Sample


class FakeEnv:
    """Three decisions of 10 s each; the queue equals the action taken."""

    max_steps = 30

    def __init__(self) -> None:
        self.t = 0
        self.seeds: list[int] = []

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        self.seeds.append(seed)
        self.t = 0
        return np.zeros(240, dtype=np.float32), {"stats": []}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.t += 10
        stats = [EnvStats(queue_length=action, arrived=1, pending=0) for _ in range(10)]
        obs = np.full(240, self.t / 30, dtype=np.float32)
        return obs, -float(action), False, self.t >= self.max_steps, {"stats": stats}


def test_run_episode_records_every_decision_and_second() -> None:
    env = FakeEnv()
    transitions: list[Sample] = []
    result = run_episode(
        env, FixedTimeController(10, 10), seed=42, on_transition=transitions.append  # type: ignore[arg-type]
    )

    assert env.seeds == [42]
    assert result.actions == [0, 1, 2]
    assert result.rewards == [0.0, -1.0, -2.0]
    assert len(result.stats) == 30

    m = result.metrics()
    assert m.total_delay == 10 * (0 + 1 + 2)
    assert m.avg_queue == 1.0
    assert m.arrived == 30
    assert m.neg_reward == -3.0


def test_transitions_chain_states_and_never_terminate_on_time_limit() -> None:
    transitions: list[Sample] = []
    run_episode(FakeEnv(), FixedTimeController(10, 10), seed=0, on_transition=transitions.append)  # type: ignore[arg-type]

    assert len(transitions) == 3  # the last decision is a transition too
    for prev, nxt in pairwise(transitions):
        assert np.array_equal(prev.next_state, nxt.state)
    assert not any(t.terminated for t in transitions)
