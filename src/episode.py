"""The one episode loop, shared by training and evaluation.

Training and evaluation, the agent and every baseline all go through
:func:`run_episode`, and every reported number goes through
:func:`EpisodeResult.metrics`. A controller's score therefore depends only on
its decisions, never on which script produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.memory import Sample
from metrics import avg_queue, insertion_delay, sum_negative_rewards, total_delay, vehicles_arrived

if TYPE_CHECKING:
    from collections.abc import Callable

    from environment import Environment, EnvStats
    from policy import Controller


@dataclass
class EpisodeMetrics:
    """The reported numbers for one episode. Definitions: ``metrics.py``."""

    total_delay: int
    avg_queue: float
    arrived: int
    insertion_delay: int
    neg_reward: float


@dataclass
class EpisodeResult:
    """Everything recorded during one episode.

    Attributes:
        seed: Traffic seed the episode ran on.
        max_steps: Episode length in simulated seconds.
        rewards: One reward per decision step.
        actions: One action per decision step.
        stats: One :class:`EnvStats` per simulated second.
    """

    seed: int
    max_steps: int
    rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    stats: list[EnvStats] = field(default_factory=list)

    def metrics(self) -> EpisodeMetrics:
        """Aggregate the per-second measurements with ``metrics.py``."""
        queue_per_step = [s.queue_length for s in self.stats]
        return EpisodeMetrics(
            total_delay=total_delay(queue_per_step),
            avg_queue=avg_queue(queue_per_step, self.max_steps),
            arrived=vehicles_arrived([s.arrived for s in self.stats]),
            insertion_delay=insertion_delay([s.pending for s in self.stats]),
            neg_reward=sum_negative_rewards(self.rewards),
        )


def run_episode(
    env: Environment,
    controller: Controller,
    seed: int,
    on_transition: Callable[[Sample], None] | None = None,
) -> EpisodeResult:
    """Run one episode of ``controller`` on the traffic generated from ``seed``.

    Args:
        env: Environment to interact with. Left open, so it can be reused.
        controller: Anything implementing :class:`~policy.Controller`.
        seed: Traffic seed passed to ``env.reset()``.
        on_transition: Called with every ``(s, a, r, s', terminated)`` transition
            as soon as it happens. Training uses it to fill the replay memory
            and take a gradient step; evaluation leaves it unset.

    Returns:
        The rewards, actions and per-second measurements of the episode.
    """
    result = EpisodeResult(seed=seed, max_steps=env.max_steps)
    controller.reset()
    state, info = env.reset(seed=seed)

    done = False
    while not done:
        action = controller.choose_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        result.actions.append(action)
        result.rewards.append(reward)
        result.stats.extend(info["stats"])

        if on_transition is not None:
            on_transition(Sample(state, action, reward, next_state, terminated))

        state = next_state
        done = terminated or truncated

    return result
