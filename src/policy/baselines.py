"""Rule-based controllers to compare a learned policy against.

All of them act through the same environment as the agent: one decision every
``green_duration`` seconds, a yellow inserted on every phase change. So the only
thing that differs between a baseline and the agent is the decision rule.

- :class:`RandomController` — a floor. A learned policy that does not beat it
  has learned nothing.
- :class:`FixedTimeController` — a fixed cycle through the four phases, the way
  most real signals run. ``comparison/fixed_phase_check.py`` checks that it
  gives exactly the same result as SUMO running the same static program itself.
- :class:`MaxPressureController` — the standard adaptive heuristic in the
  traffic-signal-control literature (Varaiya, 2013). It is the baseline a
  learned controller is usually expected to beat.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from constants import NUM_ACTIONS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

    from environment import PhaseLanes


class RandomController:
    """Picks a phase uniformly at random at every decision."""

    def __init__(self, seed: int = 0) -> None:
        """Initialize the controller.

        Args:
            seed: Seed of the controller's own random generator.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        """Nothing to reset: the generator keeps running across episodes."""

    def choose_action(self, state: NDArray[np.float32], info: Mapping[str, Any]) -> int:  # noqa: ARG002
        """Return a uniformly random action."""
        return self._rng.randrange(NUM_ACTIONS)


class FixedTimeController:
    """Cycles through the phases in order, each held for a fixed green time.

    With ``green_time=30`` and a 10 s decision step, each phase is chosen three
    decisions in a row, giving 30 s green / 4 s yellow, the plan in
    ``intersection/environment.net.xml``.
    """

    def __init__(self, green_time: int, decision_green: int) -> None:
        """Initialize the controller.

        Args:
            green_time: Seconds of green per phase per cycle.
            decision_green: The environment's ``green_duration``. ``green_time``
                must be a multiple of it.

        Raises:
            ValueError: If ``green_time`` is not a positive multiple of
                ``decision_green``.
        """
        if green_time <= 0 or green_time % decision_green != 0:
            msg = (
                f"green_time ({green_time} s) must be a positive multiple of the "
                f"environment's green_duration ({decision_green} s)"
            )
            raise ValueError(msg)
        self.repeats = green_time // decision_green
        self._decision = 0

    def reset(self) -> None:
        """Start the cycle again at phase 0."""
        self._decision = 0

    def choose_action(self, state: NDArray[np.float32], info: Mapping[str, Any]) -> int:  # noqa: ARG002
        """Return the phase the cycle is in."""
        action = (self._decision // self.repeats) % NUM_ACTIONS
        self._decision += 1
        return action


class MaxPressureController:
    """Gives green to the phase with the highest pressure.

    The pressure of a phase is the number of halted vehicles on the lanes it
    lets into the junction minus the halted vehicles on the lanes it sends
    them to. Downstream congestion lowers pressure, so the controller does not
    push vehicles into a road that cannot take them.

    Ties keep the current phase, which avoids needless yellow time.
    """

    def __init__(self) -> None:
        """Initialize the controller."""
        self._last_action: int | None = None

    def reset(self) -> None:
        """Forget the current phase."""
        self._last_action = None

    def choose_action(self, state: NDArray[np.float32], info: Mapping[str, Any]) -> int:  # noqa: ARG002
        """Return the action with the highest pressure."""
        pressures = phase_pressures(info["phase_lanes"], info["lane_queues"])
        best = max(pressures.values())
        if self._last_action is not None and pressures[self._last_action] == best:
            return self._last_action
        action = min(a for a, p in pressures.items() if p == best)
        self._last_action = action
        return action


def phase_pressures(
    phase_lanes: Mapping[int, PhaseLanes], lane_queues: Mapping[str, int]
) -> dict[int, int]:
    """Compute the max-pressure score of every action.

    Args:
        phase_lanes: ``{action: PhaseLanes}`` from the environment's info dict.
        lane_queues: ``{lane_id: halted vehicles}`` from the environment's info dict.

    Returns:
        ``{action: incoming queue - outgoing queue}``.
    """
    return {
        action: sum(lane_queues.get(lane, 0) for lane in lanes.incoming)
        - sum(lane_queues.get(lane, 0) for lane in lanes.outgoing)
        for action, lanes in phase_lanes.items()
    }
