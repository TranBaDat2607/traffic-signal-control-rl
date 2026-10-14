"""The interface every traffic-light controller implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray


class Controller(Protocol):
    """Something that picks a green phase at every decision step.

    The DQN agent, the baselines in ``policy/baselines.py`` and anything you
    add all implement this, so ``episode.run_episode()`` and ``evaluate.py``
    treat them identically: same environment, same traffic, same metrics.
    """

    def reset(self) -> None:
        """Forget per-episode state. Called before every episode."""
        ...

    def choose_action(self, state: NDArray[np.float32], info: Mapping[str, Any]) -> int:
        """Pick the next green phase.

        Args:
            state: The observation returned by the environment.
            info: The info dict returned with it (``Environment.step`` lists the
                keys). Learning agents should ignore it: it exposes queue
                counts the agent's state is not meant to contain.

        Returns:
            An action index in ``range(NUM_ACTIONS)``.
        """
        ...
