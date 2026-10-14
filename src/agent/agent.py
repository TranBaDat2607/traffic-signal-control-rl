"""DQN agent: epsilon-greedy acting and the Q-learning update."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from agent.model import Model
from constants import NUM_ACTIONS, STATE_SIZE
from policy import EpsilonGreedyPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from agent.memory import Memory
    from settings import TrainingSettings


class Agent:
    """Deep Q-Network agent (Mnih et al., 2015) with a Polyak-averaged target.

    Implements the :class:`~policy.Controller` interface, so it can be run and
    evaluated exactly like the rule-based baselines.
    """

    def __init__(
        self,
        settings: TrainingSettings,
        epsilon: float = 1.0,
        model_path: Path | None = None,
        model: Model | None = None,
    ) -> None:
        """Initialize the agent, its model, and the epsilon-greedy policy.

        Args:
            settings: Training settings with model architecture and hyperparameters.
            epsilon: Initial exploration probability in [0, 1].
            model_path: Optional directory to load trained weights from.
            model: Optional pre-built :class:`~agent.model.Model` to use
                instead of building one.  ``grid.coordinator`` passes the same
                object to every junction's agent so they share weights.  When
                provided, *model_path* is ignored.
        """
        if model is not None:
            self.model = model
        else:
            self.model = Model(
                num_layers=settings.num_layers,
                width=settings.width_layers,
                learning_rate=settings.learning_rate,
                input_dim=STATE_SIZE,
                output_dim=NUM_ACTIONS,
                model_path=model_path,
            )
        self.policy = EpsilonGreedyPolicy(self.model, NUM_ACTIONS, epsilon)
        self.tau = settings.tau

    @property
    def epsilon(self) -> float:
        """Current exploration probability (delegated to the policy)."""
        return self.policy.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Update the exploration probability.

        Args:
            epsilon: New exploration probability in [0, 1].
        """
        self.policy.set_epsilon(epsilon)

    def reset(self) -> None:
        """Nothing to reset between episodes (part of the Controller interface)."""

    def choose_action(
        self,
        state: NDArray[np.float32],
        info: Mapping[str, Any] | None = None,  # noqa: ARG002 - the agent sees only the state
    ) -> int:
        """Choose an action for the given state with the epsilon-greedy policy.

        Args:
            state: Current observation as a 1-D array.
            info: Ignored. The agent acts on its observation only.

        Returns:
            Index of the selected action.
        """
        return self.policy.select_action(state)

    def replay(self, memory: Memory, gamma: float, batch_size: int) -> float | None:
        """Sample a minibatch from replay memory and take one Q-learning step.

        The loss is the Huber loss between ``Q(s, a)`` from the online network
        and the bootstrap target

            y = r + gamma * (1 - terminated) * max_a' Q_target(s', a').

        Episodes in this project end on a time limit (truncation), never in a
        terminal state, so ``terminated`` is always 0 here and the target always
        bootstraps. Treating the time limit as terminal would teach the agent
        that the world ends at t = max_steps.

        Args:
            memory: Experience replay buffer.
            gamma: Discount factor for future rewards.
            batch_size: Number of samples to draw.

        Returns:
            The loss of this step, or ``None`` if memory is still warming up.
        """
        batch = memory.get_samples(batch_size)
        if not batch:
            return None

        device = self.model.device
        states = torch.from_numpy(np.stack([s.state for s in batch])).to(device)
        next_states = torch.from_numpy(np.stack([s.next_state for s in batch])).to(device)
        actions = torch.tensor([s.action for s in batch], device=device)
        rewards = torch.tensor([s.reward for s in batch], dtype=torch.float32, device=device)
        not_terminal = torch.tensor(
            [0.0 if s.terminated else 1.0 for s in batch], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            next_q = self.model.target_model(next_states).max(dim=1).values
            targets = rewards + gamma * not_terminal * next_q

        self.model.model.train()
        q_sa = self.model.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.model.loss_fn(q_sa, targets)

        self.model.optimize(loss)
        self.model.update_target_network(self.tau)
        return float(loss.item())

    def save_model(self, out_path: Path) -> None:
        """Save the online network's weights to ``out_path / MODEL_FILE``."""
        self.model.save_model(out_path)

    def save_checkpoint(self, out_path: Path, episode: int) -> None:
        """Save the online network's weights as ``checkpoint_ep<episode>.pt``."""
        self.model.save_weights(out_path / f"checkpoint_ep{episode}.pt")
