"""The Q-network: an MLP, its target copy, and save/load.

The Q-learning update itself is in ``agent.py``. This module only holds the
networks and the optimizer.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim

from constants import MODEL_FILE
from logger import get_logger

logger = get_logger(__name__)


class MLP(nn.Module):
    """Simple multi-layer perceptron with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        width: int,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension (one Q-value per action).
            num_layers: Number of hidden ``width x width`` layers after the
                input layer, so the network has ``num_layers + 1`` hidden layers.
            width: Number of units in each hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map states of shape (batch, input_dim) to Q-values (batch, output_dim)."""
        return self.net(x)


class Model:
    """Online Q-network, target Q-network and optimizer.

    Attributes:
        model: The online network, trained by gradient descent.
        target_model: A slowly-moving copy used for the bootstrap target.
        inference_model: A CPU copy of the online network for single-state
            action selection, which is faster than a GPU round-trip.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_layers: int,
        width: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the networks, optionally loading weights from disk.

        Architecture is always defined by the parameters; a saved file only
        supplies weights, never the structure.

        Args:
            num_layers: Number of hidden layers for the MLP.
            width: Number of units in each hidden layer.
            learning_rate: Learning rate for Adam.
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            model_path: Optional directory containing a saved ``MODEL_FILE``.

        Raises:
            FileNotFoundError: If ``model_path`` is given but has no ``MODEL_FILE``.
        """
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model device: {self.device}")

        self.model = MLP(input_dim, output_dim, num_layers, width).to(self.device)

        if model_path is not None:
            model_file = model_path / MODEL_FILE
            if not model_file.exists():
                msg = f"No trained model at {model_file}"
                raise FileNotFoundError(msg)
            logger.info(f"Loading trained model weights from {model_file}")
            state_dict = torch.load(model_file, weights_only=True, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(requires_grad=False)

        self.inference_model: MLP = copy.deepcopy(self.model).cpu()
        self.inference_model.requires_grad_(requires_grad=False)

        self.loss_fn = nn.HuberLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict_one(self, state: NDArray) -> NDArray:
        """Predict Q-values for a single state, on CPU.

        Args:
            state: 1-D array of shape (input_dim,).

        Returns:
            Array of shape (1, output_dim).
        """
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(1, self.input_dim))
            return self.inference_model(t).numpy()

    def optimize(self, loss: Tensor) -> None:
        """Take one gradient step on ``loss`` and refresh the CPU copy."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._sync_inference_model()

    def update_target_network(self, tau: float) -> None:
        """Blend online weights into the target network (Polyak averaging).

        ``target <- (1 - tau) * target + tau * online``. With ``tau = 1`` this is
        the hard copy of the original DQN paper; small ``tau`` moves the target
        smoothly instead of in periodic jumps.

        Args:
            tau: Interpolation factor in (0, 1].
        """
        with torch.no_grad():
            for p_online, p_target in zip(
                self.model.parameters(), self.target_model.parameters(), strict=True
            ):
                p_target.mul_(1.0 - tau).add_(tau * p_online)

    def save_model(self, out_path: Path) -> None:
        """Save the online weights (``state_dict``) to ``out_path / MODEL_FILE``."""
        self.save_weights(out_path / MODEL_FILE)

    def save_weights(self, file_path: Path) -> None:
        """Save the online weights (``state_dict``) to an explicit file path."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), file_path)

    def load_state_dict_np(self, weights_np: dict[str, NDArray]) -> None:
        """Load online weights from numpy arrays (used by the grid extension)."""
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in weights_np.items()})
        self._sync_inference_model()

    def _sync_inference_model(self) -> None:
        with torch.no_grad():
            for p_cpu, p_online in zip(
                self.inference_model.parameters(), self.model.parameters(), strict=True
            ):
                p_cpu.copy_(p_online.cpu())
