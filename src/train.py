"""Train the DQN agent.

    python src/train.py                                   # settings/training_settings.yaml
    python src/train.py --seed 1 --out runs/train/dqn-seed1
    python src/train.py --episodes 10 --out runs/train/smoke   # quick check

Writes to ``--out``:

- ``trained_model.pt``: weights after the last episode (the policy to evaluate),
- ``best_model.pt``: weights of the training episode with the lowest total delay,
- ``training_settings.yaml``: the settings actually used, overrides included,
- ``training_log.csv`` and ``training_curves.png``: one row per episode.

Training traffic uses seeds ``0 .. total_episodes - 1``. Evaluation
(``evaluate.py``) uses held-out seeds from ``settings/testing_settings.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from agent import Agent, Memory
from agent.early_stopping import EarlyStopping
from constants import (
    DEFAULT_SETTINGS_PATH,
    DEFAULT_TRAIN_OUT,
    TORCH_NUM_THREADS,
    TRAINING_SETTINGS_FILE,
)
from environment import Environment
from episode import run_episode
from logger import get_logger
from plots import plot_training_curves
from seeding import seed_everything
from settings import TrainingSettings, load_training_settings, save_settings

if TYPE_CHECKING:
    from agent.memory import Sample

logger = get_logger(__name__)

LOG_COLUMNS = (
    "episode",
    "epsilon",
    "total_delay",
    "avg_queue",
    "arrived",
    "insertion_delay",
    "neg_reward",
    "mean_loss",
)


def linear_epsilon(episode: int, total_episodes: int) -> float:
    """Exploration rate decaying linearly from 1 at the first episode towards 0.

    Args:
        episode: 0-based episode index.
        total_episodes: Number of training episodes.

    Returns:
        ``1 - episode / total_episodes``.
    """
    return 1.0 - episode / total_episodes


def training_session(
    settings: TrainingSettings, out_path: Path, threads: int = TORCH_NUM_THREADS
) -> list[dict[str, float]]:
    """Train an agent and write the model, settings and training log.

    Args:
        settings: Validated training settings.
        out_path: Output directory.
        threads: PyTorch CPU threads. A run is repeatable bit for bit only with
            the same thread count, which changes the order of float additions.

    Returns:
        The training log: one dict per episode, keys ``LOG_COLUMNS``.
    """
    seed_everything(settings.seed)
    torch.set_num_threads(threads)
    out_path.mkdir(parents=True, exist_ok=True)
    save_settings(settings, out_path / TRAINING_SETTINGS_FILE)

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)
    agent = Agent(settings=settings)
    early_stopping = EarlyStopping(
        patience=settings.early_stopping_patience,
        min_episode=settings.early_stopping_min_episode,
    )
    env = Environment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=settings.gui,
    )

    log: list[dict[str, float]] = []
    start = time.perf_counter()
    try:
        for episode in range(settings.total_episodes):
            agent.set_epsilon(linear_epsilon(episode, settings.total_episodes))
            losses: list[float] = []

            def learn(sample: Sample, losses: list[float] = losses) -> None:
                # Online DQN: store the transition, then take gradient steps.
                memory.add_sample(sample)
                for _ in range(settings.training_epochs):
                    loss = agent.replay(memory, settings.gamma, settings.batch_size)
                    if loss is not None:
                        losses.append(loss)

            result = run_episode(env, agent, seed=episode, on_transition=learn)
            m = result.metrics()
            log.append({
                "episode": episode + 1,
                "epsilon": round(agent.epsilon, 4),
                "total_delay": m.total_delay,
                "avg_queue": m.avg_queue,
                "arrived": m.arrived,
                "insertion_delay": m.insertion_delay,
                "neg_reward": round(m.neg_reward, 1),
                "mean_loss": round(sum(losses) / len(losses), 4) if losses else float("nan"),
            })
            logger.info(
                f"Episode {episode + 1}/{settings.total_episodes}  eps={agent.epsilon:.2f}  "
                f"delay={m.total_delay} veh-s  queue={m.avg_queue}  "
                f"arrived={m.arrived}/{settings.n_cars_generated}  "
                f"loss={log[-1]['mean_loss']}  ({time.perf_counter() - start:.0f} s)"
            )
            write_csv(log, out_path)  # every episode, so a long run can be monitored

            if settings.checkpoint_interval and (episode + 1) % settings.checkpoint_interval == 0:
                agent.save_checkpoint(out_path, episode + 1)

            # Select on the reported metric (total delay), not on the reward
            # diagnostic. EarlyStopping maximises, hence the negation.
            if early_stopping.step(-m.total_delay):
                logger.info(f"Early stopping after {episode + 1} episodes")
                break
            if early_stopping.improved:
                agent.model.save_weights(out_path / "best_model.pt")
    finally:
        env.close()

    agent.save_model(out_path)
    if log:
        plot_training_curves(
            {key: [row[key] for row in log] for key in LOG_COLUMNS},
            out_path / "training_curves.png",
        )
    logger.info(f"Training took {time.perf_counter() - start:.0f} s. Saved to {out_path}")
    return log


def write_csv(log: list[dict[str, float]], out_path: Path) -> None:
    """Write the training log so far to ``training_log.csv``."""
    with (out_path / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(log)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the DQN traffic-signal agent.")
    parser.add_argument("--settings", type=Path,
                        default=DEFAULT_SETTINGS_PATH / TRAINING_SETTINGS_FILE,
                        help="Training settings YAML (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_TRAIN_OUT / "dqn",
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Override the settings' seed")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes, e.g. for a quick check")
    parser.add_argument("--threads", type=int, default=TORCH_NUM_THREADS,
                        help="PyTorch CPU threads (default: %(default)s). Use 1-2 per run "
                             "when training several runs in parallel")
    args = parser.parse_args()

    settings = load_training_settings(args.settings)
    overrides = {"seed": args.seed, "total_episodes": args.episodes}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        settings = TrainingSettings.model_validate({**settings.model_dump(), **overrides})
    training_session(settings, args.out, threads=args.threads)


if __name__ == "__main__":
    main()
