"""Evaluation entry point for the TLCS agent."""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import torch

from agent import Agent
from constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SETTINGS_PATH,
    DEFAULT_TEST_FOLDER,
    TESTING_SETTINGS_FILE,
    TORCH_NUM_THREADS,
    TRAINING_SETTINGS_FILE,
)
from environment import Environment
from episode import run_episode
from logger import get_logger
from metrics import (
    ARRIVED_LABEL,
    AVG_QUEUE_LABEL,
    INSERTION_DELAY_LABEL,
    METRICS_VERSION,
    TOTAL_DELAY_LABEL,
    avg_queue,
    insertion_delay,
    total_delay,
    vehicles_arrived,
)
from plots import save_data_and_plot
from settings import load_testing_settings, load_training_settings

logger = get_logger(__name__)


class TestStats(TypedDict):
    """Aggregated per-episode statistics collected during evaluation.

    Metric definitions live in ``metrics.py``; see ``TrainingStats`` in
    ``train.py``. No reward diagnostic here — evaluation runs greedily and the
    reward signal is a property of training, not of controller quality.
    """

    total_delay: list[int]   # vehicle-seconds
    avg_queue: list[float]   # vehicles
    arrived: list[int]       # saturation guard
    insertion_delay: list[int]  # saturation guard: vehicle-seconds not in total_delay


def testing_session(model_path: Path, settings_file: Path, out_path: Path) -> None:
    """Evaluate a trained model over multiple episodes and report statistics.

    Loads architecture from the training settings saved alongside the model,
    then runs each test episode with epsilon=0 (fully greedy policy).

    Writes ``test_results.json`` and one ``plot_<metric>_data.txt`` per metric
    into *out_path*. The ``.txt`` files use the same names as ``train.py``'s, so
    ``compare_results.py --rl <out_path>`` compares the greedy evaluation
    against a baseline.

    Args:
        model_path: Directory containing ``trained_model.pt`` and
            ``training_settings.yaml``.
        settings_file: Path to ``testing_settings.yaml``.
        out_path: Directory to write the evaluation results into.
    """
    settings = load_testing_settings(settings_file)
    torch.set_num_threads(TORCH_NUM_THREADS)

    # Reconstruct agent architecture from the saved training settings.
    train_settings_file = model_path / TRAINING_SETTINGS_FILE
    train_settings = load_training_settings(train_settings_file)

    agent = Agent(settings=train_settings, epsilon=0.0, model_path=model_path)

    test_stats: TestStats = {
        "total_delay": [],
        "avg_queue": [],
        "arrived": [],
        "insertion_delay": [],
    }

    for episode in range(settings.total_episodes):
        seed = settings.episode_seed + episode
        logger.info(f"Test episode {episode + 1} of {settings.total_episodes} (seed={seed})")

        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        _, env_stats = run_episode(env=env, agent=agent, seed=seed)

        queue_per_step = [s.queue_length for s in env_stats]
        episode_delay = total_delay(queue_per_step)
        episode_queue = avg_queue(queue_per_step, settings.max_steps)

        episode_arrived = vehicles_arrived([s.arrived for s in env_stats])
        episode_insertion = insertion_delay([s.pending for s in env_stats])

        test_stats["total_delay"].append(episode_delay)
        test_stats["avg_queue"].append(episode_queue)
        test_stats["arrived"].append(episode_arrived)
        test_stats["insertion_delay"].append(episode_insertion)

        logger.info(f"\tTotal delay (vehicle-seconds): {episode_delay}")
        logger.info(f"\tAvg queue length: {episode_queue}")
        logger.info(
            f"\tArrived: {episode_arrived} of {settings.n_cars_generated}, "
            f"insertion delay: {episode_insertion} vehicle-seconds"
        )

    mean_delay = sum(test_stats["total_delay"]) / settings.total_episodes
    mean_queue = sum(test_stats["avg_queue"]) / settings.total_episodes

    logger.info("--- Test Summary ---")
    logger.info(f"Episodes:                           {settings.total_episodes}")
    logger.info(f"Mean total delay (vehicle-seconds): {mean_delay:.1f}")
    logger.info(f"Mean queue length:                  {mean_queue:.2f}")
    logger.info(
        f"Mean vehicles arrived:              "
        f"{sum(test_stats['arrived']) / settings.total_episodes:.1f} "
        f"of {settings.n_cars_generated}"
    )
    logger.info(
        f"Mean insertion delay (veh-s):       "
        f"{sum(test_stats['insertion_delay']) / settings.total_episodes:.1f}"
    )

    out_path.mkdir(parents=True, exist_ok=True)
    results = {
        "metrics_version": METRICS_VERSION,
        "mode": "dqn_greedy",
        "epsilon": agent.epsilon,
        "model": str(model_path),
        "total_episodes": settings.total_episodes,
        "seeds": [settings.episode_seed + i for i in range(settings.total_episodes)],
        "n_cars_generated": settings.n_cars_generated,
        "total_delays": test_stats["total_delay"],
        "avg_queues": test_stats["avg_queue"],
        "vehicles_arrived": test_stats["arrived"],
        "insertion_delays": test_stats["insertion_delay"],
    }
    (out_path / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")
    for key, filename, label in (
        ("total_delay", "delay", TOTAL_DELAY_LABEL),
        ("avg_queue", "queue", AVG_QUEUE_LABEL),
        ("arrived", "arrived", ARRIVED_LABEL),
        ("insertion_delay", "insertion_delay", INSERTION_DELAY_LABEL),
    ):
        save_data_and_plot(
            data=test_stats[key],  # type: ignore[literal-required]
            filename=filename,
            xlabel="Test episode",
            ylabel=label,
            out_folder=out_path,
        )
    logger.info(f"Results saved at: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TLCS agent.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Directory containing trained_model.pt and training_settings.yaml (default: model/)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
        help="Path to testing_settings.yaml (default: settings/testing_settings.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the evaluation results (default: <model>/test/)",
    )
    args = parser.parse_args()

    testing_session(
        model_path=args.model,
        settings_file=args.settings,
        out_path=args.out or args.model / DEFAULT_TEST_FOLDER,
    )
