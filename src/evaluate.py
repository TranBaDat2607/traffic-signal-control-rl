"""Evaluate any controller on the held-out traffic seeds.

    python src/evaluate.py dqn --model runs/train/dqn
    python src/evaluate.py fixed_time --green 30
    python src/evaluate.py fixed_time --green 10
    python src/evaluate.py max_pressure
    python src/evaluate.py random

Every controller runs through the same environment and episode loop, on the
same seeds from ``settings/testing_settings.yaml`` (10000-10099 by default,
never used for training), and is scored by ``metrics.py``. The DQN agent acts
greedily (epsilon = 0).

Writes ``eval_results.json`` (one value per episode, per metric) to ``--out``,
default ``runs/eval/<label>``. ``comparison/compare_results.py`` reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

import torch

from agent import Agent
from constants import (
    DEFAULT_EVAL_OUT,
    DEFAULT_SETTINGS_PATH,
    EVAL_RESULTS_FILE,
    TESTING_SETTINGS_FILE,
    TORCH_NUM_THREADS,
    TRAINING_SETTINGS_FILE,
)
from environment import Environment
from episode import run_episode
from logger import get_logger
from metrics import METRICS_VERSION
from policy import FixedTimeController, MaxPressureController, RandomController
from settings import TestingSettings, load_testing_settings, load_training_settings

if TYPE_CHECKING:
    from policy import Controller

logger = get_logger(__name__)

CONTROLLERS = ("dqn", "fixed_time", "max_pressure", "random")


def evaluate(controller: Controller, settings: TestingSettings) -> dict[str, list[Any]]:
    """Run ``controller`` on every evaluation seed and collect the metrics.

    Args:
        controller: Any :class:`~policy.Controller`.
        settings: The evaluation protocol.

    Returns:
        ``{"seeds": [...], "total_delays": [...], "avg_queues": [...],
        "vehicles_arrived": [...], "insertion_delays": [...]}``, one entry per
        episode.
    """
    env = Environment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=settings.gui,
    )
    out: dict[str, list[Any]] = {
        "seeds": [],
        "total_delays": [],
        "avg_queues": [],
        "vehicles_arrived": [],
        "insertion_delays": [],
    }
    try:
        for i in range(settings.total_episodes):
            seed = settings.episode_seed + i
            m = run_episode(env, controller, seed=seed).metrics()
            out["seeds"].append(seed)
            out["total_delays"].append(m.total_delay)
            out["avg_queues"].append(m.avg_queue)
            out["vehicles_arrived"].append(m.arrived)
            out["insertion_delays"].append(m.insertion_delay)
            logger.info(
                f"Episode {i + 1}/{settings.total_episodes} (seed {seed})  "
                f"delay={m.total_delay} veh-s  queue={m.avg_queue}  "
                f"arrived={m.arrived}/{settings.n_cars_generated}  "
                f"insertion delay={m.insertion_delay} veh-s"
            )
    finally:
        env.close()
    return out


def make_controller(args: argparse.Namespace, settings: TestingSettings) -> tuple[Controller, str]:
    """Build the controller named on the command line, and a label for it.

    Raises:
        SystemExit: On inconsistent arguments.
    """
    if args.controller == "dqn":
        if args.model is None:
            msg = "dqn needs --model <training output directory>"
            raise SystemExit(msg)
        train_settings = load_training_settings(args.model / TRAINING_SETTINGS_FILE)
        timing = (train_settings.green_duration, train_settings.yellow_duration)
        if timing != (settings.green_duration, settings.yellow_duration):
            msg = (
                f"The agent was trained with {timing[0]} s green / {timing[1]} s yellow, but "
                f"the evaluation settings use {settings.green_duration} s / "
                f"{settings.yellow_duration} s. Its actions would mean something else."
            )
            raise SystemExit(msg)
        agent = Agent(settings=train_settings, epsilon=0.0, model_path=args.model)
        name = args.model.name
        return agent, name if name.startswith("dqn") else f"dqn-{name}"
    if args.controller == "fixed_time":
        green = args.green or 30
        return FixedTimeController(green, settings.green_duration), f"fixed_time-{green}s"
    if args.controller == "max_pressure":
        return MaxPressureController(), "max_pressure"
    return RandomController(seed=args.seed), "random"


def summarize(results: dict[str, list[Any]], n_cars: int) -> str:
    """One-paragraph text summary of an evaluation."""
    def ms(values: list[float]) -> str:
        sd = stdev(values) if len(values) > 1 else 0.0
        return f"{mean(values):,.1f} ± {sd:,.1f}"

    seeds = results["seeds"]
    return (
        f"Episodes: {len(seeds)}, seeds {seeds[0]} to {seeds[-1]}\n"
        f"Total delay (veh-s):      {ms(results['total_delays'])}\n"
        f"Average queue (veh):      {ms(results['avg_queues'])}\n"
        f"Vehicles arrived:         {ms(results['vehicles_arrived'])} of {n_cars}\n"
        f"Insertion delay (veh-s):  {ms(results['insertion_delays'])}  (not in total delay)\n"
        f"(mean ± standard deviation over episodes)"
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a controller on held-out traffic.")
    parser.add_argument("controller", choices=CONTROLLERS)
    parser.add_argument("--model", type=Path, default=None,
                        help="dqn: training output directory with trained_model.pt")
    parser.add_argument("--green", type=int, default=None,
                        help="fixed_time: green seconds per phase (default 30)")
    parser.add_argument("--seed", type=int, default=0, help="random: controller seed")
    parser.add_argument("--settings", type=Path,
                        default=DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
                        help="Evaluation settings YAML (default: %(default)s)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Evaluate only the first N seeds, e.g. for a quick check")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: runs/eval/<label>)")
    args = parser.parse_args()

    settings = load_testing_settings(args.settings)
    if args.episodes is not None:
        settings = settings.model_copy(update={"total_episodes": args.episodes})
    torch.set_num_threads(TORCH_NUM_THREADS)

    controller, label = make_controller(args, settings)
    out_dir = args.out or DEFAULT_EVAL_OUT / label
    logger.info(f"Evaluating {label} on {settings.total_episodes} episodes")

    results = evaluate(controller, settings)
    record = {
        "metrics_version": METRICS_VERSION,
        "controller": args.controller,
        "label": label,
        "model": str(args.model) if args.model else None,
        "settings": settings.model_dump(mode="json"),
        **results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EVAL_RESULTS_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\n{label}\n{summarize(results, settings.n_cars_generated)}")
    print(f"Saved: {out_dir / EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
