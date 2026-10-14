"""
fixed_phase_check.py
--------------------
Equivalence check between the agent's measurement path and the baseline's.

Runs one episode twice on the same seed:

  1. RL path: a round-robin "agent" (actions 0, 1, 2, 3, 0, ...) driven through
     ``Environment.execute()``, exactly as ``train.py`` / ``test.py`` drive it.
  2. Baseline path: SUMO running the signal plan by itself (as
     ``run_baseline.py`` does), with the static program's durations set to the
     same green / yellow durations so that both paths show the same lights.

Both must report the same total delay. With the defaults (seed 0, 1000 cars,
5400 steps, 10 s green / 4 s yellow) that is 18,195 vehicle-seconds
(PROBLEMS.md P0-1). Any change to how data is read from SUMO (PROBLEMS.md P3-1)
must leave this number unchanged.

``--dump-states`` saves the state vector observed at every decision step, so
that two implementations of ``get_state()`` can be compared array for array.

Usage (from the project root):
    python comparison/fixed_phase_check.py
    python comparison/fixed_phase_check.py --dump-states /tmp/states.npy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from environment import Environment
from environment.generator import generate_routefile
from metrics import insertion_delay, total_delay, vehicles_arrived
from run_baseline import apply_fixed_plan
from sumo_api import backend_name

EXPECTED_DEFAULT_DELAY = 18_195


def _make_env(args: argparse.Namespace) -> Environment:
    return Environment(
        n_cars_generated=args.cars,
        max_steps=args.steps,
        yellow_duration=args.yellow,
        green_duration=args.green,
        turn_chance=0.25,
        sumocfg_file=Path("intersection/sumo_config.sumocfg"),
        gui=False,
    )


def run_rl_path(args: argparse.Namespace) -> tuple[dict, list[np.ndarray]]:
    """Round-robin actions through Environment.execute()."""
    env = _make_env(args)
    env.generate_routefile(seed=args.seed)
    env.activate()

    stats, states, action = [], [], 0
    while not env.is_over():
        states.append(env.get_state())
        stats.extend(env.execute(action))
        env.get_cumulated_waiting_time()  # the reward read, as run_episode() does
        action = (action + 1) % 4
    env.deactivate()

    return {
        "total_delay": total_delay([s.queue_length for s in stats]),
        "arrived": vehicles_arrived([s.arrived for s in stats]),
        "insertion_delay": insertion_delay([s.pending for s in stats]),
    }, states


def run_native_path(args: argparse.Namespace) -> dict:
    """SUMO runs a static program with matched durations; Python only measures."""
    env = _make_env(args)
    generate_routefile(
        seed=args.seed, n_cars_generated=args.cars, max_steps=args.steps, turn_chance=0.25
    )
    env.activate()
    apply_fixed_plan(args.green, args.yellow)

    stats = env._simulate(args.steps)  # noqa: SLF001 — same per-step measurement
    env.deactivate()

    return {
        "total_delay": total_delay([s.queue_length for s in stats]),
        "arrived": vehicles_arrived([s.arrived for s in stats]),
        "insertion_delay": insertion_delay([s.pending for s in stats]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cars", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5400)
    parser.add_argument("--green", type=int, default=10)
    parser.add_argument("--yellow", type=int, default=4)
    parser.add_argument("--dump-states", type=Path, default=None,
                        help="Save the RL path's per-decision states to this .npy file")
    args = parser.parse_args()

    t0 = time.perf_counter()
    rl, states = run_rl_path(args)
    t1 = time.perf_counter()
    native = run_native_path(args)
    t2 = time.perf_counter()

    print(f"RL path     (Environment.execute): {rl}  [{t1 - t0:.2f} s]")
    print(f"Native path (SUMO static program): {native}  [{t2 - t1:.2f} s]")
    print(f"Decision steps: {len(states)}  (SUMO backend: {backend_name()})")

    if args.dump_states is not None:
        np.save(args.dump_states, np.stack(states))
        print(f"States saved to {args.dump_states}")

    defaults = (args.seed, args.cars, args.steps, args.green, args.yellow) == (0, 1000, 5400, 10, 4)
    ok = rl["total_delay"] == native["total_delay"]
    if defaults:
        ok = ok and rl["total_delay"] == EXPECTED_DEFAULT_DELAY
        print(f"Expected total delay: {EXPECTED_DEFAULT_DELAY}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
