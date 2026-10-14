"""End-to-end check that the measurement path is right.

For each fixed-time plan, one episode is run twice on the same seed:

  1. Controller path: ``FixedTimeController`` driven through
     ``Environment.step()`` by ``episode.run_episode()``, exactly as the agent
     is trained and evaluated.
  2. Native path: SUMO running the same static signal program by itself, with
     Python only measuring.

Both must report the same total delay, vehicle for vehicle. If they do, the
environment's phase switching and its per-second measurement are faithful, and
any controller evaluated through it is measured the same way.

Plans checked (seed 0, 1000 cars, 5400 s):

  - 10 s green / 4 s yellow: 18,195 vehicle-seconds (pinned below)
  - 30 s green / 4 s yellow: the plan in ``intersection/environment.net.xml``

Run it after any change to how data is read from SUMO.

Usage (from the project root):
    python comparison/fixed_phase_check.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from constants import TL_GREEN_TO_YELLOW, TRAFFIC_LIGHT_ID
from environment import Environment
from episode import EpisodeResult, run_episode
from policy import FixedTimeController
from sumo_api import backend_name, traci

EXPECTED_10_4_DELAY = 18_195


def apply_fixed_plan(green: int, yellow: int) -> None:
    """Set the static program's durations; SUMO then runs it by itself."""
    logic = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)[0]
    yellow_phases = set(TL_GREEN_TO_YELLOW.values())
    for i, phase in enumerate(logic.phases):
        phase.duration = yellow if i in yellow_phases else green
        phase.minDur = phase.maxDur = phase.duration
    traci.trafficlight.setProgramLogic(TRAFFIC_LIGHT_ID, logic)
    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 0)


def make_env(args: argparse.Namespace) -> Environment:
    return Environment(
        n_cars_generated=args.cars,
        max_steps=args.steps,
        yellow_duration=4,
        green_duration=10,
        turn_chance=0.25,
        sumocfg_file=Path("intersection/sumo_config.sumocfg"),
    )


def controller_path(args: argparse.Namespace, green: int) -> dict:
    env = make_env(args)
    try:
        result = run_episode(env, FixedTimeController(green, env.green_duration), seed=args.seed)
    finally:
        env.close()
    return as_dict(result)


def native_path(args: argparse.Namespace, green: int) -> dict:
    env = make_env(args)
    try:
        env.reset(seed=args.seed)
        apply_fixed_plan(green, 4)
        stats = env._simulate(args.steps)  # noqa: SLF001 - same per-second measurement
    finally:
        env.close()
    return as_dict(EpisodeResult(seed=args.seed, max_steps=args.steps, stats=stats))


def as_dict(result: EpisodeResult) -> dict:
    m = result.metrics()
    return {"total_delay": m.total_delay, "arrived": m.arrived,
            "insertion_delay": m.insertion_delay}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cars", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5400)
    args = parser.parse_args()
    defaults = (args.seed, args.cars, args.steps) == (0, 1000, 5400)

    ok = True
    for green in (10, 30):
        ctrl = controller_path(args, green)
        native = native_path(args, green)
        same = ctrl == native
        if green == 10 and defaults:
            same = same and ctrl["total_delay"] == EXPECTED_10_4_DELAY
        ok = ok and same
        print(f"{green} s green / 4 s yellow")
        print(f"  controller path: {ctrl}")
        print(f"  native path:     {native}")
        if green == 10 and defaults:
            print(f"  expected total delay: {EXPECTED_10_4_DELAY}")
        print(f"  {'PASS' if same else 'FAIL'}")

    print(f"SUMO backend: {backend_name()}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
