"""The single-intersection MDP, exposed through the Gymnasium API.

One ``step()`` is one decision of the traffic light:

1. If the chosen green phase differs from the current one, the light shows the
   current phase's yellow for ``yellow_duration`` simulated seconds.
2. The chosen green phase then runs for ``green_duration`` seconds.
3. The reward is the decrease in accumulated waiting time of the vehicles on
   the incoming roads over that interval (``reward.get_cumulated_waiting_time``).

So a decision step lasts 10 s (same phase) or 14 s (phase change) with the
default settings. An episode ends after ``max_steps`` simulated seconds. That is
a time limit, not a terminal state of the traffic, so ``step()`` reports it as
``truncated=True`` and never sets ``terminated``. A learner should keep
bootstrapping from the last state.

Every simulated second is measured, not only the decision points: ``info["stats"]``
holds one :class:`EnvStats` per second, and ``metrics.py`` turns them into the
reported numbers. Rule-based controllers get what they need from
``info["lane_queues"]`` and ``info["phase_lanes"]``. The learning agent only sees
the observation.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from sumolib import checkBinary

from constants import (
    ACTION_TO_TL_PHASE,
    NUM_ACTIONS,
    STATE_SIZE,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)
from environment.generator import generate_routefile
from environment.reward import (
    get_arrived_number,
    get_cumulated_waiting_time,
    get_pending_number,
    get_queue_length,
)
from environment.state import get_lane_queues, get_state
from environment.subscriptions import subscribe
from sumo_api import select_backend, traci


@dataclass
class EnvStats:
    """Snapshot of environment statistics for a single simulation step.

    Attributes:
        queue_length: Halted vehicles on the incoming edges.
        arrived: Vehicles that left the network during this step.
        pending: Vehicles past their departure time that SUMO could not insert yet.
    """

    queue_length: int
    arrived: int = 0
    pending: int = 0


@dataclass(frozen=True)
class PhaseLanes:
    """Lanes that have a green light in one of the agent's green phases.

    Read from SUMO's traffic-light program, not hard-coded, so it always matches
    the network file.

    Attributes:
        incoming: Lanes entering the junction with a green (or yielding green) signal.
        outgoing: Lanes those green movements lead to.
    """

    incoming: frozenset[str]
    outgoing: frozenset[str]


class Environment(gym.Env[NDArray[np.float32], np.int64]):
    """A SUMO-simulated 4-way intersection as a Gymnasium environment.

    Observation: ``Box(0, 1, (240,), float32)``, see ``environment/state.py``.
    Action: ``Discrete(4)``, an index into ``ACTION_TO_TL_PHASE``.

    Usage::

        env = Environment(...)
        obs, info = env.reset(seed=0)     # seed = which traffic demand
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()

    Each instance writes its routes to its own temporary file, so several
    experiments can run side by side in separate processes.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 - Gymnasium convention

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        sumocfg_file: Path,
        gui: bool = False,
    ) -> None:
        """Initialize the environment. SUMO is started by :meth:`reset`.

        Args:
            n_cars_generated: Number of cars to generate for the episode.
            max_steps: Episode length in simulated seconds.
            yellow_duration: Seconds of yellow inserted when the phase changes.
            green_duration: Seconds a chosen green phase runs per decision.
            turn_chance: Probability for each car to turn instead of going straight.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to use the SUMO GUI binary.
        """
        super().__init__()
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.sumocfg_file = sumocfg_file
        self.gui = gui

        self.observation_space = spaces.Box(0.0, 1.0, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.sim_step = 0
        self.phase_lanes: dict[int, PhaseLanes] = {}
        self._previous_wait = 0.0
        self._routes_file: Path | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002 - Gymnasium signature
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Generate the traffic for a new episode and start SUMO.

        Args:
            seed: Seed of the traffic demand. The same seed gives the same
                vehicles, departure times and routes. ``None`` draws one from
                the environment's random generator.
            options: Unused; accepted for Gymnasium compatibility.

        Returns:
            The initial observation and an info dict (see :meth:`step`).
        """
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**31 - 1))

        if self._routes_file is None:
            fd, name = tempfile.mkstemp(prefix="tlcs_routes_", suffix=".rou.xml")
            os.close(fd)
            self._routes_file = Path(name)
        generate_routefile(
            path=self._routes_file,
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
        )

        if traci.isLoaded():
            traci.close()
        select_backend(gui=self.gui)
        traci.start(self._build_sumo_cmd())
        subscribe()
        self.phase_lanes = _read_phase_lanes()

        self.sim_step = 0
        self._previous_wait = 0.0
        return get_state(), self._info([])

    def step(
        self, action: int | np.int64
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one decision and advance the simulation until the next one.

        Args:
            action: Index of the green phase to show next.

        Returns:
            ``(observation, reward, terminated, truncated, info)``. ``info`` has:

            - ``"stats"``: one :class:`EnvStats` per simulated second of this step,
            - ``"lane_queues"``: halted vehicles per lane, at the end of the step,
            - ``"phase_lanes"``: ``{action: PhaseLanes}``, from the signal program,
            - ``"sim_step"``: simulated seconds elapsed in the episode.
        """
        if not traci.isLoaded():
            msg = "Call reset() before step()."
            raise RuntimeError(msg)

        stats = self._execute(int(action))

        current_wait = get_cumulated_waiting_time()
        reward = self._previous_wait - current_wait
        self._previous_wait = current_wait

        truncated = self.sim_step >= self.max_steps
        return get_state(), reward, False, truncated, self._info(stats)

    def close(self) -> None:
        """Stop SUMO and delete this environment's route file."""
        if traci.isLoaded():
            traci.close()
        if self._routes_file is not None:
            self._routes_file.unlink(missing_ok=True)
            self._routes_file = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line.

        Raises:
            FileNotFoundError: If the SUMO config file does not exist.
        """
        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        cmd = [
            checkBinary("sumo-gui" if self.gui else "sumo"),
            "-c",
            str(self.sumocfg_file),
            # Overrides the route file named in the .sumocfg.
            "--route-files",
            str(self._routes_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]
        if self.gui:
            # Run without waiting for the play button, and exit when TraCI
            # closes; otherwise traci.close() blocks until the window is closed.
            cmd += ["--start", "--quit-on-end"]
        return cmd

    def _info(self, stats: list[EnvStats]) -> dict[str, Any]:
        return {
            "stats": stats,
            "lane_queues": get_lane_queues(),
            "phase_lanes": dict(self.phase_lanes),
            "sim_step": self.sim_step,
        }

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the simulation, measuring every step.

        The number of steps is capped so as not to exceed ``max_steps``.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            One :class:`EnvStats` per simulated step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.sim_step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.sim_step += 1
            stats.append(
                EnvStats(
                    queue_length=get_queue_length(),
                    arrived=get_arrived_number(),
                    pending=get_pending_number(),
                )
            )

        return stats

    def _execute(self, action: int) -> list[EnvStats]:
        """Show the chosen green phase, preceded by yellow if the phase changes.

        Args:
            action: Index of the green phase in ``ACTION_TO_TL_PHASE``.

        Returns:
            One :class:`EnvStats` per simulated step.
        """
        next_green_phase = ACTION_TO_TL_PHASE[action]
        current_green_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            stats.extend(self._simulate(self.yellow_duration))

        if self.sim_step >= self.max_steps:
            return stats

        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        stats.extend(self._simulate(self.green_duration))

        return stats


def _read_phase_lanes() -> dict[int, PhaseLanes]:
    """Read which lanes each action gives green to, from the loaded program."""
    links = traci.trafficlight.getControlledLinks(TRAFFIC_LIGHT_ID)
    logic = traci.trafficlight.getAllProgramLogics(TRAFFIC_LIGHT_ID)[0]

    phase_lanes = {}
    for action, phase_index in ACTION_TO_TL_PHASE.items():
        state = logic.phases[phase_index].state
        green = [
            link
            for signal, link_group in zip(state, links, strict=True)
            if signal in "Gg"
            for link in link_group
        ]
        phase_lanes[action] = PhaseLanes(
            incoming=frozenset(link[0] for link in green),
            outgoing=frozenset(link[1] for link in green),
        )
    return phase_lanes
