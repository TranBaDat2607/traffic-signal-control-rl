# Extending the project

This guide is for using the repository as the starting point of a course project:
where each piece lives, how to change it, and what to check so the result holds up.

- [Where things are](#where-things-are)
- [Add a controller](#add-a-controller)
- [Train a different algorithm](#train-a-different-algorithm)
- [Change the reward](#change-the-reward)
- [Change the state](#change-the-state)
- [Change the demand](#change-the-demand)
- [Change the intersection](#change-the-intersection)
- [Checklist for a credible result](#checklist-for-a-credible-result)
- [Project ideas](#project-ideas)

## Where things are

| You want to change… | Edit | Keep fixed |
|---|---|---|
| The decision rule of a baseline | `src/policy/baselines.py` | everything else |
| The learning algorithm | `src/agent/agent.py` (`replay`), or your own module | `environment/`, `episode.py`, `metrics.py` |
| Exploration | `src/policy/epsilon_greedy.py`, `linear_epsilon` in `src/train.py` | |
| The network architecture | `src/agent/model.py` | |
| What the agent observes | `src/environment/state.py`, `STATE_SIZE` in `src/constants.py` | `metrics.py` |
| What the agent is rewarded for | `Environment.step` and `src/environment/reward.py` | `metrics.py` |
| Traffic demand | `src/environment/generator.py`, `n_cars_generated` / `turn_chance` in settings | the evaluation seeds |
| Phase timing | `green_duration`, `yellow_duration` in both settings files | |
| How results are scored | **nothing.** `src/metrics.py` is the yardstick; changing it makes every earlier number incomparable (bump `METRICS_VERSION` if you must) | |

## Add a controller

A controller is any object with two methods:

```python
class MyController:
    def reset(self) -> None:
        """Called before every episode."""

    def choose_action(self, state, info) -> int:
        """Return a green phase in range(4)."""
```

`state` is the 240-dimensional observation. `info` holds what a rule-based
controller may use but a learning agent should not:

- `info["lane_queues"]`: halted vehicles per lane, e.g. `{"N2TL_0": 3, "TL2E_1": 0, ...}`
- `info["phase_lanes"]`: for each action, the incoming and outgoing lanes it gives green to
- `info["sim_step"]`: seconds elapsed in the episode

Evaluate it exactly like the built-in baselines:

```python
from evaluate import evaluate
from settings import load_testing_settings

results = evaluate(MyController(), load_testing_settings(Path("settings/testing_settings.yaml")))
```

To make it available on the command line, add it to `CONTROLLERS` and
`make_controller()` in `src/evaluate.py`, and have `evaluate.py` write
`eval_results.json` so `comparison/compare_results.py` can include it.

## Train a different algorithm

`environment.Environment` is a standard Gymnasium environment
(`Box(0, 1, (240,))` observations, `Discrete(4)` actions; it passes
`gymnasium.utils.env_checker.check_env`). Any algorithm written against
Gymnasium can train on it. For example, with Stable-Baselines3
(`pip install stable-baselines3`, not a dependency of this project):

```python
from stable_baselines3 import PPO

env = Environment(n_cars_generated=1000, max_steps=5400, yellow_duration=4,
                  green_duration=10, turn_chance=0.25,
                  sumocfg_file=Path("intersection/sumo_config.sumocfg"))
model = PPO("MlpPolicy", env, seed=0).learn(total_timesteps=200_000)


class PPOController:
    def reset(self): pass
    def choose_action(self, state, info):
        return int(model.predict(state, deterministic=True)[0])
```

Three things to get right when you bring your own training loop:

1. **Traffic seeds.** `env.reset(seed=s)` decides the vehicles. Train on seeds
   `0 .. N-1` (what `train.py` does) and never on the evaluation seeds
   (10000 and up by default). With `seed=None`, the environment draws seeds from
   its own generator, which can, rarely, hit an evaluation seed.
2. **Truncation is not termination.** Episodes end on a time limit, so
   `step()` returns `truncated=True, terminated=False`. Bootstrap from the last
   state. Treating the time limit as terminal teaches the agent that the world
   ends at t = 5,400 s.
3. **One simulation per process.** The headless backend (libsumo) runs SUMO
   inside the Python process and allows only one simulation at a time. To run
   environments in parallel, use separate processes (e.g. SB3's `SubprocVecEnv`),
   not threads.

The built-in DQN is a useful reference implementation: the whole update is
`Agent.replay` (about 30 lines), and the training loop is `training_session` in
`src/train.py`.

## Change the reward

The reward is computed in `Environment.step`:

```python
current_wait = get_cumulated_waiting_time()
reward = self._previous_wait - current_wait
```

`get_cumulated_waiting_time()` sums the accumulated waiting time of the vehicles
currently on the incoming roads, so the reward is negative while queues build up
and positive when waiting vehicles leave. Alternatives worth comparing: the
negative queue length, the negative pressure, or the throughput.

Whatever reward you train on, **report results with `src/metrics.py`**, not with
the reward. Rewards differ between designs; total delay does not. The sum of
rewards is logged during training as a diagnostic only.

## Change the state

`src/environment/state.py` builds the observation from one subscription read
per step (`vehicle_data()`). To add a feature, extend `get_state()` and update
`STATE_SIZE` in `src/constants.py`; the network input size and the observation
space follow from it. Models trained with a different state size cannot be loaded
into the new one.

Ideas: the current phase as a one-hot vector (the agent cannot currently see it),
the time since the last switch, or a coarser grid.

## Change the demand

Demand comes from `src/environment/generator.py`: `n_cars_generated` vehicles per
episode with Weibull-distributed departure times (a rush-hour peak), going
straight with probability `1 - turn_chance`.

To test generalisation, train with one demand and evaluate with another by
editing a copy of `settings/testing_settings.yaml`
(e.g. `n_cars_generated: 1500`). At high demand, check the saturation guard: if
`insertion_delays` is no longer small compared with `total_delays`, vehicles are
queuing outside the network and total delay is only a lower bound.

## Change the intersection

The network is `intersection/environment.net.xml`. The lane IDs, lane groups and
the state grid in `src/constants.py` describe this network, so a new
intersection means updating `constants.py` too. The phase→lane mapping used by
max-pressure is read from SUMO and needs no change. After any change to how data
is read from SUMO, run `python comparison/fixed_phase_check.py`.

`src/grid/` and `src/grid_train.py` are an experimental multi-agent extension
(a generated N×N grid, one shared network for all junctions). It has its own
environment and is not covered by the evaluation pipeline.

## Checklist for a credible result

Most published comparisons in traffic-signal RL that fail to reproduce fail on
one of these, and so did earlier versions of this repository
(see [`comparison/archive/README.md`](../comparison/archive/README.md)).

- [ ] **Held-out traffic.** Evaluate on seeds never used in training
      (`settings/testing_settings.yaml`).
- [ ] **Greedy policy.** Evaluate with ε = 0 (`evaluate.py` does this for `dqn`).
- [ ] **Same yardstick.** Every controller is scored by `src/metrics.py`, through
      `episode.run_episode`. Never compare rewards across controllers.
- [ ] **Paired comparison.** Same seeds for every controller; report the per-seed
      difference with a confidence interval (`compare_results.py` does this).
- [ ] **Several training seeds.** At least 3 (`train.py --seed N`). Report the
      spread across runs, not only the best run.
- [ ] **Strong baselines.** Beating a 30 s fixed cycle is easy. Report max-pressure
      and the best fixed cycle you can find, not only the default.
- [ ] **No selection on test data.** Choose hyperparameters and checkpoints on
      training seeds (or a separate validation range), not on the evaluation seeds.
- [ ] **Saturation guard.** Check `vehicles_arrived` and `insertion_delays`.
- [ ] **State the budget.** Episodes, wall-clock time, hardware.
- [ ] **Commit the evidence.** Copy the outputs into `results/<run-name>/`
      (layout in [`results/README.md`](../results/README.md)).

## Project ideas

Scoped for a few weeks each. All of them reuse the environment, the baselines and
the evaluation as they are.

- **Algorithmic variants of DQN:** Double DQN (a two-line change in
  `Agent.replay`), dueling heads, n-step returns, prioritized replay. Measure
  sample efficiency: delay on held-out seeds versus training episodes.
- **Policy-gradient methods:** PPO or A2C through Gymnasium, against the same
  baselines.
- **Reward design:** waiting time vs. queue length vs. pressure as reward,
  each scored by total delay.
- **State representation:** what does the agent need? Ablate channels, add the
  current phase, or replace the grid with per-lane counts.
- **Generalisation:** train at one demand level, evaluate at others; or train on a
  mix of demand levels.
- **Robustness:** add noise to the observation (sensor error) and measure how
  performance degrades compared with max-pressure.
- **Action design:** let the agent choose the green duration as well as the phase.
