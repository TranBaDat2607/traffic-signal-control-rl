# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TLCS is a reference reinforcement-learning project for traffic signal control,
meant as a starting point for RL course projects. A Deep Q-Learning agent controls
a single 4-way intersection in the SUMO traffic simulator, choosing one of 4 green
phases every decision step, and is compared against fixed-time, max-pressure and
random baselines on held-out traffic.

## Commands

All commands must be run from the **project root** so that the relative paths in
`settings/*.yaml` resolve.

```bash
pip install -e ".[dev]"                     # runtime + dev tools

python src/train.py --seed 0 --out runs/train/dqn-seed0      # ~35-90 min, CPU
python src/train.py --episodes 5 --out runs/train/smoke      # quick check
python src/evaluate.py dqn --model runs/train/dqn-seed0      # 100 held-out seeds
python src/evaluate.py fixed_time --green 30                 # also: max_pressure, random
python comparison/compare_results.py runs/eval/fixed_time-30s runs/eval/dqn-seed0

ruff check src tests comparison
mypy src
pytest
python comparison/fixed_phase_check.py      # end-to-end measurement check (needs SUMO)
```

`comparison/fixed_phase_check.py` runs `FixedTimeController` through
`Environment.step()` and SUMO's own static program on seed 0; both must give
exactly 18,195 vehicle-seconds (10 s / 4 s) and 38,747 (30 s / 4 s). Run it after
touching anything that reads from SUMO.

Unit tests (`tests/`) need neither SUMO nor a trained model: metrics, the
statistics, the baselines (with a faked `info`), the DQN update, replay memory,
settings validation, seeding, and the episode loop against a fake environment.

## Architecture

```
src/
├── constants.py          # Network facts (lane IDs, TL phases, state grid), default paths
├── sumo_api.py           # The one SUMO import: libsumo headless, TraCI with GUI
├── metrics.py            # Episode metric definitions — the single yardstick
├── evaluation_stats.py   # Paired differences, t-intervals, across-run spread (pure Python)
├── settings.py           # Pydantic models (extra keys forbidden) + YAML load/save
├── seeding.py            # seed_everything()
├── logger.py, plots.py
├── episode.py            # run_episode(env, controller, seed) — the one loop for everything
├── train.py              # DQN training CLI (online DQN, CSV log, settings snapshot)
├── evaluate.py           # Evaluate any controller on held-out seeds -> eval_results.json
│
├── environment/          # The MDP (owns SUMO), Gymnasium API
│   ├── core.py           # Environment(gym.Env): reset(seed) / step(action); EnvStats, PhaseLanes
│   ├── generator.py      # Weibull route-file generation (one temp file per Environment)
│   ├── subscriptions.py  # subscribe() + bulk readers: one call per step, not per vehicle
│   ├── state.py          # get_state() (240-dim obs), get_lane_queues() (for baselines)
│   └── reward.py         # waiting time (reward), queue length, saturation guard reads
│
├── agent/                # DQN learner
│   ├── agent.py          # Agent: choose_action(), replay() = the Q-learning update
│   ├── model.py          # MLP + Model (online/target/CPU-inference nets, save/load)
│   ├── memory.py         # Replay ring buffer + Sample(s, a, r, s', terminated)
│   └── early_stopping.py
│
├── policy/               # Action selection
│   ├── base.py           # Controller protocol: reset(), choose_action(state, info)
│   ├── baselines.py      # FixedTime, MaxPressure, Random controllers
│   └── epsilon_greedy.py
│
├── grid/, grid_train.py  # Experimental multi-agent N×N grid; own env, not in the eval pipeline
comparison/
├── compare_results.py    # N-way paired comparison -> summary.md, comparison_summary.json, png
├── fixed_phase_check.py  # End-to-end measurement check
└── archive/              # Pre-fix results kept as a worked example of evaluation mistakes
```

**Key design decisions:**
- **One loop, one yardstick.** The agent, every baseline, training and evaluation all
  go through `episode.run_episode()` and are scored by `metrics.py` via
  `EpisodeResult.metrics()`. Do not add another aggregation or a second episode loop.
  `total_delay` is vehicle-seconds; the negative-reward sum is a training
  diagnostic and must never be compared across controllers.
- **Controllers** implement `policy.Controller` (`reset()`, `choose_action(state, info)`).
  Rule-based controllers read `info["lane_queues"]` / `info["phase_lanes"]`; the
  learning agent ignores `info`.
- **Episodes end by truncation** (time limit): `step()` never sets `terminated`;
  the DQN target bootstraps. `Sample.terminated` exists so the update is correct
  in general.
- **Reproducibility:** `TrainingSettings.seed` → `seed_everything()`; training
  traffic seeds are `0..total_episodes-1`, evaluation seeds come from
  `testing_settings.yaml` (10000+). Bit-for-bit repeatable on CPU with the same
  `--threads`. Each `Environment` writes routes to its own temp file and passes it
  with `--route-files`, so runs can go in parallel processes.
- `environment/state.py` and `environment/reward.py` are pure functions over the
  subscriptions registered in `Environment.reset()`.
- Never `import traci` directly: import `traci` from `sumo_api`. libsumo allows one
  simulation per process. Set `TLCS_SUMO_BACKEND=traci` to force the socket
  backend; both must give the same numbers.
- `agent/model.py` saves/loads `state_dict` only. Loading a missing model raises.
- Settings YAML rejects unknown keys; `train.py` writes the effective settings
  (with CLI overrides) to the output directory.

## Runtime Artifacts

Default outputs go to `runs/` (git-ignored). A run the README cites is copied
deliberately into `results/<run-name>/` (layout in `results/README.md`).

| Path | Created by | Notes |
|------|-----------|-------|
| `runs/train/<name>/` | `train.py` | `trained_model.pt`, `best_model.pt`, `training_settings.yaml`, `training_log.csv`, `training_curves.png` |
| `runs/eval/<label>/eval_results.json` | `evaluate.py` | Per-episode metrics on held-out seeds |
| `runs/comparison/` | `compare_results.py` | `summary.md`, `comparison_summary.json`, `comparison.png` |
| `results/<run>/` | copied by hand | Published runs, committed |

## Configuration

`settings/training_settings.yaml` (hyperparameters, `seed`) and
`settings/testing_settings.yaml` (the evaluation protocol). Paths inside are
relative to the project root. SUMO must be installed and `SUMO_HOME` set.
