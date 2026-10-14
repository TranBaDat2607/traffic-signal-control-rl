# Published results

Each subdirectory is one run that the top-level README cites. It is committed so
that every published number can be checked from a clean clone: the statistics
from the per-episode data (no SUMO needed), the evaluation from the committed
weights (SUMO needed, no training).

Scripts never write here. Their default outputs go to `runs/`, which git
ignores. A run is published by copying its outputs here on purpose.

## Layout of a run directory

| Path | Produced by | What it is |
|---|---|---|
| `README.md` | by hand | The exact commands, the commit, the machine, the wall-clock time |
| `testing_settings.yaml` | copied | The evaluation protocol used |
| `train/<run>/trained_model.pt` | `src/train.py` | Final weights (state_dict): the policy evaluated |
| `train/<run>/training_settings.yaml` | `src/train.py` | The settings actually used, seed included |
| `train/<run>/training_log.csv`, `training_curves.png` | `src/train.py` | One row per training episode (ε-greedy, training seeds) |
| `eval/<label>/eval_results.json` | `src/evaluate.py` | Per-episode metrics on the held-out seeds, one directory per controller |
| `comparison/summary.md`, `comparison_summary.json`, `comparison.png` | `comparison/compare_results.py` | Paired comparison of all controllers |
| `logs/` | console output | Logs of the commands |

`training_log.csv` also contains `neg_reward`, the sum of negative rewards. It
is a training diagnostic and is never compared across controllers.

## Runs

| Run | Summary |
|---|---|
| [`dqn-3seeds/`](dqn-3seeds/) | DQN, 3 training seeds × 300 episodes; greedy evaluation on 100 held-out seeds against fixed-time (30 s, 10 s), max-pressure and random |
| [`dqn-300ep/`](dqn-300ep/) | Earlier single-seed run, evaluated against fixed-time only. Superseded; kept because its numbers were published |
