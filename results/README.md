# Published results

Each subdirectory is one run that the top-level README cites. It is committed so
that every published number and chart can be re-derived from a clean clone
(PROBLEMS.md P0-6).

Scripts never write here by default. Their default outputs (`model/`,
`baseline_results/`, `comparison/comparison.png`) are scratch and are ignored by
git. A run is published by copying its outputs here on purpose.

## Layout of a run directory

| Path | Produced by | What it is |
|---|---|---|
| `README.md` | by hand | The exact commands, the commit, the machine |
| `training_settings.yaml` | `src/train.py` | Settings snapshot of the training run |
| `testing_settings.yaml` | copied by hand | Settings of the greedy evaluation |
| `trained_model.pt` | `src/train.py` | Final weights (state_dict): the policy evaluated |
| `train/plot_*_data.txt` | `src/train.py` | Per-training-episode curves (ε-greedy, training seeds) |
| `test/test_results.json`, `test/plot_*_data.txt` | `src/test.py` | Greedy (ε = 0) evaluation on held-out seeds |
| `baseline*/baseline_results.json` | `comparison/run_baseline.py` | Fixed-time baseline(s) on the *same* held-out seeds |
| `comparison*/comparison.png`, `comparison_summary.json` | `comparison/compare_results.py` | Greedy agent vs. one baseline, paired per seed |
| `logs/` | console output | Full logs of each command |

## Runs

| Run | Summary |
|---|---|
| [`dqn-300ep/`](dqn-300ep/) | 300 training episodes; greedy evaluation on 100 held-out seeds vs fixed-time 30/4 and 10/4 |

`train/plot_reward_data.txt` is a training diagnostic. It is never compared
with the baseline (PROBLEMS.md P0-1).
