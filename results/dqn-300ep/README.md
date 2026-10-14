# Run `dqn-300ep`: 300-episode training, greedy evaluation on 100 held-out seeds

Produced on 2026-10-14 on a 12-core CPU-only machine (PyTorch 2.13 CPU, SUMO /
libsumo 1.27.1, Python 3.13), from commit `e792cd1` (`main` after PR #24).
Training took 35 min 10 s.

## Commands (run from the project root, one at a time)

```bash
python src/train.py --settings settings/training_settings.yaml --out model/dqn-300ep
python src/test.py --model model/dqn-300ep
python comparison/run_baseline.py --config settings/testing_settings.yaml --out baseline_results
python comparison/run_baseline.py --config settings/testing_settings.yaml --out baseline_results_10-4 --green 10 --yellow 4
python comparison/compare_results.py --baseline baseline_results --rl model/dqn-300ep/test --out comparison_output
python comparison/compare_results.py --baseline baseline_results_10-4 --rl model/dqn-300ep/test --out comparison_output_10-4
```

The outputs were then copied here. `logs/` holds each command's console output.

## Contents

| Path | What it is |
|---|---|
| `training_settings.yaml` | Snapshot written by `train.py`; identical to `settings/training_settings.yaml` at that commit |
| `testing_settings.yaml` | Evaluation settings: 100 episodes, seeds 10000–10099, greedy |
| `trained_model.pt` | Final weights after episode 300, the policy evaluated. Not `best_model.pt`, which is selected on training seeds and is not published |
| `train/plot_*_data.txt` | One value per training episode (ε-greedy, seeds 0–299). `plot_reward_data.txt` is a training diagnostic only |
| `test/` | `test.py` output: greedy (ε = 0) on seeds 10000–10099 |
| `baseline/` | Fixed-time 30 s green / 4 s yellow (the net-file plan) on the same seeds |
| `baseline_10-4/` | Fixed-time 10 s green / 4 s yellow (the agent's own phase lengths) on the same seeds |
| `comparison/`, `comparison_10-4/` | Chart and paired statistics: DQN vs each baseline |

## Re-deriving from a clean clone

The charts and statistics need neither SUMO nor training:

```bash
python comparison/compare_results.py --baseline results/dqn-300ep/baseline --rl results/dqn-300ep/test --out /tmp/check
cmp /tmp/check/comparison.png results/dqn-300ep/comparison/comparison.png
```

Re-running the evaluation itself (needs SUMO, about 2–3 minutes) uses the
committed weights:

```bash
python src/test.py --model results/dqn-300ep --settings results/dqn-300ep/testing_settings.yaml --out /tmp/test_check
```
