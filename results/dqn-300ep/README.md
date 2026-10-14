# Run `dqn-300ep`: 300-episode training, greedy evaluation on 100 held-out seeds

> **Superseded by [`../dqn-3seeds/`](../dqn-3seeds/)**, which trains three seeds and
> adds max-pressure and random baselines. This run is kept because its numbers
> were published. It was produced by an earlier version of the scripts
> (`src/test.py` and `comparison/run_baseline.py`, since replaced by
> `src/evaluate.py`); to rerun the commands below exactly, check out commit
> `e792cd1`. Its result files are still read by the current
> `comparison/compare_results.py`, and its weights load with the current code.
>
> Against max-pressure (added later, same 100 seeds), this policy's total delay
> is 10,453 vs 10,669 vehicle-seconds: −2.0% (95% CI −3.0% to −1.1%), lower in
> 64 of 100 episodes.

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

## Re-deriving with the current code

The statistics need neither SUMO nor training (the chart's style has changed
since, so it is not byte-identical):

```bash
python comparison/compare_results.py results/dqn-300ep/baseline results/dqn-300ep/baseline_10-4 \
    results/dqn-300ep/test --out /tmp/check
```

Re-running the evaluation itself (needs SUMO, a few minutes) uses the committed
weights and gives the same per-episode numbers as `test/test_results.json`:

```bash
python src/evaluate.py dqn --model results/dqn-300ep --out /tmp/test_check
```
