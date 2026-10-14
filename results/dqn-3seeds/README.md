# Run `dqn-3seeds`: DQN with 3 training seeds vs four baselines

Produced on 2026-10-14 on a laptop (12th Gen Intel Core i5-12450H, 8 cores /
12 threads, CPU only), Python 3.13.15, PyTorch 2.13.0 (CPU), SUMO / libsumo
1.27.1, with the code in the commit that adds this directory. The three training
runs ran in parallel with 2 PyTorch threads each and took 80–93 min each.

## Results

Greedy (ε = 0) evaluation on 100 held-out traffic seeds, 10000–10099. Training
used seeds 0–299. Every controller ran on the same 100 seeds.
[`comparison/summary.md`](comparison/summary.md) has the full table against the
30 s fixed-time plan; [`comparison_vs_max_pressure/summary.md`](comparison_vs_max_pressure/summary.md)
against max-pressure.

| Controller | Total delay (veh-s), mean ± sd over episodes | vs max-pressure, paired [95% CI] | Episodes better than max-pressure |
|---|---|---|---|
| Random | 40,511 ± 6,248 | | |
| Fixed-time 30 s / 4 s | 39,823 ± 1,170 | | |
| Fixed-time 10 s / 4 s | 17,863 ± 729 | | |
| Max-pressure | 10,669 ± 574 | — | — |
| DQN, seed 0 | 10,522 ± 636 | −147 [−254, −40] (−1.4%) | 59/100 |
| DQN, seed 1 | 10,563 ± 641 | −106 [−207, −5] (−1.0%) | 60/100 |
| DQN, seed 2 | 10,536 ± 647 | −133 [−253, −13] (−1.2%) | 58/100 |
| **DQN, 3 runs** | **10,540** (95% CI across runs 10,489 to 10,592) | **−129 [−181, −77] (−1.2%)** | |

Saturation guard: every controller averaged 998.5–998.7 of 1,000 vehicles
arrived and 32.4–32.6 vehicle-seconds of insertion delay, so total delay is
complete for all of them.

Training curves (`train/*/training_log.csv`): mean training-episode delay fell
from 33,600–34,200 (episodes 1–50) to 11,450–11,580 (episodes 251–300) and was
still falling at episode 300.

## Commands (run from the project root)

```bash
for s in 0 1 2; do python src/train.py --seed $s --threads 2 --out runs/train/dqn-seed$s & done; wait
for s in 0 1 2; do python src/evaluate.py dqn --model runs/train/dqn-seed$s; done
python src/evaluate.py fixed_time --green 30
python src/evaluate.py fixed_time --green 10
python src/evaluate.py max_pressure
python src/evaluate.py random
```

The outputs were then copied here, and the comparisons generated from the copies:

```bash
R=results/dqn-3seeds
python comparison/compare_results.py $R/eval/fixed_time-30s $R/eval/fixed_time-10s $R/eval/max_pressure \
    $R/eval/random $R/eval/dqn-seed0 $R/eval/dqn-seed1 $R/eval/dqn-seed2 --out $R/comparison
python comparison/compare_results.py $R/eval/max_pressure $R/eval/dqn-seed0 $R/eval/dqn-seed1 \
    $R/eval/dqn-seed2 --out $R/comparison_vs_max_pressure
```

Those two commands regenerate the tables and charts from a clean clone, without
SUMO. To re-run an evaluation from the committed weights (needs SUMO, a few
minutes each):

```bash
python src/evaluate.py dqn --model results/dqn-3seeds/train/dqn-seed0 --out /tmp/check
```

Re-running training with the same seed and `--threads 2` on a CPU should give the
same weights. That property was checked on short runs (two 3-episode runs with the same thread count and
seed, side by side, gave identical weights), not by repeating these
80-minute runs.

## Contents

| Path | What it is |
|---|---|
| `testing_settings.yaml` | The evaluation protocol |
| `train/dqn-seed{0,1,2}/` | Final weights, settings used, per-episode training log and curves |
| `eval/<label>/eval_results.json` | Per-episode metrics of each controller on the 100 held-out seeds |
| `comparison/` | All controllers, paired against fixed-time 30 s |
| `comparison_vs_max_pressure/` | DQN runs paired against max-pressure |
| `logs/` | Console logs of training and evaluation |
