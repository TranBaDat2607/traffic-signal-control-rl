Evaluation on 100 held-out traffic seeds (10000 to 10099). Mean ± standard deviation over episodes. Δ = paired difference in total delay versus `fixed_time-30s`, with 95% CI.

| Controller | Total delay (veh-s) | Avg queue (veh) | Δ total delay vs reference | Δ % | Episodes better | Arrived | Insertion delay (veh-s) |
|---|---|---|---|---|---|---|---|
| `random` | 40,511 ± 6,248 | 7.50 ± 1.16 | +687 [-564, +1,939] | +1.7 | 56/100 | 998.6 ± 0.7 | 32.4 ± 7.6 |
| `fixed_time-30s` (reference) | 39,823 ± 1,170 | 7.37 ± 0.22 | — | — | — | 998.5 ± 0.8 | 32.4 ± 7.5 |
| `fixed_time-10s` | 17,863 ± 729 | 3.31 ± 0.14 | -21,960 [-22,206, -21,715] | -55.1 | 100/100 | 998.5 ± 0.7 | 32.5 ± 7.4 |
| `max_pressure` | 10,669 ± 574 | 1.97 ± 0.11 | -29,154 [-29,413, -28,895] | -73.2 | 100/100 | 998.6 ± 0.7 | 32.5 ± 7.5 |
| `dqn-seed1` | 10,563 ± 641 | 1.96 ± 0.12 | -29,260 [-29,522, -28,999] | -73.5 | 100/100 | 998.7 ± 0.6 | 32.6 ± 7.5 |
| `dqn-seed2` | 10,536 ± 647 | 1.95 ± 0.12 | -29,287 [-29,545, -29,029] | -73.5 | 100/100 | 998.6 ± 0.7 | 32.4 ± 7.4 |
| `dqn-seed0` | 10,522 ± 636 | 1.95 ± 0.12 | -29,301 [-29,558, -29,044] | -73.6 | 100/100 | 998.7 ± 0.6 | 32.6 ± 7.5 |

**`dqn`, 3 training runs:** mean total delay 10,540 veh-s (95% CI across runs 10,489 to 10,592); Δ vs reference -29,283 [-29,335, -29,231] veh-s (-73.5%). Per-run means: 10,522, 10,563, 10,536.
