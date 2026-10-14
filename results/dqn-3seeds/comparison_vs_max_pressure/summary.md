Evaluation on 100 held-out traffic seeds (10000 to 10099). Mean ± standard deviation over episodes. Δ = paired difference in total delay versus `max_pressure`, with 95% CI.

| Controller | Total delay (veh-s) | Avg queue (veh) | Δ total delay vs reference | Δ % | Episodes better | Arrived | Insertion delay (veh-s) |
|---|---|---|---|---|---|---|---|
| `max_pressure` (reference) | 10,669 ± 574 | 1.97 ± 0.11 | — | — | — | 998.6 ± 0.7 | 32.5 ± 7.5 |
| `dqn-seed1` | 10,563 ± 641 | 1.96 ± 0.12 | -106 [-207, -5] | -1.0 | 60/100 | 998.7 ± 0.6 | 32.6 ± 7.5 |
| `dqn-seed2` | 10,536 ± 647 | 1.95 ± 0.12 | -133 [-253, -13] | -1.2 | 58/100 | 998.6 ± 0.7 | 32.4 ± 7.4 |
| `dqn-seed0` | 10,522 ± 636 | 1.95 ± 0.12 | -147 [-254, -40] | -1.4 | 59/100 | 998.7 ± 0.6 | 32.6 ± 7.5 |

**`dqn`, 3 training runs:** mean total delay 10,540 veh-s (95% CI across runs 10,489 to 10,592); Δ vs reference -129 [-181, -77] veh-s (-1.2%). Per-run means: 10,522, 10,563, 10,536.
