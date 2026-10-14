# Known Problems — TLCS

Audit of the repository as of commit `46c64b8` (branch `main`).

This document exists to be worked through and deleted. Each entry is self-contained:
**Where** (file:line), **Evidence** (what was actually observed), **Why it's wrong**,
**Fix**, and **Done when** (the check that closes it).

Severity:

- **P0** — the published results are not defensible. Must be resolved before any
  README claims a number.
- **P1** — the repository does not run or reproduce as documented.
- **P2** — documentation or claims that overstate what the code does.
- **P3** — performance. The code is correct but slower than it needs to be.

---

## Summary

| ID | Sev | Area | Problem |
|----|-----|------|---------|
| ~~[P0-1](#p0-1)~~ | P0 | Metrics | ~~Reward panel compares two different reward definitions~~ **FIXED** |
| ~~[P0-2](#p0-2)~~ | P0 | Metrics | ~~Waiting-time panel compares seconds against queue-steps~~ **FIXED** |
| ~~[P0-3](#p0-3)~~ | P0 | Metrics | ~~`compare_results.py` silently substitutes a third unit~~ **FIXED** |
| ~~[P0-4](#p0-4)~~ | P0 | Experiment | ~~The `du100` baseline is a strawman (100-second yellow lights)~~ **FIXED** (`166e967`) |
| [P0-5](#p0-5) | P0 | Experiment | Both charts reuse the same 10-episode run; no converged policy |
| [P0-6](#p0-6) | P0 | Repro | The RL data behind the charts is not in the repository |
| [P0-7](#p0-7) | P0 | Result | On the only comparable metric, the agent loses to fixed-time |
| ~~[P1-1](#p1-1)~~ | P1 | Bug | ~~`run_baseline.py` cannot import `constants` — broken `sys.path`~~ **FIXED** (`396aa9a`) |
| [P1-2](#p1-2) | P1 | Docs | `HUONG_DAN_CHAY.md` commands are stale since files moved |
| [P1-3](#p1-3) | P1 | Packaging | `pyproject.toml`: dead entrypoint, unused dep, upstream author |
| ~~[P1-4](#p1-4)~~ | P1 | Hygiene | ~~`.gitignore` misses the paths the code actually writes to~~ **FIXED** |
| ~~[P1-5](#p1-5)~~ | P1 | Bug | ~~`early_stopping_min_episode` is silently ignored by `train.py`~~ **FIXED** |
| [P1-6](#p1-6) | P1 | Legal | MIT declared in `pyproject.toml`, no `LICENSE` file |
| [P2-1](#p2-1) | P2 | Claims | `get_neighbor_aware_state()` is dead code |
| [P2-2](#p2-2) | P2 | Claims | Three multi-agent modes documented, one implemented |
| [P2-3](#p2-3) | P2 | Cosmetic | Repository name is misspelled (`trafic`) |
| ~~[P1-7](#p1-7)~~ | P1 | Method | ~~Model selection uses the metric now demoted to a diagnostic~~ **FIXED** |
| ~~[P2-4](#p2-4)~~ | P2 | Metrics | ~~`total_delay` understates delay when vehicles cannot be inserted~~ **FIXED** |
| ~~[P3-1](#p3-1)~~ | P3 | Speed | ~~Training time is dominated by per-vehicle TraCI round-trips, not SUMO or the network~~ **FIXED** |

## Suggested order

P0-1, P0-2, P0-3 shared one root cause and were fixed as a single change (a
shared metric definition both sides import — `src/metrics.py`). P0-4 and P0-5
are re-runs that depended on that fix landing first. P0-6 and P0-7 resolve
themselves once the re-run is committed.

```
P1-1 ──► P0-1 ─┐
P1-4 ──► P0-2 ─┼──► P0-4 ──► P0-5 ──► P0-6 ──► P0-7
         P0-3 ─┘
  DONE     DONE      DONE      ▲
                               └── next
```

**Next: P0-5.** The metric is trustworthy and the baseline signal plan is
realistic (P0-4, `166e967`). Nothing can be published until there is a trained
policy to evaluate.

P1-2, P1-3, P1-5, P1-6, P1-7 and all P2 items are independent and can be done
any time.

P3-1 is independent of everything else, but P0-5 needs long training runs. Doing
P3-1 first makes P0-5, and every re-run after it, cheaper. (Done: a training
episode went from ~18–22 s to ~6–8 s.)

---

<a name="p0-1"></a>
## P0-1 — Reward panel compares two different reward definitions

**Where**

- `src/train.py:51-52` (RL side)
- `comparison/run_baseline.py:78-79` (baseline side)
- `comparison/compare_results.py:98-103` (the panel that pairs them)

**Evidence**

The RL series keeps only the negative rewards:

```python
# src/train.py:51
sum_neg_reward = sum(record.reward for record in episode_history if record.reward < 0)
```

The baseline series keeps every reward:

```python
# comparison/run_baseline.py:78-79
reward             = old_wait - current_wait
cumulative_reward += reward
```

`compare_results.py` then joins them on the key name `cumulative_rewards` and
draws both on one axis.

Measured consequence: baseline mean cumulative reward over the 10 `du10`
episodes is **0.0**. That is not a well-performing controller — `reward` is a
telescoping difference, so summing all of it over an episode collapses to
`-final_wait`, and the final wait is ~0 once the network drains. The RL curve,
summing only negatives, cannot ever reach 0 by construction.

**Why it's wrong**

The two numbers answer different questions. A reader comparing them concludes
the baseline earns ~0 reward and the agent earns −29,000, i.e. the agent is
catastrophically worse. Neither reading is supported: the quantities are not
commensurable, and one of them (the telescoping sum) is not a meaningful
performance measure at all.

**Fix**

Define the episode reward once, in a module both entry points import, and use
the same definition on both sides. Prefer a measure that does not telescope —
total accumulated waiting seconds over the episode is the natural choice, with
the negative-only sum kept (if at all) as a separate, clearly-named training
diagnostic rather than a benchmark metric.

**Done when**

Running baseline and agent on identical seeds with an identical policy (e.g. a
fixed-phase agent) produces the same reward number from both code paths, to
within floating-point noise.

> ### RESOLVED
>
> `src/metrics.py` now defines every reported episode metric, and `train.py`,
> `test.py`, `grid_train.py` and `run_baseline.py` all import it.
>
> **The reward panel is gone rather than repaired.** A fixed-time controller has
> no reward function; "cumulative reward" is an artifact of the RL formulation,
> not a measure of controller quality. The comparison figure now has two panels,
> both measured by the same function on both sides. `sum_negative_rewards()`
> survives as a training-convergence diagnostic in the training output only,
> labelled as such, and `compare_results.py` no longer reads
> `plot_reward_data.txt` at all — the number is structurally unable to reach a
> chart.
>
> **The "Done when" above was therefore met in a different form.** There is no
> longer a shared reward number to compare. The equivalent gate is the same
> *total delay* from both code paths under an identical policy. That was run: a
> round-robin fixed-phase agent through `Environment.execute()` versus SUMO-native
> static tlLogic on a matched 10 s / 4 s plan, seed 0, 5400 steps —
> **both paths returned exactly 18,195 vehicle-seconds, a gap of 0.** Integer
> sums of vehicle counts, so this is exact equality, not a tolerance.
>
> Also removed: the duplicate `get_cumulated_waiting_time()` / `get_queue_length()`
> copies in `run_baseline.py:36-50`, which shadowed `src/environment/reward.py`.
> The baseline now imports the real ones.

---

<a name="p0-2"></a>
## P0-2 — Waiting-time panel compares seconds against queue-steps

**Where**

- `src/train.py:54-58` — the value is computed
- `src/train.py:155-158` — it is labelled `"Cumulative delay (s)"`
- `comparison/run_baseline.py:77-80` — the baseline value, genuinely in seconds
- `comparison/compare_results.py:69-70, 106-111` — where they are joined and
  plotted as `"Total wait (seconds)"`

**Evidence**

The RL series is a sum of halting-vehicle *counts*:

```python
# src/train.py:54,58
sum_queue_length = sum(stats.queue_length for stats in env_stats)
training_stats["cumulative_wait"].append(sum_queue_length)
```

`stats.queue_length` comes from `traci.edge.getLastStepHaltingNumber` — a
vehicle count. Summed over 5400 steps this yields **vehicle-steps**, not
seconds. It is nonetheless written to `plot_delay_data.txt` under the axis label
`"Cumulative delay (s)"`.

The baseline series is genuinely in seconds:

```python
# comparison/run_baseline.py:77,80
current_wait     = get_cumulated_waiting_time()   # accumulated seconds
cumulative_wait += current_wait
```

`compare_results.py:70` loads `plot_delay_data.txt` into `cumulative_waits` and
`:108` labels the shared axis `"Total wait (seconds)"`.

Note that `src/test.py:75` gets this right — it calls the same quantity
`"Cumulative wait (queue-steps)"`. `train.py` is the outlier.

Measured consequence: the `+89.9%` (du10) and `+100.0%` (du100) improvement
annotations both come from this panel. Baseline `du10` mean is 304,029 seconds;
the RL value it is compared against is ≈30,707 queue-steps. The ratio is an
artifact of the unit mismatch.

**Why it's wrong**

This is the panel that produces the headline improvement figure. The figure is
a unit conversion error, not a result.

**Fix**

Two changes:

1. Rename the series to what it is (`cumulative_queue_steps`) and fix the axis
   label in `src/train.py:156-158`, matching the wording already used in
   `src/test.py`.
2. Record actual waiting seconds as a *separate* series — call
   `env.get_cumulated_waiting_time()` inside the episode loop and accumulate it
   the same way `run_baseline.py:80` does — and compare that against the
   baseline.

**Done when**

`plot_delay_data.txt` and `baseline_wait_data.txt` hold the same unit, verified
by running both paths with a fixed-phase policy and getting comparable
magnitudes.

> ### RESOLVED — but NOT by the fix prescribed above
>
> **Step 2 of that Fix is wrong and was deliberately not followed.** It says to
> record accumulated waiting seconds in `train.py` "the same way
> `run_baseline.py:80` does". That formula double-counts: it sums a quantity
> that is *already* an accumulation once per step, so a vehicle waiting 100 s
> contributes about 5,050 rather than 100. Measured on the committed data,
> episode 0 of `du10` records `cumulative_wait = 316,843` where the true delay
> is `avg_queue 3.369 × 5400 = 18,195` vehicle-seconds — **17× inflated**.
> Copying it into `train.py` would have spread the defect rather than fixed it.
>
> The claim above that "the baseline series is genuinely in seconds" is the one
> factual error in this audit entry.
>
> **What was done instead:** the shared metric is `total_delay` — vehicle-seconds
> of delay, `Σ over simulation steps of halting-vehicle count`. This is the
> quantity `train.py:54` *already computed correctly*; only its name
> (`cumulative_wait`) and its axis label (`"Cumulative delay (s)"`) were wrong.
> The baseline was changed to match it, not the other way round. Step 1 of the
> Fix stands and was applied: the series is renamed and relabelled, and the name
> `cumulative_wait` no longer exists anywhere in `src/` or `comparison/`.
>
> Verified: both paths return the same integer under an identical policy (see
> P0-1's resolution note).

---

<a name="p0-3"></a>
## P0-3 — `compare_results.py` silently substitutes a third unit

**Where** `comparison/compare_results.py:72-74`

**Evidence**

```python
# Nếu không có delay, dùng reward làm proxy
if "cumulative_waits" not in result and "cumulative_rewards" in result:
    result["cumulative_waits"] = [-r for r in result["cumulative_rewards"]]
```

**Why it's wrong**

If `plot_delay_data.txt` is missing, the script plots negated reward on an axis
labelled `"Total wait (seconds)"` — a third distinct quantity on the same axis —
and prints nothing to say it did so. A chart generated this way is
indistinguishable from a valid one.

**Fix**

Remove the fallback. If a required series is absent, raise with the path that
was expected. A missing input should stop the run, not change the meaning of
the output.

**Done when** `compare_results.py` exits non-zero with a clear message when
`plot_delay_data.txt` is absent.

> ### RESOLVED
>
> The fallback is deleted. So is the sibling fallback in `load_baseline()`
> (`:42-51`), which silently reconstructed a result dict from `.txt` files of
> unverified provenance — the same defect wearing a different hat.
>
> `compare_results.py` now exits 1, naming the expected path, in three cases,
> each verified by running it:
>
> - a required RL series is missing → names `plot_delay_data.txt` and says which
>   flag to point where;
> - `baseline_results.json` is missing → names it and gives the command to
>   generate it;
> - the baseline JSON's `metrics_version` does not match the current build →
>   says its numbers use a different definition of waiting time and must be
>   regenerated. This is what makes the archived pre-fix results fail loudly
>   instead of quietly producing a plausible wrong chart.

---

<a name="p0-4"></a>
## P0-4 — The `du100` baseline is a strawman

**Where** `intersection/environment.net.xml:127-136`

**Evidence**

All eight phases in the committed net file carry `duration="100"`:

```xml
<tlLogic id="TL" type="static" programID="0" offset="0">
    <phase duration="100" state="GGGGrrrrrrGGGGrrrrrr"/>
    <phase duration="100" state="yyyyrrrrrryyyyrrrrrr"/>   <!-- 100-second yellow -->
    ...
```

Phases at index 1, 3, 5, 7 are the yellow phases. A 100-second yellow light is
not a traffic engineering choice; it is a broken signal plan. It is also the
state currently committed on `main`.

Baseline means under this plan: mean queue **57.90** vehicles, mean wait
**75,506,007** s. Under the `du10` plan (all phases at 10s): mean queue **3.26**,
mean wait **304,029** s. An 18× difference produced entirely by the baseline
configuration.

**Why it's wrong**

`comparison/archive/comparison_du100/comparisondu100.png` reports `RL +90.2%` on queue
length against this signal plan. The agent is being credited for beating a
controller that holds yellow for 100 seconds. Any reviewer who opens the net
file will discount the entire results section.

**Fix**

Set the net file to a defensible fixed-time plan — roughly 30s green / 4s
yellow — matched to what the agent itself controls (`green_duration: 10`,
`yellow_duration: 4` in `settings/training_settings.yaml`). If a
degraded-baseline comparison is genuinely interesting, keep it as a clearly
labelled secondary experiment; never as the headline number.

**Done when** the committed net file has realistic phase durations, and no
published chart compares against the 100s plan without saying so in its caption.

> ### RESOLVED (`166e967`, PR #15)
>
> `intersection/environment.net.xml` now runs 30 s green / 4 s yellow on all
> four phases; only the `duration` attributes changed, so the phase indices in
> `src/constants.py` still line up:
>
> ```
> $ grep -n 'phase duration' intersection/environment.net.xml
> 128:        <phase duration="30" state="GGGGrrrrrrGGGGrrrrrr"/>
> 129:        <phase duration="4" state="yyyyrrrrrryyyyrrrrrr"/>
> ...
> 135:        <phase duration="4" state="rrrrrrrrryrrrrrrrrry"/>
> ```
>
> The agent is unaffected: `Environment.execute()` sets the phase itself every
> 10 s / 4 s, so the net file's durations never get to fire during training or
> testing. They are the baseline's whole behaviour.
>
> The one chart drawn against the 100 s plan,
> `comparison/archive/comparison_du100/comparisondu100.png`, is archived, and
> `comparison/archive/README.md` says that it used a 100-second yellow light.
> `compare_results.py` refuses to load that data (it has no `metrics_version`).

---

<a name="p0-5"></a>
## P0-5 — Both charts reuse the same 10-episode run

**Where**

- `settings/training_settings.yaml:3` — `total_episodes: 10`
- `src/train.py:84` — `new_epsilon = round(1.0 - (episode / tot_episodes), 2)`
- `comparison/archive/comparison_du10/`, `comparison/archive/comparison_du100/`

**Evidence**

The two comparison directories differ only in their baseline. Solving the
annotation percentages back to the RL values gives the same run in both:

| Metric | du10 baseline | du10 annotation | implied RL | du100 baseline | du100 annotation | implied RL |
|---|---|---|---|---|---|---|
| Avg queue | 3.26 | −73.5% | **5.66** | 57.90 | +90.2% | **5.67** |
| Wait series | 304,029 | +89.9% | **30,707** | 75,506,007 | +100.0% | **≈30,700** |

Identical RL values, two different baselines.

That run is 10 episodes with a linear schedule `ε = 1 − episode/10`. Episode 0
is fully random; training ends at the moment ε first reaches 0. There is no
exploitation phase.

`compare_results.py` additionally smooths with a 10-episode moving average over
a 10-point series, so every legend entry reading `MA10` is a single averaged
point, not a trend.

`comparison/HUONG_DAN_CHAY.md` already acknowledges this: *"AI mới chỉ ở bước
trải nghiệm ngẫu nhiên"* — the agent is still in random exploration.

**Why it's wrong**

There is no trained policy to report on. Whatever the charts show is the
behaviour of a partially-random controller.

**Fix**

Raise `total_episodes` to at least 200 for the single intersection (the grid
config at `settings/grid_training_settings.yaml:3` already uses 2000 with early
stopping), re-run, and evaluate with `src/test.py` at ε=0.

Note that `settings/testing_settings.yaml:9` uses `episode_seed: 10000`, which
is disjoint from the training seeds `0..N` — that is a proper held-out
evaluation set and is worth stating explicitly in the README.

**Done when** the reported numbers come from a greedy (ε=0) evaluation on
held-out seeds, with the episode count stated alongside them.

---

<a name="p0-6"></a>
## P0-6 — The RL data behind the charts is not in the repository

**Where** `model/` (absent), `.gitignore:10`

**Evidence**

`src/train.py` writes `plot_reward_data.txt`, `plot_delay_data.txt` and
`plot_queue_data.txt` into the output directory, which defaults to `model/`
(`src/constants.py:8`). No `model/` directory exists in the working tree or in
git. `.gitignore:10` ignores `models/` — plural — so the real output directory
was never ignored either; the files were simply never committed.

The committed baseline data (`baseline_results.json` and the three `.txt` files)
is present for both runs. Only the RL half is missing.

**Why it's wrong**

Neither a reader nor you can regenerate, verify, or extend the published
charts. The PNGs are the only surviving record.

**Fix**

After the P0-5 re-run, commit the RL `plot_*_data.txt` files and the settings
snapshot alongside the baseline data, in a directory named for the run.

**Done when** `compare_results.py` can be re-run from a clean clone and
reproduce the committed PNG.

---

<a name="p0-7"></a>
## P0-7 — On the only comparable metric, the agent loses

**Where** `comparison/archive/comparison_du10/comparison_du10.png`, panel 3

**Evidence**

Average queue length is the one metric computed identically on both sides —
`get_queue_length()` in `src/environment/reward.py:28-34` and
`comparison/run_baseline.py:45-50` are the same function over the same edges,
in the same unit.

Against the realistic `du10` baseline: baseline **3.26** vehicles, agent
**≈5.66**. The chart's own annotation reads `RL −73.5%`. The agent produces
queues roughly 74% longer than a plain fixed-time signal.

**Why it's wrong**

Nothing is wrong with the *measurement* — this one is trustworthy. What is
wrong is publishing the `du100` chart's `+90.2%` as the headline while this
number exists in the same repository.

**Fix**

This is a consequence of P0-5, not an independent defect. After a proper
training run, re-measure. Then report whatever the number is: if the agent wins,
the win is now credible because the baseline is fair; if it does not, report it
with the training budget and note it as the current state. A repository that
states an honest negative result reads as competent. One that states `+90%`
against a 100-second yellow light does not.

**Done when** the README's results section reports the queue metric from the
re-run, against the realistic baseline, whatever its sign.

---

<a name="p1-1"></a>
## P1-1 — `run_baseline.py` cannot import `constants`

**Where** `comparison/run_baseline.py:25`

**Evidence**

```python
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

`__file__` is `comparison/run_baseline.py`, so this resolves to
`comparison/src`, which does not exist. The imports on lines 27-28
(`from constants import INCOMING_EDGES`, `from environment.generator import
generate_routefile`) therefore fail unless `src/` happens to already be on the
path from elsewhere.

The line was correct when the script lived at the repository root; it was not
updated when the script moved into `comparison/` in commit `fd45054`.

**Fix**

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

**Done when** `python comparison/run_baseline.py --help` prints usage from a
clean checkout with SUMO installed.

---

<a name="p1-2"></a>
## P1-2 — `HUONG_DAN_CHAY.md` commands are stale

**Where** `comparison/HUONG_DAN_CHAY.md`, steps 1 and 3

**Evidence**

The document instructs, from the project root:

```bash
python run_baseline.py --config settings/training_settings.yaml --out baseline_results
python compare_results.py --baseline baseline_results --rl model/run-01 --out comparison_output
```

Both scripts are in `comparison/`. Neither path resolves from the root.

**Fix**

Update the paths to `comparison/run_baseline.py` and
`comparison/compare_results.py`. While editing: the document is the only run
guide in the repository and is Vietnamese-only — consider an English version, or
folding the pipeline into the README, for a GitHub audience.

**Done when** every command in the file runs as written from the project root.

---

<a name="p1-3"></a>
## P1-3 — `pyproject.toml`: dead entrypoint, unused dependency, upstream author

**Where** `pyproject.toml:7`, `:18`, `:35-36`

**Evidence**

```toml
authors = [{ name = "Andrea Vidali", email = "andreavidali.me@gmail.com" }]   # line 7
"typer>=0.20.0",                                                              # line 18
[project.scripts]                                                             # line 35
tlcs = "cli:app"                                                              # line 36
```

- Line 7 still credits the upstream TLCS author. This is your repository and
  will be read as your work.
- Line 36 points at `cli:app`. There is no `src/cli.py` — `pip install -e .`
  installs a `tlcs` command that raises `ModuleNotFoundError` on invocation.
- `typer` is declared but never imported anywhere in `src/`. Both entry points
  use `argparse`.

**Fix**

Set `authors` to yourself and credit Andrea Vidali's original TLCS in the README
and `LICENSE` instead. Either write the `cli.py` that `[project.scripts]`
promises (which would justify keeping `typer`), or remove both the entrypoint
and the dependency.

**Done when** `pip install -e . && tlcs --help` either works or the command no
longer exists.

---

<a name="p1-4"></a>
## P1-4 — `.gitignore` misses the paths the code actually writes to

**Where** `.gitignore:9-11`

**Evidence**

```
# Training outputs
models/
plots/
```

The code writes to `model/` (singular — `src/constants.py:7`), not `models/`.
It also writes:

- `intersection_grid/` — generated net, sumocfg and routes
  (`settings/grid_training_settings.yaml:34-36`)
- `intersection/episode_routes.rou.xml` — regenerated every episode, and
  **currently tracked in git** despite `CLAUDE.md` stating "Do not commit"
- `comparison/*_results*/` — baseline outputs

`plots/` matches nothing the code produces.

**Fix**

Replace the stale entries with the real ones. Then
`git rm --cached intersection/episode_routes.rou.xml`.

Note the interaction with [P0-6](#p0-6): the RL result `.txt` files under
`model/` *should* be committed for the published run. Either commit them to a
separate `results/` directory that is not ignored, or add a negation rule.
Decide this deliberately rather than letting the ignore rule decide.

**Done when** `git status` is clean after a full train + baseline + compare
cycle, except for files intentionally kept.

> ### RESOLVED
>
> **Decision: default outputs are scratch; published runs are copied on purpose
> into `results/<run-name>/`.** Results are not kept in `model/` with a negation
> rule. A training run overwrites `model/`, and a published number should not be
> one `train.py` invocation away from being silently replaced. `results/README.md`
> documents the layout of a run directory.
>
> - `.gitignore` now ignores `model/`, `**/episode_routes.rou.xml` (also catches
>   the stray copy created when a script is run from `src/`), `intersection_grid/`,
>   `/baseline_results*/`, `/comparison_output*/`, `/comparison/comparison.png`,
>   `*.prof` and `.claude/settings.local.json`. The stale `models/`, `plots/` and
>   `comparison/comparison_du*/` lines are gone, and so is the meaningless `!results/`.
> - `intersection/episode_routes.rou.xml` was untracked in `9d95e12`.
> - `test.py` used to print its numbers and write nothing. It now writes
>   `test_results.json` (with the seeds and `epsilon`) and `plot_*_data.txt` into
>   `<model>/test/` (the previously unused `DEFAULT_TEST_FOLDER`). It uses the
>   same filenames as `train.py`, so `compare_results.py --rl model/test` works.
>
> **Verified:** ran train → test → baseline → compare, every script at its
> default output path, then `git status --short --untracked-files=all`. It showed
> only the two source files edited by this change (`.gitignore`, `src/test.py`),
> no outputs.

---

<a name="p1-5"></a>
## P1-5 — `early_stopping_min_episode` is silently ignored by `train.py`

**Where** `src/settings.py:36`, `src/train.py:70,126`, `src/grid_train.py:186`

**Evidence**

`TrainingSettings` declares the field:

```python
early_stopping_min_episode: NonNegativeInt = 0  # don't count patience before this episode
```

`grid_train.py:186` honours it (`if ep_num >= settings.early_stopping_min_episode`).
`train.py:70` constructs `EarlyStopping(patience=settings.early_stopping_patience)`
and never reads the field, so `train.py:126` can stop during the exploration
phase — exactly what the setting exists to prevent.

The field is inherited by `GridTrainingSettings`, so it is accepted without
complaint in `settings/training_settings.yaml`, where it would do nothing.

**Why it matters**

With the linear ε schedule, early episodes are dominated by random actions and
reward barely improves. A patience counter that runs during that phase will fire
spuriously. This becomes load-bearing as soon as [P0-5](#p0-5) raises the
episode count and early stopping is switched on.

**Fix**

Apply the same guard in `train.py` as in `grid_train.py:186`, or move the
minimum-episode logic into `EarlyStopping` itself so both callers get it.

**Done when** a training run with `early_stopping_patience: 30` and
`early_stopping_min_episode: 200` does not stop before episode 200.

> ### RESOLVED
>
> The guard moved into `EarlyStopping` (`src/agent/model.py`) as a
> `min_episode` argument, and both `train.py` and `grid_train.py` pass
> `settings.early_stopping_min_episode` to it. `grid_train.py`'s own
> `if ep_num >= ...` guard is gone, so the two callers cannot drift apart again.
> Episodes below `min_episode` neither count toward patience nor set a best
> value, so an exploration-phase episode can no longer be saved as
> `best_model.pt`.
>
> Also fixed while there: `EarlyStopping.improved` was defined as
> `no_improve_count == 0`, which is `True` *before the first call*. It is now
> an explicit flag.
>
> **Unit test:** `tests/test_early_stopping.py::test_does_not_stop_before_min_episode`
> feeds 1,000 non-improving episodes with `patience=30, min_episode=200`. The first
> stop is at episode **230**: episode 200 sets the first best, and 30 more
> episodes exhaust patience. `pytest -q`: 17 passed.
>
> **Scaled-down real run** (`train.py`, 300 steps, 60 cars, `patience: 2`,
> `early_stopping_min_episode: 6`, 12 episodes, same seeds):
>
> | | stopped after | first `best_model.pt` |
> |---|---|---|
> | before (old `train.py`) | episode **3**, inside the warm-up window | episode 1 |
> | after | episode **11** | episode 6 |

---

<a name="p1-6"></a>
## P1-6 — MIT declared, no `LICENSE` file

**Where** `pyproject.toml:6`, repository root

**Evidence** `license = { text = "MIT" }` is declared; no `LICENSE` file exists.

**Why it matters**

The project derives from Andrea Vidali's TLCS. Redistribution without a license
file and without stating the provenance is both a legal gap and an attribution
gap.

**Fix**

Add an MIT `LICENSE`, and state the derivation from the original TLCS project in
both the `LICENSE` and the README credits.

**Done when** `LICENSE` exists and names both copyright holders.

---

<a name="p2-1"></a>
## P2-1 — `get_neighbor_aware_state()` is dead code

**Where** `src/grid/state.py:76-101`

**Evidence**

The function builds a 1200-element vector (the junction's own 240-element state
plus its four neighbours'). `grep -rn "get_neighbor_aware_state" --include=*.py`
returns only the definition — there is no call site.

`src/grid/coordinator.py:56` constructs the shared model with
`input_dim=STATE_SIZE`, i.e. 240. The model could not consume the 1200-element
vector without an architecture change.

**Why it matters**

This is the code that would justify a claim of "neighbour-aware coordination".
Since it is unreachable, what is implemented is shared-parameter *independent*
DQN — each junction observes only itself. The README must not claim otherwise.

**Fix**

Either wire it up (which requires `input_dim=STATE_SIZE * 5` and a retrain), or
delete it. Do not leave it as the basis for a claim.

**Done when** the function is either called or gone.

---

<a name="p2-2"></a>
## P2-2 — Three multi-agent modes documented, one implemented

**Where** `src/grid/coordinator.py:3, 25`, `notebook/improvements.ipynb`
cell 21

**Evidence**

The coordinator docstring describes SharedParameters mode and refers to "other
modes"; `Agent.__init__` in `src/agent/agent.py:30` documents the `model`
parameter as "used for SharedParameters multi-agent mode", implying
alternatives. The notebook renders a three-panel figure captioned
*Independent* / *SharedParameters* / a third mode.

Only SharedParameters exists in the code. There is no mode selector, and
`MultiAgentCoordinator.__init__` unconditionally builds one shared `Model` and
hands the same object to every agent.

**Fix**

Reduce the docstrings and the notebook figure to what is implemented, or
implement the modes. The notebook figure is the more visible of the two — it
looks like an architecture diagram of the system.

**Done when** no document in the repository describes a mode that cannot be
selected.

---

<a name="p2-3"></a>
## P2-3 — Repository name is misspelled

**Where** `https://github.com/TranBaDat2607/trafic-signal-reinforcement-learning`

**Evidence** `trafic` — one `f`.

**Fix**

Rename on GitHub; the old URL keeps redirecting, so existing clones and links
continue to work. Update the `origin` remote afterwards.

**Done when** the canonical URL reads `traffic-`.

---

<a name="p1-7"></a>
## P1-7 — Model selection uses the metric now demoted to a diagnostic

**Where** `src/train.py:137,144-147`, `src/grid_train.py:187,196-199`

**Evidence**

`EarlyStopping` is driven by the negative-reward sum, and `best_model.pt` is
saved whenever that number improves:

```python
if early_stopping.step(training_stats["neg_reward"][-1]):
...
if early_stopping.improved:
    agent.model.save_weights(out_path / "best_model.pt")
```

**Why it matters**

[P0-1](#p0-1) established that the negative-reward sum is a training diagnostic,
not a measure of controller quality — it is sampled per *decision* step, so it
is not even comparable between two runs with different `green_duration`, and a
fixed-time controller has no equivalent. The repository will now *report*
`total_delay` while *selecting the published checkpoint* by a different function
of the episode. A reviewer who accepts the P0-1 framing will immediately ask why.

This is currently latent: `settings/training_settings.yaml:25` has
`early_stopping_patience: 0`, so nothing fires. It becomes load-bearing the
moment [P0-5](#p0-5) raises the episode count and early stopping is switched on.

**Fix**

Drive `EarlyStopping` on `-total_delay` so selection matches reporting. It is a
two-line change but it alters training behaviour, so it belongs in its own
commit with a before/after run — not folded into a metrics refactor.

**Done when** the quantity that selects `best_model.pt` is the quantity the
README reports, or a comment at the call site states why it deliberately is not.

> ### RESOLVED
>
> `train.py` and `grid_train.py` now call `early_stopping.step(-total_delay)`,
> with a comment at each call site. `EarlyStopping` maximises, which is why the
> value is negated. The log lines say "best total delay" instead of "best reward",
> and the grid's per-episode log line now prints the delay as well.
>
> **Before/after run.** Same scaled config both times: 12 episodes, 600 steps,
> 150 cars, `patience: 4`, `early_stopping_min_episode: 3`, seeds 0–11. Training
> is not seeded, so the two runs follow different trajectories. What matters is
> which episode each rule selects:
>
> | | rule | `best_model.pt` ends at | lowest-delay monitored episode |
> |---|---|---|---|
> | before | max neg-reward | episode 9 (delay 2,652) | episode **8** (delay 2,524) — **not** selected |
> | after | min total delay | episode 9 (delay 3,378) | episode 9 — selected |
>
> Before the change, the published checkpoint was not the lowest-delay one.
>
> **Caveat that remains:** every training episode uses a different seed (and so
> different demand), so "lowest training-episode delay" partly reflects an easy
> seed, not only a better policy. That is why P0-5 reports a separate greedy
> evaluation on held-out seeds and does not use the training-curve minimum.

---

<a name="p2-4"></a>
## P2-4 — `total_delay` understates delay under saturation

**Where** `src/metrics.py`, `src/environment/reward.py:28-34`

**Evidence**

`get_queue_length()` counts halted vehicles on the four incoming edges. A
vehicle SUMO has not yet inserted — because the approach is full — is on no edge
and contributes zero.

**Why it matters**

At the `du10` operating point (mean queue 3.26 on 750 m approaches) this is
irrelevant. At the `du100` point (mean queue 57.90) insertion backlog is likely,
so the metric understates delay exactly where congestion is worst, and
non-linearly: a saturated controller looks *better* than it is. Any comparison
that spans both regimes is affected.

**Fix**

Record a saturation guard alongside the metrics — accumulate
`traci.simulation.getArrivedNumber()` over the episode, or read
`traci.simulation.getMinExpectedNumber()` at episode end — and write it into
`baseline_results.json` and the training log. If fewer vehicles arrived than
`n_cars_generated`, the episode did not drain and the delay figure is a lower
bound. Keep the TraCI call at the call sites so `metrics.py` stays TraCI-free.

**Done when** every published delay figure is accompanied by the arrived-vehicle
count, or a statement that the episode drained.

> ### RESOLVED
>
> **The defect is real, and arrivals alone cannot detect it.** Measured on seed 0,
> 5,400 steps, with the 30 s / 4 s baseline plan:
>
> | cars | `total_delay` | arrived | max pending | pending veh-s |
> |---|---|---|---|---|
> | 1,000 | 38,747 | 998 | 2 | 32 |
> | 4,000 | 1,096,215 | 3,998 | 4 | 669 |
> | 8,000 | 4,441,453 | 5,476 | 3,153 | **9,650,775** |
>
> At 8,000 cars, vehicles waiting outside the full approaches accumulate more
> than twice the delay that `total_delay` sees. The arrived-vehicle count alone
> is a weak guard here: `generator.py` schedules the last departure at exactly
> `max_steps`, so even a healthy episode ends with a vehicle or two in transit
> (998 of 1,000).
>
> **What was added** (`src/metrics.py`, `METRICS_VERSION` 2 → 3):
>
> - `vehicles_arrived()` — Σ `traci.simulation.getArrivedNumber()` per step.
> - `insertion_delay()` — Σ `len(traci.simulation.getPendingVehicles())` per step:
>   vehicle-seconds spent waiting to enter the network, in the same unit as
>   `total_delay` and **not included in it**. It is a separate column rather than
>   folded into `total_delay`, so the P0-1 equivalence still holds.
>
> The TraCI reads are at the call sites (`environment/reward.py`, recorded per
> step in `EnvStats`; `run_baseline.py`), so `metrics.py` stays TraCI-free. Both
> numbers go into `baseline_results.json` (`vehicles_arrived`,
> `insertion_delays`), the training log and `plot_arrived_data.txt` /
> `plot_insertion_delay_data.txt`, and the `test.py` summary. They are also in
> the `compare_results.py` table and caption, which requires them on both sides.
>
> **Verified:** `python comparison/fixed_phase_check.py` (new; it re-runs the
> P0-1 check) gives `total_delay` **18,195** on both paths, plus
> `arrived 998, insertion_delay 32` on both. `pytest -q`: 20 passed.
> A 2-episode baseline → train → compare cycle shows the guard values in the
> table and in the chart caption.
>
> The grid entry point (`grid_train.py`) does not record the guard. It
> publishes no comparison figure.

---

<a name="p3-1"></a>
## P3-1 — Training time is dominated by TraCI round-trips

**Where**

- `src/environment/state.py:70-88` — `get_state()`
- `src/environment/reward.py:18-22` — `get_cumulated_waiting_time()`
- `src/environment/reward.py:34` and `src/environment/core.py:176` —
  `get_queue_length()`, called on every simulation second
- `import traci` in `src/environment/{core,state,reward}.py`,
  `src/grid/{grid_env,state,reward}.py` and `comparison/run_baseline.py`

**Evidence**

Profiled on 2026-10-14 on the development machine: 12 CPU cores, no GPU,
CPU-only PyTorch 2.13, SUMO 1.27.1. The run was 3 episodes of
`settings/training_settings.yaml` (1000 cars, 5400 steps) under `cProfile`,
54 s in total:

| Component | Time | Share |
|---|---|---|
| `get_state()` | 19.6 s | 36.5% |
| `get_cumulated_waiting_time()` | 10.2 s | 19.1% |
| `get_queue_length()` (16,200 calls) | 5.4 s | 10.1% |
| `traci.simulationStep()` (SUMO itself) | ~6.6 s | ~12% |
| `agent.replay()` (all training) | 7.9 s | 14.7% |
| `traci.start()` (SUMO startup, ×3) | 3.0 s | 5.6% |
| `choose_action()` | 0.4 s | 0.7% |

Socket `send`/`recv` alone account for **24.1 s (45%)**. About 65% of the total
is spent reading vehicle data over TraCI. SUMO's own simulation is about 12%.

Every TraCI getter is one blocking socket round-trip. `get_state()` makes up to
4 calls per vehicle (`getLaneID`, `getLanePosition`, `getSpeed`,
`getWaitingTime`). Then `get_cumulated_waiting_time()` loops over the same
vehicles again with 2 more calls each (`getRoadID`,
`getAccumulatedWaitingTime`). `get_queue_length()` makes 4 more calls on every
simulation second, not only at each decision.

Caveat on the training share: `memory_size_min: 600` is not reached until
after episode 2 (about 437 samples per episode), so only episode 3 ran gradient
steps (464 of 1,311 `replay()` calls). In steady state the network's share is
higher than 14.7%: one episode of training took 7.9 s, against about 15 s of
environment time per episode on average. The per-episode split was not measured
directly. Even so, the environment is still the larger cost.

**Why it matters**

On this machine one episode takes about 18 s, so the ≥200-episode run that
[P0-5](#p0-5) requires takes about an hour, and every re-run or extra seed
costs the same again. The grid config (4 junctions, 2000 cars, up to 2000
episodes) makes the same per-vehicle calls per junction. It was not profiled,
but it should be affected at least as badly.

A GPU would not fix this. The network is a 4×400 MLP with batch 100. Training
is at most about a third of an episode, and for a network this size the cost
of moving work to and from a GPU would cancel most of its speedup.

**Fix**, in order of payoff per effort:

1. **Use `libsumo` instead of TraCI for headless runs.** It runs SUMO inside the
   Python process with the same API, so there are no socket round-trips:
   `pip install libsumo==1.27.1` (it must match the `eclipse-sumo` version),
   then `import libsumo as traci`. It cannot drive `sumo-gui` and allows one
   simulation per process. So select it only when `gui: false`, e.g. through
   one shared import shim that every file listed under **Where** uses. Check
   that traci-module helpers such as `traci.isLoaded()`
   (`src/environment/core.py:89`, `src/grid/grid_env.py:92`,
   `comparison/run_baseline.py:46`) exist in libsumo. The grid's
   `ProcessPoolExecutor` workers already run one simulation per process, so
   they are compatible.
2. **Read everything in one call per decision.** Use a context subscription on
   the junction (`traci.junction.subscribeContext` with `LANE_ID`,
   `LANE_POSITION`, `SPEED`, `WAITING_TIME`, `ACCUMULATED_WAITING_TIME` and
   `ROAD_ID`). Then `get_state()` and `get_cumulated_waiting_time()` both
   compute from the one result instead of each looping over vehicles. This
   helps with or without libsumo.
3. **Collect experience from several SUMO processes in parallel.** Single-
   intersection training uses one core out of 12. Run N worker processes, each
   running its own episode, feeding one replay memory. `src/grid_train.py:137`
   already uses this pattern. DQN learns off-policy, so experience from a
   slightly older policy is acceptable.
4. **Cap PyTorch threads** (`torch.set_num_threads(2)` or similar). The
   profiled run used 357% CPU, mostly PyTorch threads busy-waiting on a tiny
   network. This matters once step 3 runs several processes, so that they do
   not oversubscribe the cores.

Changes 1 and 2 must not change any number. They change how data is read, not
what is read.

**Done when**

- The fixed-phase check from [P0-1](#p0-1)'s resolution (round-robin policy,
  seed 0, 5400 steps) still returns exactly **18,195** vehicle-seconds, and
  `get_state()` returns an identical array at every decision step before and
  after the change.
- Re-running the profile below shows TraCI and socket time no longer
  dominating, with the per-episode wall time before and after recorded in this
  entry.

> ### RESOLVED — fixes 1, 2 and 4; fix 3 deliberately not done
>
> **1. libsumo.** `src/sumo_api.py` is the only place that imports SUMO's API.
> Every file listed under **Where** now does `from sumo_api import traci`.
> `select_backend(gui)` picks `libsumo` for headless runs and TraCI for
> `sumo-gui`, and `TLCS_SUMO_BACKEND=traci` forces TraCI. `traci.isLoaded()`
> exists in libsumo, so no call site needed a workaround. `libsumo` is now a
> dependency in `pyproject.toml`.
>
> **2. Subscriptions.** `src/environment/subscriptions.py` registers one junction
> context subscription (lane, position, speed, waiting time, accumulated
> waiting time, road), one edge subscription for halting numbers, and one
> simulation subscription (arrived, pending). `get_state()`,
> `get_cumulated_waiting_time()`, `get_queue_length()` and the P2-4 guard read
> those results instead of calling one getter per vehicle. The readers raise if
> nothing was subscribed, so they cannot silently return empty data. `get_state()`
> still iterates in `getIDList()` order so that its per-cell float sums are
> accumulated in exactly the old order.
>
> **4. PyTorch threads** capped at 4 (`TORCH_NUM_THREADS` in `src/constants.py`,
> set by `train.py`, `test.py`, `grid_train.py`). One `replay()` step, 4×400 MLP,
> batch 100, on this 12-core machine: 21.7 ms at 1 thread, 13.5 ms at 2,
> **9.3 ms at 4**, and 13.9 ms at 12 (the PyTorch default here is 8).
>
> **3. Parallel experience collection: not done.** Unlike 1, 2 and 4, it changes
> training dynamics (experience from an older policy, a different ordering of
> replay updates), and nothing else needs it now that a 200-episode run takes
> well under an hour.
>
> **Done-when, checked:**
>
> - `python comparison/fixed_phase_check.py` → RL path and native path both
>   **18,195** vehicle-seconds (PASS) under libsumo, and also under
>   `TLCS_SUMO_BACKEND=traci` with subscriptions.
> - `get_state()` identical at every decision step: the 386 per-decision state
>   arrays of that run, dumped before and after the change, are
>   `np.array_equal` (max abs diff 0.0). The reward signal is identical too: the
>   386 values of `get_cumulated_waiting_time()` hash the same before and after,
>   on both backends.
>
> **Per-episode wall time** (`train.py`, `settings/training_settings.yaml`, 1,000
> cars, 5,400 steps; seconds per `run_episode()` including interleaved
> `replay()`; replay starts in episode 3 once memory reaches 600 samples;
> training is unseeded, so episode lengths vary a little between runs):
>
> | episode | 1 | 2 | 3 | 4 | 5 |
> |---|---|---|---|---|---|
> | before (TraCI, per-vehicle getters) | 4.62 | 5.75 | 14.49 | 18.14 | 22.11 |
> | libsumo + subscriptions | 1.33 | 1.63 | 8.07 | 13.55 | 13.47 |
> | + 4 PyTorch threads (as committed) | 1.46 | 1.33 | 5.54 | 6.06 | 7.50 |
>
> **Profile** (the verification command below, 3 episodes, cProfile total):
>
> | Component | Before | After |
> |---|---|---|
> | total | 48.5 s | 19.2 s |
> | `get_state()` | 10.26 s | 0.48 s |
> | `get_cumulated_waiting_time()` | 5.59 s | 0.17 s |
> | `get_queue_length()` (16,200 calls) | 5.54 s | 0.15 s |
> | socket `send`/`recv` | 11.31 s | 0 |
> | `simulationStep()` (SUMO itself) | 5.37 s | 5.34 s |
> | `agent.replay()` | 7.84 s | 4.99 s |
> | SUMO start (×3) | 3.03 s | 0.02 s |
>
> Reading vehicle data went from ~21 s to ~0.8 s. What remains is SUMO's own
> simulation and the network update, which is the correct floor for this design.
>
> **Correction to the evidence above:** "the environment is still the larger
> cost" was true of the profiled run, but only because 2 of its 3 episodes ran
> no gradient steps. In steady state (episodes 3+ in the table), `replay()`
> was already the larger cost before this change: episodes 1–2 took ~5 s, and
> episodes 3–5 took 14–22 s. That is why fix 4 was worth more than the entry
> predicted.
>
> **Found while verifying, and fixed:** `grid_train.py` hung forever on Linux,
> on `main` too, before any of this change. `ProcessPoolExecutor` used the
> default `fork` start method, and a forked worker inherits PyTorch's
> initialised thread pool and deadlocks in its first forward pass. The pool now
> uses `spawn`; `parallel_worker.py` was already written for spawned workers.
> A 2-episode, 2-worker grid run now completes with libsumo in every worker.
>
> **Trade-off:** under the TraCI backend (GUI, or forced), subscriptions are
> *slower* than before: 5.7 s vs 4.1 s for the fixed-phase episode, because SUMO
> pushes every vehicle's data on every simulation step rather than at decisions
> only. The headless default is libsumo (1.2 s), so this only affects GUI runs,
> where speed is not the point.

---

## Verification commands

```bash
# P0-1/P0-2: confirm there is exactly one aggregation, and the old names are gone
grep -rn "cumulative_wait\|cumulative_reward\|sum_neg_reward" src/ comparison/ \
  --exclude-dir=archive                                                           # expect no hits
grep -rln "from metrics import" src/ comparison/                                  # expect 5 files
pytest -q                                                                          # metric definitions

# P0-3: a missing series must stop the run, not change the chart's meaning
python comparison/compare_results.py --baseline /nonexistent --rl model/   # expect exit 1

# P0-4: confirm the signal plan is realistic
grep -n 'phase duration' intersection/environment.net.xml

# P0-6 / P1-4: confirm outputs land where intended
git status --short

# P1-1: confirm the baseline script imports
python comparison/run_baseline.py --help

# P2-1: confirm no dead neighbour-state code
grep -rn "get_neighbor_aware_state" --include=*.py .

# P3-1: profile 3 training episodes (from the project root), then inspect train.prof
sed 's/^total_episodes: .*/total_episodes: 3/' settings/training_settings.yaml > /tmp/prof.yaml
python -m cProfile -o train.prof src/train.py --settings /tmp/prof.yaml --out /tmp/prof_model
python -c "import pstats; pstats.Stats('train.prof').sort_stats('cumulative').print_stats(25)"
```

## Provenance

The metric defects were introduced across two commits by two authors, neither
in isolation at fault — the mismatch only exists where they meet.

| Commit | Author | Date | Introduced |
|---|---|---|---|
| `b4dd3e1` *finish phase 6* | nammovuivui | 2026-03-16 | `src/train.py:51-58` (negatives-only reward; queue-steps named `cumulative_wait`), `src/train.py:155-158` (`"Cumulative delay (s)"` label) |
| `fd45054` *Add comparison results* | Twitchhh3105 | 2026-03-18 | all of `comparison/` — `run_baseline.py` (independent reward and wait definitions), `compare_results.py` (joins the series by key name and plots them on shared axes), both PNGs |

`b4dd3e1` created the ambiguity: a queue-step count carrying a seconds label,
internally consistent because nothing compared it to anything. `fd45054` created
the defect: a second implementation of the same metrics, joined to the first on
label rather than definition.

The structural fix is a definitions contract — one module defining episode
reward and episode wait, imported by `train.py` and `run_baseline.py` alike, so
a future divergence is a type error rather than a chart.
