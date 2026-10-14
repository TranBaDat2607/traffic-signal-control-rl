# Future Work — TLCS

Direction for turning this repository into an open-source research and
experimentation platform for traffic signal control (TSC), aimed at undergraduate
and master's students.

Researched 2026-10-14. Figures quoted below are as reported by their sources;
where a number is self-reported or unverified, that is stated next to it.

This document is about *direction*. Concrete defects in the current code live in
[PROBLEMS.md](PROBLEMS.md), and the roadmap below depends on them.

---

## Summary

The goal is sound; the original route to it is not.

The original idea — a user uploads a bird's-eye image of an intersection, sets
parameters, and an agent learns signal timings — has a research-level computer
vision problem at its core. It also cannot obtain the inputs that matter most from
an image, and its output (a live RL controller) is the form cities rarely deploy.

The recommended direction is an **intersection study kit**: OpenStreetMap +
turning-movement counts + the current signal plan go in; a fair comparison of
controllers and **recommended timing plans** come out.

---

## 1. Evaluation of the original idea

### 1.1 Technical feasibility

The idea is three problems of very different difficulty:

| Step | Difficulty | Why |
|---|---|---|
| Image → lane-level network with turn connections | **Research-level** | Active research area. A 2026 method that also uses an OSM prior reports F1 ≈ 0.88 for approach-lane counts, 0.87 for turn movements, but only 0.63 for lane types. One wrong connection invalidates the simulation. |
| Image → demand and current signal plan | **Not possible from one image** | A still image shows neither turning volumes per hour nor signal timings. Demand determines the result far more than geometry does. |
| Network + demand → trained agent | Feasible | What this repository already does — for one hard-coded intersection. |

The geometry step is also unnecessary: SUMO's `osmWebWizard.py` and `netconvert`
already build networks from OpenStreetMap. OSM networks often need manual fixes in
`netedit`, but that is far cheaper than the vision problem.

**Blocker in the current code:** `src/constants.py` hard-codes lane IDs, lane
geometry and the 4 TL phases. Any version that accepts arbitrary intersections
must first derive the state and action space from the network itself (sumo-rl
does this). This is needed regardless of which direction is chosen.

### 1.2 Real-world impact

- **RL controlling signals live is rarely deployed.** Recent surveys name the
  sim-to-real gap (sensing, action execution, traffic dynamics, objective
  mismatch) as the main obstacle. Sim2Signal (Sept 2026) found that direct
  transfer "consistently degrades performance" across all four gap sources.
- **What is deployed recommends timings.** Google's Project Green Light analyses
  data and *recommends* timing changes that city engineers apply to existing
  controllers, with no new hardware. Boston retimed 100+ intersections (114 per
  Google), and the city reports ~20% fewer unnecessary stops. Google's "up to
  30%" is an upper bound; no independent peer-reviewed evaluation was found.

Lesson: the adoptable output is **evidence-backed timing plans**, not an agent
wired into the cabinet.

### 1.3 Usefulness for students and researchers

The space is crowded, and a GUI is not what is missing:

- **sumo-rl** — Gymnasium/PettingZoo interface for SUMO, MIT, ships RESCO
  scenarios.
- **RESCO** (NeurIPS 2021) — benchmark on realistic networks.
- **LibSignal** (NeurIPS 2022) — cross-simulator library (SUMO and CityFlow).

What researchers lack is reproducibility, strong baselines and honest evaluation.
RESCO's authors found methods claiming state-of-the-art "often struggled" in
realistic scenarios, and RL-vs-max-pressure results depend heavily on how well the
baseline is tuned. This repository has already lived through that failure mode
(PROBLEMS.md P0-4: 100-second yellow baseline; P0-7: the agent loses to fixed-time
on queue length).

---

## 2. Recommended direction: an intersection study kit

### 2.1 Input

- An intersection selected from **OpenStreetMap** (`osmWebWizard.py` /
  `netconvert`), with manual fixes in `netedit` where needed.
- A **turning-movement count table** — the data traffic engineers already
  collect. SUMO's `routeSampler` converts counts to routes. It has known weaknesses
  with sparse detector data (a Munich study found it unsatisfactory; a 2025
  Mannheim paper improved it with CMA-ES) — itself a workable student topic.
- The **current signal plan**, so the status quo is the first baseline.

### 2.2 Controllers, all scored by `src/metrics.py`

| Controller | Role |
|---|---|
| Fixed-time, Webster's method | The plan a traffic engineer would compute |
| Actuated | What most modern cabinets already run |
| Max-pressure | The strong heuristic RL must beat |
| DQN / PPO | Research subject, and an upper-bound estimate for adaptive control |

### 2.3 Output

- A comparison report: multiple seeds, held-out demand, confidence intervals, and
  a "did the episode drain" guard (PROBLEMS.md P2-4).
- **Recommended fixed-time plans per time-of-day period** — deployable without new
  hardware.

### 2.4 Interoperate, don't compete

- Expose a Gymnasium/PettingZoo API compatible with sumo-rl, so students can bring
  their own algorithms and use this repository's evaluation protocol.
- Load RESCO scenarios, so results are comparable with the literature.

---

## 3. A distinctive niche

Pick one so the project is not "another sumo-rl".

**Recommended: motorcycle-dominated mixed traffic (Vietnam / Southeast Asia).**

- No work was found combining RL signal control, SUMO's sublane model (lane
  filtering) and Vietnamese traffic. The closest is a non-peer-reviewed DQN study in
  Bandung, Indonesia with dedicated two-wheeler lanes, which does not model
  filtering.
- Data is obtainable: open drone datasets (DRIFT, CitySim) and YOLO + ByteTrack
  pipelines already exist for extracting trajectories and turning counts from
  video.

**Alternative: robustness / sim-to-real stress testing** — evaluating controllers
under sensing noise, demand shift and calibration error. Sim2Signal shows growing
interest, and its code is open.

**Where the bird's-eye idea fits:** later, as an optional module for
**drone video → turning counts** (feasible with existing tools) — not
**image → network** (research-level).

---

## 4. Roadmap

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
credible    generalise  study-kit   niche
result      + baselines I/O + docs
```

**Phase 1 — A credible single-intersection result.**
Close PROBLEMS.md P0-4 → P0-7. Until one intersection produces an honest,
reproducible number, nothing built on top of it is credible. Do P3-1 (TraCI
speed-up) first or alongside — it makes every re-run cheaper.

*Done when* the README reports a greedy (ε=0) evaluation on held-out seeds against
a realistic baseline, whatever its sign.

**Phase 2 — Generalise and add baselines.**
Derive state/action from the network instead of `src/constants.py`. Add Webster,
actuated and max-pressure controllers. Add the evaluation protocol (multi-seed,
confidence intervals, drain guard). Expose the Gymnasium/PettingZoo API.

*Done when* a RESCO scenario runs end-to-end and all four controller types are
reported side by side by the same metric function.

**Phase 3 — Study-kit input/output and student docs.**
OSM + counts input, report + recommended-plan output, and a "reproduce a result in
30 minutes" guide.

*Done when* a new user can go from an OSM intersection and a count table to a
report without editing code.

**Phase 4 — Niche.**
Motorcycle sublane scenarios, then the drone video → counts module.

*Done when* at least one Vietnamese intersection scenario is calibrated against
observed counts and published with its data.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Scope — Phase 2 alone is thesis-sized | Ship phases independently; each has its own "done when" |
| SUMO speed (PROBLEMS.md P3-1) | libsumo, subscriptions, parallel workers |
| Calibration quality of OSM + count-derived scenarios | Report calibration error in every report; never present a scenario as validated when it is not |
| Overclaiming against weak baselines | The shared-metric contract in `src/metrics.py`, tuned baselines, significance tests |

---

## Sources

**Deployment and impact**
- [Google blog: Project Green Light Boston expansion](https://blog.google/outreach-initiatives/sustainability/project-green-light-boston-expansion/)
- [Boston.gov](https://www.boston.gov/node/16372851)
- [Yahoo News on Boston results](https://www.yahoo.com/news/boston-notorious-traffic-delays-heres-165116556.html)

**Sim-to-real**
- [Sim2Signal: Sim-to-Real Benchmarks for TSC (arXiv 2609.01676)](https://arxiv.org/abs/2609.01676)
- [Advances in RL for TSC: a review (OUP, 2025)](https://academic.oup.com/iti/article/doi/10.1093/iti/liaf009/8125227)
- [Deployment-oriented sim-to-real survey (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7560448)
- [Towards Real-World Deployment of RL for TSC (LemgoRL)](https://arxiv.org/pdf/2103.16223)
- [Domain randomization and meta-learning for the reality gap](https://arxiv.org/pdf/2307.11357)

**Existing platforms and evaluation practice**
- [RESCO paper (NeurIPS 2021)](https://people.engr.tamu.edu/guni/papers/NeurIPS-signals.pdf) · [RESCO repo](https://github.com/UnivOfAlabama-BittleResearchGroup/RESCO)
- [LibSignal (arXiv 2211.10649)](https://arxiv.org/abs/2211.10649v1)
- [sumo-rl on PyPI](https://pypi.org/project/sumo-rl/1.4.0/)
- [Critical review of TSC: RL vs MPC](https://arxiv.org/pdf/2211.14426)
- [Methodology for RL-based adaptive TSC](https://arxiv.org/pdf/2101.09614)

**Scenario building and calibration**
- [SUMO: Routes from Observation Points (routeSampler)](https://sumo.dlr.de/wiki/Demand/Routes_from_Observation_Points)
- [Calibrating SUMO scenarios with evolutionary algorithms](https://www.tib-op.org/ojs/index.php/scp/article/view/2590)

**Image → network**
- [Lane-level street map extraction from aerial imagery (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/html/He_Lane-Level_Street_Map_Extraction_From_Aerial_Imagery_WACV_2022_paper.html)
- [Building lane-level maps from aerial images](https://arxiv.org/pdf/2312.13449)
- [MapDreamer](https://arxiv.org/pdf/2607.01370)

**LLM-based TSC (related trend)**
- [LLMLight](https://arxiv.org/pdf/2312.16044)
- [REG-TSC](https://arxiv.org/pdf/2510.26242)
- [Traffic-R1 (ACL 2026)](https://aclanthology.org/2026.acl-long.995.pdf)

**Drone data and the niche**
- [CitySim drone dataset](https://arxiv.org/pdf/2208.11036)
- [DRIFT dataset review](https://themoonlight.io/review/drift-open-dataset-a-drone-derived-intelligence-for-traffic-analysis-in-urban-environmen)
- [Drone analytics at high-density intersections](https://arxiv.org/pdf/2411.02349)
- [Turning movement counts from cameras](https://www.arxiv.org/pdf/2511.12342)
- [Bandung DQN with two-wheeler lanes (preprint)](https://www.preprints.org/manuscript/202408.0969)
