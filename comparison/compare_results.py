"""Compare any number of controllers evaluated by ``src/evaluate.py``.

Usage (from the project root)::

    python comparison/compare_results.py runs/eval/fixed_time-30s runs/eval/max_pressure
        runs/eval/dqn-seed0 runs/eval/dqn-seed1 runs/eval/dqn-seed2 --out runs/comparison

Each argument is a directory containing ``eval_results.json``. All of them must
have been evaluated on the same seeds under the same metric definitions;
otherwise the script stops rather than compare different traffic.

The first directory is the reference (change it with ``--reference <label>``).
Every other controller is compared with it *per seed* (a paired comparison):
the table gives the mean difference in total delay with a 95% confidence
interval, and in how many episodes the controller was better.

If a trained controller appears several times (e.g. DQN trained with three
seeds), an extra line aggregates them across runs, with a confidence interval that
reflects training randomness. See ``src/evaluation_stats.py``.

Writes to ``--out``: ``summary.md`` (the table), ``comparison_summary.json``
and ``comparison.png``.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation_stats import across_runs, paired_difference
from metrics import METRICS_VERSION, TOTAL_DELAY_LABEL
from plots import INK, INK_MUTED, SERIES, SERIES_FAINT, style_axes

METRIC_KEYS = ("total_delays", "avg_queues", "vehicles_arrived", "insertion_delays")


def load_result(directory: Path) -> dict:
    """Load one evaluation, refusing anything not comparable with this build.

    Also reads the files written by earlier versions of this project
    (``test_results.json`` from ``test.py``, ``baseline_results.json`` from
    ``run_baseline.py``), which store the same metrics under the same names.
    """
    for name in ("eval_results.json", "test_results.json", "baseline_results.json"):
        path = directory / name
        if path.exists():
            break
    else:
        sys.exit(f"No eval_results.json in {directory}. Create it with src/evaluate.py.")

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("metrics_version") != METRICS_VERSION:
        sys.exit(
            f"{path} was written under metrics_version={data.get('metrics_version')!r}; this "
            f"build uses {METRICS_VERSION}. Its numbers are not comparable. Re-run evaluate.py."
        )
    if "label" not in data:  # older file formats
        if name == "baseline_results.json":
            green = data["phase_durations"][0]
            data["controller"], data["label"] = "fixed_time", f"fixed_time-{green:g}s"
        else:
            run = directory.parent.name
            data["controller"] = "dqn"
            data["label"] = run if run.startswith("dqn") else f"dqn-{run}"
            data["model"] = data.get("model") or str(directory.parent)
    return data


def fmt_ms(values: list[float], digits: int = 0) -> str:
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{fmean(values):,.{digits}f} ± {sd:,.{digits}f}"


def fmt_ci(mean: float, low: float, high: float) -> str:
    if math.isnan(low):
        return f"{mean:+,.0f}"
    return f"{mean:+,.0f} [{low:+,.0f}, {high:+,.0f}]"


def build_summary(results: list[dict], reference: dict) -> dict:
    """Per-controller statistics and paired differences against the reference."""
    ref_delay = reference["total_delays"]
    rows = []
    for r in results:
        d = paired_difference(ref_delay, r["total_delays"])
        rows.append({
            "label": r["label"],
            "controller": r["controller"],
            "episodes": len(r["seeds"]),
            "model": r.get("model"),
            **{f"{k}_mean": fmean(r[k]) for k in METRIC_KEYS},
            **{f"{k}_std": stdev(r[k]) if len(r[k]) > 1 else 0.0 for k in METRIC_KEYS},
            "delay_diff_vs_reference": d.diff.mean,
            "delay_diff_ci95": [d.diff.low, d.diff.high],
            "delay_diff_pct": d.pct,
            "episodes_lower_than_reference": d.lower,
            "episodes_higher_than_reference": d.higher,
        })

    groups = []
    by_controller = defaultdict(list)
    for row in rows:
        if row["model"]:  # only trained controllers have training runs to aggregate
            by_controller[row["controller"]].append(row)
    for controller, members in by_controller.items():
        if len(members) < 2:
            continue
        delay = across_runs([m["total_delays_mean"] for m in members])
        diff = across_runs([m["delay_diff_vs_reference"] for m in members])
        groups.append({
            "controller": controller,
            "runs": [m["label"] for m in members],
            "total_delay_mean": delay.mean,
            "total_delay_ci95_across_runs": [delay.low, delay.high],
            "delay_diff_vs_reference": diff.mean,
            "delay_diff_ci95_across_runs": [diff.low, diff.high],
            "delay_diff_pct": 100 * diff.mean / fmean(ref_delay),
            "run_delay_means": [m["total_delays_mean"] for m in members],
        })
    return {
        "metrics_version": METRICS_VERSION,
        "reference": reference["label"],
        "seeds": reference["seeds"],
        "controllers": rows,
        "groups": groups,
    }


def markdown_table(summary: dict, results: list[dict]) -> str:
    seeds = summary["seeds"]
    by_label = {r["label"]: r for r in results}
    lines = [
        (
            f"Evaluation on {len(seeds)} held-out traffic seeds ({seeds[0]} to {seeds[-1]}). "
            f"Mean ± standard deviation over episodes. Δ = paired difference in total delay "
            f"versus `{summary['reference']}`, with 95% CI."
        ),
        "",
        (
            "| Controller | Total delay (veh-s) | Avg queue (veh) | Δ total delay vs reference "
            "| Δ % | Episodes better | Arrived | Insertion delay (veh-s) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(summary["controllers"], key=lambda r: r["total_delays_mean"], reverse=True):
        r = by_label[row["label"]]
        is_ref = row["label"] == summary["reference"]
        lo, hi = row["delay_diff_ci95"]
        lines.append(
            f"| `{row['label']}`{' (reference)' if is_ref else ''} "
            f"| {fmt_ms(r['total_delays'])} | {fmt_ms(r['avg_queues'], 2)} "
            f"| {'—' if is_ref else fmt_ci(row['delay_diff_vs_reference'], lo, hi)} "
            f"| {'—' if is_ref else format(row['delay_diff_pct'], '+.1f')} "
            f"| {'—' if is_ref else f'{row["episodes_lower_than_reference"]}/{row["episodes"]}'} "
            f"| {fmt_ms(r['vehicles_arrived'], 1)} | {fmt_ms(r['insertion_delays'], 1)} |"
        )
    for g in summary["groups"]:
        lo, hi = g["delay_diff_ci95_across_runs"]
        dlo, dhi = g["total_delay_ci95_across_runs"]
        lines += [
            "",
            f"**`{g['controller']}`, {len(g['runs'])} training runs:** mean total delay "
            f"{g['total_delay_mean']:,.0f} veh-s (95% CI across runs {dlo:,.0f} to {dhi:,.0f}); "
            f"Δ vs reference {fmt_ci(g['delay_diff_vs_reference'], lo, hi)} veh-s "
            f"({g['delay_diff_pct']:+.1f}%). Per-run means: "
            + ", ".join(f"{v:,.0f}" for v in g["run_delay_means"]) + ".",
        ]
    return "\n".join(lines) + "\n"


def plot(summary: dict, results: list[dict], out_file: Path) -> None:
    """Per-episode total delay of every controller, one row each, best at the top."""
    order = sorted(results, key=lambda r: fmean(r["total_delays"]), reverse=True)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(order) + 1.6), constrained_layout=True)
    style_axes(ax, f"Total delay per episode, {len(summary['seeds'])} held-out seeds",
               TOTAL_DELAY_LABEL, "")
    ax.grid(axis="y", visible=False)
    ax.boxplot(
        [r["total_delays"] for r in order], orientation="horizontal", widths=0.5,
        patch_artist=True, showfliers=True,
        boxprops={"facecolor": SERIES_FAINT, "edgecolor": SERIES},
        medianprops={"color": SERIES, "linewidth": 2},
        whiskerprops={"color": SERIES}, capprops={"color": SERIES},
        flierprops={"marker": "o", "markersize": 3, "markeredgecolor": SERIES},
    )
    ax.set_yticks(
        range(1, len(order) + 1),
        [r["label"] + (" (reference)" if r["label"] == summary["reference"] else "")
         for r in order],
        fontsize=9, color=INK,
    )
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.text(1.0, -0.12 - 0.3 / len(order),
            "Box: middle 50% of episodes; line: median. Lower is better.",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK_MUTED)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluated controllers.")
    parser.add_argument("dirs", nargs="+", type=Path, help="Directories with eval_results.json")
    parser.add_argument("--reference", default=None,
                        help="Label of the reference controller (default: the first directory)")
    parser.add_argument("--out", type=Path, default=Path("runs/comparison"))
    args = parser.parse_args()

    results = [load_result(d) for d in args.dirs]
    labels = [r["label"] for r in results]
    if len(set(labels)) != len(labels):
        sys.exit(f"Duplicate labels {labels}; evaluate into differently named directories.")
    seeds = results[0]["seeds"]
    for r in results[1:]:
        if r["seeds"] != seeds:
            sys.exit(f"{r['label']} was evaluated on different seeds than {labels[0]}; "
                     "a paired comparison needs identical traffic.")

    reference = next((r for r in results if r["label"] == args.reference), None)
    if args.reference and reference is None:
        sys.exit(f"--reference {args.reference!r} is not one of {labels}")
    summary = build_summary(results, reference or results[0])

    args.out.mkdir(parents=True, exist_ok=True)
    table = markdown_table(summary, results)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    (args.out / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    plot(summary, results, args.out / "comparison.png")
    print(table)
    print(f"Saved: {args.out}/summary.md, comparison_summary.json, comparison.png")


if __name__ == "__main__":
    main()
