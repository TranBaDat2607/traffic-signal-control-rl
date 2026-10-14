"""Statistics for comparing controllers. Pure Python, no SUMO or NumPy needed.

Two sources of randomness matter when comparing a learned controller with a
baseline, and they need different treatment:

1. **Traffic.** Every controller is evaluated on the same traffic seeds, so
   the comparison is *paired*: take the per-seed difference, then its mean and
   confidence interval (:func:`paired_difference`). Pairing removes the
   variation between easy and hard traffic, which is usually much larger than
   the difference between controllers.
2. **Training.** A second training run with another seed gives a different
   policy. One run says nothing about this. Train several, and report the
   spread of their results (:func:`across_runs`). With few runs, this interval
   is wide, and that width is the honest answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Two-sided 95% critical values of Student's t distribution.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}
_Z95 = 1.960


def t_critical_95(df: int) -> float:
    """Two-sided 95% critical value of Student's t with ``df`` degrees of freedom.

    Exact for df <= 30 and 40, 60, 120; between those, interpolated linearly
    in 1/df (error below 0.001); the normal value 1.96 as df goes to infinity.

    Raises:
        ValueError: If ``df < 1``.
    """
    if df < 1:
        msg = f"degrees of freedom must be >= 1, got {df}"
        raise ValueError(msg)
    if df in _T95:
        return _T95[df]
    knots = sorted(_T95)
    upper = next((k for k in knots if k > df), None)
    lower = max(k for k in knots if k < df)
    x, x0 = 1 / df, 1 / lower
    x1, t1 = (1 / upper, _T95[upper]) if upper is not None else (0.0, _Z95)
    return t1 + (_T95[lower] - t1) * (x - x1) / (x0 - x1)


@dataclass(frozen=True)
class Interval:
    """A mean with a 95% confidence interval."""

    mean: float
    low: float
    high: float


def mean_ci(values: Sequence[float]) -> Interval:
    """Mean and 95% t-interval of independent samples.

    With a single value the interval is undefined and reported as NaN.
    """
    m = fmean(values)
    if len(values) < 2:  # noqa: PLR2004
        return Interval(m, math.nan, math.nan)
    half = t_critical_95(len(values) - 1) * stdev(values) / math.sqrt(len(values))
    return Interval(m, m - half, m + half)


@dataclass(frozen=True)
class PairedDifference:
    """Candidate minus reference, matched by traffic seed.

    Attributes:
        diff: Mean per-seed difference with its 95% CI.
        pct: Mean difference as a percentage of the reference mean.
        lower: Episodes where the candidate's value was lower.
        higher: Episodes where it was higher.
        n: Number of paired episodes.
    """

    diff: Interval
    pct: float
    lower: int
    higher: int
    n: int


def paired_difference(
    reference: Sequence[float], candidate: Sequence[float]
) -> PairedDifference:
    """Compare two controllers evaluated on the same seeds, in the same order.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(reference) != len(candidate):
        msg = f"paired samples differ in length: {len(reference)} vs {len(candidate)}"
        raise ValueError(msg)
    diffs = [c - r for r, c in zip(reference, candidate, strict=True)]
    ref_mean = fmean(reference)
    return PairedDifference(
        diff=mean_ci(diffs),
        pct=100 * fmean(diffs) / ref_mean if ref_mean else math.nan,
        lower=sum(d < 0 for d in diffs),
        higher=sum(d > 0 for d in diffs),
        n=len(diffs),
    )


def across_runs(run_values: Sequence[float]) -> Interval:
    """Mean and 95% t-interval across independent training runs.

    Pass one number per run (e.g. each run's mean total delay over the
    evaluation seeds). The interval reflects training randomness.
    """
    return mean_ci(run_values)
