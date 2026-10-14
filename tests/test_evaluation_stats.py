"""Tests for the paired-comparison statistics in ``src/evaluation_stats.py``."""

import math

import pytest

from evaluation_stats import across_runs, mean_ci, paired_difference, t_critical_95


@pytest.mark.parametrize(
    ("df", "expected"),
    [(1, 12.706), (2, 4.303), (30, 2.042), (99, 1.984), (1000, 1.962), (10**9, 1.960)],
)
def test_t_critical_values(df: int, expected: float) -> None:
    assert t_critical_95(df) == pytest.approx(expected, abs=1e-3)


def test_t_critical_rejects_zero_df() -> None:
    with pytest.raises(ValueError, match="degrees of freedom"):
        t_critical_95(0)


def test_mean_ci_known_values() -> None:
    ci = mean_ci([1.0, 2.0, 3.0])  # mean 2, sd 1, n 3 -> half width 4.303 / sqrt(3)
    assert ci.mean == 2.0
    assert ci.high - ci.mean == pytest.approx(4.303 / math.sqrt(3), abs=1e-3)
    assert ci.mean - ci.low == pytest.approx(ci.high - ci.mean)


def test_mean_ci_single_value_has_no_interval() -> None:
    ci = mean_ci([5.0])
    assert ci.mean == 5.0
    assert math.isnan(ci.low)
    assert math.isnan(ci.high)


def test_paired_difference_removes_shared_variation() -> None:
    # Traffic difficulty varies a lot across seeds; the candidate is always 10 better.
    reference = [100.0, 500.0, 900.0, 300.0]
    candidate = [90.0, 490.0, 890.0, 290.0]
    d = paired_difference(reference, candidate)
    assert d.diff.mean == -10.0
    assert d.diff.low == pytest.approx(-10.0)  # zero variance in the differences
    assert d.lower == 4
    assert d.higher == 0
    assert d.pct == pytest.approx(-10.0 / 450.0 * 100)


def test_paired_difference_requires_equal_length() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_difference([1.0, 2.0], [1.0])


def test_across_runs_is_wide_with_few_runs() -> None:
    ci = across_runs([10_000.0, 11_000.0, 12_000.0])
    assert ci.mean == 11_000.0
    # sd 1000, n 3: half width = 4.303 * 1000 / sqrt(3) ~ 2484
    assert ci.high - ci.mean == pytest.approx(2484.3, abs=1)
