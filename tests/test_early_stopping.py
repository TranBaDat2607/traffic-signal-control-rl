"""Tests for ``EarlyStopping`` (PROBLEMS.md P1-5).

``train.py`` used to construct ``EarlyStopping`` without reading
``early_stopping_min_episode``, so patience could run out during the
exploration phase. The guard now lives inside ``EarlyStopping`` so that every
caller gets it; these tests pin its behaviour without needing SUMO.
"""

from agent.model import EarlyStopping


def _first_stop(es: EarlyStopping, metrics: list[float]) -> int | None:
    """Feed *metrics* one episode at a time; return the 1-based stop episode."""
    for episode, metric in enumerate(metrics, start=1):
        if es.step(metric):
            return episode
    return None


def test_does_not_stop_before_min_episode() -> None:
    """The P1-5 "Done when": patience 30, min_episode 200 never stops before 200."""
    es = EarlyStopping(patience=30, min_episode=200)
    # A metric that never improves is the worst case for premature stopping.
    stop = _first_stop(es, [-100.0] * 1000)
    assert stop is not None
    assert stop >= 200
    # Episode 200 sets the first best; 30 further non-improving episodes stop it.
    assert stop == 230


def test_warmup_episodes_neither_improve_nor_set_best() -> None:
    """Exploration-phase episodes must not be selected as the best model."""
    es = EarlyStopping(patience=5, min_episode=10)
    for _ in range(9):
        es.step(1_000_000.0)  # a spectacular, but random-policy, episode
        assert not es.improved
    assert es.best == -float("inf")

    es.step(-5.0)  # episode 10: first monitored episode
    assert es.improved
    assert es.best == -5.0


def test_min_episode_zero_monitors_every_episode() -> None:
    """Default behaviour is unchanged when no minimum episode is set."""
    es = EarlyStopping(patience=3)
    assert _first_stop(es, [-10.0, -10.0, -10.0, -10.0]) == 4


def test_improvement_resets_patience() -> None:
    es = EarlyStopping(patience=2)
    assert not es.step(-10.0)
    assert not es.step(-11.0)
    assert not es.step(-9.0)
    assert es.improved
    assert not es.step(-9.5)
    assert es.step(-9.5)


def test_patience_zero_never_stops() -> None:
    es = EarlyStopping(patience=0, min_episode=5)
    assert _first_stop(es, [-1.0] * 500) is None


def test_improved_is_false_before_any_step() -> None:
    """Previously ``improved`` read True before the first call."""
    assert not EarlyStopping(patience=3).improved
