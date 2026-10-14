"""Tests for the rule-based controllers. No SUMO needed: the info dict is faked."""

import numpy as np
import pytest

from environment.core import PhaseLanes
from policy import FixedTimeController, MaxPressureController, RandomController
from policy.baselines import phase_pressures

STATE = np.zeros(240, dtype=np.float32)

# Same structure as the real intersection: action -> lanes it gives green to.
PHASE_LANES = {
    0: PhaseLanes(frozenset({"N_in", "S_in"}), frozenset({"S_out", "N_out"})),
    1: PhaseLanes(frozenset({"NL_in", "SL_in"}), frozenset({"E_out", "W_out"})),
    2: PhaseLanes(frozenset({"E_in", "W_in"}), frozenset({"W_out", "E_out"})),
    3: PhaseLanes(frozenset({"EL_in", "WL_in"}), frozenset({"N_out", "S_out"})),
}


def info(**queues: int) -> dict:
    return {"phase_lanes": PHASE_LANES, "lane_queues": queues}


def test_fixed_time_cycles_with_repeats() -> None:
    ctrl = FixedTimeController(green_time=30, decision_green=10)
    actions = [ctrl.choose_action(STATE, {}) for _ in range(13)]
    assert actions == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]


def test_fixed_time_reset_restarts_cycle() -> None:
    ctrl = FixedTimeController(green_time=10, decision_green=10)
    ctrl.choose_action(STATE, {})
    ctrl.choose_action(STATE, {})
    ctrl.reset()
    assert ctrl.choose_action(STATE, {}) == 0


@pytest.mark.parametrize("green", [0, 15, -10])
def test_fixed_time_rejects_green_not_multiple_of_decision(green: int) -> None:
    with pytest.raises(ValueError, match="multiple"):
        FixedTimeController(green_time=green, decision_green=10)


def test_pressure_is_incoming_minus_outgoing() -> None:
    p = phase_pressures(PHASE_LANES, {"N_in": 5, "S_in": 2, "N_out": 3, "E_in": 1})
    assert p == {0: 5 + 2 - 3, 1: 0, 2: 1, 3: -3}


def test_max_pressure_picks_highest_pressure() -> None:
    ctrl = MaxPressureController()
    assert ctrl.choose_action(STATE, info(E_in=4, W_in=4, N_in=3)) == 2


def test_max_pressure_downstream_congestion_lowers_pressure() -> None:
    ctrl = MaxPressureController()
    # N/S has the longer queue, but its exits are blocked.
    assert ctrl.choose_action(STATE, info(N_in=6, S_out=5, N_out=5, E_in=3)) == 2


def test_max_pressure_keeps_current_phase_on_tie() -> None:
    ctrl = MaxPressureController()
    assert ctrl.choose_action(STATE, info(E_in=2)) == 2
    assert ctrl.choose_action(STATE, info(E_in=2, N_in=2)) == 2  # tie: no switch
    assert ctrl.choose_action(STATE, info(E_in=2, N_in=3)) == 0
    ctrl.reset()
    assert ctrl.choose_action(STATE, info(E_in=2, N_in=2)) == 0  # fresh: lowest index


def test_random_controller_is_seeded() -> None:
    first = RandomController(seed=7)
    second = RandomController(seed=7)
    seq1 = [first.choose_action(STATE, {}) for _ in range(50)]
    seq2 = [second.choose_action(STATE, {}) for _ in range(50)]
    assert seq1 == seq2
    assert set(seq1) == {0, 1, 2, 3}
