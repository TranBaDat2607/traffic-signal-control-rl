"""State extraction: converts TraCI vehicle data into a 3-channel state vector.

Channels (each of length NUM_CELLS):
  0: presence      — 1 if any vehicle occupies the cell, else 0
  1: speed (norm)  — mean speed of vehicles in cell, normalised to [0, 1]
  2: wait  (norm)  — mean waiting time of vehicles in cell, normalised to [0, 1]
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from constants import (
    CELLS_PER_LANE_GROUP,
    HALTING_SPEED,
    LANE_DISTANCE_TO_CELL,
    LANE_ID_TO_GROUP,
    MAX_SPEED,
    MAX_WAIT_TIME,
    NUM_CELLS,
    ROAD_MAX_LENGTH,
)
from environment.subscriptions import vehicle_data
from sumo_api import tc, traci

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_lane_cell(lane_pos: float) -> int:
    """Map a continuous lane position to a discrete cell index.

    The lane is inverted so that cell 0 is at the traffic light and clamped
    to [0, ROAD_MAX_LENGTH].

    Args:
        lane_pos: Distance from the start of the edge in meters.

    Returns:
        Index of the discretized cell (0-based).

    Raises:
        RuntimeError: If lane_pos does not map to any known cell bucket.
    """
    lane_pos = ROAD_MAX_LENGTH - lane_pos
    lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))

    for distance, cell in LANE_DISTANCE_TO_CELL.items():
        if lane_pos <= distance:
            return cell

    msg = "Error while getting lane cell."
    raise RuntimeError(msg)


def get_state() -> NDArray:
    """Compute the 3-channel state vector from all active vehicles.

    Iterates over all active vehicles, maps each to its lane group and cell,
    and accumulates presence, speed, and waiting-time information per cell.
    Vehicle data comes from one subscription read (``vehicle_data()``). The
    loop follows ``getIDList()`` order so that the per-cell float sums are
    accumulated in the same order as before subscriptions were used.

    Returns:
        A NumPy array of shape (STATE_SIZE,) = (NUM_CELLS * 3,) with values
        in [0, 1].  Layout: [presence | speed_norm | wait_norm].

    Raises:
        ValueError: If a computed car_position is out of bounds.
    """
    presence  = np.zeros(NUM_CELLS, dtype=float)
    speed_sum = np.zeros(NUM_CELLS, dtype=float)
    wait_sum  = np.zeros(NUM_CELLS, dtype=float)
    count     = np.zeros(NUM_CELLS, dtype=float)

    data = vehicle_data()

    for car_id in traci.vehicle.getIDList():
        car = data[car_id]
        lane_group = LANE_ID_TO_GROUP.get(car[tc.VAR_LANE_ID])
        if lane_group is None:
            # Ignore cars that are not on incoming lanes.
            continue

        lane_pos: float = car[tc.VAR_LANEPOSITION]
        lane_cell = get_lane_cell(lane_pos)

        car_position = lane_group * CELLS_PER_LANE_GROUP + lane_cell

        if car_position < 0 or car_position >= NUM_CELLS:
            msg = "Out of bounds car position."
            raise ValueError(msg)

        presence[car_position] = 1.0
        speed_sum[car_position] += car[tc.VAR_SPEED]
        wait_sum[car_position]  += car[tc.VAR_WAITING_TIME]
        count[car_position]     += 1.0

    denom = np.maximum(count, 1.0)
    speed_norm = np.clip(speed_sum / denom / MAX_SPEED,     0.0, 1.0)
    wait_norm  = np.clip(wait_sum  / denom / MAX_WAIT_TIME, 0.0, 1.0)

    return np.concatenate([presence, speed_norm, wait_norm]).astype(np.float32)


def get_lane_queues() -> dict[str, int]:
    """Count halted vehicles on every lane that has at least one.

    This is not part of the agent's state. It is exposed in the environment's
    ``info`` dict for rule-based controllers such as max-pressure, which need
    queue counts on both the incoming and the outgoing lanes.

    Returns:
        ``{lane_id: number of vehicles slower than HALTING_SPEED}``.
    """
    queues: Counter[str] = Counter()
    for car in vehicle_data().values():
        if car[tc.VAR_SPEED] < HALTING_SPEED:
            queues[car[tc.VAR_LANE_ID]] += 1
    return dict(queues)
