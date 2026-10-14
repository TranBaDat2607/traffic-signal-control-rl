"""Reward and per-step measurements from TraCI data.

All reads go through the subscriptions in ``environment.subscriptions``, so
``subscribe()`` must have been called after the simulation started.
"""

from __future__ import annotations

from constants import INCOMING_EDGES
from environment.subscriptions import halting_numbers, simulation_data, vehicle_data
from sumo_api import tc


def get_cumulated_waiting_time() -> float:
    """Compute the sum of accumulated waiting times for vehicles on incoming edges.

    Returns:
        Total accumulated waiting time across all vehicles on incoming edges.
    """
    waiting_times = 0.0

    for car in vehicle_data().values():
        if car[tc.VAR_ROAD_ID] not in INCOMING_EDGES:
            continue
        waiting_times += float(car[tc.VAR_ACCUMULATED_WAITING_TIME])

    return waiting_times


def get_queue_length() -> int:
    """Return the number of stopped vehicles on all incoming edges.

    Returns:
        Total number of vehicles with speed 0 on all four incoming edges.
    """
    return sum(halting_numbers().values())


def get_arrived_number() -> int:
    """Return the number of vehicles that reached their destination in the last step.

    Returns:
        Vehicles that left the network during the last simulation step.
    """
    return int(simulation_data()[tc.VAR_ARRIVED_VEHICLES_NUMBER])


def get_pending_number() -> int:
    """Return the number of vehicles waiting to be inserted into the network.

    A vehicle is pending when its departure time has passed but SUMO could not
    insert it, typically because its approach is full. It is on no edge, so
    ``get_queue_length()`` does not count it.

    Returns:
        Vehicles past their departure time that are not yet in the network.
    """
    return len(simulation_data()[tc.VAR_PENDING_VEHICLES])
