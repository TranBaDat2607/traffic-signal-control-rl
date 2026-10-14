"""Reward and per-step measurements from TraCI data."""

from __future__ import annotations

import traci

from constants import INCOMING_EDGES


def get_cumulated_waiting_time() -> float:
    """Compute the sum of accumulated waiting times for vehicles on incoming edges.

    Returns:
        Total accumulated waiting time across all vehicles on incoming edges.
    """
    waiting_times = 0.0

    for car_id in traci.vehicle.getIDList():
        road_id = traci.vehicle.getRoadID(car_id)
        if road_id not in INCOMING_EDGES:
            continue
        wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
        waiting_times += wait_time

    return waiting_times


def get_queue_length() -> int:
    """Return the number of stopped vehicles on all incoming edges.

    Returns:
        Total number of vehicles with speed 0 on all four incoming edges.
    """
    return sum(traci.edge.getLastStepHaltingNumber(edge) for edge in INCOMING_EDGES)


def get_arrived_number() -> int:
    """Return the number of vehicles that reached their destination in the last step.

    Returns:
        Vehicles that left the network during the last simulation step.
    """
    return traci.simulation.getArrivedNumber()


def get_pending_number() -> int:
    """Return the number of vehicles waiting to be inserted into the network.

    A vehicle is pending when its departure time has passed but SUMO could not
    insert it, typically because its approach is full. It is on no edge, so
    ``get_queue_length()`` does not count it.

    Returns:
        Vehicles past their departure time that are not yet in the network.
    """
    return len(traci.simulation.getPendingVehicles())
