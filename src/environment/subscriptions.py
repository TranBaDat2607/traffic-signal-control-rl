"""TraCI subscriptions: everything the environment reads, fetched in bulk.

Reading vehicle data one getter at a time costs one call per vehicle per
variable, a blocking socket round-trip under TraCI. Subscriptions make SUMO
push the values after every simulation step, so each reader below is a
single call however many vehicles there are (PROBLEMS.md P3-1).

``subscribe()`` must be called once after ``traci.start()``. The readers
raise if it was not, rather than quietly returning empty data.
"""

from __future__ import annotations

from typing import Any

from constants import INCOMING_EDGES, TRAFFIC_LIGHT_ID
from sumo_api import tc, traci

# Vehicle variables read by get_state() and get_cumulated_waiting_time().
VEHICLE_VARS = (
    tc.VAR_LANE_ID,
    tc.VAR_LANEPOSITION,
    tc.VAR_SPEED,
    tc.VAR_WAITING_TIME,
    tc.VAR_ACCUMULATED_WAITING_TIME,
    tc.VAR_ROAD_ID,
)

# Radius (m) of the context subscription around the junction. The network's
# edges are 750 m long, so this covers every vehicle in it: the subscription
# is a bulk read, not a filter. Filtering by lane or edge stays in the callers.
CONTEXT_RANGE = 2000.0

SIMULATION_VARS = (tc.VAR_ARRIVED_VEHICLES_NUMBER, tc.VAR_PENDING_VEHICLES)


def subscribe() -> None:
    """Register every subscription the environment reads. Call after start."""
    traci.junction.subscribeContext(
        TRAFFIC_LIGHT_ID, tc.CMD_GET_VEHICLE_VARIABLE, CONTEXT_RANGE, VEHICLE_VARS
    )
    for edge in INCOMING_EDGES:
        traci.edge.subscribe(edge, (tc.LAST_STEP_VEHICLE_HALTING_NUMBER,))
    traci.simulation.subscribe(SIMULATION_VARS)


def vehicle_data() -> dict[str, dict[int, Any]]:
    """Return ``{vehicle_id: {variable: value}}`` for every vehicle in the network.

    Raises:
        RuntimeError: If vehicles exist but no subscription delivered them.
    """
    data = traci.junction.getContextSubscriptionResults(TRAFFIC_LIGHT_ID)
    if not data and traci.vehicle.getIDCount() > 0:
        msg = "No vehicle subscription results: call subscriptions.subscribe() after start."
        raise RuntimeError(msg)
    return data


def halting_numbers() -> dict[str, int]:
    """Return ``{edge_id: halting vehicles}`` for the incoming edges."""
    results = traci.edge.getAllSubscriptionResults()
    try:
        return {
            edge: results[edge][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for edge in INCOMING_EDGES
        }
    except KeyError as exc:
        msg = "No edge subscription results: call subscriptions.subscribe() after start."
        raise RuntimeError(msg) from exc


def simulation_data() -> dict[int, Any]:
    """Return the subscribed simulation variables of the last step."""
    results = traci.simulation.getSubscriptionResults()
    if not results:
        msg = "No simulation subscription results: call subscriptions.subscribe() after start."
        raise RuntimeError(msg)
    return results
