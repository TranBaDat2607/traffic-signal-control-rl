"""Single source of truth for episode-level performance metrics.

Every entry point that reports how well a controller performed — ``train.py``,
``evaluate.py`` and ``grid_train.py`` — computes its numbers here. A controller
is only comparable to another controller when both were measured by the same
function, so there must be exactly one definition of each number.

(This is a lesson learned the hard way: an earlier version of this project had
one aggregation for the agent and another for the baseline, plotted them on one
axis, and compared seconds with queue-steps without anything noticing.)

Definitions
-----------
``total_delay``
    Vehicle-seconds of delay: the per-step count of halted vehicles, summed over
    every simulation step of the episode. A vehicle stopped for 30 s
    contributes 30. This is the standard "total delay" figure and it is what
    both code paths already measured correctly.

``avg_queue``
    ``total_delay / max_steps`` — the mean number of vehicles queued at any
    instant, in vehicles.

``vehicles_arrived`` and ``insertion_delay`` — the saturation guard
    ``total_delay`` only sees vehicles that are *on* an incoming edge. A vehicle
    SUMO cannot insert because its approach is full is on no edge and
    contributes nothing, so under saturation ``total_delay`` understates delay
    exactly where congestion is worst. ``insertion_delay``
    is the delay those vehicles spend waiting outside the network:
    the per-step count of pending vehicles, summed over the episode, in
    vehicle-seconds. It is reported *alongside* ``total_delay``, not folded into
    it. When it is small relative to ``total_delay``, the delay figure is
    complete. When it is not, ``total_delay`` is a lower bound.
    ``vehicles_arrived`` counts vehicles that completed their trip. It is below
    ``n_cars_generated`` even in a healthy episode, because the route generator
    schedules the last departures right at the end of the episode.

``neg_reward``
    The sum of the negative per-step rewards. This is a *training* diagnostic
    for the learning curve, not a measure of controller quality: it is defined
    only where an agent is learning, it depends on the decision-step interval
    at which rewards are sampled, and a non-learning controller has no
    equivalent. It must never be plotted against a baseline.

Note on what is deliberately *not* defined here: summing
``get_cumulated_waiting_time()`` over the steps of an episode. That quantity is
already an accumulation, so summing it again double-counts — a vehicle that
waits 100 s contributes about 5,050 rather than 100. ``total_delay`` measures
the same thing without the double count.
"""

from collections.abc import Sequence

# Bumped whenever a definition in this module changes. Serialized alongside
# results so that data written under an older definition fails loudly instead of
# being silently compared against data written under a newer one.
METRICS_VERSION = 3

TOTAL_DELAY_LABEL = "Total delay (vehicle-seconds)"
AVG_QUEUE_LABEL = "Average queue length (vehicles)"
NEG_REWARD_LABEL = "Training diagnostic: sum of negative rewards"
ARRIVED_LABEL = "Vehicles arrived"
INSERTION_DELAY_LABEL = "Insertion delay (vehicle-seconds, not in total delay)"


def total_delay(queue_per_step: Sequence[int]) -> int:
    """Compute vehicle-seconds of delay over an episode.

    Args:
        queue_per_step: Number of halted vehicles at each simulation step.

    Returns:
        The sum of the per-step queue lengths, in vehicle-seconds.
    """
    return sum(queue_per_step)


def avg_queue(queue_per_step: Sequence[int], max_steps: int) -> float:
    """Compute the mean queue length over an episode.

    Args:
        queue_per_step: Number of halted vehicles at each simulation step.
        max_steps: Episode length in simulation steps, used as the divisor so
            that episodes ending early are still comparable.

    Returns:
        Mean number of queued vehicles per step, rounded to one decimal.
    """
    return round(total_delay(queue_per_step) / max(max_steps, 1), 1)


def vehicles_arrived(arrived_per_step: Sequence[int]) -> int:
    """Count the vehicles that completed their trip during an episode.

    Args:
        arrived_per_step: Number of vehicles that arrived at each simulation step.

    Returns:
        Total vehicles arrived over the episode.
    """
    return sum(arrived_per_step)


def insertion_delay(pending_per_step: Sequence[int]) -> int:
    """Compute vehicle-seconds spent waiting to be inserted into the network.

    This delay is invisible to :func:`total_delay`. See the module docstring.

    Args:
        pending_per_step: Vehicles past their departure time but not yet
            inserted, at each simulation step.

    Returns:
        The sum of the per-step pending counts, in vehicle-seconds.
    """
    return sum(pending_per_step)


def sum_negative_rewards(rewards: Sequence[float]) -> float:
    """Sum the negative per-step rewards of an episode.

    A training diagnostic only — see the module docstring for why this is not a
    controller-quality measure and must not be compared across controllers.

    Args:
        rewards: Per-decision-step rewards recorded during the episode.

    Returns:
        The sum of those rewards that are negative.
    """
    return sum(reward for reward in rewards if reward < 0)
