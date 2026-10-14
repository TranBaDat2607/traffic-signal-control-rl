"""Early stopping and best-model selection on a monitored metric."""


class EarlyStopping:
    """Monitors a scalar metric and signals when training should stop.

    Tracks the *maximum* of the monitored metric.  The training scripts pass
    ``-total_delay`` so that the checkpoint they select is judged by the same
    number that gets reported.
    Stops when no improvement greater than ``min_delta`` is seen for
    ``patience`` consecutive episodes.

    Set ``patience=0`` to disable early stopping entirely.

    Episodes numbered below ``min_episode`` (1-based, counted by calls to
    :meth:`step`) are ignored entirely: they neither set a best value nor
    count toward patience.  This keeps the exploration phase, where the metric
    reflects random actions rather than the policy, from triggering a stop or
    being selected as the best model.

    Args:
        patience: Number of consecutive episodes without improvement before
            stopping.  ``0`` disables early stopping.
        min_delta: Minimum change to qualify as an improvement.
        min_episode: First episode (1-based) that is monitored.  ``0`` or
            ``1`` monitors every episode.

    Example::

        es = EarlyStopping(patience=10)
        for episode in range(total):
            metric = run_episode()
            if es.step(metric):
                break  # no improvement for 10 episodes
            if es.improved:
                save_best_model()
    """

    def __init__(self, patience: int, min_delta: float = 0.0, min_episode: int = 0) -> None:
        """Initialize the monitor; the arguments are described on the class."""
        self.patience = patience
        self.min_delta = min_delta
        self.min_episode = min_episode
        self._best: float = -float("inf")
        self._no_improve_count: int = 0
        self._episode: int = 0
        self._improved: bool = False

    @property
    def improved(self) -> bool:
        """True immediately after a :meth:`step` call that set a new best."""
        return self._improved

    @property
    def best(self) -> float:
        """Best metric value seen so far."""
        return self._best

    @property
    def no_improve_count(self) -> int:
        """Number of consecutive episodes without improvement."""
        return self._no_improve_count

    def step(self, metric: float) -> bool:
        """Record a new metric value and check the stopping criterion.

        Args:
            metric: Latest episode metric (higher = better).

        Returns:
            ``True`` if training should stop (patience exhausted).  Always
            ``False`` before ``min_episode``.
        """
        self._episode += 1
        if self._episode < self.min_episode:
            self._improved = False
            return False

        self._improved = metric > self._best + self.min_delta
        if self._improved:
            self._best = metric
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self.patience > 0 and self._no_improve_count >= self.patience
