"""Single import point for the SUMO control API.

Every module that talks to SUMO imports ``traci`` from here instead of
importing the ``traci`` package directly::

    from sumo_api import select_backend, tc, traci

    select_backend(gui=False)   # before traci.start()
    traci.start(cmd)

Two backends expose the same API:

``libsumo``
    SUMO runs inside the Python process, so a getter is a function call rather
    than a blocking socket round-trip. It cannot drive ``sumo-gui`` and allows
    only one simulation per process. Used whenever ``gui`` is false and the
    ``libsumo`` package is installed.

``traci``
    SUMO runs as a separate process, controlled over a socket. Required for
    ``sumo-gui``, and the fallback when ``libsumo`` is not installed.

Both run the same SUMO version, so they give identical results.
``comparison/fixed_phase_check.py`` verifies that (PROBLEMS.md P3-1).

The backend can be switched only between simulations, never while one is
loaded.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import traci as _traci
import traci.constants as tc

if TYPE_CHECKING:
    from types import ModuleType

try:
    import libsumo as _libsumo
except ImportError:  # pragma: no cover - depends on the installation
    _libsumo = None

__all__ = ["backend_name", "select_backend", "tc", "traci"]

# Set TLCS_SUMO_BACKEND=traci to force the socket backend, e.g. to compare the two.
_FORCED = os.environ.get("TLCS_SUMO_BACKEND", "").lower()


class _Backend:
    """Forwards attribute access to the selected SUMO API module."""

    def __init__(self) -> None:
        use_libsumo = _libsumo is not None and _FORCED != "traci"
        self._module: ModuleType = _libsumo if use_libsumo else _traci

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._module, name)


traci: Any = _Backend()


def select_backend(gui: bool) -> str:
    """Choose the backend for the next simulation and return its name.

    Args:
        gui: Whether the next simulation runs under ``sumo-gui``.

    Returns:
        ``"libsumo"`` or ``"traci"``.
    """
    use_libsumo = not gui and _libsumo is not None and _FORCED != "traci"
    module = _libsumo if use_libsumo else _traci
    if module is not traci._module and traci._module.isLoaded():  # noqa: SLF001
        traci._module.close()  # noqa: SLF001
    traci._module = module  # noqa: SLF001
    return backend_name()


def backend_name() -> str:
    """Return the name of the currently selected backend."""
    return "libsumo" if traci._module is _libsumo else "traci"  # noqa: SLF001
