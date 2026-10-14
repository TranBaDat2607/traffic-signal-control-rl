"""Figures written by the training and comparison scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

SERIES = "#2a78d6"
SERIES_FAINT = "#a9c8ef"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def style_axes(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared, recessive axis style."""
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def moving_average(values: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; shorter than ``values`` by ``window - 1``."""
    data = np.asarray(values, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(
    columns: dict[str, Sequence[float]], out_file: Path, window: int = 10
) -> None:
    """Plot the per-episode training log as four small panels.

    Args:
        columns: The training log, column name -> one value per episode. Must
            contain ``total_delay``, ``neg_reward``, ``mean_loss`` and ``epsilon``.
        out_file: PNG file to write.
        window: Moving-average window, in episodes.
    """
    panels = [
        ("total_delay", "Total delay per episode", "vehicle-seconds"),
        ("neg_reward", "Sum of negative rewards (training diagnostic)", "reward"),
        ("mean_loss", "Mean Huber loss per episode", "loss"),
        ("epsilon", "Exploration rate", "epsilon"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, panels, strict=True):
        values = np.asarray(columns[key], dtype=float)
        episodes = np.arange(1, len(values) + 1)
        style_axes(ax, title, "Episode", ylabel)
        if key == "epsilon":
            ax.plot(episodes, values, color=SERIES, linewidth=2)
            continue
        if len(values) < window:
            ax.plot(episodes, values, color=SERIES, linewidth=2, marker="o", markersize=4)
            continue
        ax.plot(episodes, values, color=SERIES_FAINT, linewidth=1)
        ax.plot(episodes[window - 1 :], moving_average(values, window), color=SERIES,
                linewidth=2)
        ax.text(0.99, 0.97, f"line: {window}-episode moving average", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=INK_MUTED)
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def save_data_and_plot(  # noqa: PLR0913
    data: Sequence[float | int],
    *,
    filename: str,
    xlabel: str,
    ylabel: str,
    out_folder: Path,
    dpi: int = 96,
) -> None:
    """Save one series to ``plot_<filename>_data.txt`` and a line plot PNG.

    Used by the grid extension (``grid_train.py``).

    Args:
        data: Sequence of numeric values to plot and save.
        filename: Base name (without extension) for output files.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        out_folder: Directory where output files are written.
        dpi: Resolution in dots per inch for the saved plot image.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    style_axes(ax, ylabel, xlabel, ylabel)
    ax.plot(data, color=SERIES, linewidth=1.5)
    fig.tight_layout()
    fig.savefig(out_folder / f"plot_{filename}.png", dpi=dpi)
    plt.close(fig)

    data_text = "\n".join(str(value) for value in data) + "\n"
    (out_folder / f"plot_{filename}_data.txt").write_text(data_text, encoding="utf-8")
