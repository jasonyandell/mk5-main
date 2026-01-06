"""Visualization helpers for analysis notebooks.

This module provides consistent plotting functions:
- setup_notebook_style(): Configure matplotlib/seaborn defaults
- plot_v_distribution(): Histogram of V values
- plot_v_by_depth(): V distribution conditioned on depth
- plot_entropy_curve(): Information gain comparison
- plot_log_log(): Log-log scatter with power law fit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes


def setup_notebook_style() -> None:
    """
    Configure consistent matplotlib/seaborn styling for all notebooks.

    Call this at the start of each notebook.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Increase default figure size
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100

    # Font sizes
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # Grid
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def plot_v_distribution(
    V: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "V Distribution",
    bins: int = 85,
    color: str = "steelblue",
    show_stats: bool = True,
) -> "Axes":
    """
    Plot histogram of V values with statistics.

    Args:
        V: (N,) int8 V values in [-42, +42]
        ax: Matplotlib axes (created if None)
        title: Plot title
        bins: Number of histogram bins
        color: Bar color
        show_stats: Whether to show mean/std annotation

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(V, bins=bins, range=(-42.5, 42.5), color=color, alpha=0.7, edgecolor="black")
    ax.set_xlabel("V (Team 0 advantage)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(-45, 45)

    if show_stats:
        mean_v = np.mean(V)
        std_v = np.std(V)
        n_unique = len(np.unique(V))
        stats_text = f"Mean: {mean_v:.2f}\nStd: {std_v:.2f}\nUnique: {n_unique}"
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    return ax


def plot_v_by_depth(
    V: np.ndarray,
    depth: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "V by Depth",
    kind: str = "box",
) -> "Axes":
    """
    Plot V distribution conditioned on depth.

    Args:
        V: (N,) V values
        depth: (N,) depth values (remaining dominoes)
        ax: Matplotlib axes
        title: Plot title
        kind: "box" for boxplot, "violin" for violin plot

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    df = pd.DataFrame({"V": V, "depth": depth})

    if kind == "violin":
        sns.violinplot(data=df, x="depth", y="V", ax=ax, inner="box")
    else:
        sns.boxplot(data=df, x="depth", y="V", ax=ax)

    ax.set_xlabel("Depth (dominoes remaining)")
    ax.set_ylabel("V")
    ax.set_title(title)

    return ax


def plot_entropy_curve(
    results: list[tuple[str, float, float]],
    ax: "Axes | None" = None,
    title: str = "Information Gain by Feature",
) -> "Axes":
    """
    Plot information gain comparison across features.

    Args:
        results: List of (feature_name, mutual_info, conditional_entropy)
                 as returned by information_gain_ranking()
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    names = [r[0] for r in results]
    mutual_info = [r[1] for r in results]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, mutual_info, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mutual Information (bits)")
    ax.set_title(title)
    ax.invert_yaxis()  # Top feature at top

    # Add value labels
    for i, v in enumerate(mutual_info):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    return ax


def plot_log_log(
    x: np.ndarray,
    y: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "Log-Log Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    fit_line: bool = True,
    color: str = "steelblue",
) -> "Axes":
    """
    Log-log scatter plot with optional power law fit.

    Args:
        x: (N,) x values (must be positive)
        y: (N,) y values (must be positive)
        ax: Matplotlib axes
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        fit_line: Whether to fit and plot power law line
        color: Point color

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # Filter positive values for log
    mask = (x > 0) & (y > 0)
    x_pos = x[mask]
    y_pos = y[mask]

    ax.scatter(x_pos, y_pos, alpha=0.5, color=color, s=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if fit_line and len(x_pos) > 2:
        # Fit log-log linear regression: log(y) = a + b*log(x)
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)
        coeffs = np.polyfit(log_x, log_y, 1)
        slope, intercept = coeffs

        # Plot fit line
        x_fit = np.linspace(x_pos.min(), x_pos.max(), 100)
        y_fit = np.exp(intercept) * x_fit ** slope
        ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"y ~ x^{slope:.2f}")
        ax.legend()

    return ax


def plot_compression_comparison(
    results: dict[str, float],
    ax: "Axes | None" = None,
    title: str = "LZMA Compression by Ordering",
) -> "Axes":
    """
    Bar chart comparing compression ratios.

    Args:
        results: Dict mapping ordering name to compression ratio
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    names = list(results.keys())
    ratios = list(results.values())

    bars = ax.bar(names, ratios, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Compression Ratio")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Add horizontal line at 1.0 (no compression)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No compression")

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio:.3f}",
            ha="center",
            fontsize=10,
        )

    return ax


def plot_q_structure(
    q_stats_df: "object",  # pd.DataFrame
    ax: "Axes | None" = None,
    title: str = "Q-Value Structure",
) -> "Axes":
    """
    Multi-panel plot of Q-value statistics.

    Args:
        q_stats_df: DataFrame from features.q_stats()
        ax: Matplotlib axes (creates 2x2 if None)
        title: Overall title

    Returns:
        Matplotlib axes array
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # Q-spread distribution
    axes[0, 0].hist(q_stats_df["q_spread"], bins=50, color="steelblue", alpha=0.7)
    axes[0, 0].set_xlabel("Q-spread (best - worst)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Q-Spread Distribution")

    # Q-gap distribution
    axes[0, 1].hist(q_stats_df["q_gap"], bins=50, color="coral", alpha=0.7)
    axes[0, 1].set_xlabel("Q-gap (best - 2nd best)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Q-Gap Distribution")

    # N-legal distribution
    axes[1, 0].hist(q_stats_df["n_legal"], bins=7, range=(0.5, 7.5), color="seagreen", alpha=0.7)
    axes[1, 0].set_xlabel("Number of legal moves")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Legal Moves Distribution")

    # N-optimal distribution
    axes[1, 1].hist(q_stats_df["n_optimal"], bins=7, range=(0.5, 7.5), color="orchid", alpha=0.7)
    axes[1, 1].set_xlabel("Number of optimal moves (ties)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Optimal Moves (Ties) Distribution")

    plt.tight_layout()
    return axes


def create_summary_table(
    metrics: dict[str, float],
    title: str = "Analysis Summary",
) -> str:
    """
    Create formatted markdown table of metrics.

    Args:
        metrics: Dict mapping metric name to value
        title: Table title

    Returns:
        Markdown-formatted table string
    """
    lines = [f"### {title}", "", "| Metric | Value |", "|--------|-------|"]
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {name} | {value:.4f} |")
        else:
            lines.append(f"| {name} | {value} |")
    return "\n".join(lines)
