"""Candlewax renderer — PDF histograms as an image, one row per legal play.

Mirrors the information density of ``forge/analysis/results/web/eq_pdf_discs.html``
(the right panel — "Player N's Hand" with per-domino histograms). The image is
what clicked for a Claude instance reading the candlewax for the first time;
this tool gives that same view to Opus as a multimodal attachment.

Design choices locked in for v1:

- One row per legal play. Rows sorted by play (descending p_make) so the
  "best bet" is on top but the whole space is visible.
- X axis: Q ∈ [-42, +42], shared across rows.
- Y axis: PDF mass (normalized per row — we care about shape, not absolute
  scale).
- Color: red→yellow→green gradient keyed to bin Q value. Matches the
  screenshot's intuition that "left=bad, right=good".
- Vertical threshold line at Q = +18 (offense make point) or Q = -17 (defense
  make point), depending on `is_offense`. A faint dashed line at Q = 0 for
  reference.
- Per-row annotations on the right: μ, p_make %, offense/defense pill.
- Header: trump declaration, your seat, offense/defense.

The renderer is pure: game_state → list[OutcomeDistribution] → PNG bytes.
No I/O side effects except the final `savefig` / `to_bytes`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from burl.tools.eq_distribution import OutcomeDistribution, Q_VALUES


# --------------------------------------------------------------------------- #
# Labels                                                                       #
# --------------------------------------------------------------------------- #

_DOMINO_LABELS: dict[int, str] = {
    0: "0-0", 1: "1-0", 2: "1-1", 3: "2-0", 4: "2-1", 5: "2-2",
    6: "3-0", 7: "3-1", 8: "3-2", 9: "3-3", 10: "4-0", 11: "4-1",
    12: "4-2", 13: "4-3", 14: "4-4", 15: "5-0", 16: "5-1", 17: "5-2",
    18: "5-3", 19: "5-4", 20: "5-5", 21: "6-0", 22: "6-1", 23: "6-2",
    24: "6-3", 25: "6-4", 26: "6-5", 27: "6-6",
}


def _domino_label(d: int) -> str:
    return _DOMINO_LABELS.get(int(d), f"d{int(d)}")


# --------------------------------------------------------------------------- #
# Color palette                                                                #
# --------------------------------------------------------------------------- #

# Gradient: deep red (Q=-42) → orange → yellow (~Q=0) → green (Q=+42).
# Matches the teacher-artifact HTML's color semantics.
_CANDLEWAX_CMAP = LinearSegmentedColormap.from_list(
    "candlewax",
    [
        (0.00, "#7a1616"),   # deep red — disaster
        (0.35, "#d94a1a"),   # red-orange
        (0.50, "#d97c1a"),   # orange
        (0.55, "#d9b21a"),   # gold (~Q=0)
        (0.70, "#6fc93b"),   # lime
        (1.00, "#1a9641"),   # green — winning
    ],
)


def _bar_colors(n_bins: int = 85) -> np.ndarray:
    """RGBA colors for each Q bin."""
    # Q = [-42, +42]; normalize to [0, 1].
    t = np.linspace(0.0, 1.0, n_bins)
    return _CANDLEWAX_CMAP(t)


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #

_ROW_HEIGHT_IN = 0.85
_HEADER_HEIGHT_IN = 0.55
_FIG_WIDTH_IN = 10.0
_LEFT_LABEL_IN = 0.85    # "6-2" tag on the left
_RIGHT_STATS_IN = 1.85   # μ, p_make, offense tag on the right
_HIST_BG = "#0d1117"
_FIG_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_MUTED = "#8b949e"
_GRID_COLOR = "#30363d"
_WIN_BAND_COLOR = "#1f6feb"    # blue reference line for the make threshold
_ZERO_LINE_COLOR = "#6e7681"

_DPI = 160


# --------------------------------------------------------------------------- #
# Header rendering                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class CandlewaxHeader:
    """Header row context — renders to the top band of the figure."""
    trump: str
    is_offense: bool
    my_seat: int
    trick_no: int
    position_in_trick: int
    decision_label: str | None = None   # optional short tag e.g. "seed 900013"


def _render_header(ax: plt.Axes, hdr: CandlewaxHeader) -> None:
    ax.set_facecolor(_HIST_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)

    # Title (top line, bold) + subtitle (bottom line, muted).
    ax.text(
        0.02, 0.72, f"Player {hdr.my_seat}'s Hand",
        color=_TEXT_COLOR, fontsize=13, fontweight="bold", va="center",
    )
    if hdr.decision_label:
        ax.text(
            0.02, 0.28, hdr.decision_label,
            color=_MUTED, fontsize=9, va="center",
        )

    # Pills on the right side. Offense/defense pill, then trump.
    pill_color = "#238636" if hdr.is_offense else "#8250df"
    pill_text = (
        "Offense (win ≥ +18)"
        if hdr.is_offense
        else "Defense (bidder < 30 → Q ≥ -17)"
    )
    ax.add_patch(FancyBboxPatch(
        (0.48, 0.58), 0.26, 0.35,
        boxstyle="round,pad=0.015",
        linewidth=0, facecolor=pill_color, alpha=0.95,
    ))
    ax.text(
        0.61, 0.755, pill_text,
        color="#ffffff", fontsize=8.5, fontweight="bold", va="center", ha="center",
    )

    # Trump pill
    trump_color = "#9e7b00"
    ax.add_patch(FancyBboxPatch(
        (0.76, 0.58), 0.22, 0.35,
        boxstyle="round,pad=0.015",
        linewidth=0, facecolor=trump_color, alpha=0.95,
    ))
    ax.text(
        0.87, 0.755, f"Trump: {hdr.trump}",
        color="#ffffff", fontsize=8.5, fontweight="bold", va="center", ha="center",
    )

    # Trick / position indicator under the pills.
    ax.text(
        0.98, 0.24, f"trick {hdr.trick_no}  ·  pos {hdr.position_in_trick}/4",
        color=_MUTED, fontsize=9, va="center", ha="right",
    )


# --------------------------------------------------------------------------- #
# Row rendering                                                                #
# --------------------------------------------------------------------------- #


def _render_row(
    ax_label: plt.Axes,
    ax_hist: plt.Axes,
    ax_stats: plt.Axes,
    dist: OutcomeDistribution,
    make_threshold_q: float,
    colors: np.ndarray,
) -> None:
    # --- left label ---
    ax_label.set_facecolor(_HIST_BG)
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(0, 1)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)
    ax_label.text(
        0.5, 0.5, _domino_label(dist.play),
        color=_TEXT_COLOR, fontsize=14, fontweight="bold",
        va="center", ha="center",
    )

    # --- histogram ---
    ax_hist.set_facecolor(_HIST_BG)
    pdf = np.asarray(dist.pdf_bins, dtype=np.float32)
    ymax = max(float(pdf.max()), 1e-6)

    # Bars — one per bin, colored by bin Q value.
    ax_hist.bar(
        Q_VALUES, pdf, width=1.0, color=colors,
        edgecolor="none", linewidth=0, align="center",
    )

    # Zero reference line.
    ax_hist.axvline(0.0, color=_ZERO_LINE_COLOR, linewidth=0.8, alpha=0.6, linestyle="--")
    # Make threshold line (offense +18 or defense -17).
    ax_hist.axvline(
        make_threshold_q, color=_WIN_BAND_COLOR, linewidth=1.3, alpha=0.9,
    )

    # Mean tick at the top.
    ax_hist.plot(
        [dist.mean], [ymax * 1.05], marker="v",
        color=_TEXT_COLOR, markersize=7, clip_on=False,
    )

    ax_hist.set_xlim(-42, 42)
    ax_hist.set_ylim(0, ymax * 1.15)
    ax_hist.set_yticks([])
    # X ticks only on the bottom row — set later in the grid function.
    for spine in ax_hist.spines.values():
        spine.set_color(_GRID_COLOR)
        spine.set_linewidth(0.6)
    ax_hist.tick_params(axis="x", colors=_MUTED, labelsize=8)

    # --- right stats ---
    ax_stats.set_facecolor(_HIST_BG)
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_visible(False)

    mean_color = _mean_color(dist.mean)
    pmake_color = _pmake_color(dist.p_make)

    ax_stats.text(
        0.02, 0.75, f"p_make {dist.p_make * 100:.0f}%",
        color=pmake_color, fontsize=11, fontweight="bold", va="center",
    )
    sign = "+" if dist.mean >= 0 else ""
    ax_stats.text(
        0.02, 0.30, f"μ = {sign}{dist.mean:.1f}",
        color=mean_color, fontsize=10, va="center",
    )

    # Shape tag if non-unimodal — visual cue the shape has a story.
    if dist.distribution_shape != "unimodal":
        ax_stats.text(
            0.60, 0.50, dist.distribution_shape,
            color="#f0883e", fontsize=8, style="italic",
            va="center", ha="left",
        )


def _mean_color(mean: float) -> str:
    if mean >= 10:
        return "#2ea043"
    if mean >= 3:
        return "#6fc93b"
    if mean >= -3:
        return "#d9b21a"
    if mean >= -10:
        return "#d94a1a"
    return "#7a1616"


def _pmake_color(p: float) -> str:
    if p >= 0.7:
        return "#2ea043"
    if p >= 0.4:
        return "#d9b21a"
    return "#f85149"


# --------------------------------------------------------------------------- #
# Composition                                                                  #
# --------------------------------------------------------------------------- #


def render_candlewax(
    dists: Iterable[OutcomeDistribution],
    header: CandlewaxHeader,
    *,
    sort_by_p_make: bool = True,
) -> bytes:
    """Render the candlewax panel for one decision to PNG bytes.

    ``dists`` should be the ``OutcomeDistribution`` for each legal play in the
    decision (one row per play). ``header`` carries the game-level context that
    the screenshot shows: trump, seat, offense/defense.

    Returns raw PNG bytes; caller decides whether to write to disk or embed
    directly as a base64 multimodal block.
    """
    dist_list = list(dists)
    if not dist_list:
        raise ValueError("render_candlewax: no distributions provided")
    if sort_by_p_make:
        dist_list = sorted(dist_list, key=lambda d: -float(d.p_make))

    # All rows share the same make threshold (it's a game-state invariant,
    # not a per-play one).
    is_offense = bool(dist_list[0].is_offense)
    make_threshold_q = 18.0 if is_offense else -17.0

    n_rows = len(dist_list)
    fig_height = _HEADER_HEIGHT_IN + n_rows * _ROW_HEIGHT_IN + 0.25
    fig = plt.figure(figsize=(_FIG_WIDTH_IN, fig_height), facecolor=_FIG_BG, dpi=_DPI)

    # Grid: rows = 1 (header) + n_rows (plays) + 1 (xlabel margin).
    # Columns = [label, histogram, stats].
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=3,
        width_ratios=[_LEFT_LABEL_IN, _FIG_WIDTH_IN - _LEFT_LABEL_IN - _RIGHT_STATS_IN, _RIGHT_STATS_IN],
        height_ratios=[_HEADER_HEIGHT_IN] + [_ROW_HEIGHT_IN] * n_rows,
        hspace=0.15, wspace=0.05,
        left=0.02, right=0.98, top=0.97, bottom=0.07,
        figure=fig,
    )

    # Header spans all three columns of row 0.
    ax_header = fig.add_subplot(gs[0, :])
    _render_header(ax_header, header)

    colors = _bar_colors()

    for i, dist in enumerate(dist_list):
        row_idx = i + 1
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_hist = fig.add_subplot(gs[row_idx, 1])
        ax_stats = fig.add_subplot(gs[row_idx, 2])
        _render_row(ax_label, ax_hist, ax_stats, dist, make_threshold_q, colors)
        # Only the bottom hist row keeps its x-tick labels.
        if i < n_rows - 1:
            ax_hist.set_xticklabels([])

    # Label the bottom x-axis with the Q scale reminder.
    ax_hist.set_xlabel("Q (points, -42..+42)", color=_MUTED, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=_FIG_BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
