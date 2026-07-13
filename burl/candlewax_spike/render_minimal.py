"""Candlewax renderer — MINIMAL variant.

Strips every text annotation (pills, μ marker, p_make, shape tag, trick badge)
and keeps only the shape-bearing signal:

- The histograms themselves (same red→green Q-gradient as v1).
- A subtle translucent green band over the winning region (Q ≥ +18 for offense,
  Q ≥ -17 for defense) rendered BEHIND the bars.
- The domino label on the left ("6-2").
- Minimal header: "Player N's Hand · seed NNNNNN".
- X-axis Q ticks at the bottom.
- A very faint 0-line and a faint make-threshold line.

Theory: over many training iterations, printed text annotations become the
shortcut the model learns; hiding them forces the model to read the shapes
themselves, which is the durable skill we want.

Same ``CandlewaxHeader`` dataclass as ``render.py``. Same figure width, dpi,
and background color — only the content is stripped.
"""

from __future__ import annotations

import io
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from burl.candlewax_spike.render import (
    CandlewaxHeader,
    _CANDLEWAX_CMAP,
    _bar_colors,
    _domino_label,
)
from burl.tools.eq_distribution import OutcomeDistribution, Q_VALUES


# --------------------------------------------------------------------------- #
# Layout (mirrors render.py so the two variants are visually comparable)      #
# --------------------------------------------------------------------------- #

_ROW_HEIGHT_IN = 0.85
_HEADER_HEIGHT_IN = 0.45
_FIG_WIDTH_IN = 10.0
_LEFT_LABEL_IN = 0.85
_RIGHT_MARGIN_IN = 0.25     # tiny right gutter so bars don't touch the edge
_HIST_BG = "#0d1117"
_FIG_BG = "#0d1117"
_TEXT_COLOR = "#c9d1d9"
_MUTED = "#8b949e"
_GRID_COLOR = "#30363d"
_WIN_BAND_FILL = "#2ea043"     # translucent green fill over the winning region
_WIN_BAND_ALPHA = 0.12
_THRESHOLD_LINE_COLOR = "#2ea043"
_ZERO_LINE_COLOR = "#6e7681"

_DPI = 160


# --------------------------------------------------------------------------- #
# Header (minimal — no pills, no subtitle)                                    #
# --------------------------------------------------------------------------- #


def _render_minimal_header(ax: plt.Axes, hdr: CandlewaxHeader) -> None:
    ax.set_facecolor(_HIST_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Pull out a seed number from the decision_label if present — we only want
    # the minimal "Player N's Hand · seed NNNNNN" line, not the full label.
    seed_part = ""
    if hdr.decision_label:
        # decision_label is typically "seed 900013 · decl 7 · narrator 2";
        # keep the seed token only.
        for token in hdr.decision_label.split("·"):
            token = token.strip()
            if token.lower().startswith("seed "):
                seed_part = token
                break
    title = f"Player {hdr.my_seat}'s Hand"
    if seed_part:
        title = f"{title}  ·  {seed_part}"

    ax.text(
        0.02, 0.5, title,
        color=_TEXT_COLOR, fontsize=13, fontweight="bold", va="center",
    )


# --------------------------------------------------------------------------- #
# Row (minimal — label + histogram, no right stats column)                    #
# --------------------------------------------------------------------------- #


def _render_minimal_row(
    ax_label: plt.Axes,
    ax_hist: plt.Axes,
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

    # Translucent green band over the winning region, rendered BEHIND the bars
    # (lowest zorder). Bars/lines overlay this so readability is preserved.
    # axvspan spans the full y-range of the axes automatically.
    ax_hist.axvspan(
        make_threshold_q, 42.0,
        color=_WIN_BAND_FILL, alpha=_WIN_BAND_ALPHA,
        linewidth=0, zorder=0,
    )

    # Bars — same coloring as v1.
    ax_hist.bar(
        Q_VALUES, pdf, width=1.0, color=colors,
        edgecolor="none", linewidth=0, align="center", zorder=2,
    )

    # Faint zero reference line.
    ax_hist.axvline(
        0.0, color=_ZERO_LINE_COLOR, linewidth=0.6, alpha=0.35,
        linestyle="--", zorder=1,
    )
    # Faint make-threshold line (offense +18 or defense -17).
    ax_hist.axvline(
        make_threshold_q, color=_THRESHOLD_LINE_COLOR, linewidth=0.8,
        alpha=0.45, zorder=1,
    )

    ax_hist.set_xlim(-42, 42)
    ax_hist.set_ylim(0, ymax * 1.15)
    ax_hist.set_yticks([])
    for spine in ax_hist.spines.values():
        spine.set_color(_GRID_COLOR)
        spine.set_linewidth(0.6)
    ax_hist.tick_params(axis="x", colors=_MUTED, labelsize=8)


# --------------------------------------------------------------------------- #
# Composition                                                                  #
# --------------------------------------------------------------------------- #


def render_candlewax_minimal(
    dists: Iterable[OutcomeDistribution],
    header: CandlewaxHeader,
    *,
    sort_by_p_make: bool = True,
) -> bytes:
    """Render the MINIMAL candlewax panel for one decision to PNG bytes.

    Same signature as ``render.render_candlewax``. See module docstring for
    what's stripped relative to v1.
    """
    dist_list = list(dists)
    if not dist_list:
        raise ValueError("render_candlewax_minimal: no distributions provided")
    if sort_by_p_make:
        dist_list = sorted(dist_list, key=lambda d: -float(d.p_make))

    is_offense = bool(dist_list[0].is_offense)
    make_threshold_q = 18.0 if is_offense else -17.0

    n_rows = len(dist_list)
    fig_height = _HEADER_HEIGHT_IN + n_rows * _ROW_HEIGHT_IN + 0.25
    fig = plt.figure(figsize=(_FIG_WIDTH_IN, fig_height), facecolor=_FIG_BG, dpi=_DPI)

    # Grid: rows = 1 (header) + n_rows (plays).
    # Columns = [label, histogram]. No right stats column.
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=2,
        width_ratios=[_LEFT_LABEL_IN, _FIG_WIDTH_IN - _LEFT_LABEL_IN - _RIGHT_MARGIN_IN],
        height_ratios=[_HEADER_HEIGHT_IN] + [_ROW_HEIGHT_IN] * n_rows,
        hspace=0.15, wspace=0.05,
        left=0.02, right=0.98, top=0.97, bottom=0.07,
        figure=fig,
    )

    ax_header = fig.add_subplot(gs[0, :])
    _render_minimal_header(ax_header, header)

    colors = _bar_colors()

    ax_hist = None
    for i, dist in enumerate(dist_list):
        row_idx = i + 1
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_hist = fig.add_subplot(gs[row_idx, 1])
        _render_minimal_row(ax_label, ax_hist, dist, make_threshold_q, colors)
        if i < n_rows - 1:
            ax_hist.set_xticklabels([])

    # Label the bottom x-axis with the Q scale reminder.
    if ax_hist is not None:
        ax_hist.set_xlabel("Q (points, -42..+42)", color=_MUTED, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=_FIG_BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


__all__ = ["render_candlewax_minimal", "CandlewaxHeader", "_CANDLEWAX_CMAP"]
