"""Calibration eval for Zeb's belief head.

When Zeb says P(opponent holds domino) = 0.72, is the true frequency really
72%?  Before Burl composes Zeb's beliefs into reasoning, we need to know.

Outputs (per run, under ``burl/eval/results/belief_calibration_<ts>/``):

- ``report.md`` -- tables + headline numbers.
- ``reliability.png`` -- reliability diagrams (overall + one per opponent).
- ``brier_breakdown.png`` -- Brier score by domino features.
- ``raw_predictions.npz`` -- (pred, truth, domino_id, opp_class) for replay.

The eval is deliberately independent of ``burl.tools.zeb``: if that wrapper is
available it's used, otherwise the script falls back to calling the raw Zeb
belief head directly. Same numbers either way.

Usage::

    python -u -m burl.eval.belief_calibration --n-states 1000

Library entry point: :func:`run_calibration`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINO_HIGH, DOMINO_IS_DOUBLE, DOMINO_LOW
from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions, new_game
from forge.zeb.observation import observe
from forge.zeb.types import ZebGameState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SEED_MIN = 900_000
EVAL_SEED_MAX = 910_000  # exclusive
N_DOMINOES = 28
N_OPPONENTS = 3  # L, partner, R (relative)
OPP_NAMES = ("left", "partner", "right")

# Default checkpoint. Of the two options the task called out
# (``forge/zeb/large-belief-bootstrap.pt`` vs. the latest ``learner-cycle*.pt``),
# we pick ``forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt`` -- the latest
# bootstrap derived from the HF ``large-belief.pt`` snapshot at training cycle
# 3740 (1.7M self-play games; see ``forge/zeb/models/large-belief-recap.md``).
#
# Why not ``forge/zeb/large-belief-bootstrap.pt``? Its belief_proj weights are
# at initialization-scale (std ~0.036, max ~0.06) -- the belief head there is
# *untrained*. Loading it gives ~30% top-1, barely above the 33% random-guess
# floor. The ``lb-v-eq-3740`` snapshot has std ~0.16--0.23 and replicates the
# 72% training-style top-1 cited in OVERVIEW.
#
# Why not the ``learner-cycle*.pt`` files in ``forge/zeb/checkpoints/``? They
# are the medium-sized model (embed_dim=128) without a belief head and would
# crash the loader.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt"


# ---------------------------------------------------------------------------
# Wrapper integration: use burl.tools.zeb if present, fall back to raw forge.
# ---------------------------------------------------------------------------


@dataclass
class BeliefBackend:
    """Adapter that returns a ``[28, 3]`` softmax matrix for a given state + seat.

    Wraps either ``burl.tools.zeb`` (if available) or the raw Zeb belief head
    loaded from a checkpoint. The eval logic uses only :meth:`matrix`.
    """

    name: str  # 'burl.tools.zeb' or 'forge.zeb.raw'
    checkpoint: Path
    epoch: int | None
    _call: Callable[[ZebGameState, int], torch.Tensor]

    def matrix(self, state: ZebGameState, me_seat: int) -> torch.Tensor:
        return self._call(state, me_seat)


def _load_backend(checkpoint: Path, device: str) -> BeliefBackend:
    """Prefer ``burl.tools.zeb``; fall back to direct forge loading if unavailable."""
    try:  # pragma: no cover -- selection is empirical, not tested
        from burl.tools.zeb import get_belief_matrix, load_belief_model
    except ImportError:
        load_belief_model = None

    if load_belief_model is not None:
        try:
            bm = load_belief_model(checkpoint_path=checkpoint, device=device)
            return BeliefBackend(
                name="burl.tools.zeb",
                checkpoint=bm.checkpoint_path,
                epoch=bm.epoch,
                _call=lambda s, seat: get_belief_matrix(s, me_seat=seat, model=bm),
            )
        except Exception as exc:
            print(f"[warn] burl.tools.zeb failed ({exc}); falling back to raw forge path", file=sys.stderr)

    from forge.zeb import load_model

    model, ckpt = load_model(str(checkpoint), device=device, eval_mode=True)
    if not getattr(model, "has_belief_head", False):
        raise ValueError(
            f"Checkpoint {checkpoint} has no belief head. Pick a *-belief-*.pt snapshot."
        )
    dev = torch.device(device)

    def _direct(state: ZebGameState, me_seat: int) -> torch.Tensor:
        tokens, mask, hand_indices = observe(state, me_seat)
        hand_mask = torch.ones(7, dtype=torch.bool)
        t = tokens.unsqueeze(0).to(dev)
        m = mask.unsqueeze(0).to(dev)
        hi = hand_indices.unsqueeze(0).to(dev)
        hm = hand_mask.unsqueeze(0).to(dev)
        with torch.no_grad():
            _pol, _val, blogits = model(t, m, hi, hm)
        assert blogits is not None
        return F.softmax(blogits.float(), dim=-1).squeeze(0).cpu()

    return BeliefBackend(
        name="forge.zeb.raw",
        checkpoint=checkpoint,
        epoch=ckpt.get("epoch"),
        _call=_direct,
    )


# ---------------------------------------------------------------------------
# State sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledState:
    """One eval state sample: state + perspective seat."""

    state: ZebGameState
    me_seat: int
    seed: int
    trick_idx: int  # 0..6, inferred from len(played) // 4
    decl_id: int


def _random_legal_action(state: ZebGameState, rng: random.Random) -> int:
    legal = legal_actions(state)
    return rng.choice(legal)


def _sample_states_from_seed(
    seed: int,
    rng: random.Random,
    samples_per_game: int,
) -> list[SampledState]:
    """Replay a random rollout and snapshot ``samples_per_game`` intermediate states.

    We exclude:
    - the initial state (trick 0, no plays yet -- prior is the uniform marginal)
    - terminal states (no hidden dominoes left; belief is degenerate)
    and sample uniformly over the remaining 27 pre-trick-boundary positions.
    """
    # The observation encoding assumes phase=PLAYING, which new_game(skip_bidding=True) gives.
    state = new_game(seed=seed, skip_bidding=True)

    # Collect all non-terminal states along the random-rollout trajectory.
    # We sample *after* at least one play so there's visible play history,
    # and *before* the final play so dominoes remain hidden.
    states: list[ZebGameState] = []
    while not is_terminal(state):
        states.append(state)
        action = _random_legal_action(state, rng)
        state = apply_action(state, action)

    # states now has the 28 decision points. Drop the first (no plays yet).
    usable = states[1:]  # 27 states, trick distribution [0(partial)..6(partial)]
    if not usable:
        return []

    # Uniform-random subset -- don't bias toward any trick.
    k = min(samples_per_game, len(usable))
    indices = rng.sample(range(len(usable)), k)

    out: list[SampledState] = []
    for idx in indices:
        s = usable[idx]
        me = current_player(s)
        trick_idx = len(s.played) // 4
        out.append(
            SampledState(state=s, me_seat=me, seed=seed, trick_idx=trick_idx, decl_id=s.decl_id)
        )
    return out


def sample_eval_states(
    n_states: int,
    eval_seed: int,
    samples_per_game: int = 3,
) -> list[SampledState]:
    """Draw ``n_states`` eval states with balanced declaration coverage.

    Strategy:
    - Seeds are drawn from the held-out range 900000..909999.
    - Declarations would otherwise skew toward the per-seed random draw, so we
      filter the seed stream to keep all 10 declarations represented roughly
      uniformly (target ``n_states // 10`` per declaration).
    - Each seed contributes up to ``samples_per_game`` intermediate states,
      drawn uniformly over the 27 non-trivial positions in the trajectory.
    """
    if eval_seed < EVAL_SEED_MIN:
        raise ValueError(
            f"eval_seed must be in held-out range [{EVAL_SEED_MIN}, {EVAL_SEED_MAX}); got {eval_seed}"
        )

    rng = random.Random(eval_seed)
    target_per_decl = max(1, n_states // N_DECLS)
    per_decl_count = [0] * N_DECLS
    results: list[SampledState] = []

    seed_pool = list(range(EVAL_SEED_MIN, EVAL_SEED_MAX))
    rng.shuffle(seed_pool)

    # Pre-bin seeds by their declaration so we can interleave them. This avoids
    # the pathology where a short run (n_states ~ 50) stops before ever seeing
    # a rare declaration like ``notrump``.
    seeds_by_decl: dict[int, list[int]] = {d: [] for d in range(N_DECLS)}
    for seed in seed_pool:
        probe = new_game(seed=seed, skip_bidding=True)
        seeds_by_decl[probe.decl_id].append(seed)

    # Round-robin across declarations until we hit `n_states` or run out.
    decl_cursor = [0] * N_DECLS
    while len(results) < n_states:
        made_progress = False
        for d in range(N_DECLS):
            if len(results) >= n_states:
                break
            # Cap per-decl until every decl has at least one sample, to keep
            # coverage even when the pool is heavily shuffled.
            min_count = min(per_decl_count)
            if per_decl_count[d] > min_count and any(c == 0 for c in per_decl_count):
                continue
            if decl_cursor[d] >= len(seeds_by_decl[d]):
                continue
            seed = seeds_by_decl[d][decl_cursor[d]]
            decl_cursor[d] += 1
            samples = _sample_states_from_seed(seed, rng, samples_per_game)
            for s in samples:
                assert s.seed >= EVAL_SEED_MIN, f"eval state leaked training seed {s.seed}"
                assert not is_terminal(s.state), "sampled a terminal state"
            for s in samples:
                if len(results) >= n_states:
                    break
                results.append(s)
                per_decl_count[s.decl_id] += 1
                made_progress = True
        if not made_progress:
            break

    if len(results) < n_states:
        print(
            f"[warn] requested {n_states} states but only produced {len(results)} "
            f"(held-out pool exhausted)",
            file=sys.stderr,
        )
    return results[:n_states]


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


@dataclass
class RawPredictions:
    """Flat arrays of per-(state, domino, class) predictions + truth.

    Two top-1 regimes are tracked because they measure different things:

    - ``top1_correct`` (hidden-only): restricted to dominoes still in an
      opponent's hand at query time. This is the operationally relevant metric
      for Burl -- the wrapper rejects visible dominoes -- and is what Burl
      actually gets when it asks "who holds this hidden domino?".
    - ``top1_all_correct`` (training-style): includes already-played dominoes.
      Matches how the training loop reports belief_accuracy (mask = "not my
      original hand"). For a played domino the "correct" answer is who
      *originally* held it, which is visible in the play history; these are
      trivially-easy examples that inflate the accuracy.

    Expect a large gap between the two: the 72% cited in OVERVIEW is the
    training-style number.
    """

    pred: np.ndarray  # [N] predicted P(class is true owner)  (hidden-only)
    truth: np.ndarray  # [N] binary 0/1  (hidden-only)
    domino_id: np.ndarray  # [N] 0..27  (hidden-only)
    opp_class: np.ndarray  # [N] 0..2 (0=L, 1=partner, 2=R)
    decl_id: np.ndarray  # [N] 0..9
    trick_idx: np.ndarray  # [N] 0..6
    # Per-(state, hidden-domino) top-1.
    top1_correct: np.ndarray  # [M_hidden]
    top1_decl_id: np.ndarray  # [M_hidden]
    top1_trick_idx: np.ndarray  # [M_hidden]
    top1_domino_id: np.ndarray  # [M_hidden]
    # Per-(state, not-self-domino) top-1 -- training-style.
    top1_all_correct: np.ndarray  # [M_all]
    top1_all_is_played: np.ndarray  # [M_all] bool: was domino already played?

    def __len__(self) -> int:
        return int(self.pred.shape[0])


def _collect(
    backend: BeliefBackend,
    samples: list[SampledState],
    progress: bool = True,
) -> RawPredictions:
    """Query the belief head over every hidden domino for each sample."""
    n = len(samples)
    # Worst case: 28 dominoes * 3 classes * N states
    cap = n * N_DOMINOES * N_OPPONENTS
    pred = np.empty(cap, dtype=np.float32)
    truth = np.empty(cap, dtype=np.uint8)
    dom_id = np.empty(cap, dtype=np.int16)
    opp_cls = np.empty(cap, dtype=np.int8)
    decl = np.empty(cap, dtype=np.int8)
    trick = np.empty(cap, dtype=np.int8)

    top1_cap = n * N_DOMINOES
    top1 = np.empty(top1_cap, dtype=np.uint8)
    top1_decl = np.empty(top1_cap, dtype=np.int8)
    top1_trick = np.empty(top1_cap, dtype=np.int8)
    top1_dom = np.empty(top1_cap, dtype=np.int16)
    # Training-style: includes played dominoes.
    top1_all = np.empty(top1_cap, dtype=np.uint8)
    top1_all_played = np.empty(top1_cap, dtype=bool)

    write = 0
    t1_write = 0
    t1_all_write = 0
    skipped_states = 0

    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        matrix = backend.matrix(s.state, s.me_seat).numpy()  # [28, 3]

        # True ownership per domino, relative to me_seat.
        true_rel_owner = np.empty(N_DOMINOES, dtype=np.int8)
        for seat in range(4):
            for d in s.state.hands[seat]:
                true_rel_owner[d] = (seat - s.me_seat + 4) % 4

        # Training-style: every domino not in my original hand (includes played).
        my_original = set(s.state.hands[s.me_seat])
        not_self_ids = [d for d in range(N_DOMINOES) if d not in my_original]
        played_set = set(s.state.played)
        for d in not_self_ids:
            true_rel = int(true_rel_owner[d])
            true_cls = true_rel - 1
            top1_all[t1_all_write] = 1 if int(matrix[d].argmax()) == true_cls else 0
            top1_all_played[t1_all_write] = d in played_set
            t1_all_write += 1

        # Hidden dominoes: not in my hand and not yet played. This is the
        # subset Burl actually asks about via the wrapper.
        visible = my_original | played_set
        hidden_mask = np.array([d not in visible for d in range(N_DOMINOES)])
        hidden_ids = np.nonzero(hidden_mask)[0]
        if len(hidden_ids) == 0:
            skipped_states += 1
            continue

        for d in hidden_ids:
            true_rel = int(true_rel_owner[d])
            # Hidden-from-me means rel_owner in {1, 2, 3} (not 0/self).
            assert true_rel != 0, f"hidden domino {d} apparently owned by self"
            true_cls = true_rel - 1  # class 0=L, 1=partner, 2=R

            # Per-class (pred, truth) entries.
            for c in range(N_OPPONENTS):
                pred[write] = matrix[d, c]
                truth[write] = 1 if c == true_cls else 0
                dom_id[write] = d
                opp_cls[write] = c
                decl[write] = s.decl_id
                trick[write] = s.trick_idx
                write += 1

            # Per-(state, domino) top-1 correctness.
            top1[t1_write] = 1 if int(matrix[d].argmax()) == true_cls else 0
            top1_decl[t1_write] = s.decl_id
            top1_trick[t1_write] = s.trick_idx
            top1_dom[t1_write] = d
            t1_write += 1

        if progress and ((i + 1) % 50 == 0 or i + 1 == n):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1:5d}/{n}] {rate:.1f} states/s  "
                f"({write:,} preds, {t1_write:,} argmax decisions)"
            )

    if skipped_states:
        print(f"[info] skipped {skipped_states} states with no hidden dominoes", file=sys.stderr)

    return RawPredictions(
        pred=pred[:write],
        truth=truth[:write],
        domino_id=dom_id[:write],
        opp_class=opp_cls[:write],
        decl_id=decl[:write],
        trick_idx=trick[:write],
        top1_correct=top1[:t1_write],
        top1_decl_id=top1_decl[:t1_write],
        top1_trick_idx=top1_trick[:t1_write],
        top1_domino_id=top1_dom[:t1_write],
        top1_all_correct=top1_all[:t1_all_write],
        top1_all_is_played=top1_all_played[:t1_all_write],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean((pred - truth.astype(np.float32)) ** 2))


def _reliability_bins(pred: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> dict:
    """Return per-bin (mean_pred, empirical, count) arrays for a reliability plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive on the final edge.
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    empirical = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        count[b] = c
        if c > 0:
            mean_pred[b] = float(pred[sel].mean())
            empirical[b] = float(truth[sel].mean())
    return {
        "edges": edges,
        "mean_pred": mean_pred,
        "empirical": empirical,
        "count": count,
    }


def ece(pred: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (L1 over bins, weighted by bin population)."""
    if pred.size == 0:
        return float("nan")
    bins = _reliability_bins(pred, truth, n_bins=n_bins)
    total = bins["count"].sum()
    gap = np.abs(bins["mean_pred"] - bins["empirical"])
    return float(np.sum(gap * bins["count"]) / total)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _plot_reliability(raw: RawPredictions, out_path: Path, n_bins: int = 10) -> None:
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        ("Overall (all opponents)", None),
        (f"Opponent: {OPP_NAMES[0]} (left)", 0),
        (f"Opponent: {OPP_NAMES[1]} (partner)", 1),
        (f"Opponent: {OPP_NAMES[2]} (right)", 2),
    ]

    for ax, (title, cls) in zip(axes.flat, panels):
        if cls is None:
            sel = slice(None)
        else:
            sel = raw.opp_class == cls
        p = raw.pred[sel]
        t = raw.truth[sel]
        bins = _reliability_bins(p, t, n_bins=n_bins)
        centers = 0.5 * (bins["edges"][:-1] + bins["edges"][1:])
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfect")
        ax.plot(
            bins["mean_pred"][bins["count"] > 0],
            bins["empirical"][bins["count"] > 0],
            "o-",
            color="#1f77b4",
            lw=1.5,
            ms=6,
            label="model",
        )
        # Population histogram as bars on a secondary axis.
        ax2 = ax.twinx()
        ax2.bar(centers, bins["count"], width=1.0 / n_bins * 0.9, color="#1f77b4", alpha=0.15)
        ax2.set_ylabel("count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.spines["top"].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("empirical frequency")
        ax.set_title(f"{title}  (Brier={brier(p, t):.4f}, ECE={ece(p, t, n_bins):.4f})")
        ax.legend(loc="upper left")

    fig.suptitle("Zeb belief-head reliability (held-out eval seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_brier_breakdown(raw: RawPredictions, out_path: Path) -> None:
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: per-domino Brier, ordered by id, colored by double vs non-double.
    per_dom: list[float] = []
    is_double: list[bool] = []
    count_pts: list[int] = []
    for d in range(N_DOMINOES):
        sel = raw.domino_id == d
        per_dom.append(brier(raw.pred[sel], raw.truth[sel]))
        is_double.append(bool(DOMINO_IS_DOUBLE[d]))
        count_pts.append(int(DOMINO_COUNT_POINTS[d]))
    per_dom_arr = np.array(per_dom)
    colors = ["#d62728" if dbl else "#1f77b4" for dbl in is_double]
    ax = axes[0]
    ax.bar(np.arange(N_DOMINOES), per_dom_arr, color=colors)
    labels = [f"{DOMINO_HIGH[d]}-{DOMINO_LOW[d]}" for d in range(N_DOMINOES)]
    ax.set_xticks(np.arange(N_DOMINOES))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Brier")
    ax.set_title("Brier per domino (red = double)")
    ax.axhline(brier(raw.pred, raw.truth), ls="--", color="k", lw=1, label="overall")
    ax.legend(loc="upper right")

    # Panel 2: double vs non-double.
    ax = axes[1]
    dbl_mask = np.isin(raw.domino_id, [d for d in range(N_DOMINOES) if DOMINO_IS_DOUBLE[d]])
    groups = [("doubles", dbl_mask), ("non-doubles", ~dbl_mask)]
    vals = [brier(raw.pred[m], raw.truth[m]) for _, m in groups]
    ax.bar([g[0] for g in groups], vals, color=["#d62728", "#1f77b4"])
    ax.set_ylabel("Brier")
    ax.set_title("Brier by domino type")
    for xi, v in enumerate(vals):
        ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: by count-points (0 / 5 / 10 point dominoes).
    ax = axes[2]
    cp_groups: list[tuple[str, np.ndarray]] = []
    for pts in (0, 5, 10):
        ids = [d for d in range(N_DOMINOES) if DOMINO_COUNT_POINTS[d] == pts]
        mask = np.isin(raw.domino_id, ids)
        cp_groups.append((f"{pts} pts (n={len(ids)})", mask))
    vals = [brier(raw.pred[m], raw.truth[m]) for _, m in cp_groups]
    ax.bar([g[0] for g in cp_groups], vals, color=["#7f7f7f", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Brier")
    ax.set_title("Brier by count-points")
    for xi, v in enumerate(vals):
        ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Brier score breakdown", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_table(rows: list[tuple], headers: tuple) -> str:
    widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        s = [str(x) for x in r]
        str_rows.append(s)
        widths = [max(w, len(c)) for w, c in zip(widths, s)]

    def _fmt(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [_fmt(headers), sep]
    lines.extend(_fmt(r) for r in str_rows)
    return "\n".join(lines)


def _write_report(
    raw: RawPredictions,
    samples: list[SampledState],
    backend: BeliefBackend,
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Compute all metrics and write report.md; return headline numbers."""
    overall_brier = brier(raw.pred, raw.truth)
    overall_ece = ece(raw.pred, raw.truth)
    top1_acc = float(raw.top1_correct.mean()) if raw.top1_correct.size else float("nan")
    top1_all_acc = (
        float(raw.top1_all_correct.mean()) if raw.top1_all_correct.size else float("nan")
    )
    played_mask = raw.top1_all_is_played
    top1_played_acc = (
        float(raw.top1_all_correct[played_mask].mean()) if played_mask.any() else float("nan")
    )
    top1_unplayed_acc = (
        float(raw.top1_all_correct[~played_mask].mean()) if (~played_mask).any() else float("nan")
    )

    # Per-opponent metrics.
    opp_rows = []
    for c, name in enumerate(OPP_NAMES):
        sel = raw.opp_class == c
        opp_rows.append(
            (
                name,
                f"{sel.sum():,}",
                f"{brier(raw.pred[sel], raw.truth[sel]):.4f}",
                f"{ece(raw.pred[sel], raw.truth[sel]):.4f}",
                f"{float(raw.truth[sel].mean()):.4f}",
            )
        )

    # Brier by domino type.
    dbl_ids = [d for d in range(N_DOMINOES) if DOMINO_IS_DOUBLE[d]]
    dbl_mask = np.isin(raw.domino_id, dbl_ids)
    dom_rows = [
        (
            "doubles",
            int(dbl_mask.sum()),
            f"{brier(raw.pred[dbl_mask], raw.truth[dbl_mask]):.4f}",
            f"{ece(raw.pred[dbl_mask], raw.truth[dbl_mask]):.4f}",
        ),
        (
            "non-doubles",
            int((~dbl_mask).sum()),
            f"{brier(raw.pred[~dbl_mask], raw.truth[~dbl_mask]):.4f}",
            f"{ece(raw.pred[~dbl_mask], raw.truth[~dbl_mask]):.4f}",
        ),
    ]

    # Brier by count-points.
    cp_rows = []
    for pts in (0, 5, 10):
        ids = [d for d in range(N_DOMINOES) if DOMINO_COUNT_POINTS[d] == pts]
        mask = np.isin(raw.domino_id, ids)
        cp_rows.append(
            (
                f"{pts}-pt ({len(ids)} dominoes)",
                int(mask.sum()),
                f"{brier(raw.pred[mask], raw.truth[mask]):.4f}",
                f"{ece(raw.pred[mask], raw.truth[mask]):.4f}",
            )
        )

    # Top-1 accuracy by trick index (curriculum signal).
    trick_rows = []
    for t in sorted(set(raw.top1_trick_idx.tolist())):
        sel = raw.top1_trick_idx == t
        if sel.sum() == 0:
            continue
        trick_rows.append(
            (t, int(sel.sum()), f"{float(raw.top1_correct[sel].mean()):.4f}")
        )

    # Declaration coverage.
    decl_rows = []
    decl_counts = {d: 0 for d in range(N_DECLS)}
    for s in samples:
        decl_counts[s.decl_id] += 1
    for d in range(N_DECLS):
        decl_rows.append((d, DECL_ID_TO_NAME[d], decl_counts[d]))

    summary = {
        "n_states": len(samples),
        "n_predictions": int(len(raw)),
        "top1_accuracy_hidden": top1_acc,  # Burl-relevant
        "top1_accuracy_training_style": top1_all_acc,  # includes played dominoes
        "top1_accuracy_played": top1_played_acc,
        "top1_accuracy_unplayed": top1_unplayed_acc,
        "brier_overall": overall_brier,
        "ece_overall": overall_ece,
        "checkpoint": str(cfg["checkpoint"]),
        "backend": backend.name,
        "seeds_range": [EVAL_SEED_MIN, EVAL_SEED_MAX],
        "eval_seed": cfg["eval_seed"],
        "samples_per_game": cfg["samples_per_game"],
        "timestamp": cfg["timestamp"],
    }

    # Sanity: held-out claim.
    for s in samples:
        assert s.seed >= EVAL_SEED_MIN

    md_lines = [
        "# Zeb belief-head calibration eval",
        "",
        f"- **Checkpoint**: `{cfg['checkpoint']}`  (epoch={backend.epoch})",
        f"- **Backend**: `{backend.name}`",
        f"- **Held-out seed range**: `[{EVAL_SEED_MIN}, {EVAL_SEED_MAX})`",
        f"- **Eval-seed (rng)**: {cfg['eval_seed']}",
        f"- **States**: {len(samples):,}  (up to {cfg['samples_per_game']} per game)",
        f"- **Predictions**: {len(raw):,}  (= states * hidden_dominoes * 3 classes)",
        f"- **Device**: {cfg['device']}",
        f"- **Timestamp**: {cfg['timestamp']}",
        "",
        "## Headline numbers",
        "",
        "**Burl-relevant** (hidden dominoes only -- the set Burl's wrapper will "
        "actually query):",
        "",
        f"- **Top-1 accuracy**: {top1_acc:.4f}  "
        f"({raw.top1_correct.sum():,} / {len(raw.top1_correct):,})",
        f"- **Brier (overall)**: {overall_brier:.4f}",
        f"- **ECE (10 bins)**: {overall_ece:.4f}",
        "",
        "**Training-style** (all dominoes not in my original hand, matches the "
        "`belief_accuracy` metric in the learner's training logs):",
        "",
        f"- **Top-1 accuracy (all-non-self)**: {top1_all_acc:.4f}",
        f"- **   -- played subset**: {top1_played_acc:.4f}  "
        f"(trivially observable from the play history -- inflates the headline)",
        f"- **   -- unplayed subset** (= hidden): {top1_unplayed_acc:.4f}",
        "",
        "### Why the two numbers differ",
        "",
        "The training loop computes `belief_accuracy` over every domino that is "
        "not in the current player's *original* hand. That includes dominoes that "
        "have already been played -- for which the 'correct' answer is who "
        "*originally* held the domino, a fact the observer has already seen in the "
        "play history. These are trivially-easy examples and inflate the reported "
        "number (OVERVIEW's 72% top-1).",
        "",
        "For **Burl**, `get_belief(state, opp, dom)` rejects visible dominoes and "
        "only answers about still-hidden ones. The relevant metric is "
        f"therefore the *hidden-only* top-1 = **{top1_acc:.4f}**, which is "
        "meaningfully lower than the training-style number.",
        "",
        "If the hidden-only top-1 is below ~40% (close to the 33% random-guess "
        "floor), Burl should probably treat Zeb's outputs as *ordinal* (rank the "
        "opponents by likelihood) rather than calibrated probabilities.",
        "",
        "## Per-opponent breakdown",
        "",
        _format_table(opp_rows, ("opponent", "N", "Brier", "ECE", "base-rate")),
        "",
        "## By domino type",
        "",
        _format_table(dom_rows, ("group", "N", "Brier", "ECE")),
        "",
        "## By count-points",
        "",
        _format_table(cp_rows, ("group", "N", "Brier", "ECE")),
        "",
        "## Top-1 accuracy by trick",
        "",
        "Trick index = `len(played) // 4`. Earlier tricks have more hidden dominoes "
        "(more uncertainty); later tricks have fewer but harder constraints.",
        "",
        _format_table(trick_rows, ("trick", "N", "top-1 acc")),
        "",
        "## Declaration coverage",
        "",
        "All 10 declarations are represented -- sampling strategy keeps per-decl counts "
        "near `n_states / 10`.",
        "",
        _format_table(decl_rows, ("decl_id", "name", "states")),
        "",
        "## Filter choices (documented)",
        "",
        "- **Seeds**: drawn from held-out range `[900000, 910000)`, asserted.",
        "  Any attempt to use seed < 900000 crashes loud.",
        "- **Skipped states**: terminal states (no hidden dominoes) are dropped. "
        "  States where the observing seat has fewer than 1 hidden domino are skipped "
        "  (emits a warning).",
        "- **Perspective seat**: `current_player(state)` -- whoever has to act. This "
        "  is the same seat the belief head sees at training time.",
        "- **Rollout policy**: uniform random legal moves. We chose random rather than "
        "  E[Q] to avoid biasing the state distribution toward a specific play style; "
        "  calibration is a property of the belief head at any reachable state.",
        "- **Per-seed sampling**: up to `samples_per_game` snapshots uniformly over the "
        "  27 non-trivial decision points (trick 0 initial state dropped: no plays yet).",
        "",
        "## Files",
        "",
        "- `reliability.png` -- four-panel reliability diagram (overall + per-opp).",
        "- `brier_breakdown.png` -- Brier per-domino + by type + by count-points.",
        "- `raw_predictions.npz` -- flat arrays, for replay / secondary analysis.",
        "- `summary.json` -- headline numbers, machine-readable.",
        "",
    ]

    (out_dir / "report.md").write_text("\n".join(md_lines))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_calibration(
    *,
    n_states: int = 1000,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    eval_seed: int = EVAL_SEED_MIN,
    samples_per_game: int = 3,
    out_root: str | Path | None = None,
) -> dict:
    """Run the belief-head calibration eval end-to-end.

    Args:
        n_states: Number of eval states to sample. Each state contributes
            ``hidden_dominoes * 3`` predictions.
        checkpoint: Path to a ``*-belief-*.pt`` Zeb checkpoint.
        device: 'cuda', 'cpu', or None (auto).
        eval_seed: RNG seed for sampling (kept separate from the held-out game
            seeds). Deterministic if fixed.
        samples_per_game: Max snapshots drawn per replayed game.
        out_root: Override the default ``burl/eval/results/`` root.

    Returns:
        Summary dict written to ``summary.json``.
    """
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(out_root) if out_root is not None else Path(__file__).parent / "results"
    out_dir = out_root / f"belief_calibration_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "checkpoint": str(ckpt.resolve()),
        "device": device,
        "eval_seed": eval_seed,
        "samples_per_game": samples_per_game,
        "n_states": n_states,
        "timestamp": ts,
    }

    print(f"== Zeb belief calibration ==")
    print(f"  out_dir={out_dir}")
    print(f"  checkpoint={ckpt}")
    print(f"  device={device}  n_states={n_states}  eval_seed={eval_seed}")

    print("[1/4] sampling eval states...")
    samples = sample_eval_states(
        n_states=n_states,
        eval_seed=eval_seed,
        samples_per_game=samples_per_game,
    )
    print(f"       {len(samples)} states drawn from held-out seeds")

    print("[2/4] loading belief backend...")
    backend = _load_backend(ckpt, device)
    print(f"       backend={backend.name}  epoch={backend.epoch}")

    print("[3/4] collecting predictions...")
    raw = _collect(backend, samples)
    print(f"       {len(raw):,} predictions  ({raw.top1_correct.size:,} argmax decisions)")

    print("[4/4] writing report + plots...")
    np.savez_compressed(
        out_dir / "raw_predictions.npz",
        pred=raw.pred,
        truth=raw.truth,
        domino_id=raw.domino_id,
        opp_class=raw.opp_class,
        decl_id=raw.decl_id,
        trick_idx=raw.trick_idx,
        top1_correct=raw.top1_correct,
        top1_decl_id=raw.top1_decl_id,
        top1_trick_idx=raw.top1_trick_idx,
        top1_domino_id=raw.top1_domino_id,
        top1_all_correct=raw.top1_all_correct,
        top1_all_is_played=raw.top1_all_is_played,
    )
    _plot_reliability(raw, out_dir / "reliability.png")
    _plot_brier_breakdown(raw, out_dir / "brier_breakdown.png")

    summary = _write_report(raw, samples, backend, cfg, out_dir)

    print()
    print(f"top-1 accuracy (hidden-only, Burl-relevant): {summary['top1_accuracy_hidden']:.4f}")
    print(f"top-1 accuracy (training-style, all-non-self): {summary['top1_accuracy_training_style']:.4f}")
    print(f"  -- played subset:   {summary['top1_accuracy_played']:.4f}")
    print(f"  -- unplayed subset: {summary['top1_accuracy_unplayed']:.4f}")
    print(f"Brier (overall): {summary['brier_overall']:.4f}")
    print(f"ECE (10 bins):   {summary['ece_overall']:.4f}")
    print(f"report: {out_dir / 'report.md'}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration eval for Zeb's belief head (held-out seeds only)."
    )
    p.add_argument("--n-states", type=int, default=1000, help="eval states (default: 1000)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"path to belief-head .pt (default: {DEFAULT_CHECKPOINT.name})",
    )
    p.add_argument("--device", type=str, default=None, help="'cuda' / 'cpu' (auto)")
    p.add_argument(
        "--eval-seed", type=int, default=EVAL_SEED_MIN, help=f"rng seed (must be >= {EVAL_SEED_MIN})"
    )
    p.add_argument("--samples-per-game", type=int, default=3)
    p.add_argument("--out-root", type=str, default=None, help="override results root")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.eval_seed < EVAL_SEED_MIN:
        raise SystemExit(
            f"--eval-seed must be >= {EVAL_SEED_MIN} (held-out range). "
            f"Got {args.eval_seed}."
        )
    run_calibration(
        n_states=args.n_states,
        checkpoint=args.checkpoint,
        device=args.device,
        eval_seed=args.eval_seed,
        samples_per_game=args.samples_per_game,
        out_root=args.out_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
