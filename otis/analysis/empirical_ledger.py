"""Empirical count-fate ledger measurement over the on-policy otis corpus.

Consumes finished arena snapshot chunks (``*.snapshots.json``), replays every hand
through :func:`otis.fates.parse_game_fates` (the P1-exact parser + free
recorded-points referee), and produces the W2 measurement deliverables:

  * ``master_fates.parquet``  — 5 fate rows per hand, all metadata + source tag.
  * ``master_hands.parquet``  — one row per hand (join keys, outcome, per-team
                                tricks, bidder-relative per-tile capture, T).
  * P3: empirical joint tail P(bidder_team_pts >= B) vs the independence
    composition (per-tile Bernoulli captures convolved with the trick marginal),
    per bid_value slice; plus the 6x6 correlation matrix of the capture/trick
    indicators.
  * P2 baseline: per-tile 8-class fate base rates (capture-side-vs-BIDDING-team x
    played_mode) + marginal entropy, overall and per decl.
  * Junk-economy descriptives (walker catches, count routing, capture mechanisms).

Everything here is pure CPU. otis does not own the GPU. The suit algebra is never
reimplemented — it lives in ``otis.fates`` / ``forge.oracle``.

The bidder-relative capture indicator for tile ``t`` is
``X_t = 1`` iff the trick that took ``t`` was won by the BIDDING team
(``winner_seat % 2 == bidder % 2``). The identity that binds the whole ledger:

    bidder_team_pts == sum_t value(t) * X_t + T          (T = bidder-team tricks)

is asserted on every parsed hand — a third independent cross-check.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from otis.export_games import FATE_COLUMNS, fate_rows
from otis.fates import COUNT_TILE_PIPS, parse_game_fates
from otis.snapshots import cross_check_points, load_snapshot_hands

# Ordered count tiles and their point values (5-5, 6-4 = 10; 5-0, 4-1, 3-2 = 5).
TILE_PIPS: tuple[str, ...] = COUNT_TILE_PIPS
TILE_VALUES: dict[str, int] = {"5-5": 10, "6-4": 10, "5-0": 5, "4-1": 5, "3-2": 5}
VALUES_ORDERED: tuple[int, ...] = tuple(TILE_VALUES[p] for p in TILE_PIPS)
# Column labels for the 6-vector (X_5-5, X_6-4, X_5-0, X_4-1, X_3-2, T).
INDICATOR_LABELS: tuple[str, ...] = tuple(f"X_{p}" for p in TILE_PIPS) + ("T",)
PLAYED_MODES: tuple[str, ...] = ("led", "followed", "trumped_in", "sloughed")


# --------------------------------------------------------------------------- #
# Parse: snapshot chunks -> master tables + junk-economy aggregates
# --------------------------------------------------------------------------- #


@dataclass
class JunkCounters:
    """Incremental junk-economy aggregates (kept in-loop; trick_df not persisted)."""

    count_tricks: int = 0  # tricks carrying >= 1 count point
    walker_catches: int = 0  # count-carrying tricks won by a non-count tile
    walker_by_trump: int = 0  # of those, won by trump power
    count_tricks_by_trump: int = 0  # count-carrying tricks won by trump power
    # per-tile mechanism tables (Counters keyed by string)
    slough_route: Counter = None  # (tile) -> {"to_own","to_opp"} for sloughed tiles
    capture_route: Counter = None  # (tile) -> {"to_own","to_opp"} all modes
    mechanism: Counter = None  # (tile, "capture|mode") -> n

    def __post_init__(self) -> None:
        if self.slough_route is None:
            self.slough_route = Counter()
        if self.capture_route is None:
            self.capture_route = Counter()
        if self.mechanism is None:
            self.mechanism = Counter()


def build_tables(
    chunk_paths: list[Path],
    log_every: int = 20000,
    progress=print,
) -> tuple[pd.DataFrame, pd.DataFrame, JunkCounters, int]:
    """Parse every hand in ``chunk_paths`` into the two master tables.

    Returns ``(hands_df, fate_df, junk, n_crosscheck_ok)``. Every hand passes the
    parser P1 identity (asserted inside :func:`parse_game_fates`), the recorded-
    points referee (:func:`cross_check_points`), and the ledger identity above.
    """
    hand_rows: list[dict] = []
    fate_row_list: list[dict] = []
    junk = JunkCounters()
    n_hands = 0
    n_crosscheck = 0

    for path in chunk_paths:
        source = _source_tag(path)
        for hand in load_snapshot_hands(path):
            game = hand.game
            meta = hand.meta
            fates = parse_game_fates(game)  # P1 identity asserted internally
            cross_check_points(fates, meta)  # recorded-points referee (raises)
            n_crosscheck += 1

            bidding_team = meta.bidder % 2
            team_tricks = (fates.team0_tricks, fates.team1_tricks)
            T = team_tricks[bidding_team]

            # Bidder-relative per-tile capture indicators.
            caps: dict[str, int] = {}
            for t in fates.tiles:
                caps[t.tile] = int(t.winner_seat % 2 == bidding_team)

            # Ledger identity: pts == sum value*X + T  (third independent check).
            recomposed = sum(TILE_VALUES[p] * caps[p] for p in TILE_PIPS) + T
            if recomposed != meta.bidder_team_pts:
                raise ValueError(
                    f"{game.game_id}: ledger identity failed — recomposed "
                    f"{recomposed} != recorded bidder_team_pts {meta.bidder_team_pts}"
                )

            hand_rows.append(
                {
                    "game_id": fates.game_id,
                    "source": source,
                    "seed": meta.seed,
                    "game_idx": meta.game_idx,
                    "hand_idx": meta.hand_idx,
                    "a_team": meta.a_team,
                    "dealer": meta.dealer,
                    "bids": json.dumps(list(meta.bids)),
                    "bidder": meta.bidder,
                    "bidding_team": bidding_team,
                    "bid_value": meta.bid_value,
                    "decl_id": meta.decl_id,
                    "decl_name": fates.decl_name,
                    "bidder_team_pts": meta.bidder_team_pts,
                    "opp_team_pts": meta.opp_team_pts,
                    "made": meta.made,
                    "team0_tricks": fates.team0_tricks,
                    "team1_tricks": fates.team1_tricks,
                    "bidder_tricks": T,
                    "opp_tricks": team_tricks[1 - bidding_team],
                    "X_5-5": caps["5-5"],
                    "X_6-4": caps["6-4"],
                    "X_5-0": caps["5-0"],
                    "X_4-1": caps["4-1"],
                    "X_3-2": caps["3-2"],
                }
            )

            # Fate rows (5/hand): reuse the canonical exporter row shape + source.
            frows = fate_rows(fates, source=source, meta=meta)
            for fr, t in zip(frows, fates.tiles, strict=True):
                cap_bid = int(t.winner_seat % 2 == bidding_team)
                fr["bidding_team"] = bidding_team
                fr["capture_bidding"] = "bidding_team" if cap_bid else "opp_of_bidder"
                fate_row_list.append(fr)
                # junk: per-tile routing + mechanism (bidder-relative not needed here)
                route = "to_own" if t.capture_side == "holder_team" else "to_opp"
                junk.capture_route[(t.tile, route)] += 1
                if t.played_mode == "sloughed":
                    junk.slough_route[(t.tile, route)] += 1
                junk.mechanism[(t.tile, t.capture_side, t.played_mode)] += 1

            # junk: trick-level walker economy
            for tr in fates.tricks:
                if tr.count_points > 0:
                    junk.count_tricks += 1
                    if tr.won_by_trump:
                        junk.count_tricks_by_trump += 1
                    if not tr.winner_is_count:
                        junk.walker_catches += 1
                        if tr.won_by_trump:
                            junk.walker_by_trump += 1

            n_hands += 1
            if n_hands % log_every == 0:
                progress(f"  ...parsed {n_hands} hands ({source})", flush=True)

    hands_df = pd.DataFrame(hand_rows)
    fate_df = pd.DataFrame(fate_row_list, columns=[*FATE_COLUMNS, "bidding_team", "capture_bidding"])
    return hands_df, fate_df, junk, n_crosscheck


def _source_tag(path: Path) -> str:
    """chunk_<tag>_<n>.snapshots.json -> tag (selfplay | netwp | random)."""
    stem = path.stem
    if stem.endswith(".snapshots"):
        stem = stem[: -len(".snapshots")]
    parts = stem.split("_")  # ["chunk", tag, n]
    if len(parts) >= 3 and parts[0] == "chunk":
        return parts[1]
    raise ValueError(f"cannot infer source tag from {path.name!r}")


def add_split(df: pd.DataFrame, split_fn) -> pd.DataFrame:
    """Add a ``split`` column via the champion deal-hash convention.

    ``split_fn(seed, hand_idx)`` keeps every replay of the same deal (including
    paired halves) in one split so train/val/test never straddle a paired deal.
    """
    df = df.copy()
    df["split"] = [split_fn(int(s), int(h)) for s, h in zip(df["seed"], df["hand_idx"], strict=True)]
    return df


# --------------------------------------------------------------------------- #
# P3 — independence composition vs empirical joint tail
# --------------------------------------------------------------------------- #


def independence_pmf(caps: np.ndarray, T: np.ndarray) -> np.ndarray:
    """PMF over bidder-team points 0..42 under tile/trick independence.

    ``caps`` is ``[N,5]`` 0/1 (columns ordered as ``TILE_PIPS``); ``T`` is ``[N]``
    bidder-team trick counts 0..7. Each tile contributes an independent Bernoulli
    (0 or its point value); the trick marginal is the empirical pmf of ``T``.
    """
    p_tiles = caps.mean(axis=0)  # marginal capture probs
    dist = np.array([1.0])
    for p, v in zip(p_tiles, VALUES_ORDERED, strict=True):
        kernel = np.zeros(v + 1)
        kernel[0] = 1.0 - p
        kernel[v] = p
        dist = np.convolve(dist, kernel)
    t_pmf = np.bincount(T.astype(int), minlength=8).astype(float)
    t_pmf /= t_pmf.sum()
    dist = np.convolve(dist, t_pmf)
    # dist spans 0..42; pad/trim to 43 for safety.
    out = np.zeros(43)
    out[: min(len(dist), 43)] = dist[:43]
    return out


def tail_ge(pmf: np.ndarray, threshold: int) -> float:
    """P(points >= threshold) from a 0..42 pmf."""
    return float(pmf[threshold:].sum())


def p3_slice(points: np.ndarray, caps: np.ndarray, T: np.ndarray, threshold: int) -> dict:
    """Empirical vs independence tail at ``threshold`` for one bid-value slice.

    diff_pp = (empirical - independence) in percentage points; positive means
    independence UNDERPRICES the high tail (the registered P3 direction).
    """
    emp = float((points >= threshold).mean())
    pmf = independence_pmf(caps, T)
    ind = tail_ge(pmf, threshold)
    return {
        "n": int(len(points)),
        "threshold": int(threshold),
        "empirical_tail": emp,
        "independence_tail": ind,
        "diff_pp": (emp - ind) * 100.0,
    }


def indicator_matrix(hands: pd.DataFrame) -> np.ndarray:
    """``[N,6]`` matrix of (X_5-5, X_6-4, X_5-0, X_4-1, X_3-2, T)."""
    cols = [f"X_{p}" for p in TILE_PIPS] + ["bidder_tricks"]
    return hands[cols].to_numpy(dtype=float)


def correlation_matrix(hands: pd.DataFrame) -> np.ndarray:
    """6x6 Pearson correlation of the capture/trick indicators."""
    return np.corrcoef(indicator_matrix(hands), rowvar=False)


def tail_curve(hands: pd.DataFrame, thresholds=range(25, 43)) -> dict:
    """Empirical vs independence tail across thresholds for one hand set.

    Locates the crossover threshold where ``empirical − independence`` changes
    sign. Positive capture/trick correlation fattens both tails around a fixed
    mean, so independence overprices below the mean and underprices the high tail;
    the crossover sits near the mean of ``bidder_team_pts``.
    """
    caps = hands[[f"X_{p}" for p in TILE_PIPS]].to_numpy(dtype=float)
    T = hands["bidder_tricks"].to_numpy(dtype=float)
    pts = hands["bidder_team_pts"].to_numpy(dtype=float)
    pmf = independence_pmf(caps, T)
    rows = []
    prev_sign = None
    crossover = None
    for b in thresholds:
        emp = float((pts >= b).mean())
        ind = tail_ge(pmf, b)
        diff = (emp - ind) * 100.0
        rows.append({"threshold": int(b), "empirical": emp, "independence": ind, "diff_pp": diff})
        sign = 1 if diff >= 0 else -1
        if prev_sign is not None and sign != prev_sign and crossover is None:
            crossover = int(b)
        prev_sign = sign
    return {
        "mean_bidder_pts": float(pts.mean()),
        "std_bidder_pts": float(pts.std()),
        "crossover_threshold": crossover,
        "curve": rows,
    }


def measure_p3(selfplay_hands: pd.DataFrame, min_n: int = 500) -> dict:
    """Full P3 measurement over the selfplay hands.

    Slices: every bid_value with >= ``min_n`` hands, plus the pooled 31+ slice
    (threshold 31). Band: |diff_pp| >= 2 CONFIRMED, < 0.5 falsifier.
    """
    caps_all = selfplay_hands[[f"X_{p}" for p in TILE_PIPS]].to_numpy(dtype=float)
    T_all = selfplay_hands["bidder_tricks"].to_numpy(dtype=float)
    pts_all = selfplay_hands["bidder_team_pts"].to_numpy(dtype=float)

    slices: dict[str, dict] = {}
    for B in sorted(selfplay_hands["bid_value"].unique()):
        mask = (selfplay_hands["bid_value"] == B).to_numpy()
        if mask.sum() < min_n:
            continue
        slices[f"bid_{int(B)}"] = p3_slice(pts_all[mask], caps_all[mask], T_all[mask], int(B))

    # Pooled 31+ slice at threshold 31.
    mask31 = (selfplay_hands["bid_value"] >= 31).to_numpy()
    if mask31.sum() >= min_n:
        slices["bid_31plus"] = p3_slice(pts_all[mask31], caps_all[mask31], T_all[mask31], 31)

    corr = correlation_matrix(selfplay_hands)
    # Mean off-diagonal correlation among the five capture indicators (5x5 block).
    cap_block = corr[:5, :5]
    off = cap_block[~np.eye(5, dtype=bool)]
    verdicts = {
        k: ("CONFIRMED" if abs(v["diff_pp"]) >= 2.0 else ("FALSIFIER" if abs(v["diff_pp"]) < 0.5 else "AMBIGUOUS"))
        for k, v in slices.items()
    }
    curve = tail_curve(selfplay_hands)
    return {
        "slices": slices,
        "verdicts": verdicts,
        "correlation_labels": list(INDICATOR_LABELS),
        "correlation_matrix": corr.tolist(),
        "mean_offdiag_capture_corr": float(off.mean()),
        "tail_curve": curve,
        "direction": (
            "positive capture/trick correlation fattens BOTH tails around the "
            "fixed mean: independence OVERprices below the mean (the make region) "
            "and UNDERprices the genuine high tail (sweeps); crossover ~ the mean."
        ),
    }


# --------------------------------------------------------------------------- #
# P2 baseline — per-tile 8-class fate base rates
# --------------------------------------------------------------------------- #


def _entropy_nats(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def _class_dist(sub: pd.DataFrame) -> dict:
    """8-class distribution (capture-vs-bidding x played_mode) + entropy, for a
    per-tile subframe."""
    n = len(sub)
    counts: dict[str, int] = {}
    for cap in ("bidding_team", "opp_of_bidder"):
        for mode in PLAYED_MODES:
            counts[f"{cap}|{mode}"] = 0
    for cap, mode in zip(sub["capture_bidding"], sub["played_mode"], strict=True):
        counts[f"{cap}|{mode}"] += 1
    probs = np.array([counts[k] for k in counts]) / n if n else np.zeros(len(counts))
    return {
        "n": int(n),
        "counts": counts,
        "probs": {k: (counts[k] / n if n else 0.0) for k in counts},
        "entropy_nats": _entropy_nats(probs),
    }


def fate_base_rates(fate_df: pd.DataFrame) -> dict:
    """Per-tile 8-class fate base rates, overall and per decl_id.

    Classes = (capture relative to BIDDING team) x (played_mode). Entropy in nats
    (matching P2's NLL units). Caller restricts ``fate_df`` to the desired split.
    """
    overall: dict[str, dict] = {}
    for tile in TILE_PIPS:
        overall[tile] = _class_dist(fate_df[fate_df["tile"] == tile])

    per_decl: dict[str, dict] = {}
    for decl_id in sorted(fate_df["decl_id"].unique()):
        sub = fate_df[fate_df["decl_id"] == decl_id]
        decl_name = sub["decl_name"].iloc[0]
        per_decl[str(int(decl_id))] = {
            "decl_name": decl_name,
            "n_hands": int(len(sub) // 5),
            "tiles": {tile: _class_dist(sub[sub["tile"] == tile]) for tile in TILE_PIPS},
        }
    return {
        "classes": [f"{c}|{m}" for c in ("bidding_team", "opp_of_bidder") for m in PLAYED_MODES],
        "n_hands": int(len(fate_df) // 5),
        "overall": overall,
        "per_decl": per_decl,
    }


# --------------------------------------------------------------------------- #
# Junk economy
# --------------------------------------------------------------------------- #


def junk_summary(junk: JunkCounters) -> dict:
    """Assemble junk-economy tables from the incremental counters."""
    ct = junk.count_tricks
    per_tile: dict[str, dict] = {}
    for tile in TILE_PIPS:
        own = junk.capture_route[(tile, "to_own")]
        opp = junk.capture_route[(tile, "to_opp")]
        s_own = junk.slough_route[(tile, "to_own")]
        s_opp = junk.slough_route[(tile, "to_opp")]
        mech = {
            f"{cap}|{mode}": junk.mechanism[(tile, cap, mode)]
            for cap in ("holder_team", "opp_team")
            for mode in PLAYED_MODES
        }
        per_tile[tile] = {
            "captured_to_own_team": own,
            "captured_to_opp_team": opp,
            "frac_to_own": (own / (own + opp)) if (own + opp) else 0.0,
            "sloughed_total": s_own + s_opp,
            "sloughed_to_own": s_own,
            "sloughed_to_opp": s_opp,
            "sloughed_frac_to_opp": (s_opp / (s_own + s_opp)) if (s_own + s_opp) else 0.0,
            "mechanism": mech,
        }
    return {
        "count_carrying_tricks": ct,
        "walker_catches": junk.walker_catches,
        "walker_catch_frac": (junk.walker_catches / ct) if ct else 0.0,
        "walker_catches_by_trump": junk.walker_by_trump,
        "count_tricks_won_by_trump": junk.count_tricks_by_trump,
        "count_tricks_won_by_trump_frac": (junk.count_tricks_by_trump / ct) if ct else 0.0,
        "per_tile": per_tile,
    }
