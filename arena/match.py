"""Paired-seed arena match: two halves with rotated team assignment.

Half 1 plays A as absolute team 0, half 2 as team 1, over the same deal
seeds — so every deal is played from both sides and card luck cancels in
the comparison. When A and B differ, auctions (and therefore hand counts)
may diverge between halves; that divergence is the players' own doing and
is exactly what the arena measures.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from forge.oracle.declarations import DECL_ID_TO_NAME

from .auction import BidPolicy
from .engine import ArenaConfig, GameRecord, run_half
from .play import PlayPolicy


@dataclass
class MatchResult:
    label_a: str
    label_b: str
    cfg: ArenaConfig
    elapsed_s: float
    games: list[GameRecord]  # both halves

    @property
    def n_games(self) -> int:
        return len(self.games)

    @property
    def a_wins(self) -> int:
        return sum(1 for g in self.games if g.a_won)


def run_match(
    *,
    bid_a: BidPolicy,
    bid_b: BidPolicy,
    play_a: PlayPolicy,
    play_b: PlayPolicy,
    n_games: int,
    cfg: ArenaConfig | None = None,
    label_a: str = "A",
    label_b: str = "B",
    verbose: bool = False,
) -> MatchResult:
    """Run a full paired match; n_games is split evenly across the halves."""
    cfg = cfg or ArenaConfig()
    half = n_games // 2
    if half == 0:
        raise ValueError("n_games must be at least 2")

    t0 = time.time()
    games: list[GameRecord] = []
    for a_team in (0, 1):
        if verbose:
            print(f"  half {a_team + 1}: A=team{a_team} ({half} games)", flush=True)
        records = run_half(
            n_games=half, a_team=a_team, cfg=cfg,
            bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
            log_every_s=30.0 if verbose else None,
        )
        games.extend(records)
        if verbose:
            wins = sum(1 for g in records if g.a_won)
            print(f"    A won {wins}/{half}", flush=True)

    return MatchResult(
        label_a=label_a,
        label_b=label_b,
        cfg=cfg,
        elapsed_s=time.time() - t0,
        games=games,
    )


def _bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _offense_line(hands: list, team_is_a: bool) -> dict:
    """Contract outcomes for hands where the given side won the auction."""
    own = [h for h in hands if (h.bidder_team == h.a_team) == team_is_a]
    made = sum(1 for h in own if h.made)
    return {
        "contracts": len(own),
        "made": made,
        "made_rate": made / len(own) if own else float("nan"),
        "mean_bid": float(np.mean([h.bid_value for h in own])) if own else float("nan"),
    }


def summarize(result: MatchResult) -> dict:
    """Aggregate a match into the headline + per-phase diagnostics."""
    games = result.games
    hands = [h for g in games for h in g.hands]
    n = len(games)

    a_wins = np.array([1.0 if g.a_won else 0.0 for g in games])
    mark_margin = np.array([float(g.a_marks - g.b_marks) for g in games])
    ci_lo, ci_hi = _bootstrap_ci_mean(mark_margin)
    pt_margin = np.array([
        float(h.team_points[h.a_team] - h.team_points[1 - h.a_team])
        for h in hands
    ])

    bid_hist = Counter(h.bid_value for h in hands)
    decl_hist = Counter(DECL_ID_TO_NAME[h.decl_id] for h in hands)
    a_offense = sum(1 for h in hands if h.bidder_team == h.a_team)

    return {
        "label_a": result.label_a,
        "label_b": result.label_b,
        "n_games": n,
        "n_hands": len(hands),
        "elapsed_s": round(result.elapsed_s, 2),
        "a_wins": int(a_wins.sum()),
        "a_game_win_rate": float(a_wins.mean()),
        "a_win_rate_half1": float(np.mean([g.a_won for g in games if g.a_team == 0])),
        "a_win_rate_half2": float(np.mean([g.a_won for g in games if g.a_team == 1])),
        "mean_mark_margin": float(mark_margin.mean()),
        "mark_margin_ci_lo_95": ci_lo,
        "mark_margin_ci_hi_95": ci_hi,
        "ci_excludes_zero": int(ci_lo > 0 or ci_hi < 0),
        "mean_hands_per_game": len(hands) / n,
        "mean_hand_point_margin": float(pt_margin.mean()),
        "auction": {
            "bid_hist": dict(sorted(bid_hist.items())),
            "decl_hist": dict(decl_hist.most_common()),
            "a_offense_share": a_offense / len(hands),
            "forced_hands": sum(1 for h in hands if h.forced),
            "redealt_hands": sum(1 for h in hands if h.redeals > 0),
        },
        "contracts": {
            "a_offense": _offense_line(hands, team_is_a=True),
            "b_offense": _offense_line(hands, team_is_a=False),
        },
    }


def hand_rows(result: MatchResult) -> list[dict]:
    """Flat per-hand rows for CSV."""
    return [
        {
            "game_idx": h.game_idx,
            "hand_idx": h.hand_idx,
            "seed": h.seed,
            "a_team": h.a_team,
            "dealer": h.dealer,
            "redeals": h.redeals,
            "forced": int(h.forced),
            "bidder": h.bidder,
            "bidder_is_a": int(h.bidder_team == h.a_team),
            "bid_value": h.bid_value,
            "decl_id": h.decl_id,
            "team_a_pts": h.team_points[h.a_team],
            "team_b_pts": h.team_points[1 - h.a_team],
            "made": int(h.made),
            "marks_a_after": h.marks_after[h.a_team],
            "marks_b_after": h.marks_after[1 - h.a_team],
        }
        for g in result.games for h in g.hands
    ]


def game_rows(result: MatchResult) -> list[dict]:
    """Flat per-game rows for CSV."""
    return [
        {
            "game_idx": g.game_idx,
            "a_team": g.a_team,
            "n_hands": len(g.hands),
            "marks_a": g.a_marks,
            "marks_b": g.b_marks,
            "a_won": int(g.a_won),
        }
        for g in result.games
    ]


def snapshot_rows(result: MatchResult) -> list[dict]:
    """Per-hand deal+auction snapshots for the #26 belief-corpus bridge and
    the jud v0 realized-value head (#32).

    Each row carries everything needed to regenerate an oracle E[Q] belief
    record from a REAL auction: the seat-ordered deal, the winning declaration,
    the full per-seat bid vector, the winning seat, and the contract value.
    Hand layout matches GameRecordGPU.hands / deal_from_seed (4 x 7 ids).

    It also carries the hand's REALIZED outcome — points captured by the
    declaring team (``bidder_team_pts``) vs the opponents (``opp_team_pts``),
    and whether the contract was ``made`` — plus the ``a_team`` / ``game_idx`` /
    ``hand_idx`` / ``seed`` join keys back to per_hand.csv. ``team_points`` is
    absolute ``(team0, team1)`` and the bidder's absolute team is ``bidder %
    2``, so the declaring team's share is ``team_points[bidder % 2]`` regardless
    of which half rotated A onto which team. The two paired halves replay the
    same deal seeds, so ``a_team`` is part of the key: ``(a_team, game_idx,
    hand_idx)`` is what uniquely identifies a hand across the whole match.

    ``dealer`` is carried so auction order is reconstructible from the snapshot
    alone: bidding proceeds ``dealer + 1 .. dealer`` (the shaker bids last), so a
    consumer can recover which seats bid BEFORE the declarer — the only bids
    available at the declarer's decision time (jud v0 Step 2 encoding).
    """
    return [
        {
            "a_team": hr.a_team,
            "game_idx": hr.game_idx,
            "hand_idx": hr.hand_idx,
            "seed": hr.seed,
            "dealer": hr.dealer,
            "hands": [list(h) for h in hr.hands],
            "decl_id": hr.decl_id,
            "bids": list(hr.bids),
            "bidder": hr.bidder,
            "bid_value": hr.bid_value,
            "bidder_team_pts": hr.team_points[hr.bidder_team],
            "opp_team_pts": hr.team_points[1 - hr.bidder_team],
            "made": int(hr.made),
        }
        for g in result.games for hr in g.hands
    ]
