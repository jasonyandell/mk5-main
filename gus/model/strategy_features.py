"""Small, explicit public-state strategy features for Gus experiments.

These are intentionally humble. The goal is not to encode "good 42" directly,
but to give tiny models cheap access to human-legible state facts that the
sequence transformer otherwise has to rediscover from sparse examples.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forge.oracle.declarations import N_DECLS, has_trump_power
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    DOMINO_SUM,
    is_in_called_suit,
    led_suit_for_lead_domino,
    trick_rank,
)

from .features import _hand_list, reconstruct_prior_plays
from .voids import voids_feature_vector


STRATEGY_FEATURE_DIM = 68
STRATEGY_ACTION_FEATURE_DIM = 32

_COUNT_DOMINOES = tuple(i for i, pts in enumerate(DOMINO_COUNT_POINTS) if pts > 0)
_TOTAL_COUNT_POINTS = float(sum(DOMINO_COUNT_POINTS))


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _contains_pip(domino_id: int, pip: int) -> bool:
    return DOMINO_HIGH[domino_id] == pip or DOMINO_LOW[domino_id] == pip


def _called_rank(domino_id: int, decl_id: int) -> int:
    if 0 <= decl_id <= 6:
        if not _contains_pip(domino_id, decl_id):
            return 0
        return 14 if DOMINO_IS_DOUBLE[domino_id] else DOMINO_SUM[domino_id]
    if decl_id == 7 and DOMINO_IS_DOUBLE[domino_id]:
        return DOMINO_HIGH[domino_id]
    if decl_id == 8 and DOMINO_IS_DOUBLE[domino_id]:
        return DOMINO_HIGH[domino_id]
    return 0


def _rank_in_pip(domino_id: int) -> int:
    return 14 if DOMINO_IS_DOUBLE[domino_id] else DOMINO_SUM[domino_id]


def _natural_suit(domino_id: int) -> int:
    return DOMINO_HIGH[domino_id]


def _is_called(domino_id: int, decl_id: int) -> bool:
    return is_in_called_suit(domino_id, decl_id)


def _is_trump(domino_id: int, decl_id: int) -> bool:
    return has_trump_power(decl_id) and _is_called(domino_id, decl_id)


def extract_strategy_features(
    game_hands: list[list[int]],
    decl_id: int,
    decisions: list,
    decision_idx: int,
) -> Tensor:
    """Return a [STRATEGY_FEATURE_DIM] float tensor for the public decision state.

    Feature groups:
      - declaration one-hot
      - phase/current-trick scalars
      - current-hand shape
      - legal-action shape
      - public played/live-count shape
      - void summaries
      - visible pip coverage
      - current-trick pressure
    """
    decision = decisions[decision_idx]
    current_player = int(decision.player)
    prior_plays = reconstruct_prior_plays(game_hands, decisions, decision_idx)

    played = [int(d) for _p, d in prior_plays]
    played_set = set(played)
    played_by_me = {int(d) for p, d in prior_plays if int(p) == current_player}
    my_initial = [int(x) for x in _hand_list(game_hands[current_player]) if int(x) >= 0]
    my_current = [d for d in my_initial if d not in played_by_me]

    legal_mask = decision.legal_mask.bool()
    legal = [d for i, d in enumerate(my_current[:7]) if i < legal_mask.numel() and bool(legal_mask[i])]

    decl = [0.0] * N_DECLS
    if 0 <= int(decl_id) < N_DECLS:
        decl[int(decl_id)] = 1.0

    trick_pos = len(prior_plays) % 4
    trick_idx = min(len(prior_plays) // 4, 6)
    phase = [
        _safe_div(decision_idx, 27.0),
        _safe_div(trick_idx, 6.0),
        _safe_div(trick_pos, 3.0),
    ]

    hand_count_pts = sum(DOMINO_COUNT_POINTS[d] for d in my_current)
    hand_count_tiles = sum(1 for d in my_current if DOMINO_COUNT_POINTS[d] > 0)
    hand_called = [d for d in my_current if _is_called(d, decl_id)]
    hand_trumps = [d for d in my_current if _is_trump(d, decl_id)]
    hand_doubles = [d for d in my_current if DOMINO_IS_DOUBLE[d]]
    hand_offs = [d for d in my_current if not _is_called(d, decl_id) and not DOMINO_IS_DOUBLE[d]]
    hand_called_ranks = [_called_rank(d, decl_id) for d in hand_called]
    hand_feats = [
        _safe_div(len(my_current), 7.0),
        _safe_div(len(hand_trumps), 7.0),
        _safe_div(len(hand_called), 7.0),
        _safe_div(len(hand_doubles), 7.0),
        _safe_div(hand_count_tiles, 5.0),
        _safe_div(hand_count_pts, _TOTAL_COUNT_POINTS),
        _safe_div(len(hand_offs), 7.0),
        _safe_div(max(hand_called_ranks) if hand_called_ranks else 0, 14.0),
        1.0 if any(DOMINO_COUNT_POINTS[d] > 0 and _is_called(d, decl_id) for d in my_current) else 0.0,
        _safe_div(sum(DOMINO_COUNT_POINTS[d] for d in my_current if not _is_called(d, decl_id)), _TOTAL_COUNT_POINTS),
    ]

    legal_count_pts = sum(DOMINO_COUNT_POINTS[d] for d in legal)
    legal_feats = [
        _safe_div(len(legal), 7.0),
        _safe_div(sum(1 for d in legal if _is_trump(d, decl_id)), 7.0),
        _safe_div(sum(1 for d in legal if DOMINO_IS_DOUBLE[d]), 7.0),
        _safe_div(sum(1 for d in legal if DOMINO_COUNT_POINTS[d] > 0), 5.0),
        _safe_div(legal_count_pts, _TOTAL_COUNT_POINTS),
        1.0 if trick_pos > 0 and len(legal) < len(my_current) else 0.0,
    ]

    visible = set(my_current) | played_set
    played_count_pts = sum(DOMINO_COUNT_POINTS[d] for d in played)
    unknown_count_pts = _TOTAL_COUNT_POINTS - played_count_pts - hand_count_pts
    all_called = [d for d in range(28) if _is_called(d, decl_id)]
    played_called = sum(1 for d in played if _is_called(d, decl_id))
    hand_called_n = sum(1 for d in my_current if _is_called(d, decl_id))
    public_feats = [
        _safe_div(len(played), 28.0),
        _safe_div(played_count_pts, _TOTAL_COUNT_POINTS),
        _safe_div(max(unknown_count_pts, 0.0), _TOTAL_COUNT_POINTS),
        _safe_div(played_called, max(len(all_called), 1)),
        _safe_div(max(len(all_called) - played_called - hand_called_n, 0), max(len(all_called), 1)),
    ]

    voids = voids_feature_vector(prior_plays, int(decl_id), current_player).float().view(3, 8)
    void_feats = [
        float(voids.mean().item()),
        float(voids[0].mean().item()),
        float(voids[1].mean().item()),
        float(voids[2].mean().item()),
    ]

    # Visible pip coverage: how much of each natural pip suit is no longer hidden
    # from the current player's perspective (played or in my hand).
    pip_coverage = [
        _safe_div(sum(1 for d in visible if _contains_pip(d, pip)), 7.0)
        for pip in range(7)
    ]

    current_trick = prior_plays[-trick_pos:] if trick_pos else []
    current_trick_dominoes = [int(d) for _p, d in current_trick]
    current_winner_rel = 0
    if current_trick:
        led_suit = led_suit_for_lead_domino(int(current_trick[0][1]), int(decl_id))
        best_i = 0
        best_rank = trick_rank(int(current_trick[0][1]), led_suit, int(decl_id))
        for i, (_p, d) in enumerate(current_trick[1:], start=1):
            r = trick_rank(int(d), led_suit, int(decl_id))
            if r > best_rank:
                best_i = i
                best_rank = r
        current_winner_abs = int(current_trick[best_i][0])
        current_winner_rel = (current_winner_abs - current_player) % 4
    current_trick_feats = [
        _safe_div(sum(DOMINO_COUNT_POINTS[d] for d in current_trick_dominoes), _TOTAL_COUNT_POINTS),
        _safe_div(sum(1 for d in current_trick_dominoes if _is_trump(d, decl_id)), 4.0),
        _safe_div(sum(1 for d in current_trick_dominoes if DOMINO_COUNT_POINTS[d] > 0), 4.0),
        1.0 if any(DOMINO_COUNT_POINTS[d] >= 10 for d in current_trick_dominoes) else 0.0,
        1.0 if any(_is_trump(d, decl_id) for d in current_trick_dominoes) else 0.0,
        1.0 if current_winner_rel == 2 else 0.0,
        1.0 if current_trick and current_winner_rel in (1, 3) else 0.0,
        1.0 if trick_pos == 0 else 0.0,
        1.0 if trick_pos == 3 else 0.0,
    ]

    my_pip_counts = [
        _safe_div(sum(1 for d in my_current if _contains_pip(d, pip)), 7.0)
        for pip in range(7)
    ]
    unseen_count_by_pip = []
    for pip in range(7):
        pts = sum(
            DOMINO_COUNT_POINTS[d]
            for d in _COUNT_DOMINOES
            if d not in visible and _contains_pip(d, pip)
        )
        unseen_count_by_pip.append(_safe_div(pts, _TOTAL_COUNT_POINTS))

    features = (
        decl
        + phase
        + hand_feats
        + legal_feats
        + public_feats
        + void_feats
        + pip_coverage
        + current_trick_feats
        + my_pip_counts
        + unseen_count_by_pip
    )
    if len(features) != STRATEGY_FEATURE_DIM:
        raise AssertionError(f"strategy feature dim drifted: {len(features)} != {STRATEGY_FEATURE_DIM}")
    return torch.tensor(features, dtype=torch.float32)


def extract_strategy_action_features(
    game_hands: list[list[int]],
    decl_id: int,
    decisions: list,
    decision_idx: int,
) -> Tensor:
    """Return per-hand-slot public/action features with shape [7, 14]."""
    decision = decisions[decision_idx]
    current_player = int(decision.player)
    prior_plays = reconstruct_prior_plays(game_hands, decisions, decision_idx)
    played_by_me = {int(d) for p, d in prior_plays if int(p) == current_player}
    my_initial = [int(x) for x in _hand_list(game_hands[current_player]) if int(x) >= 0]
    my_current = [d for d in my_initial if d not in played_by_me]
    legal_mask = decision.legal_mask.bool()

    trick_pos = len(prior_plays) % 4
    current_trick = prior_plays[-trick_pos:] if trick_pos else []
    led_suit = None
    current_best_rank = None
    if current_trick:
        lead_domino = int(current_trick[0][1])
        led_suit = led_suit_for_lead_domino(lead_domino, int(decl_id))
        current_best_rank = max(trick_rank(int(d), led_suit, int(decl_id)) for _p, d in current_trick)
        current_best_player = None
        best_rank = -1
        for p, d in current_trick:
            r = trick_rank(int(d), led_suit, int(decl_id))
            if r > best_rank:
                best_rank = r
                current_best_player = int(p)
        current_best_rel = (int(current_best_player) - current_player) % 4 if current_best_player is not None else None
    else:
        current_best_rel = None

    played_set = {int(d) for _p, d in prior_plays}
    visible = played_set | set(my_current)

    def suit_live_stats(domino_id: int) -> tuple[float, float, float, float]:
        """Counts for the suit this domino would lead/follow as a natural tile."""
        if _is_called(domino_id, int(decl_id)):
            suit_members = [d for d in range(28) if _is_called(d, int(decl_id))]
            rank = _called_rank(domino_id, int(decl_id))
            higher = [d for d in suit_members if _called_rank(d, int(decl_id)) > rank]
        else:
            pip = _natural_suit(domino_id)
            suit_members = [
                d for d in range(28)
                if _contains_pip(d, pip) and not _is_called(d, int(decl_id))
            ]
            rank = _rank_in_pip(domino_id)
            higher = [d for d in suit_members if _rank_in_pip(d) > rank]
        live = [d for d in suit_members if d not in visible]
        live_higher = [d for d in higher if d not in visible]
        live_count = sum(DOMINO_COUNT_POINTS[d] for d in live)
        live_higher_count = sum(DOMINO_COUNT_POINTS[d] for d in live_higher)
        return (
            _safe_div(len(live), max(len(suit_members), 1)),
            _safe_div(len(live_higher), max(len(suit_members), 1)),
            _safe_div(live_count, _TOTAL_COUNT_POINTS),
            _safe_div(live_higher_count, _TOTAL_COUNT_POINTS),
        )

    def pip_count_risk(pip: int) -> float:
        pts = sum(
            DOMINO_COUNT_POINTS[d]
            for d in _COUNT_DOMINOES
            if d not in visible and _contains_pip(d, pip)
        )
        return _safe_div(pts, _TOTAL_COUNT_POINTS)

    rows: list[list[float]] = []
    for slot in range(7):
        if slot >= len(my_current):
            rows.append([0.0] * STRATEGY_ACTION_FEATURE_DIM)
            continue
        d = int(my_current[slot])
        legal = bool(legal_mask[slot]) if slot < legal_mask.numel() else False
        count_pts = DOMINO_COUNT_POINTS[d]
        called = _is_called(d, int(decl_id))
        trump = _is_trump(d, int(decl_id))
        rank = _called_rank(d, int(decl_id))
        follows_led = 0.0
        beats_current = 0.0
        current_partner_winning = 0.0
        current_opp_winning = 0.0
        trump_in = 0.0
        point_dump = 0.0
        if led_suit is not None:
            r = trick_rank(d, led_suit, int(decl_id))
            follows_led = 1.0 if r > 0 and (not trump or led_suit == 7) else 0.0
            beats_current = 1.0 if legal and current_best_rank is not None and r > current_best_rank else 0.0
            current_partner_winning = 1.0 if current_best_rel == 2 else 0.0
            current_opp_winning = 1.0 if current_best_rel in (1, 3) else 0.0
            trump_in = 1.0 if legal and trump and led_suit != 7 else 0.0
            point_dump = 1.0 if legal and count_pts > 0 and not beats_current else 0.0
        live_frac, live_higher_frac, live_count_frac, live_higher_count_frac = suit_live_stats(d)
        high = DOMINO_HIGH[d]
        low = DOMINO_LOW[d]
        same_high_double = next((x for x in my_current if DOMINO_IS_DOUBLE[x] and DOMINO_HIGH[x] == high and x != d), None)
        same_low_double = next((x for x in my_current if DOMINO_IS_DOUBLE[x] and DOMINO_HIGH[x] == low and x != d), None)
        protected_by_my_double = same_high_double is not None or same_low_double is not None
        pips = {high, low}
        pip_risk = max(pip_count_risk(pip) for pip in pips)
        min_pip_coverage = min(_safe_div(sum(1 for x in visible if _contains_pip(x, pip)), 7.0) for pip in pips)
        max_pip_coverage = max(_safe_div(sum(1 for x in visible if _contains_pip(x, pip)), 7.0) for pip in pips)
        rows.append([
            1.0,                                      # present
            1.0 if legal else 0.0,
            1.0 if called else 0.0,
            1.0 if trump else 0.0,
            1.0 if DOMINO_IS_DOUBLE[d] else 0.0,
            _safe_div(count_pts, 10.0),
            _safe_div(rank, 14.0),
            _safe_div(DOMINO_HIGH[d], 6.0),
            _safe_div(DOMINO_LOW[d], 6.0),
            1.0 if not called and not DOMINO_IS_DOUBLE[d] else 0.0,
            follows_led,
            beats_current,
            1.0 if count_pts >= 10 else 0.0,
            1.0 if trick_pos == 0 else 0.0,
            current_partner_winning,
            current_opp_winning,
            trump_in,
            point_dump,
            live_frac,
            live_higher_frac,
            live_count_frac,
            live_higher_count_frac,
            pip_risk,
            min_pip_coverage,
            max_pip_coverage,
            1.0 if protected_by_my_double else 0.0,
            1.0 if same_high_double is not None else 0.0,
            1.0 if same_low_double is not None else 0.0,
            _safe_div(sum(1 for x in my_current if _contains_pip(x, high)), 7.0),
            _safe_div(sum(1 for x in my_current if _contains_pip(x, low)), 7.0),
            1.0 if legal and count_pts > 0 and current_partner_winning else 0.0,
            1.0 if legal and count_pts > 0 and current_opp_winning else 0.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)
