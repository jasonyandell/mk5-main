from __future__ import annotations

from dataclasses import dataclass

from .declarations import DOUBLES_SUIT, DOUBLES_TRUMP, NOTRUMP, PIP_TRUMP_IDS, has_trump_power


PIPS = tuple(range(7))
N_DOMINOES = 28


def create_dominoes() -> list[tuple[int, int]]:
    """Return the double-six set as (high, low) pairs, ordered by high then low."""
    dominoes: list[tuple[int, int]] = []
    for high in range(7):
        for low in range(high + 1):
            dominoes.append((high, low))
    if len(dominoes) != N_DOMINOES:
        raise AssertionError("unexpected domino count")
    return dominoes


DOMINOES: list[tuple[int, int]] = create_dominoes()
DOMINO_HIGH: tuple[int, ...] = tuple(h for h, _l in DOMINOES)
DOMINO_LOW: tuple[int, ...] = tuple(l for _h, l in DOMINOES)
DOMINO_SUM: tuple[int, ...] = tuple(h + l for h, l in DOMINOES)
DOMINO_IS_DOUBLE: tuple[bool, ...] = tuple(h == l for h, l in DOMINOES)


def _count_points_for_domino(high: int, low: int) -> int:
    if (high, low) in ((5, 5), (6, 4)):
        return 10
    if (high, low) in ((5, 0), (4, 1), (3, 2)):
        return 5
    return 0


DOMINO_COUNT_POINTS: tuple[int, ...] = tuple(_count_points_for_domino(h, l) for h, l in DOMINOES)


def domino_contains_pip(domino_id: int, pip: int) -> bool:
    high = DOMINO_HIGH[domino_id]
    low = DOMINO_LOW[domino_id]
    return pip == high or pip == low


def is_in_called_suit(domino_id: int, decl_id: int) -> bool:
    if decl_id in PIP_TRUMP_IDS:
        return domino_contains_pip(domino_id, decl_id)
    if decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
        return DOMINO_IS_DOUBLE[domino_id]
    if decl_id == NOTRUMP:
        return False
    raise ValueError(f"unknown decl_id: {decl_id}")


def led_suit_for_lead_domino(lead_domino_id: int, decl_id: int) -> int:
    """Return led suit in {0..6, 7=called suit} when `lead_domino_id` leads the trick."""
    if decl_id == NOTRUMP:
        return DOMINO_HIGH[lead_domino_id]

    if is_in_called_suit(lead_domino_id, decl_id):
        return 7
    return DOMINO_HIGH[lead_domino_id]


def can_follow(domino_id: int, led_suit: int, decl_id: int) -> bool:
    if led_suit == 7:
        return is_in_called_suit(domino_id, decl_id)
    return domino_contains_pip(domino_id, led_suit) and not is_in_called_suit(domino_id, decl_id)


def _rank_in_pip_suit(domino_id: int) -> int:
    if DOMINO_IS_DOUBLE[domino_id]:
        return 14
    return DOMINO_SUM[domino_id]


def _rank_in_called_suit(domino_id: int, decl_id: int) -> int:
    if decl_id in PIP_TRUMP_IDS:
        return _rank_in_pip_suit(domino_id)
    if decl_id == DOUBLES_TRUMP:
        return DOMINO_HIGH[domino_id]
    raise ValueError(f"decl has no called-suit rank: {decl_id}")


def _rank_in_doubles_suit(domino_id: int) -> int:
    return DOMINO_HIGH[domino_id]


def trick_rank(domino_id: int, led_suit: int, decl_id: int) -> int:
    """Return a 6-bit ordering key; higher wins. Matches docs/SUIT_ALGEBRA tiering."""
    if has_trump_power(decl_id) and is_in_called_suit(domino_id, decl_id):
        tier = 2
        rank = _rank_in_called_suit(domino_id, decl_id)
        return (tier << 4) + rank

    if can_follow(domino_id, led_suit, decl_id):
        tier = 1
        rank = _rank_in_doubles_suit(domino_id) if led_suit == 7 else _rank_in_pip_suit(domino_id)
        return (tier << 4) + rank

    return 0


@dataclass(frozen=True)
class TrickOutcome:
    winner_offset: int
    points: int


def score_trick(domino_ids: tuple[int, int, int, int]) -> int:
    return 1 + sum(DOMINO_COUNT_POINTS[d] for d in domino_ids)


def resolve_trick(leader_domino_id: int, domino_ids: tuple[int, int, int, int], decl_id: int) -> TrickOutcome:
    led_suit = led_suit_for_lead_domino(leader_domino_id, decl_id)
    ranks = [trick_rank(d, led_suit, decl_id) for d in domino_ids]
    best = 0
    best_rank = ranks[0]
    for i in range(1, 4):
        r = ranks[i]
        if r > best_rank:
            best = i
            best_rank = r
    return TrickOutcome(winner_offset=best, points=score_trick(domino_ids))

