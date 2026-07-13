"""``full_board_snapshot`` — Burl's "give me everything in one read" tool.

Pure rule-based introspection of a ``ZebGameState``. No oracle, no Gus —
fast (sub-millisecond) state read. The model uses this to orient before
calling the heavier tools (``belief_trajectory``, ``explore_game``).

Output format
-------------
Plain prose, structured the way Burl learned to read state during training:
matching ``id(p-p)`` rendering, seat-relative team labels, explicit trump
ranking, count-domino ledger, and bid math. Paragraphs are short and
scan-able — Burl asked for "clear data, fast decisions".
"""
from __future__ import annotations

from typing import Any

from burl.harness.agent_runner import _DOMINO_LABELS
from forge.oracle.declarations import (
    DECL_ID_TO_NAME,
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    NOTRUMP,
    PIP_TRUMP_IDS,
)
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    DOMINO_SUM,
    led_suit_for_lead_domino,
)


def _label(d: int) -> str:
    return f"{int(d)}({_DOMINO_LABELS[int(d)]})"


def _trump_dominoes_high_to_low(decl_id: int) -> list[int]:
    """Return trump-domino ids in rank order (highest first)."""
    if decl_id in PIP_TRUMP_IDS:
        # Pip-suit trump: every domino containing the trump pip. Double of
        # the trump suit ranks highest, then by the OTHER pip descending.
        trump_pip = decl_id
        candidates = [d for d in range(28) if trump_pip in (DOMINO_HIGH[d], DOMINO_LOW[d])]
        # Rank: doubles first (rank=14 in tables), then by other-pip DESC.
        def rank(d: int) -> int:
            if DOMINO_IS_DOUBLE[d]:
                return 100
            other = DOMINO_HIGH[d] if DOMINO_LOW[d] == trump_pip else DOMINO_LOW[d]
            return other
        return sorted(candidates, key=rank, reverse=True)
    if decl_id == DOUBLES_TRUMP:
        # Doubles-as-trump: only the 7 doubles, 6-6 high to 0-0 low.
        return sorted(
            [d for d in range(28) if DOMINO_IS_DOUBLE[d]],
            key=lambda d: DOMINO_HIGH[d],
            reverse=True,
        )
    # doubles-suit / notrump: no trump.
    return []


def _trump_summary_line(decl_id: int) -> str:
    name = DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}")
    if decl_id == NOTRUMP:
        return "Trump: notrump (no suit has trump power; highest of led suit wins each trick)."
    if decl_id == DOUBLES_SUIT:
        return "Trump: doubles-suit (no trump; doubles form their own suit but have no trump power)."
    if decl_id == DOUBLES_TRUMP:
        return "Trump: doubles-trump (only the 7 doubles are trump, 6-6 high → 0-0 low)."
    if decl_id in PIP_TRUMP_IDS:
        return f"Trump: {name} (every domino containing a {decl_id} is trump; the {decl_id}-{decl_id} is highest)."
    return f"Trump: decl_{decl_id}."


def _seat_role(seat: int, bidder: int) -> str:
    """Return a label like 'OFFENSE (bidder)' / 'DEFENSE'."""
    bidder_team = bidder % 2
    seat_team = seat % 2
    on_offense = seat_team == bidder_team
    return "OFFENSE" if on_offense else "DEFENSE"


_COUNT_TIERS = (10, 5)


def _count_dominoes_still_in_play(played: frozenset) -> dict[int, list[int]]:
    """Group remaining count dominoes by point value (10-pt, 5-pt)."""
    out: dict[int, list[int]] = {tier: [] for tier in _COUNT_TIERS}
    for d in range(28):
        pts = DOMINO_COUNT_POINTS[d]
        if pts == 0 or d in played:
            continue
        out[pts].append(d)
    return out


def _format_trick_history(play_history: tuple, decl_id: int) -> list[str]:
    """One-line-per-trick render of completed tricks. Does not recompute
    winner (that requires resolve_trick); shows the play sequence with the
    leading seat marked."""
    if not play_history:
        return []
    lines: list[str] = []
    # Walk in chunks of 4. play_history is ((seat, dom), ...) in play order.
    for trick_idx in range(0, len(play_history) // 4):
        chunk = play_history[trick_idx * 4 : trick_idx * 4 + 4]
        leader_seat = chunk[0][0]
        plays_str = ", ".join(f"s{s}:{_label(d)}" for s, d in chunk)
        lines.append(f"  T{trick_idx + 1}  led by seat {leader_seat}:  {plays_str}")
    return lines


def _current_trick_render(
    current_trick: tuple, trick_leader: int, me_abs: int, decl_id: int,
) -> list[str]:
    """Render the in-progress trick (lead suit, plays so far)."""
    n_played = len(current_trick)
    if n_played == 0:
        return [f"  leader: {'me' if trick_leader == me_abs else f'seat {trick_leader}'}    plays: (none — leader's turn)"]
    lead_dom = int(current_trick[0])
    led_suit = led_suit_for_lead_domino(lead_dom, decl_id)
    if led_suit == 7:
        led_label = "trump (the lead is in trump)"
    else:
        led_label = f"{led_suit}s (pip-suit; trump can still win)"
    # Reconstruct seat-per-position from leader.
    pos_seats = [(trick_leader + i) % 4 for i in range(n_played)]
    plays_str = ", ".join(
        f"s{s}:{_label(d)}" for s, d in zip(pos_seats, current_trick)
    )
    return [
        f"  leader: seat {trick_leader}    led suit: {led_label}",
        f"  plays so far: {plays_str}",
    ]


def render_full_board_snapshot(game_state: Any, me_abs: int) -> str:
    """Produce the snapshot prose for the current decision."""
    gs = game_state
    decl_id = int(gs.decl_id)
    bidder = int(gs.bidder)
    bid_value = int(gs.bid_state.high_bid)
    played: frozenset = gs.played

    my_team = me_abs % 2
    partner_seat = (me_abs + 2) % 4
    left_opp = (me_abs + 1) % 4
    right_opp = (me_abs + 3) % 4

    my_initial_hand = list(gs.hands[me_abs])
    my_remaining = sorted(d for d in my_initial_hand if d not in played)

    n_completed_tricks = len(gs.play_history) // 4
    cur_trick_n = n_completed_tricks + 1
    pos_in_trick = len(gs.current_trick) + 1

    lines: list[str] = []
    lines.append(
        f"SNAPSHOT — trick {cur_trick_n}/7, position {pos_in_trick}/4 "
        f"({'YOU LEAD' if pos_in_trick == 1 and gs.trick_leader == me_abs else 'mid-trick'})"
    )
    lines.append("")

    # Seats / teams
    me_role = _seat_role(me_abs, bidder)
    partner_role = _seat_role(partner_seat, bidder)
    lopp_role = _seat_role(left_opp, bidder)
    ropp_role = _seat_role(right_opp, bidder)
    lines.append(
        f"Me: seat {me_abs} ({me_role})    Partner: seat {partner_seat} ({partner_role})    "
        f"Left opp: seat {left_opp} ({lopp_role})    Right opp: seat {right_opp} ({ropp_role})"
    )
    lines.append(
        f"Bid: seat {bidder} bid {bid_value} (Team {bidder % 2})."
    )
    lines.append("")

    # Trump
    lines.append(_trump_summary_line(decl_id))
    trumps = _trump_dominoes_high_to_low(decl_id)
    if trumps:
        my_trumps = [d for d in trumps if d in my_remaining]
        played_trumps = [d for d in trumps if d in played]
        unseen_trumps = [d for d in trumps if d not in my_remaining and d not in played]
        lines.append(
            "  ranking high→low: " + ", ".join(_label(d) for d in trumps)
        )
        lines.append(
            f"  in my hand: {', '.join(_label(d) for d in my_trumps) if my_trumps else '(none)'}"
        )
        lines.append(
            f"  already played: {', '.join(_label(d) for d in played_trumps) if played_trumps else '(none)'}"
        )
        lines.append(
            f"  unseen (with partner or opps): "
            f"{', '.join(_label(d) for d in unseen_trumps) if unseen_trumps else '(none — all trumps gone)'}"
        )
    lines.append("")

    # My hand
    lines.append(
        "My remaining hand: "
        + (", ".join(_label(d) for d in my_remaining) if my_remaining else "(none)")
    )
    lines.append(
        "Partner's exact hand is hidden information — call belief_trajectory() "
        "for per-domino seat posteriors."
    )
    lines.append("")

    # Current trick
    lines.append(f"Current trick (#{cur_trick_n}):")
    lines.extend(_current_trick_render(
        gs.current_trick, int(gs.trick_leader), me_abs, decl_id,
    ))
    lines.append("")

    # Score
    t0_pts = int(gs.team_points[0])
    t1_pts = int(gs.team_points[1])
    my_team_pts = t0_pts if my_team == 0 else t1_pts
    opp_team_pts = t1_pts if my_team == 0 else t0_pts
    bidder_team = bidder % 2
    bidder_team_pts = t0_pts if bidder_team == 0 else t1_pts
    bid_remaining = max(0, bid_value - bidder_team_pts)
    lines.append("Score so far (count + trick points combined; each trick = 1 trick point + count dominoes won):")
    lines.append(
        f"  Team 0 (seats 0+2): {t0_pts}    Team 1 (seats 1+3): {t1_pts}    "
        f"(total distributed: {t0_pts + t1_pts} of 42)"
    )
    lines.append(
        f"  My team (Team {my_team}): {my_team_pts}    "
        f"Opp team: {opp_team_pts}"
    )
    lines.append(
        f"  Bidder team (Team {bidder_team}) needs {bid_remaining} more "
        f"to make {bid_value}."
    )
    lines.append("")

    # Count dominoes still in play
    counts = _count_dominoes_still_in_play(played)
    lines.append("Count dominoes still in play:")
    for tier in _COUNT_TIERS:
        if not counts[tier]:
            lines.append(f"  {tier:>2}-pt: (none remaining)")
        else:
            lines.append(
                f"  {tier:>2}-pt: " + ", ".join(_label(d) for d in counts[tier])
            )
    captured_count_doms = [
        d for d in range(28) if DOMINO_COUNT_POINTS[d] > 0 and d in played
    ]
    if captured_count_doms:
        lines.append(
            "  already played (captured or in current trick): "
            + ", ".join(_label(d) for d in captured_count_doms)
        )
    lines.append("")

    # Trick history
    if n_completed_tricks > 0:
        lines.append(f"Tricks completed ({n_completed_tricks}):")
        lines.extend(_format_trick_history(gs.play_history, decl_id))
        lines.append("")

    lines.append(
        "For opp-hand probabilities: belief_trajectory().    "
        "For a candidate play's outcome distribution: explore_game(play=ID)."
    )
    return "\n".join(lines)


def render_full_board_structured(game_state: Any, me_abs: int) -> dict:
    """Companion structured payload — same data, machine-readable."""
    gs = game_state
    decl_id = int(gs.decl_id)
    played: frozenset = gs.played
    n_completed = len(gs.play_history) // 4
    my_remaining = sorted(d for d in gs.hands[me_abs] if d not in played)
    trumps = _trump_dominoes_high_to_low(decl_id)
    counts = _count_dominoes_still_in_play(played)
    return {
        "me_abs": int(me_abs),
        "my_team": int(me_abs % 2),
        "partner_seat": (int(me_abs) + 2) % 4,
        "left_opp_seat": (int(me_abs) + 1) % 4,
        "right_opp_seat": (int(me_abs) + 3) % 4,
        "decl_id": decl_id,
        "decl_name": DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}"),
        "bidder": int(gs.bidder),
        "bid_value": int(gs.bid_state.high_bid),
        "trick_index": n_completed + 1,
        "position_in_trick": len(gs.current_trick) + 1,
        "trick_leader": int(gs.trick_leader),
        "current_trick_plays": [int(d) for d in gs.current_trick],
        "play_history": [[int(s), int(d)] for s, d in gs.play_history],
        "team_points": [int(gs.team_points[0]), int(gs.team_points[1])],
        "my_remaining_hand": my_remaining,
        "trump_dominoes_high_to_low": trumps,
        "count_dominoes_remaining": {str(k): v for k, v in counts.items()},
    }
