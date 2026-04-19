"""Rules-as-tools for Burl — replaces the prose primer with callable answers.

Each tool takes the same duck-typed ``game_state`` the engine tools accept
(see ``burl.tools.engine`` module docstring for the required fields and
seat-convention rules). Each tool returns JSON-serialisable primitives so the
native harness observation formatter round-trips it cleanly.

No rule is reimplemented here. Every behavioural call routes through
``forge.oracle.tables`` / ``forge.oracle.declarations``. The only logic this
module owns is *presentation* — which primitives to return and how to label
them — so if the engine is right, these tools are right.

Design rationale: see ``scratch/burl_p5_iter2_prep/rules_as_tools_design.md``.
"""

from __future__ import annotations

from typing import Any

from forge.oracle.declarations import (
    DOUBLES_SUIT,
    NOTRUMP,
    N_DECLS,
    has_trump_power,
)
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    resolve_trick,
    trick_rank,
)


# --------------------------------------------------------------------------- #
# Local helpers — duplicated from engine.py so rules.py has no intra-burl dep #
# on private names (engine.py is in another teammate's file scope).            #
# --------------------------------------------------------------------------- #


_SUIT_NAMES = (
    "blanks", "ones", "twos", "threes", "fours", "fives", "sixes", "called",
)


def _domino_label(domino_id: int) -> str:
    return f"{DOMINO_HIGH[domino_id]}-{DOMINO_LOW[domino_id]}"


def _abs_current_player(game_state: Any) -> int:
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = game_state.leader
    return (int(leader) + len(game_state.current_trick)) % 4


def _current_trick_domino_ids(game_state: Any) -> tuple[int, ...]:
    trick = game_state.current_trick
    if not trick:
        return ()
    first = trick[0]
    if isinstance(first, int):
        return tuple(trick)
    return tuple(d for _p, d in trick)


def _seat_role(me: int, partner: int, seat: int) -> str:
    if seat == me:
        return "me"
    if seat == partner:
        return "partner"
    if seat == (me + 1) % 4:
        return "left_opponent"
    return "right_opponent"


def _check_domino_id(domino_id: int) -> None:
    if not isinstance(domino_id, int) or isinstance(domino_id, bool):
        raise ValueError(
            f"domino_id must be int, got {type(domino_id).__name__}"
        )
    if not 0 <= domino_id < N_DOMINOES:
        raise ValueError(
            f"domino_id out of range [0, {N_DOMINOES}): {domino_id}"
        )


def _check_decl(decl_id: int) -> None:
    if not 0 <= decl_id < N_DECLS:
        raise ValueError(f"state.decl_id out of range [0, {N_DECLS}): {decl_id}")


def _count_entry(domino_id: int) -> dict:
    return {
        "domino_id": int(domino_id),
        "pip_label": _domino_label(domino_id),
        "count_value": int(DOMINO_COUNT_POINTS[domino_id]),
    }


def _domino_report(game_state: Any, domino_id: int, led_suit: int) -> dict:
    """Shared rank + trump + follow breakdown for a domino under a led suit."""
    decl = game_state.decl_id
    return {
        "domino_id": int(domino_id),
        "pip_label": _domino_label(domino_id),
        "rank": int(trick_rank(domino_id, led_suit, decl)),
        "is_trump": bool(
            has_trump_power(decl) and is_in_called_suit(domino_id, decl)
        ),
        "can_follow": bool(can_follow(domino_id, led_suit, decl)),
    }


def _beats_reason(
    game_state: Any,
    winner_id: int,
    loser_id: int,
    led_suit: int,
) -> str:
    decl = game_state.decl_id
    if has_trump_power(decl) and is_in_called_suit(winner_id, decl):
        if is_in_called_suit(loser_id, decl):
            return "winner is the higher trump"
        return "winner is trump; loser is not"
    if can_follow(winner_id, led_suit, decl) and not can_follow(
        loser_id, led_suit, decl
    ):
        return "winner follows the led suit; loser is off-suit"
    return "winner is the higher-ranked domino in the led suit"


# --------------------------------------------------------------------------- #
# count_dominoes_remaining                                                     #
# --------------------------------------------------------------------------- #


def count_dominoes_remaining(game_state: Any) -> dict:
    """Which count dominoes are still live and who has captured what so far.

    The double-six set has exactly 5 count dominoes (total 35 count points):
    5-5 and 6-4 worth 10 each; 5-0, 4-1, 3-2 worth 5 each. This tool reports
    which of them are still unplayed, which have been captured, and the
    running team totals from ``team_points`` (0s if the field is absent).

    All seats are absolute; my team is my absolute seat's parity (0 or 1).
    """
    _check_decl(game_state.decl_id)
    me = _abs_current_player(game_state)
    my_team = me % 2

    unplayed_5: list[dict] = []
    unplayed_10: list[dict] = []
    played_5: list[dict] = []
    played_10: list[dict] = []

    for d in range(N_DOMINOES):
        pts = DOMINO_COUNT_POINTS[d]
        if pts == 0:
            continue
        entry = _count_entry(d)
        if d in game_state.played:
            (played_5 if pts == 5 else played_10).append(entry)
        else:
            (unplayed_5 if pts == 5 else unplayed_10).append(entry)

    loose = sum(e["count_value"] for e in unplayed_5 + unplayed_10)

    team_points = getattr(game_state, "team_points", (0, 0))
    if team_points is None or len(team_points) != 2:
        team_points = (0, 0)

    return {
        "unplayed_5pt": unplayed_5,
        "unplayed_10pt": unplayed_10,
        "played_5pt": played_5,
        "played_10pt": played_10,
        "loose_count_points": int(loose),
        "my_team_captured": int(team_points[my_team]),
        "opp_team_captured": int(team_points[1 - my_team]),
    }


# --------------------------------------------------------------------------- #
# trick_winner_if                                                              #
# --------------------------------------------------------------------------- #


def _trick_leader(game_state: Any) -> int:
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = getattr(game_state, "leader", 0)
    return int(leader)


def _count_value(domino_id: int) -> int:
    return int(DOMINO_COUNT_POINTS[domino_id])


def trick_winner_if(game_state: Any, domino_id: int) -> dict:
    """Simulate playing ``domino_id`` into the current trick.

    Three cases:

    1. No trick in progress (I am about to lead). Reports the led suit my
       lead would set; no winner yet.
    2. Trick partially played and my play does not complete it. Reports
       the led suit, whether my play would currently lead among the plays
       so far, and how many plays remain.
    3. My play completes the trick. Runs the full resolution through
       ``forge.oracle.tables.resolve_trick`` and reports the winner seat
       (absolute + role), whether my team takes it, and the points at
       stake on this trick.

    The tool DOES NOT check legality; call ``is_legal`` first if you want
    that. This is the "what happens if I played X" primitive, not the
    "should I play X" primitive.
    """
    _check_domino_id(domino_id)
    _check_decl(game_state.decl_id)

    me = _abs_current_player(game_state)
    partner = (me + 2) % 4
    my_team = me % 2
    leader = _trick_leader(game_state)
    decl = game_state.decl_id

    trick_ids = _current_trick_domino_ids(game_state)
    position = len(trick_ids)

    if position == 0:
        led_suit = led_suit_for_lead_domino(domino_id, decl)
        return {
            "completes_trick": False,
            "i_am_leading": True,
            "led_suit_if_played": _SUIT_NAMES[led_suit],
            "winner_seat_role": None,
            "winner_seat_absolute": None,
            "my_team_wins": None,
            "points_in_trick_so_far": int(_count_value(domino_id)),
            "note": "my lead sets the led suit; winner TBD when others play",
        }

    lead = trick_ids[0]
    led_suit = led_suit_for_lead_domino(lead, decl)
    full_plays = list(trick_ids) + [domino_id]

    if position == 3:
        outcome = resolve_trick(lead, tuple(full_plays), decl)
        winner_seat = (leader + outcome.winner_offset) % 4
        winner_role = _seat_role(me, partner, winner_seat)
        return {
            "completes_trick": True,
            "i_am_leading": False,
            "led_suit": _SUIT_NAMES[led_suit],
            "winner_seat_role": winner_role,
            "winner_seat_absolute": int(winner_seat),
            "my_team_wins": bool(winner_seat % 2 == my_team),
            "points_at_stake": int(outcome.points),
            "note": (
                f"trick completes: {winner_role} (seat {winner_seat}) "
                f"wins {outcome.points} points"
            ),
        }

    my_rank = trick_rank(domino_id, led_suit, decl)
    ranks = [trick_rank(d, led_suit, decl) for d in trick_ids]
    best_so_far = max(range(len(trick_ids)), key=lambda i: ranks[i])

    if my_rank > ranks[best_so_far]:
        leading_seat = me
        leading_role = "me"
    else:
        leading_seat = (leader + best_so_far) % 4
        leading_role = _seat_role(me, partner, leading_seat)

    remaining = 3 - position
    points_so_far = sum(_count_value(d) for d in full_plays)
    return {
        "completes_trick": False,
        "i_am_leading": False,
        "led_suit": _SUIT_NAMES[led_suit],
        "currently_leading_seat_role": leading_role,
        "currently_leading_seat_absolute": int(leading_seat),
        "my_play_would_lead_partial": bool(leading_seat == me),
        "plays_remaining_after_me": int(remaining),
        "points_in_trick_so_far": int(points_so_far),
        "note": (
            f"partial trick: after my play, {leading_role} leads; "
            f"{remaining} more play(s) before resolution"
        ),
    }


# --------------------------------------------------------------------------- #
# what_beats_what                                                              #
# --------------------------------------------------------------------------- #


def what_beats_what(
    game_state: Any,
    domino_a: int,
    domino_b: int,
    lead_domino: int | None = None,
) -> dict:
    """Compare A and B as trick plays under the current declaration.

    Comparison depends on the led suit of the trick, which depends on both
    the declaration and the lead domino. If a trick is in progress,
    ``lead_domino`` defaults to that trick's lead. If no trick is in
    progress, pass a ``lead_domino`` explicitly (e.g. "if I led 21, what
    would beat it?"). A ``ValueError`` is raised if no led suit can be
    determined.

    The two dominoes are treated symmetrically — there is no "leader" among
    them; the question is "if both were played into a trick led by
    ``lead_domino``, which would rank higher?"
    """
    _check_domino_id(domino_a)
    _check_domino_id(domino_b)
    _check_decl(game_state.decl_id)

    if lead_domino is None:
        trick_ids = _current_trick_domino_ids(game_state)
        if trick_ids:
            lead_domino = int(trick_ids[0])

    if lead_domino is None:
        raise ValueError(
            "lead_domino is required when no trick is in progress"
        )
    _check_domino_id(lead_domino)

    decl = game_state.decl_id
    led_suit = led_suit_for_lead_domino(lead_domino, decl)

    rank_a = trick_rank(domino_a, led_suit, decl)
    rank_b = trick_rank(domino_b, led_suit, decl)

    if rank_a > rank_b:
        winner = "a"
        reason = _beats_reason(game_state, domino_a, domino_b, led_suit)
    elif rank_b > rank_a:
        winner = "b"
        reason = _beats_reason(game_state, domino_b, domino_a, led_suit)
    else:
        winner = "neither"
        reason = (
            "both are off-suit (neither follows the led suit nor trumps); "
            "whichever is played later is irrelevant to the trick winner"
        )

    return {
        "winner": winner,
        "reason": reason,
        "led_suit": _SUIT_NAMES[led_suit],
        "lead_domino_id": int(lead_domino),
        "lead_domino_label": _domino_label(lead_domino),
        "a": _domino_report(game_state, domino_a, led_suit),
        "b": _domino_report(game_state, domino_b, led_suit),
    }


# --------------------------------------------------------------------------- #
# contract_progress                                                            #
# --------------------------------------------------------------------------- #


def contract_progress(game_state: Any) -> dict:
    """Progress toward (or away from) the bid contract.

    Reports who bid, what their target is, what each team has captured so
    far (from ``team_points``), how much count is still loose, and a
    coarse status tag. When ``bidder`` or ``bid_state.high_bid`` is absent
    or nonsensical, falls back to a 30-count floor and a
    ``"bidder_unknown"`` status rather than raising — some early scaffolding
    states omit those fields.
    """
    _check_decl(game_state.decl_id)
    me = _abs_current_player(game_state)
    my_team = me % 2
    opp_team = 1 - my_team

    bidder_raw = getattr(game_state, "bidder", -1)
    bidder = int(bidder_raw) if bidder_raw is not None else -1

    bid_state = getattr(game_state, "bid_state", None)
    high_bid = 0
    if bid_state is not None:
        high_bid = int(getattr(bid_state, "high_bid", 0) or 0)
    bid_target = high_bid if high_bid >= 30 else 30

    team_points = getattr(game_state, "team_points", (0, 0))
    if team_points is None or len(team_points) != 2:
        team_points = (0, 0)
    my_team_captured = int(team_points[my_team])
    opp_team_captured = int(team_points[opp_team])

    loose_count = sum(
        DOMINO_COUNT_POINTS[d]
        for d in range(N_DOMINOES)
        if d not in game_state.played
    )
    tricks_completed = len(game_state.play_history) // 4
    tricks_remaining = max(0, 7 - tricks_completed)
    hand_points_remaining = int(loose_count) + int(tricks_remaining)

    bidder_team: int | None = None
    my_team_role = "unknown"
    offense_captured: int | None = None
    defense_captured: int | None = None
    offense_needed: int | None = None
    defense_set_threshold: int | None = None

    if 0 <= bidder < 4:
        bidder_team = bidder % 2
        if bidder_team == my_team:
            my_team_role = "offense"
            offense_captured = my_team_captured
            defense_captured = opp_team_captured
        else:
            my_team_role = "defense"
            offense_captured = opp_team_captured
            defense_captured = my_team_captured
        offense_needed = max(0, bid_target - offense_captured)
        defense_set_threshold = max(0, 42 - bid_target + 1)

    if offense_captured is None:
        status = "bidder_unknown"
    elif offense_captured >= bid_target:
        status = "offense_has_made_bid"
    elif offense_captured + hand_points_remaining < bid_target:
        status = "defense_has_set_bid"
    else:
        status = "contract_in_play"

    return {
        "bidder_seat_absolute": int(bidder),
        "bidder_team": bidder_team,
        "my_team_role": my_team_role,
        "bid_target": int(bid_target),
        "my_team_captured": my_team_captured,
        "opp_team_captured": opp_team_captured,
        "offense_captured": offense_captured,
        "defense_captured": defense_captured,
        "offense_count_still_needed": offense_needed,
        "defense_points_to_set_bid": defense_set_threshold,
        "loose_count_points": int(loose_count),
        "trick_points_remaining": int(tricks_remaining),
        "tricks_completed": int(tricks_completed),
        "tricks_remaining": int(tricks_remaining),
        "hand_points_remaining": hand_points_remaining,
        "status": status,
    }


__all__ = [
    "count_dominoes_remaining",
    "trick_winner_if",
    "what_beats_what",
    "contract_progress",
]
