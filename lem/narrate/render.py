"""Turn a GameRecordGPU into prose from one player's seat.

Second-person for the narrator ("you"), third-person for everyone else.
Partner is "your partner". Opponents are "Player N".
"""

from __future__ import annotations

from forge.oracle.declarations import NOTRUMP, DOUBLES_SUIT, DOUBLES_TRUMP, has_trump_power
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    resolve_trick,
)


# Declaration → human phrase for the bid announcement.
def _decl_phrase(decl_id: int) -> str:
    names = {
        0: "blanks",
        1: "ones",
        2: "twos",
        3: "threes",
        4: "fours",
        5: "fives",
        6: "sixes",
    }
    if decl_id in names:
        return f"called {names[decl_id]} as trump"
    if decl_id == DOUBLES_TRUMP:
        return "called doubles as trump"
    if decl_id == DOUBLES_SUIT:
        return "called doubles as their own suit (no trump)"
    if decl_id == NOTRUMP:
        return "called no trump"
    raise ValueError(f"unknown decl_id {decl_id}")


def _dom(dom_id: int) -> str:
    return f"{DOMINO_HIGH[dom_id]}-{DOMINO_LOW[dom_id]}"


def _format_hand(ids: list[int]) -> str:
    # Sort by high desc, low desc — reads natural (6-6, 5-3, ...)
    ordered = sorted(ids, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i]))
    return ", ".join(_dom(d) for d in ordered)


def _player_ref(player: int, narrator: int, partner: int, *, cap: bool = False) -> str:
    if player == narrator:
        s = "you"
    elif player == partner:
        s = "your partner"
    else:
        s = f"Player {player}"
    return s[0].upper() + s[1:] if cap else s


def _classify_follow(domino_id: int, lead_domino: int, decl_id: int) -> str:
    """Classify a non-leading play. Returns one of: 'follow', 'trump', 'sluff'."""
    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    if has_trump_power(decl_id) and is_in_called_suit(domino_id, decl_id):
        if led_suit == 7:
            # Trump was led and they played trump — that's following suit
            return "follow"
        return "trump"
    if can_follow(domino_id, led_suit, decl_id):
        return "follow"
    return "sluff"


# (subject-is-narrator, classification) → verb phrase
_FOLLOW_VERBS = {
    # narrator form ("you follow")
    (True, "follow"): "follow",
    (True, "trump"): "trump in",
    (True, "sluff"): "sluff",
    # third-person form ("partner follows")
    (False, "follow"): "follows",
    (False, "trump"): "trumps in",
    (False, "sluff"): "sluffs",
}


def render_narration(
    record,
    narrator: int,
    bid: int,
    bidder: int = 0,
    stop_at_decision: int | None = None,
) -> str:
    """Render a GameRecordGPU as second-person prose from `narrator`'s seat.

    Args:
        record: GameRecordGPU with 28 decisions in play order.
        narrator: Player index (0-3) whose POV we narrate.
        bid: Bid value the bidder committed to (e.g. 30).
        bidder: Player index that won the bid (default 0 — matches simulator convention).
        stop_at_decision: If set, truncate before this decision index and emit a
            "what do you play?" prompt. The decision at this index must belong to
            `narrator`.
    """
    decl_id = record.decl_id
    initial_hands = record.hands
    decisions = record.decisions
    partner = (narrator + 2) % 4
    narrator_team = narrator % 2
    bidder_team = bidder % 2

    narrator_turns = [i for i, d in enumerate(decisions) if d.player == narrator]

    if stop_at_decision is not None:
        if not (0 <= stop_at_decision < len(decisions)):
            raise ValueError(
                f"stop_at_decision={stop_at_decision} out of range [0, {len(decisions)})"
            )
        stop_player = decisions[stop_at_decision].player
        if stop_player != narrator:
            raise ValueError(
                f"stop_at_decision={stop_at_decision} is Player {stop_player}'s turn, "
                f"but narrator is Player {narrator}. "
                f"Narrator's decisions in this game: {narrator_turns}"
            )

    opp_a = (narrator + 1) % 4
    opp_b = (narrator + 3) % 4

    lines: list[str] = []
    lines.append("You are playing Texas 42.")
    lines.append(
        f"You are Player {narrator}. Your partner is Player {partner}. "
        f"Your opponents are Player {opp_a} and Player {opp_b}."
    )
    lines.append(f"Your hand: {_format_hand(initial_hands[narrator])}.")
    lines.append("")

    bidder_ref = _player_ref(bidder, narrator, partner, cap=True)
    lines.append(f"{bidder_ref} bid {bid} and {_decl_phrase(decl_id)}.")
    lines.append("")

    team_points = [0, 0]
    assert len(decisions) == 28, f"expected 28 decisions, got {len(decisions)}"

    def _render_play(pref_line_indent: str, is_leader: bool, player: int, dom_id: int,
                     lead_dom: int, dec) -> str:
        forced = int(dec.legal_mask.sum().item()) == 1
        tag = " (forced)" if forced else ""
        pref = _player_ref(player, narrator, partner, cap=True)
        if is_leader:
            verb = "lead" if player == narrator else "leads"
            return f"{pref_line_indent}{pref} {verb} the {_dom(dom_id)}{tag}."
        kind = _classify_follow(dom_id, lead_dom, decl_id)
        verb = _FOLLOW_VERBS[(player == narrator, kind)]
        return f"{pref_line_indent}{pref} {verb} with the {_dom(dom_id)}{tag}."

    for trick_num in range(7):
        trick_start = trick_num * 4
        trick_decisions = decisions[trick_start : trick_start + 4]
        plays: list[tuple[int, int]] = []
        for d in trick_decisions:
            dom_id = initial_hands[d.player][d.action_taken]
            plays.append((d.player, dom_id))

        stopping_here = (
            stop_at_decision is not None
            and trick_start <= stop_at_decision < trick_start + 4
        )
        stop_offset = (stop_at_decision - trick_start) if stopping_here else 4

        lines.append(f"Trick {trick_num + 1}:")

        lead_dom = plays[0][1]
        for i in range(stop_offset):
            player, dom_id = plays[i]
            lines.append(
                _render_play("  ", i == 0, player, dom_id, lead_dom, trick_decisions[i])
            )

        if stopping_here:
            # Build the decision block and return.
            lines.append("")
            lines.append("--")
            # Compute narrator's remaining hand
            played_by_narrator: set[int] = set()
            for past in decisions[:stop_at_decision]:
                if past.player == narrator:
                    played_by_narrator.add(
                        initial_hands[narrator][past.action_taken]
                    )
            remaining = [
                d for d in initial_hands[narrator] if d not in played_by_narrator
            ]

            lines.append(f"It is your turn on trick {trick_num + 1}.")
            if stop_offset == 0:
                lines.append("You have the lead.")
            else:
                lead_player, lead_dom_local = plays[0]
                led_ref = _player_ref(lead_player, narrator, partner, cap=True)
                lines.append(f"{led_ref} led the {_dom(lead_dom_local)}.")
            lines.append(f"Your remaining dominoes: {_format_hand(remaining)}.")
            lines.append(
                f"Score so far — your team: {team_points[narrator_team]}, "
                f"theirs: {team_points[1 - narrator_team]}."
            )
            lines.append("")
            lines.append(
                "Before you play, fill in this scratchpad to organize your thinking:\n"
                "HAND: [list your remaining dominoes]\n"
                "VOIDS: [list known voids from play history, e.g. \"Player 2: fives\" — "
                "or \"none observed\" if no voids are known]\n"
                "COUNTS: [for each count domino (5-5=10pts, 6-4=10pts, 5-0=5pts, "
                "4-1=5pts, 3-2=5pts), state played or still out]\n"
                "PLAY: [the domino you choose to play]"
            )
            return "\n".join(lines)

        # Full trick — render the result line and carry score forward.
        dominoes = tuple(p[1] for p in plays)
        outcome = resolve_trick(lead_dom, dominoes, decl_id)
        winner_player = plays[outcome.winner_offset][0]
        winner_team = winner_player % 2
        team_points[winner_team] += outcome.points

        winner_ref = _player_ref(winner_player, narrator, partner, cap=True)
        takes_verb = "take" if winner_player == narrator else "takes"
        lines.append(
            f"  → {winner_ref} {takes_verb} the trick — {outcome.points} point"
            f"{'s' if outcome.points != 1 else ''}. "
            f"[Your team: {team_points[narrator_team]}, theirs: {team_points[1 - narrator_team]}]"
        )
        lines.append("")

    # Closing
    your_total = team_points[narrator_team]
    their_total = team_points[1 - narrator_team]
    lines.append(f"Final: your team {your_total}, their team {their_total}.")

    bidder_points = team_points[bidder_team]
    bidder_label = _player_ref(bidder, narrator, partner, cap=True)
    if bidder_points >= bid:
        over = bidder_points - bid
        if over == 0:
            lines.append(f"{bidder_label} bid {bid} and made it exactly.")
        else:
            lines.append(f"{bidder_label} bid {bid} and made it by {over}.")
    else:
        short = bid - bidder_points
        lines.append(f"{bidder_label} bid {bid} and was set by {short}.")

    return "\n".join(lines)
