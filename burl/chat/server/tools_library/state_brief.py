DESCRIPTION = "State synthesis in Burl's preferred labeled format: [GAME STATE], [CONTEXT & GOAL], [PROTOCOL]. Same data as the user-message dump but parsed into terse bulleted sections instead of narrative prose. Pure rule-based, sub-ms. Use as the first read on any decision."

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
    led_suit_for_lead_domino,
)


def _label(d):
    return f"{int(d)}({_DOMINO_LABELS[int(d)]})"


def _trumps_high_to_low(decl_id):
    if decl_id in PIP_TRUMP_IDS:
        trump_pip = decl_id
        cands = [d for d in range(28) if trump_pip in (DOMINO_HIGH[d], DOMINO_LOW[d])]
        def rank(d):
            if DOMINO_IS_DOUBLE[d]:
                return 100
            other = DOMINO_HIGH[d] if DOMINO_LOW[d] == trump_pip else DOMINO_LOW[d]
            return other
        return sorted(cands, key=rank, reverse=True)
    if decl_id == DOUBLES_TRUMP:
        return sorted(
            [d for d in range(28) if DOMINO_IS_DOUBLE[d]],
            key=lambda d: DOMINO_HIGH[d],
            reverse=True,
        )
    return []


def _trump_summary(decl_id):
    name = DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}")
    if decl_id == NOTRUMP:
        return "notrump (no trump suit; highest of led suit wins)"
    if decl_id == DOUBLES_SUIT:
        return "doubles-suit (no trump; doubles form their own suit)"
    if decl_id == DOUBLES_TRUMP:
        return "doubles-trump (only the 7 doubles are trump, 6-6 → 0-0)"
    if decl_id in PIP_TRUMP_IDS:
        return f"{name} (every domino with a {decl_id}; the {decl_id}-{decl_id} is highest)"
    return name


def tool(ctx, **kwargs):
    gs = ctx.game_state
    me = int(ctx.me_abs)
    decl_id = int(gs.decl_id)
    bidder = int(gs.bidder)
    bid_value = int(gs.bid_state.high_bid)
    played = gs.played

    my_team = me % 2
    bidder_team = bidder % 2
    on_offense = my_team == bidder_team
    role = "OFFENSE" if on_offense else "DEFENSE"
    partner = (me + 2) % 4
    left = (me + 1) % 4
    right = (me + 3) % 4

    my_hand = sorted(d for d in gs.hands[me] if d not in played)
    cur = tuple(int(x) for x in gs.current_trick)
    n_done = len(gs.play_history) // 4
    trick_n = n_done + 1
    pos_in_trick = len(cur) + 1

    trumps = _trumps_high_to_low(decl_id)
    my_trumps = [d for d in trumps if d in my_hand]
    played_trumps = [d for d in trumps if d in played]
    unseen_trumps = [d for d in trumps if d not in my_hand and d not in played]

    if not cur:
        action_line = "you lead this trick"
        trick_line = "(no plays yet)"
    else:
        lead = cur[0]
        led_suit = led_suit_for_lead_domino(lead, decl_id)
        if led_suit == 7:
            led_label = "TRUMP"
        else:
            led_label = f"{led_suit}s"
        seat_per_pos = [(int(gs.trick_leader) + i) % 4 for i in range(len(cur))]
        plays = ", ".join(f"s{s}:{_label(d)}" for s, d in zip(seat_per_pos, cur))
        action_line = f"position {pos_in_trick}/4; led suit = {led_label}"
        trick_line = f"led by seat {gs.trick_leader} — {plays}"

    if gs.play_history:
        history_lines = []
        for ti in range(n_done):
            chunk = gs.play_history[ti * 4 : ti * 4 + 4]
            leader_seat = chunk[0][0]
            plays = ", ".join(f"s{s}:{_label(d)}" for s, d in chunk)
            history_lines.append(f"  T{ti + 1} (led by s{leader_seat}): {plays}")
        history_block = "\n".join(history_lines)
    else:
        history_block = "  (none)"

    t0_pts = int(gs.team_points[0])
    t1_pts = int(gs.team_points[1])
    bidder_team_pts = t0_pts if bidder_team == 0 else t1_pts
    bid_remaining = max(0, bid_value - bidder_team_pts)

    count_5 = sorted(d for d in range(28) if DOMINO_COUNT_POINTS[d] == 5 and d not in played)
    count_10 = sorted(d for d in range(28) if DOMINO_COUNT_POINTS[d] == 10 and d not in played)
    count_5_str = ", ".join(_label(d) for d in count_5) or "(none)"
    count_10_str = ", ".join(_label(d) for d in count_10) or "(none)"

    protocol_line = (
        "1. belief_trajectory()           — Gus's posterior over opponent hands\n"
        "2. explore_game(play=X)          — outcome distribution for one candidate\n"
        "3. probe_best_case(play=X) /     — condition on upside or downside catalyst\n"
        "   probe_worst_case(play=X)\n"
        "4. commit_play(domino_id=X)      — final play (after >=1 probe)"
    )

    lines = []
    lines.append("[GAME STATE]")
    lines.append(f"- Trick {trick_n}/7, {action_line}")
    lines.append(
        f"- You: seat {me} (Team {my_team}, {role})    "
        f"Partner: seat {partner}    Left opp: seat {left}    Right opp: seat {right}"
    )
    lines.append("- My hand: " + (", ".join(_label(d) for d in my_hand) or "(empty)"))
    lines.append(f"- Trump: {_trump_summary(decl_id)}")
    if trumps:
        lines.append(
            f"  Trumps high→low: " + ", ".join(_label(d) for d in trumps)
        )
        lines.append(
            f"  In my hand: " + (", ".join(_label(d) for d in my_trumps) or "(none)")
        )
        lines.append(
            f"  Already played: " + (", ".join(_label(d) for d in played_trumps) or "(none)")
        )
        lines.append(
            f"  Unseen (with partner or opps): "
            + (", ".join(_label(d) for d in unseen_trumps) or "(none — all out)")
        )
    lines.append(f"- Current trick: {trick_line}")
    lines.append("- Tricks completed:")
    lines.append(history_block)
    lines.append("")
    lines.append("[CONTEXT & GOAL]")
    lines.append(f"- Bid: seat {bidder} bid {bid_value} (Team {bidder_team})")
    if on_offense:
        lines.append(f"- My role: OFFENSE — must capture >= {bid_value} count")
    else:
        lines.append(
            f"- My role: DEFENSE — keep Team {bidder_team} below {bid_value} count to SET them"
        )
    lines.append(f"- Score: Team 0 = {t0_pts}    Team 1 = {t1_pts}    (42 total)")
    lines.append(f"- Bidder team needs {bid_remaining} more to make the bid.")
    lines.append(f"- Count loose: 10-pt: {count_10_str}    5-pt: {count_5_str}")
    lines.append("")
    lines.append("[PROTOCOL]")
    lines.append(protocol_line)

    structured = {
        "trick_index": trick_n,
        "position_in_trick": pos_in_trick,
        "you_lead": len(cur) == 0,
        "seat": me,
        "team": my_team,
        "role": role,
        "partner_seat": partner,
        "left_opp_seat": left,
        "right_opp_seat": right,
        "decl_id": decl_id,
        "decl_name": DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}"),
        "bidder_seat": bidder,
        "bidder_team": bidder_team,
        "bid_value": bid_value,
        "bid_remaining": bid_remaining,
        "team_points": [t0_pts, t1_pts],
        "my_hand": my_hand,
        "my_trumps": my_trumps,
        "trumps_high_to_low": trumps,
        "trumps_played": played_trumps,
        "trumps_unseen": unseen_trumps,
        "current_trick_plays": [int(d) for d in cur],
        "play_history": [[int(s), int(d)] for s, d in gs.play_history],
        "count_dominoes_loose": {"5": count_5, "10": count_10},
    }
    return {"prose": "\n".join(lines), "structured": structured}
