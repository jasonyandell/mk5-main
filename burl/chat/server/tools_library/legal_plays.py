DESCRIPTION = 'Names the led suit (with the rule that decided it), then lists which dominoes in your hand are legal vs illegal under follow-suit. Pure rule-based; says WHAT IS legal, not what to play. Call this before commit_play whenever you are not leading.'

from burl.harness.agent_runner import _DOMINO_LABELS
from forge.oracle.declarations import (
    DECL_ID_TO_NAME,
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    NOTRUMP,
    PIP_TRUMP_IDS,
)
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    can_follow,
    led_suit_for_lead_domino,
)


def _label(d):
    return f"{int(d)}({_DOMINO_LABELS[int(d)]})"


def _led_suit_explanation(lead_dom, decl_id, led_suit):
    decl_name = DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}")
    hi, lo = DOMINO_HIGH[lead_dom], DOMINO_LOW[lead_dom]
    if led_suit == 7:
        if decl_id in PIP_TRUMP_IDS:
            return (
                f"led suit = TRUMP ({decl_name}). The lead {_label(lead_dom)} "
                f"contains a {decl_id}, so it is trump under {decl_name}-trump."
            )
        if decl_id == DOUBLES_TRUMP:
            return (
                f"led suit = TRUMP (doubles-as-trump). The lead {_label(lead_dom)} "
                f"is a double, and only doubles are trump."
            )
        if decl_id == DOUBLES_SUIT:
            return (
                f"led suit = doubles-suit. Lead {_label(lead_dom)} is a double; "
                f"doubles form their own suit (no trump power)."
            )
        return f"led suit = called suit (decl_id {decl_id})."
    if decl_id == NOTRUMP:
        return (
            f"led suit = {led_suit}s (notrump). With no trump, the led suit is "
            f"the higher pip of the lead {_label(lead_dom)} ({hi})."
        )
    return (
        f"led suit = {led_suit}s. The lead {_label(lead_dom)} is NOT trump under "
        f"{decl_name}-trump; the led suit is the higher pip ({hi}), not the lower ({lo})."
    )


def tool(ctx, **kwargs):
    gs = ctx.game_state
    me = int(ctx.me_abs)
    decl_id = int(gs.decl_id)
    decl_name = DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}")
    played = gs.played
    my_hand = sorted(d for d in gs.hands[me] if d not in played)
    cur = tuple(int(x) for x in gs.current_trick)

    lines = []
    if not cur:
        lines.append(
            f"YOU LEAD this trick. There is no led suit yet — every domino in your "
            f"hand is legal."
        )
        legal = list(my_hand)
        illegal = []
        led_suit = None
        lead_dom = None
    else:
        lead_dom = cur[0]
        led_suit = led_suit_for_lead_domino(lead_dom, decl_id)
        lines.append(_led_suit_explanation(lead_dom, decl_id, led_suit))
        followers = [d for d in my_hand if can_follow(d, led_suit, decl_id)]
        if followers:
            legal = list(followers)
            illegal = [d for d in my_hand if d not in followers]
            lines.append(
                f"You hold {len(followers)} domino(s) of the led suit; you MUST "
                f"follow suit. Off-suit and trump plays are ILLEGAL on this trick."
            )
        else:
            legal = list(my_hand)
            illegal = []
            lines.append(
                f"You are VOID in the led suit. Any domino in your hand is legal, "
                f"including trump."
            )

    lines.append("")
    lines.append(
        "LEGAL plays: "
        + (", ".join(_label(d) for d in legal) if legal else "(none — hand is empty)")
    )
    if illegal:
        lines.append(
            "ILLEGAL on this trick: " + ", ".join(_label(d) for d in illegal)
        )
    lines.append("")
    lines.append(
        f"Trump declaration is {decl_name} (decl_id={decl_id}). This tool tells you "
        f"WHAT IS legal. It does not tell you what to pick."
    )

    structured = {
        "decl_id": decl_id,
        "decl_name": decl_name,
        "lead_domino_id": int(lead_dom) if lead_dom is not None else None,
        "led_suit": int(led_suit) if led_suit is not None else None,
        "led_suit_is_trump": led_suit == 7 if led_suit is not None else None,
        "you_lead": len(cur) == 0,
        "void_in_led_suit": (
            False if not cur else not any(
                can_follow(d, led_suit, decl_id) for d in my_hand
            )
        ),
        "must_follow": (
            False if not cur else any(
                can_follow(d, led_suit, decl_id) for d in my_hand
            )
        ),
        "legal_plays": [int(d) for d in legal],
        "illegal_plays": [int(d) for d in illegal],
        "my_hand": [int(d) for d in my_hand],
    }
    return {"prose": "\n".join(lines), "structured": structured}
