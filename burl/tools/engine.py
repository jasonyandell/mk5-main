"""Engine-backed epistemic tools for Burl.

Tools answer "what IS the state?", never "what SHOULD you do?". Each function is
a thin wrapper around canonical forge helpers; no rule is reimplemented here.

Accepted game_state: duck-typed. Any object with the fields used below works.
Tested against both ``forge.eq.game.GameState`` and ``forge.zeb.types.ZebGameState``.
Required fields:
    decl_id                 int, declaration id (0..9)
    hands                   tuple[tuple[int, ...], ...], 4 hands of domino ids
    played                  frozenset[int], globally played domino ids
    play_history            tuple of (abs_player, domino_id) or
                            (abs_player, domino_id, lead_domino_id)
    current_trick           tuple[int | (int, int), ...], current trick plays
    trick_leader or leader  int, abs seat who led the current trick

Relative-seat convention (matches ``forge.zeb.observation.FEAT_PLAYER``):
    0 = me (the player to act)
    1 = left opponent
    2 = partner
    3 = right opponent

Suit representation (matches ``forge.oracle.tables``):
    0..6 = pip suits (blanks..sixes)
    7    = the called suit (trump when decl has trump power, else doubles-suit)
"""

from __future__ import annotations

from typing import Any

from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
)


def _abs_current_player(state: Any) -> int:
    leader = getattr(state, "trick_leader", None)
    if leader is None:
        leader = state.leader
    return (leader + len(state.current_trick)) % 4


def _current_trick_domino_ids(state: Any) -> tuple[int, ...]:
    """Return played-domino ids for the current trick, regardless of tuple shape."""
    trick = state.current_trick
    if not trick:
        return ()
    first = trick[0]
    if isinstance(first, int):
        return tuple(trick)  # ZebGameState: tuple[int, ...]
    return tuple(d for _p, d in trick)  # eq.GameState: tuple[(player, domino_id)]


def _lead_domino_for_play(trick_plays: list[tuple[int, int]]) -> int:
    """Return lead domino id for the trick containing trick_plays[-1].

    play_history is absolute-player tuples in the order played, in 4-play trick
    groups. The lead is the first play whose position-in-trick is 0.
    """
    return trick_plays[0][1]


def _plays_with_leads(state: Any) -> list[tuple[int, int, int]]:
    """Normalize play_history to (player, domino_id, lead_domino_id) tuples.

    eq.GameState already stores the lead. ZebGameState does not, so we recover
    it by walking the history in 4-play trick chunks and taking each chunk's
    first domino as its lead.
    """
    history = state.play_history
    if not history:
        return []
    if len(history[0]) == 3:
        return list(history)

    # ZebGameState shape: (abs_player, domino_id)
    out: list[tuple[int, int]] = []
    trick: list[tuple[int, int]] = []
    for entry in history:
        trick.append(entry)
        if len(trick) == 4:
            lead = trick[0][1]
            for p, d in trick:
                out.append((p, d, lead))
            trick = []
    if trick:
        lead = trick[0][1]
        for p, d in trick:
            out.append((p, d, lead))
    return out


def _check_domino_id(domino_id: int) -> None:
    if not isinstance(domino_id, (int,)) or isinstance(domino_id, bool):
        raise ValueError(f"domino_id must be int, got {type(domino_id).__name__}")
    if not 0 <= domino_id < N_DOMINOES:
        raise ValueError(f"domino_id out of range [0, {N_DOMINOES}): {domino_id}")


def _check_decl(decl_id: int) -> None:
    if not 0 <= decl_id < N_DECLS:
        raise ValueError(f"state.decl_id out of range [0, {N_DECLS}): {decl_id}")


def is_legal(game_state: Any, domino_id: int) -> tuple[bool, str]:
    """Return (legal, reason). ``reason`` is '' when legal.

    Legality is judged for the player currently on turn ("me"). If the domino
    is not in my hand, that's the reason. Otherwise follow-suit rules apply.
    """
    _check_domino_id(domino_id)
    _check_decl(game_state.decl_id)

    me = _abs_current_player(game_state)
    hand = game_state.hands[me]
    remaining = tuple(d for d in hand if d not in game_state.played)

    if domino_id not in remaining:
        if domino_id in hand:
            return False, f"domino {domino_id} already played this hand"
        return False, f"domino {domino_id} not in current player's hand"

    current_trick_ids = _current_trick_domino_ids(game_state)
    if not current_trick_ids:
        return True, ""

    lead = current_trick_ids[0]
    led_suit = led_suit_for_lead_domino(lead, game_state.decl_id)

    if can_follow(domino_id, led_suit, game_state.decl_id):
        return True, ""

    # Must follow if any domino in hand can follow.
    if any(can_follow(d, led_suit, game_state.decl_id) for d in remaining):
        human_suit = "called suit" if led_suit == 7 else f"suit {led_suit}"
        return False, f"must follow {human_suit} (led by {lead})"

    return True, ""


def is_trump(game_state: Any, domino_id: int) -> bool:
    """True if ``domino_id`` belongs to the called suit under the declaration.

    "Trump" here means the called suit. Under no-trump (decl 9) nothing is
    trump; under doubles-suit (decl 8) doubles are their own suit but not
    ranking trump — ``is_trump`` reports False for doubles-suit because the
    suit has no trump power. Use ``trump_declared`` to disambiguate.
    """
    _check_domino_id(domino_id)
    _check_decl(game_state.decl_id)

    from forge.oracle.declarations import DOUBLES_SUIT, NOTRUMP

    if game_state.decl_id in (NOTRUMP, DOUBLES_SUIT):
        return False
    return is_in_called_suit(domino_id, game_state.decl_id)


def unseen(game_state: Any) -> set[int]:
    """Return domino ids that are neither in my hand nor in the played set.

    This is the pool Zeb's belief distribution is defined over — dominoes whose
    seat placement among the three opponents is still uncertain to "me".
    """
    _check_decl(game_state.decl_id)
    me = _abs_current_player(game_state)
    my_hand = set(game_state.hands[me])
    seen = my_hand | set(game_state.played)
    return {d for d in range(N_DOMINOES) if d not in seen}


def void_audit(game_state: Any, player_seat: int, suit: int) -> bool:
    """True if ``player_seat`` has been proven void in ``suit`` by play history.

    ``player_seat`` is RELATIVE (0=me, 1=left, 2=partner, 3=right).
    ``suit`` is 0..6 (pip suit) or 7 (called suit).
    """
    if not isinstance(player_seat, int) or isinstance(player_seat, bool):
        raise ValueError(f"player_seat must be int, got {type(player_seat).__name__}")
    if not 0 <= player_seat < 4:
        raise ValueError(f"player_seat out of range [0, 4): {player_seat}")
    if not isinstance(suit, int) or isinstance(suit, bool):
        raise ValueError(f"suit must be int, got {type(suit).__name__}")
    if not 0 <= suit <= 7:
        raise ValueError(f"suit out of range [0, 7]: {suit}")
    _check_decl(game_state.decl_id)

    me = _abs_current_player(game_state)
    abs_player = (me + player_seat) % 4

    for p, d, lead in _plays_with_leads(game_state):
        if p != abs_player:
            continue
        led_suit = led_suit_for_lead_domino(lead, game_state.decl_id)
        if led_suit != suit:
            continue
        if not can_follow(d, led_suit, game_state.decl_id):
            return True
    return False


def trump_declared(game_state: Any) -> str:
    """Return the canonical name of the current declaration.

    Uses ``forge.oracle.declarations.DECL_ID_TO_NAME``:
        'blanks' | 'ones' | 'twos' | 'threes' | 'fours' | 'fives' | 'sixes'
        | 'doubles-trump' | 'doubles-suit' | 'notrump'
    """
    _check_decl(game_state.decl_id)
    return DECL_ID_TO_NAME[game_state.decl_id]


_SUIT_NAMES = (
    "blanks", "ones", "twos", "threes", "fours", "fives", "sixes", "called",
)


def _domino_label(d: int) -> str:
    return f"{DOMINO_HIGH[d]}-{DOMINO_LOW[d]}"


def _count_value(d: int) -> int:
    """Point value of domino `d` in 42 counter scoring (0, 5, or 10).

    The only counters in a double-6 set are 5-0 / 4-1 / 3-2 (total 5) and
    5-5 / 6-4 (total 10). No other pair totals 5 or 10.
    """
    total = DOMINO_HIGH[d] + DOMINO_LOW[d]
    if total == 5:
        return 5
    if total == 10:
        return 10
    return 0


def game_summary(game_state: Any) -> dict:
    """One-shot consolidated view of the decision context.

    Exists because 20 trace-inspections showed Gemma repeatedly asking for
    info that's in the prompt but hard to extract from prose ("What's the
    trump?" "Which plays are in the current trick?" "Who's my partner?"). We
    give it the answers pre-parsed, in one call, so reasoning can start from
    facts instead of from re-parsing.

    All returned seats are ABSOLUTE (0..3). All labels are pip-form "H-L".
    Opponents' voids are keyed by relative seat label for Gemma's convenience.

    The response is deliberately JSON-serializable with primitives only — it
    needs to round-trip through the harness tool-observation formatter.
    """
    _check_decl(game_state.decl_id)
    decl_id = game_state.decl_id
    me = _abs_current_player(game_state)
    partner = (me + 2) % 4
    left_opp = (me + 1) % 4
    right_opp = (me + 3) % 4

    ctrick_ids = _current_trick_domino_ids(game_state)
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = game_state.leader

    current_trick_plays = [
        {
            "seat_absolute": (leader + i) % 4,
            "seat_role": _seat_role(me, partner, (leader + i) % 4),
            "domino_id": int(d),
            "pip_label": _domino_label(d),
            "is_trump": bool(is_trump(game_state, d)),
        }
        for i, d in enumerate(ctrick_ids)
    ]

    if ctrick_ids:
        lead_suit_id = led_suit_for_lead_domino(ctrick_ids[0], decl_id)
        lead_suit_name = _SUIT_NAMES[lead_suit_id]
        i_am_leading = False
    else:
        lead_suit_id = None
        lead_suit_name = None
        i_am_leading = True

    my_hand = game_state.hands[me]
    remaining = [d for d in my_hand if d not in game_state.played]
    legal_plays = []
    for d in remaining:
        ok, _reason = is_legal(game_state, d)
        if not ok:
            continue
        legal_plays.append({
            "domino_id": int(d),
            "pip_label": _domino_label(d),
            "is_trump": bool(is_trump(game_state, d)),
            "follows_lead_suit": (
                bool(can_follow(d, lead_suit_id, decl_id))
                if lead_suit_id is not None else None
            ),
            "count_value": _count_value(d),
        })

    points_in_current_trick = sum(_count_value(d) for d in ctrick_ids)

    opponent_voids = {
        "left_opponent": [
            _SUIT_NAMES[s] for s in range(7) if void_audit(game_state, 1, s)
        ],
        "partner": [
            _SUIT_NAMES[s] for s in range(7) if void_audit(game_state, 2, s)
        ],
        "right_opponent": [
            _SUIT_NAMES[s] for s in range(7) if void_audit(game_state, 3, s)
        ],
    }

    unseen_ids = unseen(game_state)
    unseen_counters = [
        {"domino_id": d, "pip_label": _domino_label(d), "count_value": _count_value(d)}
        for d in sorted(unseen_ids)
        if _count_value(d) > 0
    ]
    my_counters = [
        {"domino_id": d, "pip_label": _domino_label(d), "count_value": _count_value(d)}
        for d in sorted(remaining)
        if _count_value(d) > 0
    ]

    tricks_played = len(game_state.play_history) // 4
    position_in_trick = len(ctrick_ids) + 1

    return {
        "trump_declaration": trump_declared(game_state),
        "seats": {
            "me_absolute": int(me),
            "partner_absolute": int(partner),
            "left_opponent_absolute": int(left_opp),
            "right_opponent_absolute": int(right_opp),
        },
        "tricks_completed": int(tricks_played),
        "position_in_current_trick": int(position_in_trick),
        "i_am_leading": bool(i_am_leading),
        "current_trick_plays": current_trick_plays,
        "lead_suit": lead_suit_name,
        "points_in_current_trick": int(points_in_current_trick),
        "my_legal_plays": legal_plays,
        "my_counter_dominoes_in_hand": my_counters,
        "unseen_counter_dominoes": unseen_counters,
        "opponent_voids_inferred": opponent_voids,
    }


def _seat_role(me: int, partner: int, seat: int) -> str:
    """Return 'me', 'partner', 'left_opponent', or 'right_opponent' for an absolute seat."""
    if seat == me:
        return "me"
    if seat == partner:
        return "partner"
    # left = me+1 mod 4, right = me+3 mod 4
    if seat == (me + 1) % 4:
        return "left_opponent"
    return "right_opponent"


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from forge.zeb.game import apply_action, new_game
    from forge.zeb.game import legal_actions as zeb_legal_slots

    def _zeb_legal_domino_ids(state):
        me = _abs_current_player(state)
        hand = state.hands[me]
        return tuple(hand[s] for s in zeb_legal_slots(state))

    # 1. Deterministic real-game state.
    state = new_game(seed=2026, skip_bidding=True)
    me = _abs_current_player(state)

    print(f"seed=2026 decl_id={state.decl_id} bidder={state.bidder} me(abs)={me}")
    print(f"declaration: {trump_declared(state)}")

    # 2. Hand + legality audit (leading — all legal).
    hand = state.hands[me]
    remaining = [d for d in hand if d not in state.played]
    engine_legal = set(_zeb_legal_domino_ids(state))

    print(f"my hand (abs seat {me}): {remaining}")
    tool_legal = set()
    for d in remaining:
        ok, reason = is_legal(state, d)
        if ok:
            tool_legal.add(d)
        else:
            print(f"  tool rejected {d}: {reason}")
    assert tool_legal == engine_legal, (tool_legal, engine_legal)
    print(f"  OK: is_legal agrees with engine on {len(remaining)} hand dominoes")

    # 3. Trumps in hand.
    trumps_in_hand = [d for d in remaining if is_trump(state, d)]
    print(f"trumps in my hand: {trumps_in_hand}")

    # 4. Unseen pool.
    unseen_set = unseen(state)
    print(f"unseen pool size: {len(unseen_set)} (expected {28 - len(remaining)})")
    assert len(unseen_set) == 28 - len(remaining)

    # 5. Mid-game: drive forward until a player sluffs, then assert void_audit.
    from forge.oracle.tables import can_follow as _cf
    import random as _r

    rng = _r.Random(0)
    sluff_found = False
    cur = state
    for _ in range(28):
        if not cur.current_trick:
            legal_slots = zeb_legal_slots(cur)
            if not legal_slots:
                break
            cur = apply_action(cur, rng.choice(legal_slots))
            continue

        actor = _abs_current_player(cur)
        lead = cur.current_trick[0]
        led_suit = led_suit_for_lead_domino(lead, cur.decl_id)
        legal_slots = zeb_legal_slots(cur)
        legal_dom_ids = [cur.hands[actor][s] for s in legal_slots]
        if not any(_cf(d, led_suit, cur.decl_id) for d in legal_dom_ids):
            # actor is about to sluff
            cur = apply_action(cur, legal_slots[0])
            rel = (actor - _abs_current_player(cur) + 4) % 4  # relative to next-me
            # Easier: compute rel to the me that will *observe* the sluff next turn.
            observer = _abs_current_player(cur)
            rel = (actor - observer + 4) % 4
            # For void_audit we pass game_state whose "me" is observer.
            # With our cur, me == observer already (we just advanced).
            voided = void_audit(cur, rel, led_suit)
            print(
                f"sluff: abs_player={actor} led_suit={led_suit} "
                f"observer(me)={observer} rel_seat={rel} void_audit={voided}"
            )
            assert voided, "void_audit failed to detect a known sluff"
            sluff_found = True
            break
        cur = apply_action(cur, rng.choice(legal_slots))

    if not sluff_found:
        print("no sluff occurred in rollout (not a failure; rare for random play)")

    print("self-test: OK")
