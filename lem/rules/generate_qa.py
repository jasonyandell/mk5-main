#!/usr/bin/env python3
"""Generate synthetic Q&A corpus for Stage 0 rules adapter.

Each question/answer pair is engine-verified — ground truth, no hallucination risk.
Categories target the specific gaps found in first Gemma contact:
  1. Trump membership
  2. Led suit determination
  3. Legal moves
  4. Trick winners
  5. Count math
  6. Hand-state tracking (after playing N tricks)
  7. Void inference from play history

Usage:
    python -m lem.rules.generate_qa --n-per-category 500 --output lem/rules/qa_corpus.jsonl
    python -m lem.rules.generate_qa --n-per-category 50 --output scratch/qa_small.jsonl  # dev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from forge.oracle.declarations import (
    DECL_ID_TO_NAME,
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    N_DECLS,
    NOTRUMP,
    has_trump_power,
)
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    resolve_trick,
    trick_rank,
)
from forge.eq.game import GameState


def dom(dom_id: int) -> str:
    """Format domino as H-L."""
    return f"{DOMINO_HIGH[dom_id]}-{DOMINO_LOW[dom_id]}"


def pips_to_id(high: int, low: int) -> int:
    hi, lo = max(high, low), min(high, low)
    return hi * (hi + 1) // 2 + lo


def decl_name(decl_id: int) -> str:
    names = {
        0: "blanks", 1: "ones", 2: "twos", 3: "threes",
        4: "fours", 5: "fives", 6: "sixes",
        DOUBLES_TRUMP: "doubles", DOUBLES_SUIT: "doubles as own suit",
        NOTRUMP: "no trump",
    }
    return names[decl_id]


# ============================================================================
# Category 1: Trump membership
# ============================================================================

def gen_trump_membership(rng: random.Random) -> dict:
    """Is domino X a trump when Y is trump?"""
    dom_id = rng.randint(0, N_DOMINOES - 1)
    decl_id = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])

    is_trump = False
    if decl_id == NOTRUMP:
        is_trump = False
    elif has_trump_power(decl_id) and is_in_called_suit(dom_id, decl_id):
        is_trump = True

    q = f"Trump is {decl_name(decl_id)}. Is the {dom(dom_id)} a trump domino?"
    a = "Yes." if is_trump else "No."

    if is_trump and decl_id in range(7):
        a += f" It contains the trump pip ({decl_id})."
    elif is_trump and decl_id == DOUBLES_TRUMP:
        a += " It is a double, and doubles are trump."
    elif not is_trump and decl_id in range(7):
        hi, lo = DOMINO_HIGH[dom_id], DOMINO_LOW[dom_id]
        a += f" Its pips are {hi} and {lo}, neither is the trump pip ({decl_id})."
    elif not is_trump and decl_id == DOUBLES_TRUMP:
        a += " It is not a double."
    elif decl_id == NOTRUMP:
        a += " There is no trump suit."

    return {"category": "trump_membership", "question": q, "answer": a}


# ============================================================================
# Category 2: Led suit determination
# ============================================================================

def gen_led_suit(rng: random.Random) -> dict:
    """What suit is led when X leads under Y trump?"""
    dom_id = rng.randint(0, N_DOMINOES - 1)
    decl_id = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])

    led_suit = led_suit_for_lead_domino(dom_id, decl_id)

    suit_names = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
                  4: "fours", 5: "fives", 6: "sixes", 7: "trump"}

    q = (f"Trump is {decl_name(decl_id)}. "
         f"A player leads the {dom(dom_id)}. What suit is led?")

    if led_suit == 7:
        a = f"Trump is led. The {dom(dom_id)} is a trump domino, so leading it leads the trump suit."
    else:
        hi = DOMINO_HIGH[dom_id]
        a = f"The suit of {suit_names[led_suit]} is led. "
        if hi == led_suit:
            a += f"The {dom(dom_id)} is not a trump, so the led suit is its higher pip ({led_suit})."
        else:
            a += f"The higher pip of the {dom(dom_id)} is {hi}, which determines the led suit."

    return {"category": "led_suit", "question": q, "answer": a}


# ============================================================================
# Category 3: Legal moves
# ============================================================================

def gen_legal_moves(rng: random.Random) -> dict:
    """Given a hand and a lead, what are the legal plays?"""
    seed = rng.randint(0, 999999)
    hands = deal_from_seed(seed)
    decl_id = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])

    state = GameState.from_hands(hands, decl_id, leader=0)

    # Play 0-4 random tricks to get varied positions
    n_tricks = rng.randint(0, 4)
    for _ in range(n_tricks):
        for _ in range(4):
            if state.is_complete():
                break
            legal = state.legal_actions()
            state = state.apply_action(rng.choice(legal))

    if state.is_complete():
        return gen_legal_moves(rng)  # retry

    # Now pick a mid-trick position where someone is following
    if len(state.current_trick) == 0:
        # Need to lead first, then ask about a follower
        legal = state.legal_actions()
        lead = rng.choice(legal)
        state = state.apply_action(lead)

    if state.is_complete():
        return gen_legal_moves(rng)

    player = state.current_player()
    hand = state.hands[player]
    legal = state.legal_actions()
    lead_dom = state.current_trick[0][1]

    hand_str = ", ".join(dom(d) for d in sorted(hand, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))
    legal_str = ", ".join(dom(d) for d in sorted(legal, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))

    q = (f"Trump is {decl_name(decl_id)}. "
         f"The {dom(lead_dom)} was led. "
         f"Your hand is: {hand_str}. "
         f"What are your legal plays?")

    if set(legal) == set(hand):
        a = (f"Your legal plays are: {legal_str}. "
             f"You cannot follow the led suit, so you may play any domino.")
    else:
        a = (f"Your legal plays are: {legal_str}. "
             f"You must follow the led suit.")

    return {"category": "legal_moves", "question": q, "answer": a}


# ============================================================================
# Category 4: Trick winners
# ============================================================================

def gen_trick_winner(rng: random.Random) -> dict:
    """Who wins this trick?"""
    seed = rng.randint(0, 999999)
    hands = deal_from_seed(seed)
    decl_id = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])

    state = GameState.from_hands(hands, decl_id, leader=0)

    # Optionally play some tricks first
    n_tricks = rng.randint(0, 5)
    for _ in range(n_tricks):
        for _ in range(4):
            if state.is_complete():
                break
            legal = state.legal_actions()
            state = state.apply_action(rng.choice(legal))

    if state.is_complete():
        return gen_trick_winner(rng)

    # Play one complete trick
    trick_plays = []
    leader = state.leader
    for i in range(4):
        if state.is_complete():
            return gen_trick_winner(rng)
        player = state.current_player()
        legal = state.legal_actions()
        action = rng.choice(legal)
        trick_plays.append((player, action))
        state = state.apply_action(action)

    if len(trick_plays) != 4:
        return gen_trick_winner(rng)

    lead_dom = trick_plays[0][1]
    domino_ids = tuple(d for _, d in trick_plays)
    outcome = resolve_trick(lead_dom, domino_ids, decl_id)
    winner_player = trick_plays[outcome.winner_offset][0]
    winning_dom = trick_plays[outcome.winner_offset][1]

    plays_str = ". ".join(
        f"Player {p} plays the {dom(d)}" for p, d in trick_plays
    )

    q = (f"Trump is {decl_name(decl_id)}. "
         f"Player {trick_plays[0][0]} leads. "
         f"{plays_str}. "
         f"Who wins the trick?")

    a = f"Player {winner_player} wins with the {dom(winning_dom)}."

    # Explain why
    if has_trump_power(decl_id) and is_in_called_suit(winning_dom, decl_id):
        a += " It is the highest trump played."
    else:
        a += " It is the highest domino of the led suit."

    return {"category": "trick_winner", "question": q, "answer": a}


# ============================================================================
# Category 5: Count math
# ============================================================================

def gen_count_math(rng: random.Random) -> dict:
    """How many count points are in this trick / set of dominoes?"""
    variant = rng.choice(["trick", "identify", "total_remaining"])

    if variant == "trick":
        # Random 4 dominoes as a trick
        all_doms = list(range(N_DOMINOES))
        rng.shuffle(all_doms)
        trick_doms = all_doms[:4]
        trick_str = ", ".join(dom(d) for d in trick_doms)
        count_pts = sum(DOMINO_COUNT_POINTS[d] for d in trick_doms)
        total_pts = count_pts + 1  # +1 trick point

        q = f"A trick contains: {trick_str}. How many count points are in this trick (not including the trick point)?"
        a = f"{count_pts} count points."
        if count_pts > 0:
            counters = [dom(d) for d in trick_doms if DOMINO_COUNT_POINTS[d] > 0]
            details = [f"{dom(d)} ({DOMINO_COUNT_POINTS[d]})" for d in trick_doms if DOMINO_COUNT_POINTS[d] > 0]
            a += f" The count dominoes are: {', '.join(details)}."

    elif variant == "identify":
        dom_id = rng.randint(0, N_DOMINOES - 1)
        pts = DOMINO_COUNT_POINTS[dom_id]
        q = f"How many count points is the {dom(dom_id)} worth?"
        if pts > 0:
            a = f"The {dom(dom_id)} is worth {pts} count points."
        else:
            a = f"The {dom(dom_id)} is worth 0 count points. It is not a count domino."

    else:  # total_remaining
        # Given some played dominoes, how many count points remain?
        n_played = rng.randint(4, 20)
        all_doms = list(range(N_DOMINOES))
        rng.shuffle(all_doms)
        played = all_doms[:n_played]
        remaining = all_doms[n_played:]
        played_count = sum(DOMINO_COUNT_POINTS[d] for d in played)
        remaining_count = sum(DOMINO_COUNT_POINTS[d] for d in remaining)
        played_str = ", ".join(dom(d) for d in sorted(played, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))

        q = (f"The following dominoes have been played: {played_str}. "
             f"How many count points are still out (in unplayed dominoes)?")
        a = f"{remaining_count} count points remain."
        if remaining_count > 0:
            rem_counters = [d for d in remaining if DOMINO_COUNT_POINTS[d] > 0]
            details = [f"{dom(d)} ({DOMINO_COUNT_POINTS[d]})" for d in rem_counters]
            a += f" The remaining count dominoes are: {', '.join(details)}."
        else:
            a += " All count dominoes have been played."

    return {"category": "count_math", "question": q, "answer": a}


# ============================================================================
# Category 6: Hand-state tracking
# ============================================================================

def gen_hand_tracking(rng: random.Random) -> dict:
    """Given initial hand + play history, what's still in your hand?"""
    seed = rng.randint(0, 999999)
    hands = deal_from_seed(seed)
    decl_id = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])
    narrator = rng.randint(0, 3)

    state = GameState.from_hands(hands, decl_id, leader=0)
    initial_hand = list(hands[narrator])

    # Play 1-5 tricks randomly
    n_tricks = rng.randint(1, 5)
    history_lines = []
    for t in range(n_tricks):
        trick_plays = []
        for _ in range(4):
            if state.is_complete():
                break
            player = state.current_player()
            legal = state.legal_actions()
            action = rng.choice(legal)
            trick_plays.append((player, action))
            state = state.apply_action(action)

        if len(trick_plays) == 4:
            plays_str = ", ".join(
                f"{'You' if p == narrator else f'Player {p}'} played the {dom(d)}"
                for p, d in trick_plays
            )
            history_lines.append(f"Trick {t+1}: {plays_str}.")

    if not history_lines:
        return gen_hand_tracking(rng)

    remaining = list(state.hands[narrator])
    initial_str = ", ".join(dom(d) for d in sorted(initial_hand, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))
    remaining_str = ", ".join(dom(d) for d in sorted(remaining, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))

    history = " ".join(history_lines)

    q = (f"You are Player {narrator}. Your initial hand was: {initial_str}. "
         f"{history} "
         f"What dominoes are still in your hand?")

    a = f"Your remaining dominoes are: {remaining_str}."
    played_by_me = [d for d in initial_hand if d not in remaining]
    if played_by_me:
        played_str = ", ".join(dom(d) for d in played_by_me)
        a += f" You played: {played_str}."

    return {"category": "hand_tracking", "question": q, "answer": a}


# ============================================================================
# Category 7: Void inference
# ============================================================================

def gen_void_inference(rng: random.Random) -> dict:
    """From play history, can we tell if a player is void in a suit?"""
    seed = rng.randint(0, 999999)
    hands = deal_from_seed(seed)
    decl_id = rng.choice(list(range(7)))  # pip-suit trump for clearer void signals
    narrator = rng.randint(0, 3)

    state = GameState.from_hands(hands, decl_id, leader=0)

    # Play 2-5 tricks to build history
    n_tricks = rng.randint(2, 5)
    history_lines = []
    voids_revealed: dict[int, set[int]] = {p: set() for p in range(4)}

    for t in range(n_tricks):
        trick_plays = []
        lead_dom = None
        for play_idx in range(4):
            if state.is_complete():
                break
            player = state.current_player()
            legal = state.legal_actions()
            action = rng.choice(legal)

            if play_idx == 0:
                lead_dom = action
            else:
                # Check if player failed to follow suit → void
                led_suit = led_suit_for_lead_domino(lead_dom, decl_id)
                if not can_follow(action, led_suit, decl_id):
                    # Player didn't follow → void in led suit
                    voids_revealed[player].add(led_suit)

            trick_plays.append((player, action))
            state = state.apply_action(action)

        if len(trick_plays) == 4:
            def pname(p):
                return "You" if p == narrator else f"Player {p}"
            plays_str = ", ".join(f"{pname(p)} played the {dom(d)}" for p, d in trick_plays)
            history_lines.append(f"Trick {t+1}: {pname(trick_plays[0][0])} led the {dom(trick_plays[0][1])}. {plays_str}.")

    if not history_lines:
        return gen_void_inference(rng)

    # Pick a player and suit to ask about
    target_player = rng.choice([p for p in range(4) if p != narrator])
    suit_names = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
                  4: "fours", 5: "fives", 6: "sixes", 7: "trump"}
    ask_suit = rng.randint(0, 6)

    history = " ".join(history_lines)
    is_void = ask_suit in voids_revealed[target_player]

    q = (f"Trump is {decl_name(decl_id)}. You are Player {narrator}. "
         f"{history} "
         f"Based on the play history, is Player {target_player} known to be void in {suit_names[ask_suit]}?")

    if is_void:
        a = (f"Yes. Player {target_player} failed to follow {suit_names[ask_suit]} "
             f"when it was led, which means they have no {suit_names[ask_suit]} in their hand.")
    else:
        a = (f"No. There is no evidence from the play history that Player {target_player} "
             f"is void in {suit_names[ask_suit]}.")

    return {"category": "void_inference", "question": q, "answer": a}


# ============================================================================
# Main
# ============================================================================

GENERATORS = {
    "trump_membership": gen_trump_membership,
    "led_suit": gen_led_suit,
    "legal_moves": gen_legal_moves,
    "trick_winner": gen_trick_winner,
    "count_math": gen_count_math,
    "hand_tracking": gen_hand_tracking,
    "void_inference": gen_void_inference,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q&A corpus for Stage 0")
    parser.add_argument("--n-per-category", type=int, default=500,
                        help="Number of examples per category (default: 500)")
    parser.add_argument("--output", type=str, default="lem/rules/qa_corpus.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    category_counts: dict[str, int] = {}

    with open(output_path, "w") as f:
        for cat_name, gen_fn in GENERATORS.items():
            print(f"[{cat_name}] generating {args.n_per_category} examples...",
                  file=sys.stderr, flush=True)
            for i in range(args.n_per_category):
                try:
                    example = gen_fn(rng)
                    f.write(json.dumps(example) + "\n")
                    total += 1
                    category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                except Exception as e:
                    print(f"  [warn] {cat_name} example {i} failed: {e}",
                          file=sys.stderr)

    print(f"\nGenerated {total} examples to {output_path}", file=sys.stderr)
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
