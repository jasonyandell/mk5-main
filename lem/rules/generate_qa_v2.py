#!/usr/bin/env python3
"""Stage 0 v2: Kerry-structured Q&A generator for Texas 42.

Curriculum based on Kerry Newberry's Learner's Guide for SC 42 Group,
scaled to thousands of examples via the game engine.

Four sections in pedagogical order:
  A. Getting to Know the Dominoes — suits, ranks, counts
  B. Understanding the Game — true/false on rules and scoring
  C. Following Suit — the HARD part (deliberately tricky trump scenarios)
  D. Who Wins the Trick? How Many Points? — trick resolution

Section C is weighted heaviest because trump-membership confusion is the
#1 stumbling block for both humans and Gemma 4 E2B.

Usage:
    python -m lem.rules.generate_qa_v2 --output lem/rules/qa_kerry_v2.jsonl
    python -m lem.rules.generate_qa_v2 --output scratch/qa_test.jsonl --n-total 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from forge.oracle.declarations import (
    DOUBLES_TRUMP,
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


def dom(d: int) -> str:
    return f"{DOMINO_HIGH[d]}-{DOMINO_LOW[d]}"


def pips_to_id(hi: int, lo: int) -> int:
    h, l = max(hi, lo), min(hi, lo)
    return h * (h + 1) // 2 + l


SUIT_NAMES = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
              4: "fours", 5: "fives", 6: "sixes"}
DECL_NAMES = {**SUIT_NAMES, DOUBLES_TRUMP: "doubles", NOTRUMP: "no trump"}

# All pip-suit trump IDs for tricky C scenarios
PIP_TRUMPS = list(range(7))

# Count domino facts
COUNT_DOMS = [(pips_to_id(5, 5), 10), (pips_to_id(6, 4), 10),
              (pips_to_id(5, 0), 5), (pips_to_id(4, 1), 5),
              (pips_to_id(3, 2), 5)]
COUNT_IDS = {d for d, _ in COUNT_DOMS}


# ============================================================================
# Section A: Getting to Know the Dominoes
# ============================================================================

def gen_section_a(rng: random.Random) -> dict:
    """Suit membership, ranking, count identification."""
    variant = rng.choice([
        "suit_members", "suit_highest", "count_value", "count_per_suit",
        "suit_count_total", "is_double", "suit_rank",
    ])

    if variant == "suit_members":
        suit = rng.randint(0, 6)
        members = [i for i in range(N_DOMINOES)
                   if DOMINO_HIGH[i] == suit or DOMINO_LOW[i] == suit]
        members_str = ", ".join(dom(d) for d in sorted(members, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))
        q = f"List all dominoes in the {SUIT_NAMES[suit]} suit, from highest to lowest."
        a = f"The {SUIT_NAMES[suit]} suit contains: {members_str}. The {suit}-{suit} (double) is highest."
        return {"section": "A", "category": "suit_members", "question": q, "answer": a}

    elif variant == "suit_highest":
        suit = rng.randint(0, 6)
        double_id = pips_to_id(suit, suit)
        q = f"What is the highest domino in the {SUIT_NAMES[suit]} suit?"
        a = f"The {dom(double_id)} (the double) is the highest domino in the {SUIT_NAMES[suit]} suit."
        return {"section": "A", "category": "suit_highest", "question": q, "answer": a}

    elif variant == "count_value":
        d = rng.randint(0, N_DOMINOES - 1)
        pts = DOMINO_COUNT_POINTS[d]
        q = f"Is the {dom(d)} a count domino? If so, how many points is it worth?"
        if pts > 0:
            a = f"Yes. The {dom(d)} is worth {pts} count points."
        else:
            a = f"No. The {dom(d)} is worth 0 count points."
        return {"section": "A", "category": "count_value", "question": q, "answer": a}

    elif variant == "count_per_suit":
        suit = rng.randint(0, 6)
        suit_doms = [i for i in range(N_DOMINOES)
                     if DOMINO_HIGH[i] == suit or DOMINO_LOW[i] == suit]
        suit_counts = [(i, DOMINO_COUNT_POINTS[i]) for i in suit_doms if DOMINO_COUNT_POINTS[i] > 0]
        n = len(suit_counts)
        q = f"How many count dominoes are in the {SUIT_NAMES[suit]} suit?"
        if n == 0:
            a = f"The {SUIT_NAMES[suit]} suit has no count dominoes."
        else:
            details = ", ".join(f"{dom(d)} ({pts}pts)" for d, pts in suit_counts)
            a = f"The {SUIT_NAMES[suit]} suit has {n} count domino{'s' if n > 1 else ''}: {details}."
        return {"section": "A", "category": "count_per_suit", "question": q, "answer": a}

    elif variant == "suit_count_total":
        q = "How many total points are available in a hand of Texas 42?"
        a = "42 points total: 35 count points (from the five count dominoes) plus 7 trick points (1 per trick)."
        return {"section": "A", "category": "total_points", "question": q, "answer": a}

    elif variant == "is_double":
        d = rng.randint(0, N_DOMINOES - 1)
        is_dbl = DOMINO_IS_DOUBLE[d]
        q = f"Is the {dom(d)} a double?"
        a = f"{'Yes' if is_dbl else 'No'}. The {dom(d)} {'is' if is_dbl else 'is not'} a double."
        return {"section": "A", "category": "is_double", "question": q, "answer": a}

    else:  # suit_rank
        suit = rng.randint(0, 6)
        d1 = rng.choice([i for i in range(N_DOMINOES) if DOMINO_HIGH[i] == suit or DOMINO_LOW[i] == suit])
        d2 = rng.choice([i for i in range(N_DOMINOES) if (DOMINO_HIGH[i] == suit or DOMINO_LOW[i] == suit) and i != d1])
        # Compare within suit (no trump context — natural ranking)
        r1 = trick_rank(d1, suit, NOTRUMP)
        r2 = trick_rank(d2, suit, NOTRUMP)
        if r1 > r2:
            higher, lower = d1, d2
        elif r2 > r1:
            higher, lower = d2, d1
        else:
            higher, lower = d1, d2  # tie
        q = f"In the {SUIT_NAMES[suit]} suit (no trump), which ranks higher: the {dom(d1)} or the {dom(d2)}?"
        a = f"The {dom(higher)} ranks higher than the {dom(lower)} in the {SUIT_NAMES[suit]} suit."
        return {"section": "A", "category": "suit_rank", "question": q, "answer": a}


# ============================================================================
# Section B: Understanding the Game (True/False + Short Answer)
# ============================================================================

_SECTION_B_TF = [
    ("The 6-6 is always the most valuable domino in 42.", False,
     "False. A trump domino beats any non-trump, regardless of pip count. And 6-6 is worth 0 count points."),
    ("All dominoes with five dots on one side are worth 5 points.", False,
     "False. The 5-5 is worth 10 points (dots add to 10), and dominoes like 5-4 or 5-3 are worth 0 points (dots don't add to 5 or 10)."),
    ("To make a bid of 42 or higher, you must win all the tricks.", True,
     "True. 42 points = all 35 count + all 7 trick points. You must sweep."),
    ("You don't have to follow suit if you win the bid.", False,
     "False. Following suit is always mandatory regardless of who bid. Every player must follow suit if they can."),
    ("For each hand, there is only one round of bidding.", True,
     "True. Each player gets exactly one chance to bid or pass, going clockwise."),
    ("A trump domino is higher than every domino in every other suit.", True,
     "True. Any trump beats any non-trump, even the double of the led suit."),
    ("When fives are trump, the 5-0 is a five, not a blank.", True,
     "True. When a pip suit is trump, every domino containing that pip is in the trump suit and removed from its other suit."),
    ("When doubles are trump, the 5-5 is both a five and a trump.", False,
     "False. When doubles are trump, doubles are their own suit. The 5-5 is a trump, not a five."),
    ("Each trick is worth 1 point plus any count dominoes on it.", True,
     "True. Every trick is worth at least 1 point (the trick point) plus the count value of any count dominoes played on that trick."),
    ("The maximum possible points on a single trick is 21.", True,
     "True. If both 10-count dominoes (5-5 and 6-4) and one 5-count domino are on the same trick: 10+10+5+1 = 26. Actually, with all 4 plays: max is 10+10+5+1 = 26 if three count dominoes land on one trick. But the max with 4 dominoes is 10+10+5+5+1 = 31... wait. A trick has exactly 4 dominoes. The max count is 5-5(10) + 6-4(10) + 5-0(5) + 4-1(5) = 30 + 1 trick = 31. True only if the question says 21, which would be wrong. Let me fix this."),
]

def gen_section_b(rng: random.Random) -> dict:
    """True/false and short-answer about game rules."""
    variant = rng.choice(["tf", "scoring", "trump_basics"])

    if variant == "tf":
        # Pick from curated true/false (matching Kerry's style)
        tf_items = [
            ("The 6-6 is always the most valuable domino in 42.", False,
             "False. A trump domino beats any non-trump. And 6-6 is worth 0 count points."),
            ("All dominoes with five dots on one side are worth 5 points.", False,
             "False. The 5-5 is worth 10 (dots add to 10). Dominoes like 5-4 are worth 0 (dots don't add to 5 or 10)."),
            ("To make a bid of 42, you must win all the tricks.", True,
             "True. 42 = 35 count + 7 tricks. You must take every point."),
            ("You don't have to follow suit if you win the bid.", False,
             "False. Every player must follow suit if they can, always."),
            ("A trump domino is higher than every non-trump domino.", True,
             "True. Any trump beats any non-trump."),
            ("When fives are trump, the 5-0 is a trump, not a blank.", True,
             "True. It contains a five, so it's in the trump suit."),
            ("When doubles are trump, the 5-5 is both a five and a trump.", False,
             "False. When doubles are trump, 5-5 is trump only. It's not in the fives suit."),
            ("Each trick is worth at least 1 point.", True,
             "True. Every trick earns 1 trick point, plus any count."),
            ("If your team bids 30 and takes exactly 30 points, you make the bid.", True,
             "True. You need at least as many points as the bid."),
            ("The 3-2 is worth 5 count points because its dots add to 5.", True,
             "True. Dominoes whose dots total 5 or 10 are count dominoes."),
        ]
        stmt, truth, explanation = rng.choice(tf_items)
        q = f"True or false: {stmt}"
        return {"section": "B", "category": "true_false", "question": q, "answer": explanation}

    elif variant == "scoring":
        bid = rng.choice([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
        points_taken = rng.randint(0, 42)
        made = points_taken >= bid if bid <= 41 else points_taken == 42
        q = f"A team bids {bid} and takes {points_taken} points. Did they make the bid or get set?"
        if made:
            a = f"They made the bid. They needed {bid} and took {points_taken}."
        else:
            short = bid - points_taken
            a = f"They got set. They needed {bid} but only took {points_taken} (short by {short})."
        return {"section": "B", "category": "scoring", "question": q, "answer": a}

    else:  # trump_basics
        decl = rng.choice(PIP_TRUMPS + [DOUBLES_TRUMP, NOTRUMP])
        d = rng.randint(0, N_DOMINOES - 1)
        if decl == NOTRUMP:
            is_trump = False
            q = f"With no trump declared, is the {dom(d)} a trump?"
            a = "No. There is no trump suit when no trump is declared."
        elif decl == DOUBLES_TRUMP:
            is_trump = DOMINO_IS_DOUBLE[d]
            q = f"When doubles are trump, is the {dom(d)} a trump?"
            if is_trump:
                a = f"Yes. The {dom(d)} is a double, and all doubles are trump."
            else:
                a = f"No. The {dom(d)} is not a double."
        else:
            is_trump = has_trump_power(decl) and is_in_called_suit(d, decl)
            q = f"When {SUIT_NAMES[decl]} are trump, is the {dom(d)} a trump?"
            hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]
            if is_trump:
                a = f"Yes. The {dom(d)} contains a {decl}, so it is in the trump suit."
            else:
                a = f"No. The {dom(d)} has pips {hi} and {lo}. Neither is {decl}, so it is not trump."
        return {"section": "B", "category": "trump_basics", "question": q, "answer": a}


# ============================================================================
# Section C: Following Suit — THE HARD PART
# Deliberately generates tricky scenarios where trump membership is confusing.
# ============================================================================

def _build_tricky_hand(decl_id: int, led_suit: int, rng: random.Random,
                       hand_size: int = 4) -> list[int]:
    """Build a hand that's tricky for following suit.

    Includes dominoes that LOOK like they should follow but are actually trump,
    and vice versa. This mimics Kerry's exercise design.
    """
    all_doms = list(range(N_DOMINOES))
    rng.shuffle(all_doms)

    hand = []
    # Try to include: a trump that shares a pip with the led suit (confusing!)
    for d in all_doms:
        if len(hand) >= hand_size:
            break
        if d in hand:
            continue
        # Prioritize confusing dominoes
        hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]
        is_trump = has_trump_power(decl_id) and is_in_called_suit(d, decl_id)
        touches_led = (hi == led_suit or lo == led_suit) if led_suit < 7 else False

        # A trump that touches the led suit = maximally confusing
        if is_trump and touches_led and len(hand) < hand_size:
            hand.append(d)
        # A non-trump that follows the led suit
        elif not is_trump and touches_led and len(hand) < hand_size:
            hand.append(d)

    # Fill remaining slots with random dominoes not in hand
    for d in all_doms:
        if len(hand) >= hand_size:
            break
        if d not in hand:
            hand.append(d)

    return hand[:hand_size]


def gen_section_c(rng: random.Random) -> dict:
    """Following suit — Kerry's Section C, engine-scaled.

    Generates deliberately tricky hands where trump membership matters.
    """
    # Pick a pip-suit trump (most confusion happens here)
    decl_id = rng.choice(PIP_TRUMPS + [DOUBLES_TRUMP])

    # Pick a lead domino
    lead_id = rng.randint(0, N_DOMINOES - 1)
    led_suit = led_suit_for_lead_domino(lead_id, decl_id)

    # Build a tricky hand
    hand_size = rng.choice([3, 4, 5])
    hand = _build_tricky_hand(decl_id, led_suit, rng, hand_size)

    # Compute legal plays using the engine
    legal = []
    for d in hand:
        if can_follow(d, led_suit, decl_id):
            legal.append(d)
    if not legal:
        legal = list(hand)  # can't follow → play anything

    hand_str = ", ".join(dom(d) for d in hand)
    legal_str = ", ".join(dom(d) for d in legal)

    suit_name = "trump" if led_suit == 7 else SUIT_NAMES.get(led_suit, f"suit {led_suit}")
    q = (f"Trump is {DECL_NAMES[decl_id]}. "
         f"The {dom(lead_id)} is led (led suit: {suit_name}). "
         f"Your hand: {hand_str}. "
         f"Which dominoes can you legally play?")

    # Build explanation
    if set(legal) == set(hand):
        explanation = f"You have no {suit_name} in your hand, so you may play any domino."
        a = f"You can play any of: {legal_str}. {explanation}"
    else:
        # Explain each domino
        parts = []
        for d in hand:
            hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]
            is_trump = has_trump_power(decl_id) and is_in_called_suit(d, decl_id)
            follows = can_follow(d, led_suit, decl_id)

            if follows:
                parts.append(f"{dom(d)} can follow (it is in the led suit)")
            elif is_trump:
                parts.append(f"{dom(d)} cannot follow (it is a trump, not in the {suit_name} suit)")
            else:
                parts.append(f"{dom(d)} cannot follow (not in the {suit_name} suit)")

        a = f"Legal plays: {legal_str}. " + "; ".join(parts) + "."

    return {"section": "C", "category": "following_suit", "question": q, "answer": a}


# ============================================================================
# Section D: Who Wins the Trick? How Many Points?
# ============================================================================

def gen_section_d(rng: random.Random) -> dict:
    """Trick resolution — who wins and for how many points."""
    seed = rng.randint(0, 999999)
    hands = deal_from_seed(seed)
    decl_id = rng.choice(PIP_TRUMPS + [DOUBLES_TRUMP, NOTRUMP])

    state = GameState.from_hands(hands, decl_id, leader=rng.randint(0, 3))

    # Play some tricks to get variety
    n_pre = rng.randint(0, 4)
    for _ in range(n_pre):
        for _ in range(4):
            if state.is_complete():
                break
            legal = state.legal_actions()
            state = state.apply_action(rng.choice(legal))

    if state.is_complete():
        return gen_section_d(rng)

    # Play one complete trick
    trick_plays = []
    leader = state.leader
    for _ in range(4):
        if state.is_complete():
            return gen_section_d(rng)
        player = state.current_player()
        legal = state.legal_actions()
        action = rng.choice(legal)
        trick_plays.append((player, action))
        state = state.apply_action(action)

    if len(trick_plays) != 4:
        return gen_section_d(rng)

    lead_dom = trick_plays[0][1]
    domino_ids = tuple(d for _, d in trick_plays)
    outcome = resolve_trick(lead_dom, domino_ids, decl_id)
    winner = trick_plays[outcome.winner_offset][0]
    winner_dom = trick_plays[outcome.winner_offset][1]
    points = outcome.points

    names = ["Willie", "Nanci", "Waylon", "Guy"]
    plays_str = ". ".join(f"{names[p]} plays the {dom(d)}" for p, d in trick_plays)

    q = (f"Trump is {DECL_NAMES[decl_id]}. "
         f"{names[trick_plays[0][0]]} leads. "
         f"{plays_str}. "
         f"Willie and Waylon are partners. Nanci and Guy are partners. "
         f"Who wins the trick, and how many points does the winning team earn?")

    # Explain why
    is_trump_win = has_trump_power(decl_id) and is_in_called_suit(winner_dom, decl_id)
    if is_trump_win:
        reason = f"because the {dom(winner_dom)} is the highest trump played"
    else:
        reason = f"because the {dom(winner_dom)} is the highest domino of the led suit"

    # Count breakdown
    count_parts = []
    for _, d in trick_plays:
        cp = DOMINO_COUNT_POINTS[d]
        if cp > 0:
            count_parts.append(f"{dom(d)}={cp}")
    if count_parts:
        count_detail = f" Count on this trick: {', '.join(count_parts)}."
    else:
        count_detail = " No count dominoes on this trick."

    a = (f"{names[winner]} wins the trick {reason}. "
         f"The winning team earns {points} point{'s' if points != 1 else ''} "
         f"(1 trick point{' + ' + ' + '.join(f'{DOMINO_COUNT_POINTS[d]} count' for _, d in trick_plays if DOMINO_COUNT_POINTS[d] > 0) if count_parts else ''})."
         f"{count_detail}")

    return {"section": "D", "category": "trick_winner", "question": q, "answer": a}


# ============================================================================
# Main
# ============================================================================

GENERATORS = {
    "A": gen_section_a,
    "B": gen_section_b,
    "C": gen_section_c,
    "D": gen_section_d,
}

# Kerry-weighted distribution: Section C gets 40% because it's the hardest
SECTION_WEIGHTS = {"A": 0.15, "B": 0.20, "C": 0.40, "D": 0.25}


def main():
    parser = argparse.ArgumentParser(description="Generate Kerry-structured Q&A")
    parser.add_argument("--output", type=str, default="lem/rules/qa_kerry_v2.jsonl")
    parser.add_argument("--n-total", type=int, default=15000,
                        help="Total examples across all sections")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    section_counts: dict[str, int] = {}

    with open(output_path, "w") as f:
        for section, weight in SECTION_WEIGHTS.items():
            n = int(args.n_total * weight)
            gen_fn = GENERATORS[section]
            print(f"[Section {section}] generating {n} examples "
                  f"({weight*100:.0f}% of total)...", file=sys.stderr, flush=True)
            for i in range(n):
                try:
                    example = gen_fn(rng)
                    f.write(json.dumps(example) + "\n")
                    total += 1
                    section_counts[section] = section_counts.get(section, 0) + 1
                except Exception as e:
                    print(f"  [warn] Section {section} example {i} failed: {e}",
                          file=sys.stderr)

    print(f"\nGenerated {total} examples to {output_path}", file=sys.stderr)
    for section, count in sorted(section_counts.items()):
        print(f"  Section {section}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
