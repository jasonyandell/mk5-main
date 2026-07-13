#!/usr/bin/env python3
"""Stage 0 v3: Targeted trump membership drilling.

The #1 remaining weakness after Kerry training: the model confuses which
dominoes are trumps under different declarations. This generates focused
Q&A on the hardest trump cases.

Usage:
    python -m lem.rules.generate_trump_drill --output lem/rules/qa_trump_drill.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP, has_trump_power
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
)

SUIT_NAMES = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
              4: "fours", 5: "fives", 6: "sixes"}
DECL_NAMES = {**SUIT_NAMES, DOUBLES_TRUMP: "doubles", NOTRUMP: "no trump"}


def dom(d: int) -> str:
    return f"{DOMINO_HIGH[d]}-{DOMINO_LOW[d]}"


def pips_to_id(hi: int, lo: int) -> int:
    h, l = max(hi, lo), min(hi, lo)
    return h * (h + 1) // 2 + l


# ============================================================================
# Drill 1: "Is X a trump when Y is trump?" — the core question
# ============================================================================

def gen_is_trump(rng: random.Random) -> dict:
    decl = rng.choice(list(range(7)) + [DOUBLES_TRUMP, NOTRUMP])
    d = rng.randint(0, N_DOMINOES - 1)
    hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]

    if decl == NOTRUMP:
        q = f"When no trump is declared, is the {dom(d)} a trump?"
        a = "No. There is no trump suit when no trump is declared."
    elif decl == DOUBLES_TRUMP:
        is_t = DOMINO_IS_DOUBLE[d]
        q = f"When doubles are trump, is the {dom(d)} a trump?"
        if is_t:
            a = f"Yes. The {dom(d)} is a double, and all seven doubles are trump when doubles are declared."
        else:
            a = f"No. The {dom(d)} is not a double. Only the seven doubles (0-0 through 6-6) are trump."
    else:
        is_t = has_trump_power(decl) and is_in_called_suit(d, decl)
        q = f"When {SUIT_NAMES[decl]} are trump, is the {dom(d)} a trump?"
        if is_t:
            a = (f"Yes. The {dom(d)} contains a {decl}, so it is in the {SUIT_NAMES[decl]} "
                 f"trump suit. Every domino with a {decl} on either side is trump.")
        else:
            a = (f"No. The {dom(d)} has pips {hi} and {lo}. Neither pip is {decl}, "
                 f"so the {dom(d)} is not a trump when {SUIT_NAMES[decl]} are trump.")

    return {"category": "is_trump", "question": q, "answer": a}


# ============================================================================
# Drill 2: "List ALL trumps when X is trump"
# ============================================================================

def gen_list_trumps(rng: random.Random) -> dict:
    decl = rng.choice(list(range(7)) + [DOUBLES_TRUMP])

    trumps = [i for i in range(N_DOMINOES)
              if has_trump_power(decl) and is_in_called_suit(i, decl)]
    trumps_str = ", ".join(dom(d) for d in sorted(trumps, key=lambda i: (-DOMINO_HIGH[i], -DOMINO_LOW[i])))

    q = f"List all trump dominoes when {DECL_NAMES[decl]} are trump, from highest to lowest."
    if decl in range(7):
        a = (f"The {SUIT_NAMES[decl]} trump suit contains {len(trumps)} dominoes: {trumps_str}. "
             f"Every domino with a {decl} on either side is trump. The {dom(pips_to_id(decl, decl))} "
             f"(double) is the highest.")
    else:
        a = (f"The doubles trump suit contains 7 dominoes: {trumps_str}. "
             f"Ranked from 6-6 (highest) to 0-0 (lowest).")

    return {"category": "list_trumps", "question": q, "answer": a}


# ============================================================================
# Drill 3: "Which of these are trumps?" — batch classification
# ============================================================================

def gen_which_are_trumps(rng: random.Random) -> dict:
    decl = rng.choice(list(range(7)) + [DOUBLES_TRUMP])
    n = rng.randint(3, 6)
    doms = rng.sample(range(N_DOMINOES), n)

    trumps = [d for d in doms if has_trump_power(decl) and is_in_called_suit(d, decl)]
    non_trumps = [d for d in doms if d not in trumps]

    doms_str = ", ".join(dom(d) for d in doms)
    q = f"When {DECL_NAMES[decl]} are trump, which of these dominoes are trumps: {doms_str}?"

    if trumps:
        t_str = ", ".join(dom(d) for d in trumps)
        nt_str = ", ".join(dom(d) for d in non_trumps) if non_trumps else "none"
        parts = []
        for d in doms:
            hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]
            if d in trumps:
                if decl in range(7):
                    parts.append(f"{dom(d)} is trump (contains {decl})")
                else:
                    parts.append(f"{dom(d)} is trump (it's a double)")
            else:
                if decl in range(7):
                    parts.append(f"{dom(d)} is NOT trump (pips {hi},{lo} — no {decl})")
                else:
                    parts.append(f"{dom(d)} is NOT trump (not a double)")
        a = f"Trumps: {t_str}. Not trumps: {nt_str}. " + "; ".join(parts) + "."
    else:
        a = f"None of these are trumps. " + "; ".join(
            f"{dom(d)} has pips {DOMINO_HIGH[d]},{DOMINO_LOW[d]} — no {decl}"
            for d in doms
        ) + "."

    return {"category": "which_trumps", "question": q, "answer": a}


# ============================================================================
# Drill 4: "Was that play a trump or a follow?" — in-context classification
# ============================================================================

def gen_trump_or_follow(rng: random.Random) -> dict:
    decl = rng.choice(list(range(7)))  # pip trumps for maximum confusion
    lead = rng.randint(0, N_DOMINOES - 1)
    led_suit = led_suit_for_lead_domino(lead, decl)
    play = rng.randint(0, N_DOMINOES - 1)

    is_trump = has_trump_power(decl) and is_in_called_suit(play, decl)
    follows = can_follow(play, led_suit, decl)

    suit_name = "trump" if led_suit == 7 else SUIT_NAMES.get(led_suit, str(led_suit))

    q = (f"Trump is {SUIT_NAMES[decl]}. The {dom(lead)} was led (led suit: {suit_name}). "
         f"A player plays the {dom(play)}. Did they follow suit, play a trump, or sluff?")

    hi, lo = DOMINO_HIGH[play], DOMINO_LOW[play]
    if is_trump and led_suit == 7:
        a = (f"They followed suit. The {dom(play)} is a trump (contains {decl}), "
             f"and trump was led, so playing trump is following suit.")
    elif is_trump:
        a = (f"They played a trump. The {dom(play)} contains a {decl}, making it a trump. "
             f"It is NOT in the {suit_name} suit even though it might look like it — "
             f"trump membership takes priority.")
    elif follows:
        a = (f"They followed suit. The {dom(play)} is in the {suit_name} suit "
             f"and is not a trump (no {decl} pip).")
    else:
        a = (f"They sluffed. The {dom(play)} is neither in the {suit_name} suit "
             f"nor a trump. Pips are {hi} and {lo}.")

    return {"category": "trump_or_follow", "question": q, "answer": a}


# ============================================================================
# Drill 5: Count dominoes that are also trumps — the sneaky case
# ============================================================================

def gen_count_trump_interaction(rng: random.Random) -> dict:
    """Is a count domino also a trump? Changes which suit it belongs to."""
    count_doms = [pips_to_id(5, 5), pips_to_id(6, 4), pips_to_id(5, 0),
                  pips_to_id(4, 1), pips_to_id(3, 2)]
    d = rng.choice(count_doms)
    decl = rng.choice(list(range(7)) + [DOUBLES_TRUMP])
    pts = DOMINO_COUNT_POINTS[d]

    is_trump = has_trump_power(decl) and is_in_called_suit(d, decl)
    hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]

    q = (f"The {dom(d)} is worth {pts} count points. "
         f"When {DECL_NAMES[decl]} are trump, is the {dom(d)} also a trump?")

    if is_trump:
        if decl in range(7):
            a = (f"Yes. The {dom(d)} contains a {decl}, so it is a trump AND a {pts}-point "
                 f"count domino. It belongs to the trump suit, not the "
                 f"{SUIT_NAMES[hi] if hi != decl else SUIT_NAMES[lo]} suit.")
        else:
            a = (f"Yes. The {dom(d)} is a double, so when doubles are trump it is both "
                 f"a trump and a {pts}-point count domino.")
    else:
        if decl in range(7):
            a = (f"No. The {dom(d)} has pips {hi} and {lo}. Neither is {decl}, so it is "
                 f"not a trump. It is still worth {pts} count points but belongs to its "
                 f"regular suit(s), not the trump suit.")
        else:
            a = (f"No. The {dom(d)} is not a double, so it is not a trump when doubles "
                 f"are trump. It is still worth {pts} count points.")

    return {"category": "count_trump", "question": q, "answer": a}


# ============================================================================
# Main
# ============================================================================

GENERATORS = [
    (gen_is_trump, 0.25),
    (gen_list_trumps, 0.10),
    (gen_which_are_trumps, 0.20),
    (gen_trump_or_follow, 0.30),
    (gen_trump_or_follow, 0.00),  # extra weight handled by trump_or_follow's 0.30
    (gen_count_trump_interaction, 0.15),
]


def main():
    parser = argparse.ArgumentParser(description="Generate trump-focused drilling Q&A")
    parser.add_argument("--output", type=str, default="lem/rules/qa_trump_drill.jsonl")
    parser.add_argument("--n-total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gens = [(fn, w) for fn, w in GENERATORS if w > 0]
    fns = [fn for fn, _ in gens]
    weights = [w for _, w in gens]

    total = 0
    cats: dict[str, int] = {}

    with open(output_path, "w") as f:
        for i in range(args.n_total):
            fn = rng.choices(fns, weights=weights, k=1)[0]
            try:
                ex = fn(rng)
                ex["section"] = "C_trump"
                f.write(json.dumps(ex) + "\n")
                total += 1
                cats[ex["category"]] = cats.get(ex["category"], 0) + 1
            except Exception as e:
                print(f"[warn] example {i} failed: {e}", file=sys.stderr)

    print(f"Generated {total} trump drill examples to {output_path}", file=sys.stderr)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
