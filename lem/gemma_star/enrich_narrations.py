#!/usr/bin/env python3
"""Enrich narration JSONL with ground truth for scratchpad validation.

Reads existing narrations, extracts ground truth (hand, played dominoes, voids)
from the prompt text, updates the prompt ending to the scratchpad format,
and writes an enriched JSONL.

No GPU needed — pure text parsing + game logic.

Usage:
    python -m lem.gemma_star.enrich_narrations \
        --input lem/data/narrations_train.jsonl \
        --output lem/data/narrations_train_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Suit name → suit id mapping
SUIT_NAME_TO_ID = {
    "blanks": 0, "ones": 1, "twos": 2, "threes": 3,
    "fours": 4, "fives": 5, "sixes": 6,
}

# Count dominoes
COUNT_DOMINOES = {"5-5", "6-4", "5-0", "4-1", "3-2"}

# The new scratchpad prompt ending
SCRATCHPAD_PROMPT = (
    "Before you play, fill in this scratchpad to organize your thinking:\n"
    "HAND: [list your remaining dominoes]\n"
    "VOIDS: [list known voids from play history, e.g. \"Player 2: fives\" — "
    "or \"none observed\" if no voids are known]\n"
    "COUNTS: [for each count domino (5-5=10pts, 6-4=10pts, 5-0=5pts, "
    "4-1=5pts, 3-2=5pts), state played or still out]\n"
    "PLAY: [the domino you choose to play]"
)


def _normalize_dom(d: str) -> str:
    """Normalize domino string to H-L form."""
    a, b = d.split("-")
    hi, lo = max(int(a), int(b)), min(int(a), int(b))
    return f"{hi}-{lo}"


def _extract_true_hand(prompt: str) -> list[str]:
    """Extract narrator's remaining hand from 'Your remaining dominoes: ...'"""
    m = re.search(r"Your remaining dominoes:\s*(.+?)\.", prompt)
    if m:
        doms = re.findall(r"\d-\d", m.group(1))
        return [_normalize_dom(d) for d in doms]
    return []


def _extract_plays(prompt: str) -> list[dict]:
    """Extract all plays from trick history.

    Returns list of {player_ref, domino, is_leader, verb, trick_num}.
    """
    plays = []
    trick_num = 0

    for line in prompt.split("\n"):
        trick_match = re.match(r"Trick (\d+):", line.strip())
        if trick_match:
            trick_num = int(trick_match.group(1))
            continue

        # Leader: "Player 0 leads the 5-3."
        lead_match = re.match(
            r"\s+(.+?)\s+leads?\s+the\s+(\d-\d)", line
        )
        if lead_match:
            plays.append({
                "player_ref": lead_match.group(1).strip(),
                "domino": _normalize_dom(lead_match.group(2)),
                "is_leader": True,
                "verb": "lead",
                "trick_num": trick_num,
            })
            continue

        # Follower: "Player 2 follows with the 6-5."
        follow_match = re.match(
            r"\s+(.+?)\s+(follows?|trumps? in|sluffs?)\s+with\s+the\s+(\d-\d)", line
        )
        if follow_match:
            plays.append({
                "player_ref": follow_match.group(1).strip(),
                "domino": _normalize_dom(follow_match.group(3)),
                "is_leader": False,
                "verb": follow_match.group(2).rstrip("s"),
                "trick_num": trick_num,
            })

    return plays


def _extract_played_dominoes(plays: list[dict]) -> list[str]:
    """Get all dominoes that have been played."""
    return [p["domino"] for p in plays]


def _extract_voids(plays: list[dict], decl_name: str) -> dict[str, list[str]]:
    """Infer voids from sluffs in the play history.

    When a player sluffs, they're void in the led suit of that trick.
    Returns {player_ref: [suit_name, ...]}.
    """
    voids: dict[str, set[str]] = {}

    # Group plays by trick to know the led suit
    tricks: dict[int, list[dict]] = {}
    for p in plays:
        tricks.setdefault(p["trick_num"], []).append(p)

    # Determine trump suit for led-suit computation
    trump_pip = SUIT_NAME_TO_ID.get(decl_name)  # None for doubles/notrump

    for trick_num, trick_plays in tricks.items():
        if not trick_plays:
            continue

        lead_dom = trick_plays[0]["domino"]
        lead_hi = int(lead_dom.split("-")[0])
        lead_lo = int(lead_dom.split("-")[1])
        is_double = lead_hi == lead_lo

        # Determine led suit
        if trump_pip is not None and (lead_hi == trump_pip or lead_lo == trump_pip):
            led_suit = decl_name  # trump was led
        elif decl_name == "doubles" and is_double:
            led_suit = "trump"
        else:
            # Higher pip determines suit
            pip_to_suit = {0: "blanks", 1: "ones", 2: "twos", 3: "threes",
                           4: "fours", 5: "fives", 6: "sixes"}
            led_suit = pip_to_suit.get(lead_hi, f"suit-{lead_hi}")

        # Check followers for sluffs
        for p in trick_plays[1:]:
            if p["verb"] in ("sluff", "trump"):
                # Sluff = void in led suit. Trump = also void in led suit (chose trump instead)
                if p["verb"] == "sluff" or (p["verb"] == "trump" and led_suit != "trump"):
                    player = p["player_ref"]
                    voids.setdefault(player, set()).add(led_suit)

    return {k: sorted(v) for k, v in voids.items()}


def _extract_count_status(played_dominoes: list[str]) -> dict[str, str]:
    """Determine which count dominoes have been played vs still out."""
    played_set = set(played_dominoes)
    return {
        dom: "played" if dom in played_set else "out"
        for dom in sorted(COUNT_DOMINOES)
    }


def enrich_example(ex: dict) -> dict:
    """Enrich a single narration example with ground truth + new prompt."""
    prompt = ex["prompt"]
    plays = _extract_plays(prompt)

    true_hand = _extract_true_hand(prompt)
    played_dominoes = _extract_played_dominoes(plays)
    true_voids = _extract_voids(plays, ex.get("decl_name", ""))
    count_status = _extract_count_status(played_dominoes)

    # Replace old prompt ending with scratchpad
    new_prompt = prompt.replace("What do you play?", SCRATCHPAD_PROMPT)

    enriched = dict(ex)
    enriched["prompt"] = new_prompt
    enriched["true_hand"] = true_hand
    enriched["played_dominoes"] = played_dominoes
    enriched["true_voids"] = true_voids
    enriched["count_status"] = count_status

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich narrations with ground truth")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    examples = [json.loads(line) for line in input_path.read_text().strip().split("\n")]
    print(f"Read {len(examples)} examples from {input_path}", file=sys.stderr)

    enriched = [enrich_example(ex) for ex in examples]

    # Sanity checks
    n_with_hand = sum(1 for e in enriched if e["true_hand"])
    n_with_voids = sum(1 for e in enriched if any(e["true_voids"].values()))
    n_with_counts = sum(1 for e in enriched if e["count_status"])
    n_prompt_updated = sum(1 for e in enriched if "HAND:" in e["prompt"])

    print(f"  hand extracted: {n_with_hand}/{len(enriched)}", file=sys.stderr)
    print(f"  voids found: {n_with_voids}/{len(enriched)}", file=sys.stderr)
    print(f"  counts computed: {n_with_counts}/{len(enriched)}", file=sys.stderr)
    print(f"  prompt updated: {n_prompt_updated}/{len(enriched)}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")

    print(f"Wrote {len(enriched)} enriched examples to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
