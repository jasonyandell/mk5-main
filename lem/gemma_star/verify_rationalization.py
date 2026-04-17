#!/usr/bin/env python3
"""Verifier for rationalization responses.

Takes a model response + the decision's structured state, and checks every
factual claim in the response against engine truth. Returns a verdict and
a list of errors. A response is clean iff the errors list is empty.

Purpose: filter rationalization candidates to produce a training set without
hallucinated reasoning. Anchors SFT to the model's own words about correct
facts — the model learned these facts in comprehension training at 99%
accuracy, so any hallucination here is a red flag.

Usage:
    python -m lem.gemma_star.verify_rationalization \
        --responses scratch/scout_B_rationalize.jsonl \
        --decisions lem/data/decisions_scout.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# suit_name → pip id (0-6) for pip suits; 7 = trump; 8 = doubles-as-suit
SUIT_NAMES: dict[str, int] = {
    "blanks": 0, "blank": 0,
    "ones": 1, "one": 1,
    "twos": 2, "two": 2,
    "threes": 3, "three": 3,
    "fours": 4, "four": 4,
    "fives": 5, "five": 5,
    "sixes": 6, "six": 6,
    "doubles": 8,
    "no-trump": 9, "notrump": 9, "no trump": 9,
}


def _normalize_dom(s: str) -> str:
    """Normalize to H-L with H >= L."""
    a, b = s.split("-")
    a, b = int(a), int(b)
    return f"{max(a, b)}-{min(a, b)}"


def _dom_to_pips(s: str) -> tuple[int, int]:
    a, b = s.split("-")
    a, b = int(a), int(b)
    return max(a, b), min(a, b)


def _is_valid_dom(s: str) -> bool:
    if not re.fullmatch(r"\d-\d", s):
        return False
    h, lo = _dom_to_pips(s)
    return 0 <= lo <= h <= 6


def _all_28_doms() -> set[str]:
    return {f"{h}-{lo}" for h in range(7) for lo in range(h + 1)}


def _is_trump_dom(dom: str, decl_id: int) -> bool:
    """Is this domino trump under the given declaration?"""
    h, lo = _dom_to_pips(dom)
    if decl_id <= 6:
        return h == decl_id or lo == decl_id
    if decl_id == 7:  # doubles trump
        return h == lo
    if decl_id == 8:  # doubles as suit
        return False  # no trump, doubles have their own suit
    if decl_id == 9:  # notrump
        return False
    return False


def _strip_think(response: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _strip_special(response: str) -> str:
    """Remove model special tokens."""
    response = re.sub(r"<\|[^|]*\|>", "", response)
    response = re.sub(r"<think>", "", response)
    response = re.sub(r"</think>", "", response)
    return response.strip()


def _clean(response: str) -> str:
    """Full cleanup — think blocks and special tokens."""
    return _strip_special(_strip_think(response))


# ---------------------------------------------------------------------------
# Individual check functions
# Each returns a list of error strings (empty = no errors for that check)
# ---------------------------------------------------------------------------

def check_domino_validity(response: str) -> list[str]:
    """Every X-Y reference must be a real domino."""
    errors = []
    for match in re.findall(r"\b\d-\d\b", response):
        if not _is_valid_dom(match):
            errors.append(f"Invalid domino reference: '{match}'")
    return errors


def check_hand_claims(response: str, decision: dict) -> list[str]:
    """Claims like 'I hold X' or 'your hand: X' must match remaining_hand."""
    errors = []
    hand = {_normalize_dom(d) for d in decision["remaining_hand"]}

    # Patterns: "I hold X", "you have X", "your hand: X, Y", "I have X"
    patterns = [
        r"(?:I|you) (?:hold|have)(?:\s+the)?\s+(\d-\d)",
        r"(?:my|your) hand(?:[^.]*?):\s*([\d\-,\s]+)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, response, re.IGNORECASE):
            doms_str = m.group(1)
            for d in re.findall(r"\d-\d", doms_str):
                d_norm = _normalize_dom(d)
                if d_norm not in hand:
                    errors.append(
                        f"Claims to hold {d_norm}, but actual hand is {sorted(hand)}"
                    )
    return errors


def check_trump_declaration(response: str, decision: dict) -> list[str]:
    """Claims like 'Fives are trump' must match decl_name."""
    errors = []
    decl_name = decision["decl_name"].lower()

    # Pattern: "X are trump" / "X is trump" where X is a suit name
    for m in re.finditer(r"\b([a-z\-]+)\s+(?:are|is)\s+trump\b", response, re.IGNORECASE):
        claimed = m.group(1).lower()
        # Normalize plural/singular
        claimed_norm = claimed.rstrip("s")
        decl_norm = decl_name.rstrip("s")
        if claimed_norm != decl_norm:
            # Check if claimed is a known suit name
            if claimed in SUIT_NAMES or claimed_norm in SUIT_NAMES:
                errors.append(
                    f"Claims '{claimed} are trump' but actual trump is '{decl_name}'"
                )
    return errors


def check_trump_membership(response: str, decision: dict) -> list[str]:
    """Claims like 'X is trump' / 'X is a trump' must match engine for that domino."""
    errors = []
    decl_id = decision["decl_id"]

    # Patterns: "X is trump", "X is a trump", "X-Y is trump"
    # Positive
    for m in re.finditer(r"(\d-\d)\s+is\s+(?:a\s+)?trump\b", response, re.IGNORECASE):
        dom = _normalize_dom(m.group(1))
        if not _is_trump_dom(dom, decl_id):
            errors.append(
                f"Claims '{dom} is trump' but under {decision['decl_name']}, "
                f"{dom} is not trump"
            )
    # Negative
    for m in re.finditer(r"(\d-\d)\s+is\s+not\s+(?:a\s+)?trump\b", response, re.IGNORECASE):
        dom = _normalize_dom(m.group(1))
        if _is_trump_dom(dom, decl_id):
            errors.append(
                f"Claims '{dom} is not trump' but under {decision['decl_name']}, "
                f"{dom} IS trump"
            )
    return errors


def check_action_match(response: str, decision: dict) -> list[str]:
    """The stated 'Play X.' must not contradict bot_action.

    Lenient: missing/implicit play statements ('Play it', 'Play now') are OK
    because the bot_action is revealed in the prompt. What we REJECT is stating
    a different specific domino.
    """
    bot = _normalize_dom(decision["bot_action"])
    plays = list(re.finditer(r"[Pp]lay(?:\s+the)?\s+(\d-\d)", response))
    if not plays:
        return []  # no specific play stated — implicit agreement
    stated = _normalize_dom(plays[-1].group(1))
    if stated != bot:
        return [f"Stated 'Play {stated}' but bot_action is {bot}"]
    return []


def check_references_are_visible(response: str, decision: dict) -> list[str]:
    """Every domino mentioned must be in-hand or already-played. No ghost dominoes."""
    errors = []
    visible = {_normalize_dom(d) for d in decision["remaining_hand"]}
    for play in decision["plays"]:
        visible.add(_normalize_dom(play["dom"]))

    mentioned = {_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", response)
                 if _is_valid_dom(d)}

    invisible = mentioned - visible
    if invisible:
        errors.append(
            f"References dominoes not in hand or played: {sorted(invisible)}"
        )
    return errors


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

CHECKS = [
    ("domino_validity", check_domino_validity, False),  # False = needs no decision
    ("references_visible", check_references_are_visible, True),
    ("hand_claims", check_hand_claims, True),
    ("trump_declaration", check_trump_declaration, True),
    ("trump_membership", check_trump_membership, True),
    ("action_match", check_action_match, True),
]


def verify(response: str, decision: dict) -> dict:
    """Run all checks on a rationalization response.

    Args:
        response: the raw model response string
        decision: the decision dict with structured state (remaining_hand, plays, etc.)

    Returns:
        {
            "valid": bool,
            "errors": dict[str, list[str]],  # check_name -> errors
            "n_errors": int,
        }
    """
    cleaned = _clean(response)
    errors_by_check: dict[str, list[str]] = {}
    for name, fn, needs_decision in CHECKS:
        try:
            if needs_decision:
                errs = fn(cleaned, decision)
            else:
                errs = fn(cleaned)
        except Exception as e:
            errs = [f"check failed with exception: {e}"]
        if errs:
            errors_by_check[name] = errs

    n_errors = sum(len(errs) for errs in errors_by_check.values())
    return {
        "valid": n_errors == 0,
        "errors": errors_by_check,
        "n_errors": n_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True,
                        help="JSONL with 'response' field and decision fields")
    parser.add_argument("--decisions", default=None,
                        help="Optional JSONL with decision structured state. "
                             "If omitted, responses file must contain those fields "
                             "(scout_play_qwen.py output does).")
    parser.add_argument("--show", type=int, default=3,
                        help="Show N failure examples in detail")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    responses = [json.loads(l) for l in responses_path.read_text().strip().split("\n") if l.strip()]

    # If responses don't have decision structure, pair with decisions file
    if "remaining_hand" not in responses[0]:
        if not args.decisions:
            print("[error] responses lack structured state; pass --decisions",
                  file=sys.stderr)
            sys.exit(1)
        decisions = [json.loads(l) for l in Path(args.decisions).read_text().strip().split("\n") if l.strip()]
        # Pair by (seed, narrator, decl_name)
        decision_by_key = {
            (d["seed"], d["narrator"], d.get("decl_name")): d for d in decisions
        }
        paired = []
        for r in responses:
            key = (r["seed"], r["narrator"], r.get("decl_name"))
            if key in decision_by_key:
                merged = {**r, **decision_by_key[key]}
                paired.append(merged)
            else:
                print(f"[warn] no decision match for {key}", file=sys.stderr)
        responses = paired

    # Verify all
    results = []
    for ex in responses:
        v = verify(ex["response"], ex)
        results.append((ex, v))

    # Stats
    n_valid = sum(1 for _, v in results if v["valid"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Valid: {n_valid}/{total} ({100*n_valid/total:.0f}%)")
    print()

    # Error breakdown
    error_counts: dict[str, int] = {}
    for _, v in results:
        for check_name in v["errors"]:
            error_counts[check_name] = error_counts.get(check_name, 0) + 1
    if error_counts:
        print("Errors by check type:")
        for name, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")
        print()

    # Show failures
    failures = [(ex, v) for ex, v in results if not v["valid"]]
    for i, (ex, v) in enumerate(failures[:args.show]):
        print(f"--- FAILURE {i+1}: bot={ex['bot_action']} ---")
        cleaned = _clean(ex["response"])
        print(f"Response: {cleaned[:400]}")
        for check_name, errs in v["errors"].items():
            for e in errs:
                print(f"  [{check_name}] {e}")
        print()

    # Show passes
    passes = [(ex, v) for ex, v in results if v["valid"]]
    if passes:
        print(f"--- SAMPLE PASS (of {len(passes)}) ---")
        ex, v = passes[0]
        cleaned = _clean(ex["response"])
        print(f"bot={ex['bot_action']} gap={ex.get('eq_gap', '?')}")
        print(f"Response: {cleaned[:400]}")


if __name__ == "__main__":
    main()
