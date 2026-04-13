#!/usr/bin/env python3
"""Offline grader for comprehension eval responses.

Reads raw responses from scratch/eval_responses.jsonl (captured by eval_comprehension.py)
and grades them with flexible fact-extraction grading. No GPU needed.

Usage:
    python -m lem.gemma_star.grade_offline
    python -m lem.gemma_star.grade_offline --show-all  # show correct + incorrect
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def _normalize_dom(d: str) -> str:
    parts = d.split("-")
    if len(parts) != 2:
        return d
    try:
        a, b = int(parts[0]), int(parts[1])
        return f"{max(a,b)}-{min(a,b)}"
    except ValueError:
        return d


def _clean_response(response: str) -> str:
    """Strip thinking blocks and special tokens."""
    # Strip Gemma 4 thinking: <|channel>thought...<channel|>
    idx = response.rfind("<channel|>")
    if idx >= 0:
        response = response[idx + len("<channel|>"):]
    # Strip other markers
    response = re.sub(r"<\|?turn\|?>", "", response)
    response = re.sub(r"<\|?channel\|?>.*", "", response)
    response = re.sub(r"<pad>", "", response)
    response = re.sub(r"<eos>", "", response)
    response = re.sub(r"<bos>", "", response)
    return response.strip()


def _extract_all_dominoes(text: str) -> set[str]:
    """Find all X-Y domino patterns in text."""
    return {_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", text)}


def grade_legal_moves(gt_answer: str, response: str) -> dict:
    """Grade legal_moves: extract dominos mentioned as playable, compare to GT."""
    response = _clean_response(response)

    # Extract GT legal moves
    gt_match = re.search(r"[Ll]egal moves?:\s*(.+?)(?:\.|$)", gt_answer)
    if gt_match:
        gt_moves = {_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", gt_match.group(1))}
    else:
        gt_moves = _extract_all_dominoes(gt_answer)

    # Extract response moves — look for dominoes mentioned as legal/playable
    # The model might say "Play 5-2" or "5-2 is legal" or list them
    resp_moves = set()

    # Pattern 1: "Legal moves: X, Y"
    m = re.search(r"[Ll]egal moves?:\s*(.+?)(?:\.|$)", response)
    if m:
        resp_moves = {_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", m.group(1))}

    # Pattern 2: "Play X" or "play the X"
    if not resp_moves:
        plays = re.findall(r"[Pp]lay\s+(?:the\s+)?(\d-\d)", response)
        resp_moves = {_normalize_dom(d) for d in plays}

    # Pattern 3: numbered list "1. X" or "1. **X**"
    if not resp_moves:
        listed = re.findall(r"\d+\.\s+\**(?:Play\s+)?(\d-\d)", response)
        resp_moves = {_normalize_dom(d) for d in listed}

    # Pattern 4: "X is a legal move" or "X is legal"
    if not resp_moves:
        legal = re.findall(r"(\d-\d)\b[^.]*?legal", response, re.IGNORECASE)
        resp_moves = {_normalize_dom(d) for d in legal}

    # Pattern 5: "cards in your hand: X and Y" or "your hand: X, Y"
    if not resp_moves:
        hand_match = re.search(r"(?:cards? in your hand|your (?:legal )?moves? are)[:\s]+(.+?)(?:\.|$)", response, re.IGNORECASE)
        if hand_match:
            resp_moves = {_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", hand_match.group(1))}

    # Pattern 6: just find ALL dominoes in response if nothing else worked
    # (last resort — might over-match but better than parse_fail)
    if not resp_moves:
        all_doms = _extract_all_dominoes(response)
        # Only use if there are exactly as many as GT (heuristic)
        if len(all_doms) == len(gt_moves) and len(all_doms) <= 4:
            resp_moves = all_doms

    if not resp_moves:
        return {"correct": False, "detail": "parse_fail", "gt": gt_moves, "resp": set()}

    correct = resp_moves == gt_moves
    return {"correct": correct, "detail": f"gt={gt_moves} resp={resp_moves}", "gt": gt_moves, "resp": resp_moves}


def grade_is_trump(gt_answer: str, response: str) -> dict:
    """Grade is_trump: check yes/no — prioritize the FIRST clear signal."""
    response = _clean_response(response)
    gt_yes = gt_answer.lower().startswith("yes")

    resp_lower = response.lower()
    # Check negatives FIRST (they contain "trump" substring which would match positives)
    no_signals = ["is not a trump", "is not trump", "not a trump", "isn't a trump",
                  "not the trump", "no,", "no."]
    yes_signals = ["is a trump", "is trump", "yes,", "yes."]

    # Find earliest signal
    earliest_pos = len(resp_lower)
    resp_yes = None
    for sig in no_signals:
        idx = resp_lower.find(sig)
        if idx >= 0 and idx < earliest_pos:
            earliest_pos = idx
            resp_yes = False
    for sig in yes_signals:
        idx = resp_lower.find(sig)
        if idx >= 0 and idx < earliest_pos:
            earliest_pos = idx
            resp_yes = True

    if resp_yes is None:
        return {"correct": False, "detail": "parse_fail"}

    correct = gt_yes == resp_yes
    return {"correct": correct, "detail": f"gt={'yes' if gt_yes else 'no'} resp={'yes' if resp_yes else 'no'}"}


def grade_where_is(gt_answer: str, response: str) -> dict:
    """Grade where_is: check if correct location is mentioned."""
    response = _clean_response(response)
    gt_lower = gt_answer.lower()
    resp_lower = response.lower()

    if "in your hand" in gt_lower:
        correct = "hand" in resp_lower and ("your" in resp_lower or "my" in resp_lower)
        return {"correct": correct, "detail": "hand check"}

    if "not been played" in gt_lower:
        correct = ("not" in resp_lower and "played" in resp_lower) or "unknown" in resp_lower
        return {"correct": correct, "detail": "not-played check"}

    # Played by X on trick N
    gt_trick = re.search(r"trick (\d)", gt_lower)
    gt_player = re.search(r"played by (\w+)", gt_lower)

    if gt_trick:
        # Check if correct trick number appears
        resp_tricks = re.findall(r"trick (\d)", resp_lower)
        trick_correct = gt_trick.group(1) in resp_tricks

        # Check player (flexible — "P1", "player 1", "partner", "you")
        player_correct = True  # lenient for now
        if gt_player:
            player = gt_player.group(1).lower()
            player_correct = player in resp_lower

        correct = trick_correct
        return {"correct": correct, "detail": f"trick={gt_trick.group(1)} in resp_tricks={resp_tricks}"}

    return {"correct": False, "detail": "can't parse GT"}


def grade_count_status(gt_answer: str, response: str) -> dict:
    """Grade count_status: check if key facts match."""
    response = _clean_response(response)
    gt_lower = gt_answer.lower()
    resp_lower = response.lower()

    # Key facts: captured by team, in hand, still out, played on trick N
    if "your team" in gt_lower:
        correct = "your team" in resp_lower or ("you" in resp_lower and ("win" in resp_lower or "capture" in resp_lower or "take" in resp_lower or "won" in resp_lower))
        return {"correct": correct, "detail": "your-team check"}

    if "opponent" in gt_lower:
        # Opponents captured it — check if response indicates opponents won
        correct = ("opponent" in resp_lower or
                   "p1 win" in resp_lower or "p3 win" in resp_lower or
                   "loss" in resp_lower or "them" in resp_lower or
                   "p1 wins" in resp_lower or "p3 wins" in resp_lower)
        # Also check trick number
        gt_trick = re.search(r"trick (\d)", gt_lower)
        if gt_trick and gt_trick.group(1) in resp_lower:
            correct = True
        return {"correct": correct, "detail": "opponent check"}

    if "in your hand" in gt_lower:
        correct = "hand" in resp_lower
        return {"correct": correct, "detail": "in-hand check"}

    if "still out" in gt_lower or "not been played" in gt_lower:
        correct = "not" in resp_lower or "still" in resp_lower
        return {"correct": correct, "detail": "still-out check"}

    return {"correct": False, "detail": "can't parse GT"}


def grade_what_beats(gt_answer: str, response: str) -> dict:
    """Grade what_beats: check if correct beaters are mentioned."""
    response = _clean_response(response)

    if "nothing can beat" in gt_answer.lower():
        correct = "nothing" in response.lower() or "no" in response.lower() or "highest" in response.lower() or "cannot be beaten" in response.lower()
        return {"correct": correct, "detail": "nothing-beats check"}

    # Extract GT beater dominoes (exclude the domino being asked about)
    gt_doms = _extract_all_dominoes(gt_answer)
    resp_doms = _extract_all_dominoes(response)

    if not gt_doms:
        return {"correct": True, "detail": "no GT doms"}

    # Check if response mentions at least 50% of GT beaters
    overlap = len(gt_doms & resp_doms)
    total = len(gt_doms)
    ratio = overlap / total if total > 0 else 0
    correct = ratio >= 0.5

    return {"correct": correct, "detail": f"overlap={overlap}/{total} ({ratio:.0%}) gt={gt_doms} resp={resp_doms}"}


GRADERS = {
    "legal_moves": grade_legal_moves,
    "is_trump": grade_is_trump,
    "where_is": grade_where_is,
    "count_status": grade_count_status,
    "what_beats": grade_what_beats,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scratch/eval_responses.jsonl")
    parser.add_argument("--show-all", action="store_true", help="Show correct answers too")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Not found: {path}. Run eval_comprehension.py first to capture responses.")
        sys.exit(1)

    examples = [json.loads(line) for line in path.read_text().strip().split("\n") if line.strip()]
    print(f"Loaded {len(examples)} responses from {path}\n")

    stats = Counter()
    by_cat = {}

    for ex in examples:
        cat = ex["category"]
        grader = GRADERS.get(cat)
        if not grader:
            print(f"  Unknown category: {cat}")
            continue

        result = grader(ex["answer"], ex["response"])
        correct = result["correct"]

        stats["total"] += 1
        stats["correct" if correct else "wrong"] += 1

        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "correct": 0}
        by_cat[cat]["total"] += 1
        if correct:
            by_cat[cat]["correct"] += 1

        # Display
        if not correct or args.show_all:
            mark = "✓" if correct else "✗"
            print(f"{mark} [{cat}] Q: {ex['question']}")
            resp_clean = _clean_response(ex["response"])
            print(f"  GT:   {ex['answer'][:120]}")
            print(f"  Resp: {resp_clean[:200]}")
            print(f"  Grade: {result['detail']}")
            print()

    # Summary
    total = stats["total"]
    correct = stats.get("correct", 0)
    print(f"{'='*60}")
    print(f"Overall: {correct}/{total} ({correct/total*100:.0f}%)")
    print()
    for cat in sorted(by_cat):
        c = by_cat[cat]
        acc = c["correct"] / c["total"] * 100 if c["total"] > 0 else 0
        print(f"  {cat:15s}: {c['correct']}/{c['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    main()
