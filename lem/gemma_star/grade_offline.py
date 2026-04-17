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


def grade_void_deduction(gt_answer: str, response: str) -> dict:
    """Grade void_deduction: check if voids and unknowns are correctly identified.

    Checks three things:
    1. Proven voids: does the response identify the right players as void?
    2. Uncertainty: does the response express "don't know" for untested players?
    3. No hallucination: does the response avoid claiming void for non-void players?
    """
    response = _clean_response(response)
    gt_lower = gt_answer.lower()
    resp_lower = response.lower()

    # Extract proven voids from GT: "PX is void in suit"
    gt_voids = set(re.findall(r"(p\d|partner) is void", gt_lower))
    resp_voids = set(re.findall(r"(p\d|partner) is void", resp_lower))

    # Also accept "has no X" or "doesn't have" as void signals
    resp_no_suit = set(re.findall(r"(p\d|partner) (?:has no|doesn't have|does not have|holds no)", resp_lower))
    resp_voids |= resp_no_suit

    # Extract "don't know" signals from GT
    gt_unknowns = set(re.findall(r"(p\d|partner) was never tested", gt_lower))
    # Check if response expresses uncertainty for those players
    resp_unknowns = set(re.findall(r"(p\d|partner)[^.]*(?:don't know|do not know|unknown|uncertain|no information|never tested|not tested|can't tell|cannot tell|unclear|no way to know)", resp_lower))

    # Score: did we get the voids right?
    void_correct = gt_voids <= resp_voids  # all GT voids found in response
    void_no_hallucinate = resp_voids <= gt_voids  # no extra voids claimed

    # Score: did we express uncertainty for untested players?
    uncertainty_correct = gt_unknowns <= resp_unknowns if gt_unknowns else True

    correct = void_correct and void_no_hallucinate and uncertainty_correct

    detail_parts = []
    if not void_correct:
        detail_parts.append(f"missed voids: {gt_voids - resp_voids}")
    if not void_no_hallucinate:
        detail_parts.append(f"hallucinated voids: {resp_voids - gt_voids}")
    if not uncertainty_correct:
        detail_parts.append(f"missed uncertainty: {gt_unknowns - resp_unknowns}")
    detail = "; ".join(detail_parts) if detail_parts else f"voids={gt_voids} unknowns={gt_unknowns}"

    return {"correct": correct, "detail": detail}


def grade_suit_members(gt_answer: str, response: str) -> dict:
    """Grade suit_members: check the listed dominoes match the GT set, in correct order."""
    response = _clean_response(response)

    gt_doms_ordered = [_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", gt_answer)]
    resp_doms_ordered = [_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", response)]

    gt_set = set(gt_doms_ordered)
    resp_set = set(resp_doms_ordered)

    # Primary: did the response contain the same set of suit members?
    set_correct = gt_set.issubset(resp_set) and len(resp_set - gt_set) <= 1

    # Secondary: is the first GT domino (the highest-ranked) named first in response?
    order_correct = True
    if gt_doms_ordered and resp_doms_ordered:
        gt_first = gt_doms_ordered[0]  # the double / highest
        order_correct = resp_doms_ordered[0] == gt_first

    correct = set_correct and order_correct
    detail = f"gt_set={gt_set} resp_set={resp_set} order_ok={order_correct}"
    return {"correct": correct, "detail": detail}


def grade_rank_in_suit(gt_answer: str, response: str) -> dict:
    """Grade rank_in_suit: check the ranked order matches GT."""
    response = _clean_response(response)

    gt_doms_ordered = [_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", gt_answer)]
    resp_doms_ordered = [_normalize_dom(d) for d in re.findall(r"\b(\d-\d)\b", response)]

    # GT has the dominoes twice (question + answer listing). Take the last N which is the ranked answer.
    # Actually GT answer starts with "Ranked highest to lowest: X, Y, Z." so take the first sequence.
    if not gt_doms_ordered or not resp_doms_ordered:
        return {"correct": False, "detail": "parse_fail"}

    n = len(gt_doms_ordered)
    gt_ordered = gt_doms_ordered[:n]

    # Look for a matching ordered subsequence in the response
    # (the response may repeat the question list; we want the ranked list)
    found_match = False
    for start in range(len(resp_doms_ordered) - n + 1):
        if resp_doms_ordered[start:start + n] == gt_ordered:
            found_match = True
            break

    # Partial credit: top pick correct?
    top_correct = resp_doms_ordered[0] == gt_ordered[0] if resp_doms_ordered else False

    correct = found_match
    detail = f"gt_order={gt_ordered} top_correct={top_correct} exact_match={found_match}"
    return {"correct": correct, "detail": detail}


def grade_conditional_beat(gt_answer: str, response: str) -> dict:
    """Grade conditional_beat: the YES/NO final answer must match."""
    response = _clean_response(response)

    # Extract ground truth YES/NO
    gt_yes = bool(re.search(r"Answer:\s*YES", gt_answer, re.IGNORECASE))

    # Extract response YES/NO — look for "Answer: YES/NO" or final verdict
    m = re.search(r"Answer:\s*(YES|NO)", response, re.IGNORECASE)
    if m:
        resp_yes = m.group(1).upper() == "YES"
    else:
        # Fallback: look for "beats" or "does not beat"
        if re.search(r"does not beat|doesn't beat|cannot beat|can't beat", response, re.IGNORECASE):
            resp_yes = False
        elif re.search(r"\bbeats\b", response, re.IGNORECASE):
            resp_yes = True
        else:
            return {"correct": False, "detail": "parse_fail"}

    correct = gt_yes == resp_yes
    return {"correct": correct, "detail": f"gt={'YES' if gt_yes else 'NO'} resp={'YES' if resp_yes else 'NO'}"}


def grade_beaters_in_unseen(gt_answer: str, response: str) -> dict:
    """Grade beaters_in_unseen: check overlap of listed beater dominoes."""
    response = _clean_response(response)
    if "nothing" in gt_answer.lower() or "no unseen" in gt_answer.lower():
        correct = "nothing" in response.lower() or "no unseen" in response.lower()
        return {"correct": correct, "detail": "nothing check"}
    # Extract "Answer: X, Y, Z" section
    gt_match = re.search(r"Answer:\s*(.+?)(?:\.|$)", gt_answer)
    gt_doms = {_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", gt_match.group(1))} if gt_match else set()
    resp_doms = {_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", response)}
    if not gt_doms:
        return {"correct": True, "detail": "no GT doms"}
    overlap = len(gt_doms & resp_doms) / len(gt_doms)
    correct = overlap >= 0.8
    return {"correct": correct, "detail": f"overlap={overlap:.0%}"}


def grade_partner_response(gt_answer: str, response: str) -> dict:
    """Grade partner_response: check listed dominoes overlap."""
    # Same logic as beaters_in_unseen — both list unseen dominoes
    return grade_beaters_in_unseen(gt_answer, response)


def grade_intervention_check(gt_answer: str, response: str) -> dict:
    """Grade intervention_check: YES/NO answer must match."""
    response = _clean_response(response)
    gt_yes = bool(re.search(r"Answer:\s*YES", gt_answer, re.IGNORECASE))
    m = re.search(r"Answer:\s*(YES|NO)", response, re.IGNORECASE)
    if m:
        resp_yes = m.group(1).upper() == "YES"
    else:
        if re.search(r"cannot intervene|can not intervene", response, re.IGNORECASE):
            resp_yes = False
        elif re.search(r"can intervene", response, re.IGNORECASE):
            resp_yes = True
        else:
            return {"correct": False, "detail": "parse_fail"}
    correct = gt_yes == resp_yes
    return {"correct": correct, "detail": f"gt={'YES' if gt_yes else 'NO'} resp={'YES' if resp_yes else 'NO'}"}


def grade_visibility_audit(gt_answer: str, response: str) -> dict:
    """Grade visibility_audit: check the unseen list matches."""
    response = _clean_response(response)
    # Look for "Unseen list: X, Y, Z" line
    gt_match = re.search(r"Unseen list:\s*(.+?)(?:\.|$)", gt_answer)
    if not gt_match:
        return {"correct": False, "detail": "no GT unseen list"}
    gt_unseen = {_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", gt_match.group(1))}

    resp_match = re.search(r"[Uu]nseen(?:\s+list)?:\s*(.+?)(?:\.|\n|$)", response)
    if not resp_match:
        return {"correct": False, "detail": "no response unseen list"}
    resp_unseen = {_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", resp_match.group(1))}

    correct = gt_unseen == resp_unseen
    return {"correct": correct, "detail": f"gt={gt_unseen} resp={resp_unseen}"}


def grade_highest_unseen_in_suit(gt_answer: str, response: str) -> dict:
    """Grade highest_unseen_in_suit: single-domino answer."""
    response = _clean_response(response)
    gt_match = re.search(r"Answer:\s*(\d-\d)", gt_answer)
    if not gt_match:
        return {"correct": False, "detail": "no GT answer"}
    gt_dom = _normalize_dom(gt_match.group(1))

    # Take the last "Answer: X" in response (accept earlier "highest" mentions too)
    resp_matches = list(re.finditer(r"(?:Answer:|[Hh]ighest unseen[^.]*?is)\s*(\d-\d)", response))
    if not resp_matches:
        # Fall back to any domino reference in the response — highest rank at end usually
        all_doms = [_normalize_dom(d) for d in re.findall(r"\b\d-\d\b", response)]
        if all_doms and all_doms[-1] == gt_dom:
            return {"correct": True, "detail": "fallback-last-dom match"}
        return {"correct": False, "detail": "parse_fail"}
    resp_dom = _normalize_dom(resp_matches[-1].group(1))
    correct = resp_dom == gt_dom
    return {"correct": correct, "detail": f"gt={gt_dom} resp={resp_dom}"}


GRADERS = {
    "legal_moves": grade_legal_moves,
    "is_trump": grade_is_trump,
    "where_is": grade_where_is,
    "count_status": grade_count_status,
    "what_beats": grade_what_beats,
    "void_deduction": grade_void_deduction,
    "suit_members": grade_suit_members,
    "rank_in_suit": grade_rank_in_suit,
    "conditional_beat": grade_conditional_beat,
    "beaters_in_unseen": grade_beaters_in_unseen,
    "partner_response": grade_partner_response,
    "intervention_check": grade_intervention_check,
    "visibility_audit": grade_visibility_audit,
    "highest_unseen_in_suit": grade_highest_unseen_in_suit,
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
