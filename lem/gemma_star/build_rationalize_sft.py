#!/usr/bin/env python3
"""Build rationalization SFT data from verified-clean scout responses.

Takes rationalization responses from scout_play_qwen.py --rationalize,
filters through the verifier, and outputs (prompt, open-ended question,
clean-rationalization) SFT pairs for training a reasoning adapter.

Key design choice: we train on the OPEN-ENDED question ("What do you play
and why?") with the filtered rationalization as the target. This way the
model learns to produce reasoning in response to the prompt shape it'll
actually see at inference time (not the rationalize-style "strong player
chose X" prompt).

Usage:
    python -m lem.gemma_star.build_rationalize_sft \
      --responses scratch/rationalize_v9_500.jsonl \
      --decisions lem/data/decisions_500.jsonl \
      --output lem/data/rationalize_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from lem.gemma_star.verify_rationalization import verify, _clean

    responses = [json.loads(l) for l in Path(args.responses).read_text().strip().split("\n") if l.strip()]
    decisions = [json.loads(l) for l in Path(args.decisions).read_text().strip().split("\n") if l.strip()]
    dec_by_key = {(d["seed"], d["narrator"], d.get("decl_name")): d for d in decisions}

    kept = []
    n_total = 0
    n_unmatched = 0
    for r in responses:
        n_total += 1
        key = (r["seed"], r["narrator"], r.get("decl_name"))
        if key not in dec_by_key:
            n_unmatched += 1
            continue
        d = dec_by_key[key]
        ex = {**r, **d}
        v = verify(ex["response"], ex)
        if v["valid"]:
            cleaned = _clean(ex["response"])
            # Strip trailing model-chat tokens that snuck past _clean
            cleaned = re.sub(r"<\|[^|]*\|>.*$", "", cleaned, flags=re.DOTALL).strip()
            # Must contain a final "Play X" for training discipline
            if not re.search(r"[Pp]lay\s+(?:the\s+)?" + re.escape(d["bot_action"]), cleaned):
                # Ensure the rationalization ends with the correct action
                cleaned = cleaned.rstrip(".") + f". Play {d['bot_action']}."
            kept.append({
                "category": "rationalize_play",
                "prompt": d["prompt"],
                "question": "What do you play and why?",
                "answer": cleaned,
                "seed": d["seed"],
                "decl_name": d["decl_name"],
                "narrator": d["narrator"],
                "bot_action": d["bot_action"],
                "eq_gap": d.get("eq_gap"),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"Responses read: {n_total}", file=sys.stderr)
    print(f"Unmatched (missing decision record): {n_unmatched}", file=sys.stderr)
    print(f"Clean / kept for SFT: {len(kept)} ({100 * len(kept) / max(1, n_total - n_unmatched):.0f}%)", file=sys.stderr)
    print(f"Wrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
