#!/usr/bin/env python3
"""Local STaR runner — watch Gemma think about Texas 42 in real time.

Uses llama.cpp (CPU) with the local GGUF. No adapter for now (base model only —
PEFT adapters need conversion to GGUF LoRA format for llama.cpp).

Usage:
    # Run on a random example
    python -m lem.gemma_star.local_star

    # Run on a specific example index
    python -m lem.gemma_star.local_star --index 42

    # Use the eval set
    python -m lem.gemma_star.local_star --narrations lem/data/narrations_eval.jsonl

    # Run N examples and show summary stats
    python -m lem.gemma_star.local_star --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Reuse grading logic from the Modal harness
from lem.gemma_star.star_harness import grade_k1, parse_play

GGUF_PATH = "scratch/gemma-4-E2B-it-Q4_K_M.gguf"
DEFAULT_NARRATIONS = "lem/data/narrations_train.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local STaR runner with llama.cpp")
    parser.add_argument("--narrations", type=str, default=DEFAULT_NARRATIONS)
    parser.add_argument("--index", type=int, default=None, help="Specific example index")
    parser.add_argument("--count", type=int, default=1, help="Number of examples to run")
    parser.add_argument("--gguf", type=str, default=GGUF_PATH)
    parser.add_argument("--n-ctx", type=int, default=8192, help="Context window size")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for example selection")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    # Load examples
    narrations_path = Path(args.narrations)
    if not narrations_path.exists():
        print(f"[error] Not found: {narrations_path}", file=sys.stderr)
        sys.exit(1)

    with open(narrations_path) as f:
        examples = [json.loads(line) for line in f]
    print(f"[data] Loaded {len(examples)} examples from {narrations_path}", file=sys.stderr)

    # Load model
    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"[error] GGUF not found: {gguf_path}", file=sys.stderr)
        print("  Run: python -m lem.gemma_star.local_star --help", file=sys.stderr)
        sys.exit(1)

    print(f"[model] Loading {gguf_path} (this takes a few seconds)...", file=sys.stderr)
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=args.n_ctx,
        n_threads=8,
        verbose=False,
    )
    print("[model] Ready.", file=sys.stderr)

    # Select examples
    rng = random.Random(args.seed)
    if args.index is not None:
        indices = [args.index]
    elif args.count == 1:
        indices = [rng.randint(0, len(examples) - 1)]
    else:
        indices = rng.sample(range(len(examples)), min(args.count, len(examples)))

    # Stats
    stats = {"pass": 0, "fail": 0, "illegal": 0, "parse_fail": 0}

    for run_num, idx in enumerate(indices):
        ex = examples[idx]
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Example {idx}: seed={ex['seed']}, decl={ex['decl_name']}, "
              f"narrator=P{ex['narrator']}", file=sys.stderr)
        print(f"Legal: {ex['legal_actions']}", file=sys.stderr)
        print(f"Bot played: {ex['bot_action']} (E[Q]={ex['bot_eq']})", file=sys.stderr)
        print(f"Best: {ex['best_action']} (E[Q]={ex['best_eq']})", file=sys.stderr)
        print(f"All E[Q]: {ex['all_eq']}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        # Run inference
        messages = [{"role": "user", "content": ex["prompt"]}]
        t_start = time.time()

        if args.no_stream:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.6,
            )
            full_text = response["choices"][0]["message"]["content"]
            print(full_text)
        else:
            # Stream so we can watch thinking in real time
            print("\n--- GEMMA THINKING ---", file=sys.stderr)
            full_text = ""
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.6,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_text += content
            print()  # newline after streaming

        elapsed = time.time() - t_start
        print(f"\n--- [{elapsed:.1f}s] ---", file=sys.stderr)

        # Grade
        gemma_action = parse_play(full_text)
        grade = grade_k1(
            gemma_action, ex["bot_action"], ex["bot_eq"],
            ex["all_eq"], ex["legal_actions"],
        )
        stats[grade["grade"]] = stats.get(grade["grade"], 0) + 1

        # Display result
        if grade["grade"] == "pass":
            print(f"\n✓ PASS — played {gemma_action}, "
                  f"E[Q]={grade.get('gemma_eq', '?'):.1f} >= bot's {grade.get('bot_eq', '?'):.1f} "
                  f"(delta={grade.get('delta', 0):+.1f})", file=sys.stderr)
        elif grade["grade"] == "fail":
            print(f"\n✗ FAIL — played {gemma_action} "
                  f"(E[Q]={grade.get('gemma_eq', '?'):.1f}) "
                  f"but bot played {ex['bot_action']} "
                  f"(E[Q]={grade.get('bot_eq', '?'):.1f})", file=sys.stderr)
        elif grade["grade"] == "illegal":
            print(f"\n✗ ILLEGAL — played {gemma_action}, "
                  f"not in {ex['legal_actions']}", file=sys.stderr)
        else:
            print(f"\n✗ PARSE FAIL — couldn't extract action from response", file=sys.stderr)

    # Summary
    if len(indices) > 1:
        total = sum(stats.values())
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Summary: {total} examples", file=sys.stderr)
        print(f"  Pass:       {stats['pass']} ({100*stats['pass']/total:.0f}%)", file=sys.stderr)
        print(f"  Fail:       {stats['fail']} ({100*stats['fail']/total:.0f}%)", file=sys.stderr)
        print(f"  Illegal:    {stats['illegal']} ({100*stats['illegal']/total:.0f}%)", file=sys.stderr)
        print(f"  Parse fail: {stats['parse_fail']} ({100*stats['parse_fail']/total:.0f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
