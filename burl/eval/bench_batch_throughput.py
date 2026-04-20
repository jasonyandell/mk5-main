"""Benchmark MLX-LM batch_generate aggregate tok/s on real Burl prompts.

Sweeps batch size against real N=16 held-out decisions rendered through the
iter-3-rules prompt shape (rules-as-tools + primer + 42-framing; ~2378
tokens per prompt). Prints per-batch stats and stops when aggregate tok/s
regresses below PLATEAU_RATIO of the running peak.

Canonical result on M5 Max (48 GB, bf16, 2026-04-19):
  batch=16  -> 512 tok/s  (6.2x single-stream)
  batch=64  -> 1206 tok/s (14.5x; knee of the curve, recommended default)
  batch=128 -> 1334 tok/s (16.1x; aggregate peak, 15.5 GB peak mem)
  batch=256 -> 1309 tok/s (plateau)

Reproducer:
  PYTHONPATH=. python -u -m burl.eval.bench_batch_throughput
"""

from __future__ import annotations

import sys
import time
import traceback

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from burl.eval.decision_dataset import load_dataset
from burl.harness.agent_runner import _current_player, _visible_history
from burl.harness.agent_runner_native import (
    build_tool_schemas,
    render_native_messages,
)

MODEL_REPO = "mlx-community/gemma-4-e2b-it-bf16"
DATASET = "burl/eval/data/move4_decisions_n50.jsonl"
MAX_TOKENS = 128
BATCH_SIZES = [16, 32, 48, 64, 96, 128, 192, 256]
PLATEAU_RATIO = 0.90


def main() -> int:
    print(f"Loading {MODEL_REPO} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL_REPO)
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    print(f"Loading decisions from {DATASET} ...")
    decisions = load_dataset(DATASET)[:16]
    tool_schemas = build_tool_schemas(enable_rules_tools=True)

    def build_prompt(decision) -> list[int]:
        state = decision.game_state
        me_abs = _current_player(state)
        hand = [d for d in state.hands[me_abs] if d not in state.played]
        history = _visible_history(state)
        system, user = render_native_messages(
            state, hand, history,
            enable_rules_tools=True,
            enable_primer=True,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=tool_schemas,
        )
        return tokenizer.encode(rendered)

    base_prompts = [build_prompt(d) for d in decisions]
    lens = [len(p) for p in base_prompts]
    print(f"prompt lens mean={sum(lens)/len(lens):.0f} min={min(lens)} max={max(lens)}\n")

    sampler = make_sampler(temp=0.6)
    results: list[tuple[int, float, float, float, float]] = []
    best_agg_tps = 0.0

    for batch in BATCH_SIZES:
        prompts = [base_prompts[i % 16] for i in range(batch)]
        print(f"--- batch={batch} ---", flush=True)
        try:
            t0 = time.time()
            resp = batch_generate(
                model, tokenizer,
                prompts=prompts,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
                verbose=False,
                completion_batch_size=batch,
            )
            wall = time.time() - t0
        except Exception as e:
            print(f"  FAILED at batch={batch}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            break

        stats = resp.stats
        total_gen = stats.generation_tokens
        agg = total_gen / stats.generation_time if stats.generation_time > 0 else 0
        per_seq = agg / batch
        results.append((batch, wall, stats.peak_memory, agg, per_seq))
        print(
            f"  wall={wall:.2f}s  gen_tokens={total_gen}  "
            f"prompt_tps={stats.prompt_tps:.0f}  gen_tps={stats.generation_tps:.1f}  "
            f"aggregate={agg:.1f} tok/s  per_seq={per_seq:.1f} tok/s  "
            f"peak_mem={stats.peak_memory:.1f}GB",
            flush=True,
        )

        if agg > best_agg_tps:
            best_agg_tps = agg
        elif agg < PLATEAU_RATIO * best_agg_tps:
            print(
                f"  PLATEAU: aggregate {agg:.1f} < "
                f"{PLATEAU_RATIO:.0%} * peak {best_agg_tps:.1f} — stopping",
                flush=True,
            )
            break
        print()

    print("\n=== summary ===")
    print(f"{'batch':>6}  {'wall_s':>7}  {'peak_GB':>7}  {'agg_tps':>9}  {'per_seq':>8}")
    for batch, wall, mem, agg, per_seq in results:
        print(f"{batch:>6}  {wall:>7.2f}  {mem:>7.1f}  {agg:>9.1f}  {per_seq:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
