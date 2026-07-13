"""iter-2 training launcher — thin wrapper around star.py for the blended corpus.

Reuses the `train_iter0` remote function from `burl/train/star.py` (which is
recipe-neutral — it trains a LoRA adapter on any `{"messages": [...]}` JSONL
corpus) and supplies iter-2-specific defaults:

  - corpus: `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl`
            (33% short / 67% long blend from T4 — regenerate when the real
            iter-2 raw corpus lands and point `--corpus` at it)
  - adapter: `jasonyandell/gemma-4-e2b-texas42-burl-iter2` (private HF repo)
  - epochs / LR / rank / batch / grad-accum match iter-1's recipe verbatim

Invocation (team-lead-fire-once-T1-and-T3-land):

    modal run burl/train/star_iter2.py::main_iter2 --corpus <path>

A smoke variant is available via --smoke; it samples 5 rows, max_steps=10,
and pushes to the shared `-burl-smoke` repo instead of `-iter2`.

DO NOT import this module from notebooks / evaluation code — it only wires
CLI arguments into `train_iter0.remote(...)`. See star.py for the training
body itself.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from burl.train.star import SMOKE_REPO, app, train_iter0

DEFAULT_CORPUS = "scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl"
DEFAULT_ADAPTER_NAME = "gemma-4-e2b-texas42-burl-iter2"
ADAPTER_ORG = "jasonyandell"

# iter-1's recipe — copied verbatim so iter-2 is an apples-to-apples comparison.
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_RANK = 16
DEFAULT_BATCH = 2
DEFAULT_GRAD_ACCUM = 4


@app.local_entrypoint()
def main_iter2(
    corpus: str = DEFAULT_CORPUS,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    rank: int = DEFAULT_RANK,
    batch: int = DEFAULT_BATCH,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    smoke: bool = False,
    n_examples: int = 0,
) -> None:
    """Fire iter-2 SFT on the blended corpus.

    Params mirror the flags the team-lead asked for in T6.
    """
    path = Path(corpus)
    if not path.exists():
        print(f"[error] corpus not found: {path}", file=sys.stderr)
        sys.exit(1)

    lines = [
        line for line in path.read_text().strip().split("\n") if line.strip()
    ]
    print(f"[local] corpus: {path} ({len(lines)} entries)", file=sys.stderr)

    if smoke:
        rng = random.Random(42)
        sample = rng.sample(lines, 5)
        text = "\n".join(sample)
        adapter_repo = SMOKE_REPO
        run_name = f"{adapter_name}-smoke"
        epochs_use = 1
        max_steps = 10
        per_device_batch_size = 1
        gradient_accumulation_steps = 2
        lr_use = lr
        rank_use = rank
        print(
            f"[local] SMOKE mode: 5 examples, max_steps=10, "
            f"batch={per_device_batch_size}*grad_accum={gradient_accumulation_steps}, "
            f"push -> {adapter_repo}",
            file=sys.stderr,
        )
    else:
        if n_examples > 0 and n_examples < len(lines):
            rng = random.Random(42)
            sample = rng.sample(lines, n_examples)
            text = "\n".join(sample)
            print(
                f"[local] FULL mode: subsample {n_examples}/{len(lines)}",
                file=sys.stderr,
            )
        else:
            text = "\n".join(lines)
            print(
                f"[local] FULL mode: all {len(lines)} examples",
                file=sys.stderr,
            )
        adapter_repo = f"{ADAPTER_ORG}/{adapter_name}"
        run_name = adapter_name
        epochs_use = epochs
        max_steps = -1
        per_device_batch_size = batch
        gradient_accumulation_steps = grad_accum
        lr_use = lr
        rank_use = rank
        print(
            f"[local] epochs={epochs_use} lr={lr_use} rank={rank_use} "
            f"batch={per_device_batch_size}*grad_accum={gradient_accumulation_steps}, "
            f"push -> {adapter_repo}",
            file=sys.stderr,
        )

    result = train_iter0.remote(
        corpus_jsonl=text,
        adapter_repo=adapter_repo,
        wandb_run_name=run_name,
        epochs=epochs_use,
        max_steps=max_steps,
        lr=lr_use,
        lora_rank=rank_use,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print(f"\n{'='*60}", file=sys.stderr)
    print("RESULT", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(json.dumps(result, indent=2), file=sys.stderr)
