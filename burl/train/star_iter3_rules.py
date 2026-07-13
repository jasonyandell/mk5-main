"""iter-3-rules training launcher — thin wrapper around star.py.

Reuses ``train_iter0`` from ``burl/train/star.py`` (recipe-neutral SFT on any
``{"messages": [...]}`` JSONL corpus) and supplies iter-3-rules defaults:

  - corpus: ``burl/data/star_iter3_rules_corpus.jsonl`` (produced by
    ``run_move4_star_rollout --enable-rules-tools`` + EQ-gate, per T12)
  - adapter: ``jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules``
  - recipe: matches iter-1/iter-2 verbatim so iter-3-rules is an
    apples-to-apples prompt-shape A/B against them.

Invocation (team-lead-greenlit):

    modal run burl/train/star_iter3_rules.py::main_iter3_rules --corpus <path>

A smoke variant (--smoke) samples 5 rows, max_steps=10, and pushes to the
shared ``-burl-smoke`` repo instead of ``-iter3-rules``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from burl.train.star import SMOKE_REPO, app, train_iter0

DEFAULT_CORPUS = "burl/data/star_iter3_rules_corpus.jsonl"
DEFAULT_ADAPTER_NAME = "gemma-4-e2b-texas42-burl-iter3-rules"
ADAPTER_ORG = "jasonyandell"

DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_RANK = 16
DEFAULT_BATCH = 2
DEFAULT_GRAD_ACCUM = 4


@app.local_entrypoint()
def main_iter3_rules(
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
    """Fire iter-3-rules SFT on the rules-as-tools corpus.

    The corpus must have been generated with
    ``run_move4_star_rollout --enable-rules-tools`` so the system-prompt
    shape matches what eval-time will send.
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
