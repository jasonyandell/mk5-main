"""iter-4-thoughts training launcher — preserve_thoughts SFT on iter-3-rules corpus.

Reuses ``train_iter0`` from ``burl/train/star.py`` but sets
``preserve_thoughts=True`` so Gemma 4's ``strip_thinking()`` macro is
bypassed via a ``formatting_func``. Thought blocks inside the assistant
content reach the SFT loss function; labels on those positions are not
masked. See ``burl/train/star.py`` module docstring and
``burl/train/test_formatting_func.py`` for the empirical pin.

Scientific question this unlocks (answered downstream in a separate eval
task): with thoughts visible to the loss, does iter-4 trained on the
iter-3-rules corpus develop a richer reasoning style at inference, vs.
iter-3-rules's current tool-call-first output? iter-4-thoughts does NOT
rebuild the corpus — it simply re-trains the iter-3-rules raw data with
thoughts visible.

Defaults:

  - corpus: ``burl/data/star_iter3_rules_corpus.jsonl`` (from T12).
  - adapter: ``jasonyandell/gemma-4-e2b-texas42-burl-iter4-thoughts``.
  - epochs / LR / rank / batch / grad-accum match iter-1/iter-2/iter-3-v2/
    iter-3-rules verbatim so the variable under test is ONLY the
    ``preserve_thoughts`` flag.

Invocation (team-lead-greenlit, not yet fired):

    modal run burl/train/star_iter4_thoughts.py::main_iter4_thoughts

A smoke variant (``--smoke``) samples 5 rows, max_steps=10, and pushes to
the shared ``-burl-smoke`` repo instead of ``-iter4-thoughts``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from burl.train.star import SMOKE_REPO, app, train_iter0

DEFAULT_CORPUS = "burl/data/star_iter3_rules_corpus.jsonl"
DEFAULT_ADAPTER_NAME = "gemma-4-e2b-texas42-burl-iter4-thoughts"
ADAPTER_ORG = "jasonyandell"

DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_RANK = 16
DEFAULT_BATCH = 2
DEFAULT_GRAD_ACCUM = 4


@app.local_entrypoint()
def main_iter4_thoughts(
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
    """Fire iter-4-thoughts SFT with preserve_thoughts=True."""
    path = Path(corpus)
    if not path.exists():
        print(f"[error] corpus not found: {path}", file=sys.stderr)
        sys.exit(1)

    lines = [
        line for line in path.read_text().strip().split("\n") if line.strip()
    ]
    print(f"[local] corpus: {path} ({len(lines)} entries)", file=sys.stderr)
    print(
        "[local] preserve_thoughts=True (formatting_func bypass of strip_thinking)",
        file=sys.stderr,
    )

    if smoke:
        rng = random.Random(42)
        sample = rng.sample(lines, min(5, len(lines)))
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
            f"[local] SMOKE mode: {len(sample)} examples, max_steps=10, "
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
        preserve_thoughts=True,
    )

    print(f"\n{'='*60}", file=sys.stderr)
    print("RESULT", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(json.dumps(result, indent=2), file=sys.stderr)
