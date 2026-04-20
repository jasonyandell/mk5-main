"""LoRA fine-tune skeleton for Qwen3.6-35B-A3B-4bit on the Candlewax corpus.

Research writeup: scratch/candlewax_spike/mlx_lora_research.md

This module is intentionally a scaffold:
  * import-safe on any machine that has mlx-vlm installed (heavy imports are
    lazy inside ``run_training`` so ``--dry-run`` and unit-import work cold)
  * ``--dry-run`` is the default; the actual training call is gated behind
    ``--execute``
  * The inner call to ``mlx_vlm.trainer.sft_trainer.train`` is left in place
    but the script will refuse to invoke it unless ``--execute`` is passed
    AND the corpus file exists.

Design choices match ``scratch/candlewax_spike/mlx_lora_research.md`` §5-6:
  * Model:  mlx-community/Qwen3.6-35B-A3B-4bit (model_type=qwen3_5_moe)
  * LoRA:   rank 8, alpha 16, dropout 0.0  (mlx-vlm defaults)
  * Targets: all nn.Linear/nn.QuantizedLinear in language_model excl. lm_head
             => attention q/k/v/o, shared-expert MLP, MoE router (gate),
                shared_expert_gate.  128-way expert bank (SwitchGLU) stays
                frozen — see research doc §2 + §6.
  * Budget: batch=1, grad-accum=4, grad-checkpoint, max_seq_length=2048,
            iters=50, lr=1e-5  (fits the 48 GB unified-memory envelope on M5)
  * Loss:   train_on_completions=True; assistant_id is resolved from the Qwen
            tokenizer at runtime (the 77091 default is Llama, not Qwen — see
            research doc §6 gotcha #3).

Corpus path: scratch/candlewax_spike/star_corpus_v1.jsonl
  Our existing rows have keys
    {decision_idx, mode, seed, image_path, system_prompt, user_prompt,
     completion, filter_metadata}
  — ``_star_row_to_messages`` adapts that to the
  ``{images, messages}`` shape mlx-vlm's VisionDataset expects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --- repo-relative defaults ---------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
DEFAULT_CORPUS = REPO_ROOT / "scratch" / "candlewax_spike" / "star_corpus_v1.jsonl"
DEFAULT_ADAPTER_DIR = REPO_ROOT / "scratch" / "candlewax_spike" / "qwen_adapter"


# --- config -------------------------------------------------------------------


@dataclass(frozen=True)
class LoraConfig:
    """LoRA hyperparameters.  Defaults match research doc §5 recommendation."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    """Training loop hyperparameters.

    Values tuned to fit 35B-A3B-4bit in 48 GB unified memory on an M5 Max.
    """

    model_path: str = DEFAULT_MODEL
    corpus_path: Path = DEFAULT_CORPUS
    adapter_dir: Path = DEFAULT_ADAPTER_DIR
    iters: int = 50
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_seq_length: int = 2048
    learning_rate: float = 1e-5
    grad_clip: float | None = 1.0
    steps_per_report: int = 5
    steps_per_save: int = 25
    grad_checkpoint: bool = True
    train_on_completions: bool = True


# --- corpus adapter -----------------------------------------------------------


def _star_row_to_messages(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one star_corpus_v1.jsonl row to mlx-vlm VisionDataset shape.

    Our rows are image-bearing; Qwen3.5-MoE uses embedded-image chat
    templates, so we pass the image both in the 'images' column (for
    load_dataset) and inside the user message as a content part.
    """

    image_path = row.get("image_path")
    user_text = row["user_prompt"]
    system_text = row["system_prompt"]
    completion_text = row["completion"]

    # Keep ALL message content as plain strings so pyarrow can serialize the
    # batch into a HF Dataset (mixed list/string content across rows causes
    # `ArrowInvalid: cannot mix list and non-list`). Qwen3.6's chat template
    # will inject the image placeholder at the user turn from the separate
    # "images" column; we don't need to embed a content-list here.
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": completion_text},
    ]

    out: dict[str, Any] = {"messages": messages}
    if image_path:
        out["images"] = [str(image_path)]
    return out


def load_corpus_as_messages(path: Path) -> list[dict[str, Any]]:
    """Read star_corpus_v1.jsonl and return a list of VisionDataset rows."""

    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(_star_row_to_messages(json.loads(line)))
    return rows


# --- training -----------------------------------------------------------------


def _resolve_qwen_assistant_id(tokenizer: Any) -> int:
    """Return the token id for the Qwen assistant-start marker.

    Falls back to the mlx-vlm default (77091 = Llama) if the Qwen marker
    isn't in the vocab, which would indicate we loaded the wrong model.
    """

    for marker in ("<|im_start|>assistant", "<|im_start|>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if ids:
            return int(ids[0])
    return 77091


def run_training(train_cfg: TrainConfig, lora_cfg: LoraConfig) -> None:
    """Execute the LoRA fine-tune.  Heavy imports deliberately lazy.

    This function is the only path that actually touches GPU/MLX; CLI
    ``--dry-run`` (the default) short-circuits before this is called.
    """

    # Lazy imports — keeps ``python qwen_lora_train.py --dry-run`` cheap and
    # keeps the module importable on machines without mlx-vlm.
    import mlx.optimizers as optim  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from mlx_vlm.trainer.datasets import VisionDataset  # noqa: PLC0415
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train  # noqa: PLC0415
    from mlx_vlm.trainer.utils import (  # noqa: PLC0415
        find_all_linear_names,
        get_peft_model,
        print_trainable_parameters,
    )
    from mlx_vlm.utils import load  # noqa: PLC0415

    if not train_cfg.corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {train_cfg.corpus_path}. "
            "Populate it before running with --execute."
        )

    print(f"[load] model={train_cfg.model_path}")
    model, processor = load(
        train_cfg.model_path,
        processor_config={"trust_remote_code": True},
    )

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    print(f"[load] model_type={model_type!r}")
    if model_type != "qwen3_5_moe":
        print(
            f"[warn] expected qwen3_5_moe, got {model_type!r} — "
            "proceeding anyway.",
            file=sys.stderr,
        )

    # Dataset — build in-memory so we don't need load_dataset's CLI wiring.
    rows = load_corpus_as_messages(train_cfg.corpus_path)
    print(f"[data] rows={len(rows)}")
    hf_dataset = Dataset.from_list(rows)
    train_dataset = VisionDataset(
        hf_dataset,
        model.config.__dict__,
        processor,
        image_resize_shape=None,
    )

    # LoRA wrap — see research doc §2 for which linears this catches.
    modules = find_all_linear_names(model.language_model)
    print(f"[lora] target module suffixes={sorted(modules)}")
    model = get_peft_model(
        model,
        modules,
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        verbose=False,
    )
    print_trainable_parameters(model)

    optimizer = optim.Adam(learning_rate=train_cfg.learning_rate)

    tokenizer = getattr(processor, "tokenizer", processor)
    assistant_id = _resolve_qwen_assistant_id(tokenizer)
    print(f"[loss] assistant_id={assistant_id}")

    train_cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = train_cfg.adapter_dir / "adapters.safetensors"

    training_args = TrainingArgs(
        batch_size=train_cfg.batch_size,
        iters=train_cfg.iters,
        steps_per_report=train_cfg.steps_per_report,
        steps_per_eval=10**9,  # we have no val set — skip eval
        steps_per_save=train_cfg.steps_per_save,
        val_batches=0,
        max_seq_length=train_cfg.max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=train_cfg.grad_checkpoint,
        learning_rate=train_cfg.learning_rate,
        grad_clip=train_cfg.grad_clip,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        full_finetune=False,
    )

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        train_on_completions=train_cfg.train_on_completions,
        assistant_id=assistant_id,
    )

    print(f"[done] adapter saved at {adapter_file}")


# --- CLI ----------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tune Qwen3.6-35B-A3B-4bit on the Candlewax corpus. "
            "Dry-run by default; pass --execute to actually train."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the config and exit without loading the model (default).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run training.  Overrides --dry-run.",
    )
    return parser


def _summarize(train_cfg: TrainConfig, lora_cfg: LoraConfig) -> None:
    print("=== qwen_lora_train — planned run ===")
    print(f"  model        : {train_cfg.model_path}")
    print(f"  corpus       : {train_cfg.corpus_path}")
    print(f"  corpus exists: {train_cfg.corpus_path.exists()}")
    print(f"  adapter dir  : {train_cfg.adapter_dir}")
    print(f"  lora         : rank={lora_cfg.rank} alpha={lora_cfg.alpha} "
          f"dropout={lora_cfg.dropout}")
    print(f"  iters        : {train_cfg.iters}")
    print(f"  batch/accum  : {train_cfg.batch_size} / "
          f"{train_cfg.grad_accum_steps}")
    print(f"  max seq len  : {train_cfg.max_seq_length}")
    print(f"  lr           : {train_cfg.learning_rate}")
    print(f"  grad ckpt    : {train_cfg.grad_checkpoint}")
    print(f"  loss mask    : train_on_completions="
          f"{train_cfg.train_on_completions}")
    print("=====================================")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    lora_cfg = LoraConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    train_cfg = TrainConfig(
        model_path=args.model,
        corpus_path=args.corpus,
        adapter_dir=args.adapter_dir,
        iters=args.iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        learning_rate=args.lr,
    )

    _summarize(train_cfg, lora_cfg)

    if args.execute:
        run_training(train_cfg, lora_cfg)
        return 0

    # --dry-run path
    print("[dry-run] no model load, no training. Pass --execute to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
