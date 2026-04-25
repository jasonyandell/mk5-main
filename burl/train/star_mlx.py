"""Local MLX-LM LoRA trainer for Burl STaR (Apple Silicon M5 Max).

Port of ``burl/train/star.py`` (Modal+B200+PEFT+SFTTrainer) to on-device
mlx-lm. Targets ``mlx-community/gemma-4-e2b-it-bf16`` by default. Saves an
adapter dir (``adapters.safetensors`` + ``adapter_config.json``) compatible
with ``mlx_lm.load(adapter_path=...)``.

Load-bearing detail: ``preserve_thoughts=True``. Gemma 4's chat template
contains a ``strip_thinking(text)`` jinja macro that erases every
``<|channel>thought ... <channel|>`` region from assistant content before
tokenization. ``ChatDataset`` calls ``apply_chat_template`` so it invokes
``strip_thinking`` too. ``PreserveThoughtsDataset`` bypasses the chat
template on the assistant side -- pin in ``test_star_mlx.py``.

mlx-lm quirks (0.31.2):
  * ``tuner.utils.load_adapters`` reads ``config.num_layers``. We write
    ``num_layers`` (functional) + ``lora_layers`` (task-spec alias).
  * ``iterate_batches`` refuses ``len(dataset) < batch_size``; we clamp.
  * ``steps_per_report=1`` so the callback fires every iter (star.py parity
    with ``SFTConfig(logging_steps=1)``).
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GEMMA4_TURN_TERMINATOR = "<turn|>\n"
DEFAULT_MODEL_REPO = "mlx-community/gemma-4-e2b-it-bf16"

# Module subpaths inside each DecoderLayer (attention + MLP). Matches
# star.py:222-228 target_modules.
LORA_TARGET_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


class PreserveThoughtsDataset:
    """Chat dataset that bypasses Gemma 4's strip_thinking on the assistant side.

    Duck-typed like ``mlx_lm.tuner.datasets.ChatDataset``: ``__len__``,
    ``__getitem__``, ``process(d) -> (tokens, offset)``.

    For each row ``{"messages": [{user}, {assistant}]}``:
      1. User turn via ``apply_chat_template(add_generation_prompt=True)``
         -- canonical ``<bos><|turn>user\\n...<turn|>\\n<|turn>model\\n`` prefix.
      2. Assistant ``content`` appended verbatim (no apply_chat_template ->
         no strip_thinking).
      3. ``<turn|>\\n`` terminator so the model learns when to stop.
      4. Single ``encode(..., add_special_tokens=False)`` pass.

    Returns ``(tokens, 0)``: every non-pad token is trainable (thought tokens
    included). Matches preserve_thoughts=True in star.py.
    """

    def __init__(self, data: list[dict[str, Any]], tokenizer) -> None:
        self._data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._data[idx]

    def process(self, d: dict[str, Any]) -> tuple[list[int], int]:
        msgs = d["messages"]
        if (
            len(msgs) != 2
            or msgs[0].get("role") != "user"
            or msgs[1].get("role") != "assistant"
        ):
            raise ValueError(
                "preserve_thoughts dataset expects [user, assistant]; got "
                f"roles={[m.get('role') for m in msgs]!r}"
            )
        user_prefix = self.tokenizer.apply_chat_template(
            [msgs[0]], tokenize=False, add_generation_prompt=True
        )
        text = user_prefix + msgs[1]["content"] + GEMMA4_TURN_TERMINATOR
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return (tokens, 0)


class _EarlyStopSignal(Exception):
    """Raised from the val-loss callback to abort the train loop early.

    Caught by ``train_mlx`` so the adapter is still saved (with the
    best-checkpoint snapshot if one was captured) and ``early_stop`` is
    recorded in ``adapter_config.json``.
    """


@dataclass
class _TrajectoryCollector:
    """mlx-lm TrainingCallback that accumulates per-iter training loss
    and (optionally) tracks val loss + early-stop + best-checkpoint snapshot.
    """

    losses: list[dict[str, float]]
    val_losses: list[dict[str, float]] = field(default_factory=list)
    best_val_loss: float = math.inf
    best_val_iter: int = -1
    best_params: dict | None = None
    over_threshold_streak: int = 0

    # When set, the callback also snapshots model.trainable_parameters at
    # each new best-val-loss and aborts via _EarlyStopSignal once val_loss
    # has stayed > best_val_loss * early_stop_val_rise for
    # early_stop_patience consecutive evals.
    model: Any | None = None
    early_stop_val_rise: float | None = None
    early_stop_patience: int = 2

    def on_train_loss_report(self, train_info: dict) -> None:
        self.losses.append(
            {
                "step": int(train_info["iteration"]),
                "loss": float(train_info["train_loss"]),
                "lr": float(train_info.get("learning_rate", 0.0)),
            }
        )

    def on_val_loss_report(self, val_info: dict) -> None:
        # mlx-lm passes iteration-1 in val_info; use as-is so the val and
        # train series share the same iteration axis.
        step = int(val_info["iteration"])
        val_loss = float(val_info["val_loss"])
        self.val_losses.append({"step": step, "val_loss": val_loss})

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_iter = step
            self.over_threshold_streak = 0
            if self.model is not None:
                # Snapshot trainable params via tree_flatten so the dict is
                # stable and serializable with mx.save_safetensors later.
                from mlx.utils import tree_flatten

                self.best_params = dict(tree_flatten(self.model.trainable_parameters()))
            return

        # val_loss did not improve; check early-stop trigger.
        if self.early_stop_val_rise is None:
            return
        threshold = self.best_val_loss * float(self.early_stop_val_rise)
        if val_loss > threshold:
            self.over_threshold_streak += 1
            print(
                f"[early-stop] val {val_loss:.4f} > {threshold:.4f} "
                f"(best={self.best_val_loss:.4f} @ iter {self.best_val_iter}); "
                f"streak={self.over_threshold_streak}/{self.early_stop_patience}",
                flush=True,
            )
            if self.over_threshold_streak >= self.early_stop_patience:
                raise _EarlyStopSignal(
                    f"val_loss > {self.early_stop_val_rise}x best for "
                    f"{self.over_threshold_streak} consecutive evals; "
                    f"best={self.best_val_loss:.4f} @ iter {self.best_val_iter}"
                )
        else:
            # Streak only counts consecutive evals above threshold.
            self.over_threshold_streak = 0


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_dataset(rows, tokenizer, preserve_thoughts: bool):
    if preserve_thoughts:
        return PreserveThoughtsDataset(rows, tokenizer)
    from mlx_lm.tuner.datasets import ChatDataset

    return ChatDataset(rows, tokenizer, chat_key="messages", mask_prompt=False)


def _build_lora_model(model, rank: int, dropout: float) -> int:
    """Freeze base model, wrap target linear layers as LoRA. Return num_layers."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    n_layers = len(model.layers)
    linear_to_lora_layers(
        model,
        num_layers=n_layers,
        config={
            "rank": rank,
            "scale": 2.0 * rank,
            "dropout": dropout,
            "keys": LORA_TARGET_KEYS,
        },
    )
    return n_layers


def _build_optimizer(lr: float, total_iters: int, warmup_ratio: float = 0.1):
    """AdamW with linear-warmup -> cosine-decay. Matches SFTConfig in star.py."""
    import mlx.optimizers as optim
    from mlx.optimizers import schedulers

    warmup_steps = max(1, int(total_iters * warmup_ratio))
    decay_steps = max(1, total_iters - warmup_steps)
    warmup_fn = schedulers.linear_schedule(0.0, lr, warmup_steps)
    cosine_fn = schedulers.cosine_decay(lr, decay_steps, end=0.0)
    schedule = schedulers.join_schedules([warmup_fn, cosine_fn], [warmup_steps + 1])
    return optim.AdamW(learning_rate=schedule)


def _compute_iters(
    n_rows: int,
    epochs: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
) -> int:
    """Total iters (grad-accum micro-steps). ``max_steps`` is in update units."""
    if max_steps and max_steps > 0:
        return max_steps * gradient_accumulation_steps
    batches_per_epoch = max(1, n_rows // per_device_batch_size)
    return max(gradient_accumulation_steps, epochs * batches_per_epoch)


def train_mlx(
    corpus_path: Path,
    adapter_out_dir: Path,
    model_repo: str = DEFAULT_MODEL_REPO,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_rank: int = 16,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    preserve_thoughts: bool = False,
    max_steps: int = -1,
    seed: int = 42,
    max_seq_length: int = 4096,
    val_corpus_path: Path | None = None,
    steps_per_eval: int = 50,
    val_batches: int = -1,
    early_stop_val_rise: float | None = None,
    early_stop_patience: int = 2,
) -> dict:
    """Train a LoRA adapter on a JSONL chat corpus."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import print_trainable_parameters

    corpus_path = Path(corpus_path)
    adapter_out_dir = Path(adapter_out_dir)
    adapter_out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_corpus(corpus_path)
    if not rows:
        raise ValueError(f"corpus is empty: {corpus_path}")
    print(f"[data] {len(rows)} rows from {corpus_path}", flush=True)

    mx.random.seed(seed)
    print(f"[model] loading {model_repo}...", flush=True)
    t_load = time.time()
    model, tokenizer = mlx_load(model_repo)
    print(f"[model] loaded in {time.time() - t_load:.1f}s", flush=True)

    # Unwrap TokenizerWrapper -- we want raw HF tokenizer for add_special_tokens=False.
    raw_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    dataset = _build_dataset(rows, raw_tok, preserve_thoughts)

    val_rows: list[dict[str, Any]] = []
    val_dataset_built = None
    if val_corpus_path is not None:
        val_corpus_path = Path(val_corpus_path)
        val_rows = _load_corpus(val_corpus_path)
        if not val_rows:
            raise ValueError(f"val corpus is empty: {val_corpus_path}")
        print(f"[data] {len(val_rows)} val rows from {val_corpus_path}", flush=True)
        val_dataset_built = _build_dataset(val_rows, raw_tok, preserve_thoughts)

    effective_batch = min(per_device_batch_size, len(dataset))
    if effective_batch < per_device_batch_size:
        print(
            f"[warn] batch {per_device_batch_size} > rows {len(dataset)}; "
            f"clamping to {effective_batch}",
            flush=True,
        )
    total_iters = _compute_iters(
        n_rows=len(rows),
        epochs=epochs,
        per_device_batch_size=effective_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
    )
    print(
        f"[train] iters={total_iters} epochs={epochs} rows={len(rows)} "
        f"batch={effective_batch} grad_accum={gradient_accumulation_steps}",
        flush=True,
    )
    n_layers = _build_lora_model(model, rank=lora_rank, dropout=0.05)
    print_trainable_parameters(model)
    optimizer = _build_optimizer(lr=lr, total_iters=total_iters)

    adapter_file = adapter_out_dir / "adapters.safetensors"

    # Auto-size val_batches: cover the val set in one pass, capped at 50 so
    # an eval call doesn't dominate wall time on a large val_dataset.
    if val_dataset_built is not None:
        if val_batches <= 0:
            val_batches = max(1, min(50, math.ceil(len(val_rows) / max(1, effective_batch))))
        eval_steps = steps_per_eval
        print(
            f"[val] val_batches={val_batches} steps_per_eval={eval_steps} "
            f"early_stop_val_rise={early_stop_val_rise} patience={early_stop_patience}",
            flush=True,
        )
    else:
        val_batches = 0
        eval_steps = 10**9

    training_args = TrainingArgs(
        batch_size=effective_batch,
        iters=total_iters,
        val_batches=val_batches,
        steps_per_report=1,
        steps_per_eval=eval_steps,
        steps_per_save=10**9,
        max_seq_length=max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=True,
        grad_accumulation_steps=gradient_accumulation_steps,
    )

    callback = _TrajectoryCollector(
        losses=[],
        model=model if early_stop_val_rise is not None else None,
        early_stop_val_rise=early_stop_val_rise,
        early_stop_patience=early_stop_patience,
    )
    early_stopped = False
    t_train = time.time()
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=CacheDataset(dataset),
            val_dataset=CacheDataset(val_dataset_built) if val_dataset_built is not None else None,
            args=training_args,
            training_callback=callback,
        )
    except _EarlyStopSignal as exc:
        early_stopped = True
        print(f"[early-stop] {exc}", flush=True)
    elapsed = time.time() - t_train

    # Save best-checkpoint snapshot if one was captured (early-stop with
    # val tracking); otherwise save current weights. Defensive final save:
    # steps_per_save semantics vary across mlx-lm versions.
    if callback.best_params is not None:
        print(
            f"[save] writing best-val checkpoint (iter {callback.best_val_iter}, "
            f"val_loss {callback.best_val_loss:.4f})",
            flush=True,
        )
        adapter_weights = callback.best_params
    else:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_file), adapter_weights)

    # `num_layers` + `lora_parameters` are what load_adapters actually reads.
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": n_layers,
        "lora_layers": n_layers,  # alias for task spec
        "lora_parameters": {
            "rank": lora_rank,
            "scale": 2.0 * lora_rank,
            "dropout": 0.05,
            "keys": LORA_TARGET_KEYS,
        },
        "num_epochs": epochs,
        "preserve_thoughts": preserve_thoughts,
        "lr": lr,
        "batch_size": effective_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model": model_repo,
        "n_train_rows": len(rows),
        "n_val_rows": len(val_rows),
        "seed": seed,
        "early_stop": early_stopped,
        "best_val_iter": callback.best_val_iter,
        "best_val_loss": (
            callback.best_val_loss if math.isfinite(callback.best_val_loss) else None
        ),
        "early_stop_val_rise": early_stop_val_rise,
        "early_stop_patience": early_stop_patience,
    }
    with open(adapter_out_dir / "adapter_config.json", "w") as fid:
        json.dump(adapter_config, fid, indent=2)

    loss_trajectory = callback.losses
    val_loss_trajectory = callback.val_losses
    final_loss = loss_trajectory[-1]["loss"] if loss_trajectory else math.nan
    n_steps = int(total_iters // gradient_accumulation_steps)

    result = {
        "adapter_path": str(adapter_out_dir),
        "n_rows": len(rows),
        "n_val_rows": len(val_rows),
        "n_steps": n_steps,
        "n_iters": total_iters,
        "final_loss": final_loss,
        "loss_trajectory": loss_trajectory,
        "val_loss_trajectory": val_loss_trajectory,
        "early_stop": early_stopped,
        "best_val_iter": callback.best_val_iter,
        "best_val_loss": (
            callback.best_val_loss if math.isfinite(callback.best_val_loss) else None
        ),
        "train_seconds": round(elapsed, 1),
    }
    print(
        f"[done] iters={total_iters} steps={n_steps} "
        f"final_loss={final_loss:.4f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


# --- CLI --------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLX-LM LoRA trainer for Burl STaR.")
    p.add_argument("--corpus", type=Path, help="JSONL chat corpus path")
    p.add_argument("--adapter-out", type=Path, help="Adapter output dir")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_REPO)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--preserve-thoughts", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--val-corpus", type=Path, default=None,
                   help="Optional JSONL val corpus; enables periodic eval.")
    p.add_argument("--steps-per-eval", type=int, default=50,
                   help="Iters between val-loss evals (in micro-steps).")
    p.add_argument("--val-batches", type=int, default=-1,
                   help="Batches per eval; -1 = auto-cover val set, capped 50.")
    p.add_argument("--early-stop-val-rise", type=float, default=None,
                   help="Abort when val_loss > this multiple of best (e.g. 1.3). "
                        "Off by default.")
    p.add_argument("--early-stop-patience", type=int, default=2,
                   help="Consecutive evals above threshold required to abort.")
    p.add_argument("--smoke", action="store_true", help="In-memory 5-row smoke test.")
    return p


def _smoke_rows() -> list[dict[str, Any]]:
    """5 synthetic rows with thought regions (same shape as test fixtures)."""
    thought = "<|channel>thought\nSMOKE_THOUGHT_{i}\nPick the best play.\n<channel|>"
    tool = (
        "<|tool_call>call:trump_declared{{}}<tool_call|>"
        "<|tool_response>{{\"declaration\":\"sevens\"}}<tool_response|>"
        "<|tool_call>call:commit_play{{domino_id:{did}}}<tool_call|>"
    )
    return [
        {
            "messages": [
                {"role": "user", "content": f"Game {i}: pick the next play."},
                {
                    "role": "assistant",
                    "content": thought.format(i=i) + tool.format(did=20 + i),
                },
            ]
        }
        for i in range(5)
    ]


def _run_smoke() -> int:
    print("[smoke] building 5-row in-memory corpus", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        corpus = td_path / "smoke.jsonl"
        with open(corpus, "w") as fid:
            for r in _smoke_rows():
                fid.write(json.dumps(r) + "\n")
        adapter_dir = td_path / "adapter"
        result = train_mlx(
            corpus_path=corpus,
            adapter_out_dir=adapter_dir,
            epochs=1,
            lora_rank=4,
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            preserve_thoughts=True,
            max_steps=10,
        )
        traj = result["loss_trajectory"]
        assert len(traj) >= 5, f"[smoke] trajectory too short: {len(traj)}"
        loss_start, loss_end = traj[0]["loss"], traj[-1]["loss"]
        assert loss_end < loss_start, (
            f"[smoke] loss did not decrease: {loss_start:.4f}->{loss_end:.4f}"
        )
        print(
            f"[smoke] loss start={loss_start:.4f} -> end={loss_end:.4f} "
            f"({len(traj)} pts, {result['train_seconds']}s)",
            flush=True,
        )
        from mlx_lm import load as mlx_load

        mlx_load(DEFAULT_MODEL_REPO, adapter_path=str(adapter_dir))
        print("[smoke] adapter round-trip OK via mlx_lm.load(adapter_path=...)")
    print("[smoke] PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _run_smoke()
    if args.corpus is None or args.adapter_out is None:
        parser.error("--corpus and --adapter-out are required unless --smoke")
    result = train_mlx(
        corpus_path=args.corpus,
        adapter_out_dir=args.adapter_out,
        model_repo=args.model,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.rank,
        per_device_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        preserve_thoughts=args.preserve_thoughts,
        max_steps=args.max_steps,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        val_corpus_path=args.val_corpus,
        steps_per_eval=args.steps_per_eval,
        val_batches=args.val_batches,
        early_stop_val_rise=args.early_stop_val_rise,
        early_stop_patience=args.early_stop_patience,
    )
    summary = {k: v for k, v in result.items()
               if k not in {"loss_trajectory", "val_loss_trajectory"}}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
