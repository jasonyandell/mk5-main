"""Phase 3: Train LoRA adapter on B200 for Burl STaR iter0 corpus.

Modal app that loads `burl/data/star_iter0_corpus.jsonl`, fine-tunes a LoRA
adapter on top of Gemma 4 E2B, and pushes it to HuggingFace as
`jasonyandell/gemma-4-e2b-texas42-burl-iter0` (private).

Usage:
    # Smoke: 5 examples, max_steps=10, push to -burl-smoke repo
    modal run burl/train/star.py --smoke

    # Full run on all 50 corpus entries, push to -burl-iter0 repo
    modal run burl/train/star.py

The chat-format corpus already has `<|tool_call>` / `<|tool_response>` /
`<|channel>` special tokens in assistant content; the Gemma 4 tokenizer
round-trips these as single tokens, so SFTTrainer's apply_chat_template
flow preserves them end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_REPO = "jasonyandell/gemma-4-e2b-texas42-burl-iter0"
SMOKE_REPO = "jasonyandell/gemma-4-e2b-texas42-burl-smoke"
GPU_TYPE = "B200"

app = modal.App("burl-star-train")

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "torch>=2.6",
        "transformers==5.5.0",
        "accelerate>=1.2",
        "peft>=0.14",
        "trl>=0.15",
        "datasets>=3.0",
        "huggingface_hub>=0.27",
        "wandb>=0.19",
        "pillow",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def _patch_clippable_linear():
    """Patch Gemma4ClippableLinear for PEFT compatibility (training only).

    Mirrors lem/gemma_star/star_loop.py::_patch_clippable_linear.
    """
    import torch
    try:
        from transformers.models.gemma4 import modeling_gemma4

        class PatchedClippableLinear(torch.nn.Linear):
            def __init__(self, config, in_features, out_features):
                torch.nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
                if self.use_clipped_linears:
                    self.register_buffer("input_min", torch.tensor(-float("inf")))
                    self.register_buffer("input_max", torch.tensor(float("inf")))
                    self.register_buffer("output_min", torch.tensor(-float("inf")))
                    self.register_buffer("output_max", torch.tensor(float("inf")))

            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = torch.nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out

        modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
        print("[patch] ClippableLinear patched", flush=True)
    except ImportError:
        pass


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    timeout=7200,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-api-key"),
    ],
    volumes={
        "/model-cache": modal.Volume.from_name(
            "gemma-e2b-cache", create_if_missing=True
        )
    },
)
def train_iter0(
    corpus_jsonl: str,
    adapter_repo: str,
    wandb_run_name: str,
    epochs: int = 1,
    max_steps: int = -1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
) -> dict:
    """Train a LoRA adapter on the Burl STaR corpus and push to HuggingFace."""
    import gc
    import os
    import time

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    raw = [
        json.loads(line)
        for line in corpus_jsonl.strip().split("\n")
        if line.strip()
    ]
    traces = [{"messages": t["messages"]} for t in raw]
    print(f"[data] {len(traces)} training traces loaded", flush=True)
    if traces:
        asst = traces[0]["messages"][-1]["content"]
        print(
            f"[data] first assistant content: {len(asst)} chars, "
            f"special tokens present: "
            f"channel={'<|channel>' in asst} "
            f"tool_call={'<|tool_call>' in asst} "
            f"tool_response={'<|tool_response>' in asst}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _patch_clippable_linear()

    t_load = time.time()
    print(f"[model] Loading {MODEL_ID} (bf16, sdpa)...", flush=True)
    train_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"[model] Loaded in {time.time()-t_load:.0f}s", flush=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    train_model = get_peft_model(train_model, lora_config)
    train_model.print_trainable_parameters()

    dataset = Dataset.from_list(traces).shuffle(seed=42)

    wandb.init(project="burl-star", name=wandb_run_name, reinit=True)

    training_args = SFTConfig(
        output_dir="/tmp/burl-star-train",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        seed=42,
    )

    trainer = SFTTrainer(
        model=train_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    t_train = time.time()
    result = trainer.train()
    train_elapsed = time.time() - t_train
    print(
        f"[train] Done: final loss={result.training_loss:.4f} "
        f"in {train_elapsed:.0f}s "
        f"({trainer.state.global_step} steps)",
        flush=True,
    )

    loss_trajectory = [
        {"step": log.get("step"), "loss": log["loss"]}
        for log in trainer.state.log_history
        if "loss" in log
    ]
    if loss_trajectory:
        print(
            f"[train] loss trajectory: "
            f"start={loss_trajectory[0]['loss']:.4f} "
            f"-> end={loss_trajectory[-1]['loss']:.4f}",
            flush=True,
        )

    print(f"[push] Pushing adapter -> {adapter_repo} (private)...", flush=True)
    train_model.push_to_hub(adapter_repo, private=True)
    tokenizer.push_to_hub(adapter_repo, private=True)
    print("[push] Done.", flush=True)

    n_steps = int(trainer.state.global_step)
    wandb.log({
        "train_loss_final": result.training_loss,
        "n_traces": len(traces),
        "train_seconds": train_elapsed,
        "n_steps": n_steps,
    })
    wandb.finish()

    del trainer, train_model, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "adapter_repo": adapter_repo,
        "n_traces": len(traces),
        "n_steps": n_steps,
        "training_loss": float(result.training_loss),
        "loss_start": float(loss_trajectory[0]["loss"]) if loss_trajectory else None,
        "loss_end": float(loss_trajectory[-1]["loss"]) if loss_trajectory else None,
        "train_seconds": round(train_elapsed),
        "loss_trajectory": loss_trajectory,
    }


@app.local_entrypoint()
def main(
    corpus_path: str = "burl/data/star_iter0_corpus.jsonl",
    smoke: bool = False,
    epochs: int = 3,
    n_examples: int = 0,
    adapter_suffix: str = "iter0",
):
    """Entrypoint: `modal run burl/train/star.py [--smoke] [--epochs N] [--corpus-path ...]`.

    Defaults to 3 epochs on the full corpus (50 entries × batch 2 × grad_accum 4
    = 6 steps/epoch; 3 epochs puts us in the ~18-step range the team-lead asked
    for).
    """
    import random
    import sys

    path = Path(corpus_path)
    if not path.exists():
        print(f"[error] corpus not found: {path}", file=sys.stderr)
        sys.exit(1)

    lines = [
        line for line in path.read_text().strip().split("\n") if line.strip()
    ]
    print(
        f"[local] corpus: {path} ({len(lines)} entries)",
        file=sys.stderr,
    )

    if smoke:
        rng = random.Random(42)
        sample = rng.sample(lines, 5)
        text = "\n".join(sample)
        adapter_repo = SMOKE_REPO
        run_name = "burl-iter0-smoke"
        epochs_use = 1
        max_steps = 10
        # Shrink batch for a 5-example smoke so each step is a real update.
        per_device_batch_size = 1
        gradient_accumulation_steps = 2
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
        adapter_repo = f"jasonyandell/gemma-4-e2b-texas42-burl-{adapter_suffix}"
        run_name = f"burl-{adapter_suffix}"
        epochs_use = epochs
        max_steps = -1
        per_device_batch_size = 2
        gradient_accumulation_steps = 4
        print(
            f"[local] epochs={epochs_use}, "
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
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print(f"\n{'='*60}", file=sys.stderr)
    print("RESULT", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(json.dumps(result, indent=2), file=sys.stderr)
