"""Train LoRA on STaR traces (winning reasoning + rationalizations).

Takes the output of star_harness.py and trains a new LoRA adapter.
The traces already contain chat messages (user prompt + assistant response
with thinking channel), so we train directly on those.

Usage:
    # Train on iteration 0 traces, push to HF
    modal run lem/gemma_star/train_star.py \
        --traces lem/data/star_iter0.jsonl \
        --output-repo jasonyandell/gemma-4-e2b-texas42-star-iter0

    # With a base adapter to build on top of
    modal run lem/gemma_star/train_star.py \
        --traces lem/data/star_iter0.jsonl \
        --base-adapter jasonyandell/gemma-4-e2b-texas42-stage0 \
        --output-repo jasonyandell/gemma-4-e2b-texas42-star-iter0
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"

app = modal.App("lem-star-train")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52",
        "accelerate>=1.2",
        "peft>=0.14",
        "trl>=0.15",
        "datasets>=3.0",
        "huggingface_hub>=0.27",
        "pillow",
        "wandb>=0.19",
    )
)


@app.function(
    image=train_image,
    gpu="A100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def train(
    traces_jsonl: str,
    output_repo: str,
    base_adapter: str = "",
    epochs: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    dry_run: bool = False,
    iteration: int = 0,
) -> dict:
    """Train LoRA on STaR traces."""
    import json
    import os

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    # --- Load traces ---
    traces = [json.loads(line) for line in traces_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] Loaded {len(traces)} traces")
    types = {}
    for t in traces:
        types[t["type"]] = types.get(t["type"], 0) + 1
    for typ, count in sorted(types.items()):
        print(f"  {typ}: {count}")

    # Format: each trace already has "messages" with user + assistant
    formatted = [{"messages": t["messages"]} for t in traces]
    dataset = Dataset.from_list(formatted)
    dataset = dataset.shuffle(seed=42)

    # Split — but for small datasets, keep more for training
    if len(formatted) > 20:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds = dataset
        val_ds = None

    print(f"[data] Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}")

    if dry_run:
        return {"status": "dry_run", "n_train": len(train_ds)}

    # --- Patch ClippableLinear ---
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
    print("[patch] ClippableLinear patched")

    # --- Load model ---
    print(f"[model] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )

    # If building on a base adapter, merge it first then apply fresh LoRA
    if base_adapter:
        print(f"[model] Merging base adapter: {base_adapter}")
        model = PeftModel.from_pretrained(model, base_adapter)
        model = model.merge_and_unload()

    # --- Fresh LoRA ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Train ---
    wandb.init(project="lem-star", name=f"star-iter{iteration}-r{lora_rank}-e{epochs}")

    training_args = SFTConfig(
        output_dir="/tmp/star-output",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("[train] Starting...")
    result = trainer.train()
    print(f"[train] Done. Loss: {result.training_loss:.4f}")

    # --- Push ---
    print(f"[save] Pushing to {output_repo}...")
    model.push_to_hub(output_repo, private=True)
    tokenizer.push_to_hub(output_repo, private=True)
    print("[save] Done.")

    wandb.finish()
    return {
        "status": "complete",
        "train_loss": result.training_loss,
        "n_traces": len(traces),
        "iteration": iteration,
    }


@app.local_entrypoint()
def main(
    traces: str = "lem/data/star_iter0.jsonl",
    base_adapter: str = "jasonyandell/gemma-4-e2b-texas42-stage0",
    output_repo: str = "jasonyandell/gemma-4-e2b-texas42-star-iter0",
    epochs: int = 1,
    lr: float = 1e-4,
    iteration: int = 0,
    dry_run: bool = False,
):
    """Train on STaR traces."""
    import sys

    traces_path = Path(traces)
    if not traces_path.exists():
        print(f"[error] Traces not found: {traces_path}", file=sys.stderr)
        sys.exit(1)

    traces_text = traces_path.read_text()
    n_traces = traces_text.strip().count("\n") + 1
    print(f"[local] {n_traces} traces from {traces_path}", file=sys.stderr)
    print(f"[local] Base adapter: {base_adapter or '(none)'}", file=sys.stderr)
    print(f"[local] Output: {output_repo}", file=sys.stderr)

    result = train.remote(
        traces_jsonl=traces_text,
        output_repo=output_repo,
        base_adapter=base_adapter,
        epochs=epochs,
        lr=lr,
        dry_run=dry_run,
        iteration=iteration,
    )

    print(f"\nResult: {result}", file=sys.stderr)
