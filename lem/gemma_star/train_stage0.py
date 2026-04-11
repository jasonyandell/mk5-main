"""Stage 0: LoRA training on synthetic Q&A corpus for Texas 42 rules.

Trains Gemma 4 E2B to understand 42 rules via engine-generated Q&A pairs.
Runs on Modal with an L4 GPU. Saves the LoRA adapter for later inference.

Usage:
    # Train with defaults (3500 examples, 3 epochs)
    modal run lem/gemma_star/train_stage0.py

    # Custom settings
    modal run lem/gemma_star/train_stage0.py \
        --corpus lem/rules/qa_corpus.jsonl \
        --epochs 5 \
        --lr 2e-4

    # Dry run (just check data loading)
    modal run lem/gemma_star/train_stage0.py --dry-run
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_REPO = "jasonyandell/gemma-4-e2b-texas42-stage0-kerry"

app = modal.App("lem-stage0-train")

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
        "bitsandbytes>=0.45",
    )
)


@app.function(
    image=train_image,
    gpu="B200",  # 192GB — no memory pressure, full batching
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def train(
    corpus_jsonl: str,
    primer_text: str,
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_len: int = 1024,
    dry_run: bool = False,
    push_to_hub: bool = True,
) -> dict:
    """Train LoRA adapter on Q&A corpus."""
    import json
    import os

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    # --- Load corpus ---
    examples = [json.loads(line) for line in corpus_jsonl.strip().split("\n")]
    print(f"[data] Loaded {len(examples)} examples across categories:")
    cats = {}
    for ex in examples:
        cats[ex["category"]] = cats.get(ex["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # --- Format as chat conversations ---
    # System prompt = rules primer, then Q&A as user/assistant turns
    def format_example(ex):
        return {
            "messages": [
                {"role": "system", "content": primer_text},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }

    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)

    # Shuffle and split 95/5 train/val
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[data] Train: {len(train_ds)}, Val: {len(val_ds)}")

    if dry_run:
        print("[dry-run] Data loaded successfully. Exiting.")
        return {"status": "dry_run", "n_train": len(train_ds), "n_val": len(val_ds)}

    # --- Patch Gemma4ClippableLinear for PEFT compatibility ---
    # Gemma 4's ClippableLinear inherits from nn.Module, but PEFT only targets
    # nn.Linear. This monkey-patch re-inherits from nn.Linear while preserving
    # the clamping behavior. Must be applied BEFORE model loading.
    # See: https://huggingface.co/google/gemma-4-31B/discussions/3
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
    print("[patch] Gemma4ClippableLinear patched for PEFT compatibility")

    # --- Load model + tokenizer ---
    print(f"[model] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        modules_to_save=None,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Training ---
    wandb.init(project="lem-stage0", name=f"stage0-r{lora_rank}-e{epochs}-lr{lr}")

    training_args = SFTConfig(
        output_dir="/tmp/stage0-output",
        num_train_epochs=epochs,
        max_steps=max_steps,  # -1 = use epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # effective batch = 16
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="no",  # eval OOMs on L4; skip and save directly
        save_strategy="no",  # save manually at end
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
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("[train] Starting training...")
    result = trainer.train()
    print(f"[train] Done. Loss: {result.training_loss:.4f}")

    # --- Save ---
    if push_to_hub:
        print(f"[save] Pushing adapter to {ADAPTER_REPO}...")
        model.push_to_hub(ADAPTER_REPO, private=True)
        tokenizer.push_to_hub(ADAPTER_REPO, private=True)
        print("[save] Done.")
    else:
        save_path = "/tmp/stage0-adapter"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[save] Saved locally to {save_path}")

    wandb.finish()

    return {
        "status": "complete",
        "train_loss": result.training_loss,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs": epochs,
        "trainable_params": trainable,
    }


@app.local_entrypoint()
def main(
    corpus: str = "lem/rules/qa_corpus.jsonl",
    primer: str = "lem/rules/primer.md",
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    lora_rank: int = 16,
    dry_run: bool = False,
    no_push: bool = False,
):
    """Train Stage 0 rules adapter."""
    import sys

    corpus_path = Path(corpus)
    primer_path = Path(primer)

    if not corpus_path.exists():
        print(f"[error] Corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)
    if not primer_path.exists():
        print(f"[error] Primer not found: {primer_path}", file=sys.stderr)
        sys.exit(1)

    corpus_text = corpus_path.read_text()
    primer_text = primer_path.read_text()

    n_examples = corpus_text.count("\n")
    print(f"[local] Corpus: {n_examples} examples from {corpus_path}", file=sys.stderr)
    print(f"[local] Primer: {len(primer_text)} chars from {primer_path}", file=sys.stderr)
    steps_str = f", max_steps={max_steps}" if max_steps > 0 else ""
    print(f"[local] Config: epochs={epochs}{steps_str}, lr={lr}, rank={lora_rank}", file=sys.stderr)

    result = train.remote(
        corpus_jsonl=corpus_text,
        primer_text=primer_text,
        epochs=epochs,
        max_steps=max_steps,
        lr=lr,
        lora_rank=lora_rank,
        dry_run=dry_run,
        push_to_hub=not no_push,
    )

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Result: {result}", file=sys.stderr)
