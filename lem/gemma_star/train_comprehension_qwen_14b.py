"""Stage 0: LoRA training on comprehension Q&A — Qwen 3 14B via Unsloth.

Capacity experiment — tests whether the rationalization plateau at ~68/100
with Qwen 3 1.7B is a model-size ceiling or a curriculum/bootstrap problem.

Usage:
    modal run lem/gemma_star/train_comprehension_qwen_14b.py
    modal run lem/gemma_star/train_comprehension_qwen_14b.py --batch-size 8
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "unsloth/Qwen3-14B"
ADAPTER_REPO = "jasonyandell/qwen3-14b-texas42-stage0-v9"

app = modal.App("lem-comprehension-train-qwen-14b")

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "torch>=2.6",
        "unsloth>=2026.4.4",
        "unsloth_zoo",
        "transformers==5.5.0",
        "accelerate>=1.5",
        "peft>=0.14",
        "trl>=0.15",
        "datasets>=3.0",
        "huggingface_hub>=0.27",
        "bitsandbytes>=0.45",
        "wandb>=0.19",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=train_image,
    gpu="B200",
    timeout=21600,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("qwen3-14b-cache", create_if_missing=True)},
)
def train(
    corpus_jsonl: str,
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 1e-4,  # lower LR for bigger model
    batch_size: int = 16,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    push_to_hub: bool = True,
) -> dict:
    """Train LoRA adapter on Qwen 3 14B via Unsloth."""
    import json
    import os

    from unsloth import FastLanguageModel
    import torch
    import wandb
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    examples = [json.loads(line) for line in corpus_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] Loaded {len(examples)} examples", flush=True)
    cats = {}
    for ex in examples:
        cats[ex["category"]] = cats.get(ex["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}", flush=True)

    def format_example(ex):
        user_content = ex["prompt"].rstrip() + "\n\n" + ex["question"]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }

    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted).shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[data] Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    if dry_run:
        print("[dry-run] Data loaded. Exiting.", flush=True)
        return {"status": "dry_run", "n_train": len(train_ds), "n_val": len(val_ds)}

    print(f"[model] Loading {MODEL_ID} via Unsloth...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False,
    )

    train_ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        for ex in train_ds
    ])
    val_ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        for ex in val_ds
    ])

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    run_name = f"stage0-v9-qwen3-14b-bs{batch_size}-lr{lr}-e{epochs}"
    wandb.init(project="lem-stage0", name=run_name)

    training_args = SFTConfig(
        output_dir="/tmp/comprehension-train",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        bf16=True,
        max_length=512,
        dataset_text_field="text",
        dataset_num_proc=1,
        report_to="wandb",
        seed=42,
    )

    from transformers import TrainerCallback

    class PushPerEpochCallback(TrainerCallback):
        def __init__(self, base_name, tokenizer, do_push):
            self.base_name = base_name
            self.tokenizer = tokenizer
            self.do_push = do_push

        def on_save(self, args, state, control, model=None, **kwargs):
            if not self.do_push or model is None:
                return
            epoch = int(state.epoch)
            repo = f"{self.base_name}-ep{epoch}"
            print(f"[save] Epoch {epoch} — pushing to {repo}...", flush=True)
            try:
                model.push_to_hub(repo, private=True)
                self.tokenizer.push_to_hub(repo, private=True)
                print(f"[save] Epoch {epoch} done.", flush=True)
            except Exception as e:
                print(f"[save] Epoch {epoch} push failed: {e}", flush=True)

    push_cb = PushPerEpochCallback(adapter_name, tokenizer, push_to_hub)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[push_cb],
    )

    print("[train] Starting...", flush=True)
    result = trainer.train()
    print(f"[train] Done. Loss: {result.training_loss:.4f}", flush=True)

    if push_to_hub:
        print(f"[save] Pushing final to {adapter_name}...", flush=True)
        model.push_to_hub(adapter_name, private=True)
        tokenizer.push_to_hub(adapter_name, private=True)
        print("[save] Done.", flush=True)

    wandb.finish()

    return {
        "status": "complete",
        "train_loss": result.training_loss,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs": epochs,
        "adapter": adapter_name,
    }


@app.local_entrypoint()
def main(
    corpus: str = "lem/data/comprehension_train_v9.jsonl",
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 1e-4,
    batch_size: int = 16,
    lora_rank: int = 16,
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    no_push: bool = False,
):
    """Train Qwen 3 14B comprehension adapter on v9 data."""
    import sys

    corpus_path = Path(corpus)
    if not corpus_path.exists():
        print(f"[error] Not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    corpus_text = corpus_path.read_text()
    n = corpus_text.strip().count("\n") + 1
    print(f"[local] {n} examples from {corpus_path}", file=sys.stderr)
    print(f"[local] Qwen 3 14B, bs={batch_size}, lr={lr}, epochs={epochs}", file=sys.stderr)

    result = train.remote(
        corpus_jsonl=corpus_text,
        epochs=epochs,
        max_steps=max_steps,
        lr=lr,
        batch_size=batch_size,
        lora_rank=lora_rank,
        adapter_name=adapter_name,
        dry_run=dry_run,
        push_to_hub=not no_push,
    )

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Result: {result}", file=sys.stderr)
