"""Stage 0: LoRA training on comprehension Q&A for Texas 42 via Unsloth + Qwen 3 1.7B Instruct.

Qwen 3 is dense pure-GQA (no linear-attention layers like Qwen 3.5), so all
standard fast paths work without needing flash-linear-attention / causal-conv1d.

Usage:
    modal run lem/gemma_star/train_comprehension_qwen.py
    modal run lem/gemma_star/train_comprehension_qwen.py --epochs 3 --lr 2e-4 --batch-size 64
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "unsloth/Qwen3-1.7B"
ADAPTER_REPO = "jasonyandell/qwen3-1.7b-texas42-stage0-v5"

app = modal.App("lem-comprehension-train-qwen")

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
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("qwen3-cache", create_if_missing=True)},
)
def train(
    corpus_jsonl: str,
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    batch_size: int = 64,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    push_to_hub: bool = True,
) -> dict:
    """Train LoRA adapter on Qwen 3.5 2B via Unsloth."""
    import json
    import os

    from unsloth import FastLanguageModel  # must import before trl/transformers/peft
    import torch
    import wandb
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    # --- Load corpus ---
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

    # --- Load model via Unsloth ---
    print(f"[model] Loading {MODEL_ID} via Unsloth...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=512,  # our examples max at ~476 tokens
        dtype=None,  # auto (bf16 on B200)
        load_in_4bit=False,
    )

    # Build text column without datasets.map (Unsloth-patched tokenizer isn't dill-picklable)
    train_ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        for ex in train_ds
    ])
    val_ds = Dataset.from_list([
        {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        for ex in val_ds
    ])

    # --- LoRA ---
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

    # --- Train ---
    run_name = f"stage0-v5-qwen3-1.7b-bs{batch_size}-lr{lr}-e{epochs}"
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
        dataset_num_proc=1,  # Unsloth-patched tokenizer isn't dill-picklable
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
    corpus: str = "lem/data/comprehension_train_v5.jsonl",
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    batch_size: int = 64,
    lora_rank: int = 16,
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    no_push: bool = False,
):
    """Train Stage 0 comprehension adapter on Qwen 3.5 2B via Unsloth."""
    import sys

    corpus_path = Path(corpus)
    if not corpus_path.exists():
        print(f"[error] Not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    corpus_text = corpus_path.read_text()
    n = corpus_text.strip().count("\n") + 1
    steps_str = f", max_steps={max_steps}" if max_steps > 0 else ""
    print(f"[local] {n} examples from {corpus_path}", file=sys.stderr)
    print(f"[local] epochs={epochs}{steps_str}, lr={lr}, batch={batch_size}, rank={lora_rank}", file=sys.stderr)
    print(f"[local] Push to: {adapter_name}", file=sys.stderr)

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
