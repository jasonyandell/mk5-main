"""Stage 0 v4: LoRA training on comprehension Q&A for Texas 42.

Trains Gemma 4 E2B on game-context questions: trump identification, domino
tracking, count status, legal moves, domino ranking. Each example is a compact
game state + one question with engine-verified ground truth.

Usage:
    modal run lem/gemma_star/train_comprehension.py
    modal run lem/gemma_star/train_comprehension.py --epochs 3 --lr 2e-4
    modal run lem/gemma_star/train_comprehension.py --dry-run
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_REPO = "jasonyandell/gemma-4-e2b-texas42-stage0-v4"

app = modal.App("lem-comprehension-train")

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
        "pillow",
        "wandb>=0.19",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=train_image,
    gpu="B200",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def train(
    corpus_jsonl: str,
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    start_adapter: str = "",
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    push_to_hub: bool = True,
) -> dict:
    """Train LoRA adapter on comprehension Q&A."""
    import json
    import os

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # --- Format as chat: user=(game state + question), assistant=answer ---
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

    # 95/5 split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[data] Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    if dry_run:
        print("[dry-run] Data loaded. Exiting.", flush=True)
        return {"status": "dry_run", "n_train": len(train_ds), "n_val": len(val_ds)}

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
    print("[patch] ClippableLinear patched", flush=True)

    # --- Load model (optionally with previous adapter merged) ---
    print(f"[model] Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )

    if start_adapter:
        from peft import PeftModel
        print(f"[model] Merging previous adapter: {start_adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, start_adapter)
        model = model.merge_and_unload()
        print("[model] Adapter merged.", flush=True)

    # --- LoRA ---
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
    print(f"[model] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

    # --- Train ---
    run_name = f"stage0-v4-r{lora_rank}-lr{lr}"
    if max_steps > 0:
        run_name += f"-{max_steps}steps"
    else:
        run_name += f"-e{epochs}"
    wandb.init(project="lem-stage0", name=run_name)

    training_args = SFTConfig(
        output_dir="/tmp/comprehension-train",
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
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
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("[train] Starting...", flush=True)
    result = trainer.train()
    print(f"[train] Done. Loss: {result.training_loss:.4f}", flush=True)

    # --- Save ---
    if push_to_hub:
        print(f"[save] Pushing to {adapter_name}...", flush=True)
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
    corpus: str = "lem/data/comprehension_train.jsonl",
    epochs: int = 3,
    max_steps: int = -1,
    lr: float = 2e-4,
    lora_rank: int = 16,
    start_adapter: str = "",
    adapter_name: str = ADAPTER_REPO,
    dry_run: bool = False,
    no_push: bool = False,
):
    """Train Stage 0 v4 comprehension adapter."""
    import sys

    corpus_path = Path(corpus)
    if not corpus_path.exists():
        print(f"[error] Not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    corpus_text = corpus_path.read_text()
    n = corpus_text.strip().count("\n") + 1
    steps_str = f", max_steps={max_steps}" if max_steps > 0 else ""
    adapter_str = f", resume from {start_adapter}" if start_adapter else ""
    print(f"[local] {n} examples from {corpus_path}", file=sys.stderr)
    print(f"[local] epochs={epochs}{steps_str}, lr={lr}, rank={lora_rank}{adapter_str}", file=sys.stderr)
    print(f"[local] Push to: {adapter_name}", file=sys.stderr)

    result = train.remote(
        corpus_jsonl=corpus_text,
        epochs=epochs,
        max_steps=max_steps,
        lr=lr,
        lora_rank=lora_rank,
        start_adapter=start_adapter,
        adapter_name=adapter_name,
        dry_run=dry_run,
        push_to_hub=not no_push,
    )

    print(f"\n{'='*40}", file=sys.stderr)
    print(f"Result: {result}", file=sys.stderr)
