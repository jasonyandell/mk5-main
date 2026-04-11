"""Single-GPU STaR loop: HF generate with flash attention + batching + LoRA.

One GPU, model loaded once per iteration. Each iteration:
  1. Batch inference on all prompts (flash attention + torch.compile)
  2. Grade K1 (beat the bot)
  3. Batch rationalization on failures (reuse loaded model)
  4. Train LoRA on winning traces + rationalizations
  5. Push adapter to HuggingFace, repeat

B200 with 192GB: model is ~10GB, leaving 180GB+ for KV cache and batches.
Flash attention + padding-based batching = massive throughput on this tiny model.

Usage:
    modal run lem/gemma_star/star_loop.py --iterations 3
    modal run lem/gemma_star/star_loop.py --iterations 1 --limit 5   # smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_BASE = "jasonyandell/gemma-4-e2b-texas42"
GPU_TYPE = "B200"  # B200 ($6.25/hr, 192GB) or "H100" ($3.95/hr, 80GB)

app = modal.App("lem-star-loop")

loop_image = (
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


# --- Grading logic (inlined to avoid lem module mount issues) ---

import re


def parse_play(response_text: str) -> str | None:
    """Extract the domino the model chose to play from its response."""
    patterns = [
        r"[Pp]lay[:\s]+(?:the\s+)?(\d-\d)",
        r"[Aa]nswer[:\s]+(?:the\s+)?(\d-\d)",
        r"[Cc]hoice[:\s]+(?:the\s+)?(\d-\d)",
        r"I (?:would |will |should )?play (?:the )?(\d-\d)",
        r"\*\*(\d-\d)\*\*",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)
    all_doms = re.findall(r"\b(\d-\d)\b", response_text)
    if all_doms:
        return all_doms[-1]
    return None


def grade_k1(gemma_action: str | None, bot_action: str, bot_eq: float,
             all_eq: dict[str, float], legal_actions: list[str]) -> dict:
    """Grade K1: did Gemma beat the bot?"""
    if gemma_action is None:
        return {"grade": "parse_fail", "gemma_action": None}
    if gemma_action not in legal_actions:
        return {"grade": "illegal", "gemma_action": gemma_action}
    gemma_eq = all_eq.get(gemma_action, float("-inf"))
    if gemma_eq >= bot_eq:
        return {"grade": "pass", "gemma_action": gemma_action,
                "gemma_eq": gemma_eq, "bot_eq": bot_eq,
                "delta": round(gemma_eq - bot_eq, 3)}
    return {"grade": "fail", "gemma_action": gemma_action,
            "gemma_eq": gemma_eq, "bot_eq": bot_eq,
            "delta": round(gemma_eq - bot_eq, 3)}


# --- Model management ---

def _patch_clippable_linear():
    """Patch Gemma4ClippableLinear for PEFT compatibility (training only)."""
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


def _load_model(adapter_repo: str | None = None):
    """Load Gemma 4 E2B with flash attention, optionally with LoRA adapter merged."""
    import time
    import torch
    from transformers import AutoModelForCausalLM

    _patch_clippable_linear()

    print(f"[model] Loading {MODEL_ID} with SDPA...", flush=True)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",  # PyTorch native flash attention, no extra package
    )

    if adapter_repo:
        from peft import PeftModel
        print(f"[model] Loading LoRA adapter: {adapter_repo}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()
        print(f"[model] Adapter merged.", flush=True)

    model.eval()
    model = torch.compile(model, mode="reduce-overhead")

    elapsed = time.time() - t
    print(f"[model] Ready in {elapsed:.0f}s (sdpa + torch.compile)", flush=True)
    return model


def _batch_infer(model, tokenizer, prompts: list[str],
                 max_tokens: int = 1024, temperature: float = 0.6,
                 batch_size: int = 0) -> list[str]:
    """Batch inference with left-padding for efficient GPU utilization."""
    import time
    import torch

    # Auto batch size: fit everything at once on B200 (192GB, model is 10GB)
    if batch_size <= 0:
        batch_size = len(prompts)

    # Format with chat template + thinking mode
    all_messages = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        for msgs in all_messages
    ]

    # Left-pad for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = []
    t = time.time()

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        input_lens = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # Decode only the generated tokens for each sequence
        for j, (out, in_len) in enumerate(zip(outputs, input_lens)):
            generated = out[in_len:]
            text = tokenizer.decode(generated, skip_special_tokens=False)
            responses.append(text)

        done = min(i + batch_size, len(formatted))
        elapsed = time.time() - t
        # Count total generated tokens
        total_gen_tokens = sum(len(out) - in_len for out, in_len in zip(outputs, input_lens))
        print(f"  [{done}/{len(formatted)}] {elapsed:.0f}s, "
              f"~{total_gen_tokens/elapsed:.0f} tok/s this batch", flush=True)

    return responses


@app.function(
    image=loop_image,
    gpu=GPU_TYPE,
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-api-key")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def run_loop(
    narrations_jsonl: str,
    start_adapter: str = "",
    n_iterations: int = 3,
    lr: float = 1e-4,
    lora_rank: int = 16,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    start_iteration: int = 0,
) -> str:
    """Run N STaR iterations on a single GPU."""
    import gc
    import os
    import time

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    # --- Load narrations ---
    examples = [json.loads(line) for line in narrations_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} narration prompts loaded", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    current_adapter = start_adapter
    all_results = []

    for iteration in range(start_iteration, start_iteration + n_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  ITERATION {iteration}", flush=True)
        print(f"  Adapter: {current_adapter or '(base model)'}", flush=True)
        print(f"  GPU: {GPU_TYPE}", flush=True)
        print(f"{'='*60}", flush=True)

        # =====================================================================
        # Load model (flash attention + adapter + torch.compile)
        # =====================================================================
        infer_model = _load_model(adapter_repo=current_adapter or None)

        # =====================================================================
        # Phase 1: Batch inference
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 1: Batch inference ({len(examples)} prompts)...", flush=True)
        t1 = time.time()

        prompts = [ex["prompt"] for ex in examples]
        responses = _batch_infer(
            infer_model, tokenizer, prompts,
            max_tokens=max_new_tokens, temperature=temperature,
        )

        elapsed = time.time() - t1
        print(f"[iter {iteration}] Phase 1 done: {len(responses)} responses in {elapsed:.0f}s "
              f"({len(responses)/elapsed:.1f} prompts/s)", flush=True)

        # =====================================================================
        # Phase 2: Grade K1
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 2: Grading...", flush=True)
        stats = {"pass": 0, "fail": 0, "illegal": 0, "parse_fail": 0}
        graded = []

        for ex, response in zip(examples, responses):
            action = parse_play(response)
            grade = grade_k1(action, ex["bot_action"], ex["bot_eq"],
                             ex["all_eq"], ex["legal_actions"])
            stats[grade["grade"]] += 1
            graded.append({"example": ex, "response": response, "action": action, "grade": grade})

        total = sum(stats.values())
        pass_rate = stats["pass"] / total * 100
        print(f"[iter {iteration}] Grading: {stats['pass']}/{total} pass ({pass_rate:.0f}%)", flush=True)
        print(f"  fail={stats['fail']} illegal={stats['illegal']} parse_fail={stats['parse_fail']}", flush=True)

        # =====================================================================
        # Phase 3: Rationalize failures (reuse loaded model)
        # =====================================================================
        legal_failures = [g for g in graded if g["grade"]["grade"] == "fail"]
        discarded = [g for g in graded if g["grade"]["grade"] in ("illegal", "parse_fail")]
        print(f"\n[iter {iteration}] Phase 3: Rationalizing {len(legal_failures)} legal failures "
              f"(discarding {len(discarded)} illegal/unparseable)...", flush=True)
        t3 = time.time()

        rational_responses = []
        if legal_failures:
            rat_prompts = []
            for g in legal_failures:
                rp = (g["example"]["prompt"].rstrip()
                      + f"\n\nThe correct play here is {g['example']['best_action']}. "
                      f"Explain why {g['example']['best_action']} is the best choice.")
                rat_prompts.append(rp)

            rational_responses = _batch_infer(
                infer_model, tokenizer, rat_prompts,
                max_tokens=max_new_tokens, temperature=temperature,
            )
            print(f"[iter {iteration}] Phase 3 done: {len(rational_responses)} rationalizations "
                  f"in {time.time()-t3:.0f}s", flush=True)

        # =====================================================================
        # Free inference model to reclaim GPU for training
        # =====================================================================
        del infer_model
        gc.collect()
        torch.cuda.empty_cache()

        # =====================================================================
        # Phase 4: Compile training traces
        # =====================================================================
        traces = []

        for g in graded:
            if g["grade"]["grade"] == "pass":
                traces.append({
                    "messages": [
                        {"role": "user", "content": g["example"]["prompt"]},
                        {"role": "assistant", "content": g["response"]},
                    ]
                })

        for g, rr in zip(legal_failures, rational_responses):
            rp = (g["example"]["prompt"].rstrip()
                  + f"\n\nThe correct play here is {g['example']['best_action']}. "
                  f"Explain why {g['example']['best_action']} is the best choice.")
            traces.append({
                "messages": [
                    {"role": "user", "content": rp},
                    {"role": "assistant", "content": rr},
                ]
            })

        print(f"[iter {iteration}] {len(traces)} training traces "
              f"({stats['pass']} wins + {len(legal_failures)} rationalizations, "
              f"{len(discarded)} discarded)", flush=True)

        # =====================================================================
        # Phase 5: Train LoRA
        # =====================================================================
        _patch_clippable_linear()
        print(f"\n[iter {iteration}] Phase 5: Training LoRA...", flush=True)
        t5 = time.time()

        train_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )

        if current_adapter:
            train_model = PeftModel.from_pretrained(train_model, current_adapter)
            train_model = train_model.merge_and_unload()

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        train_model = get_peft_model(train_model, lora_config)

        dataset = Dataset.from_list(traces).shuffle(seed=42 + iteration)

        wandb.init(project="lem-star", name=f"star-iter{iteration}",
                   reinit=True)

        training_args = SFTConfig(
            output_dir="/tmp/star-train",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
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
            seed=42 + iteration,
        )

        trainer = SFTTrainer(
            model=train_model, args=training_args,
            train_dataset=dataset, processing_class=tokenizer,
        )
        result = trainer.train()
        print(f"[iter {iteration}] Training done: loss={result.training_loss:.4f} in {time.time()-t5:.0f}s", flush=True)

        # Push adapter
        new_adapter_repo = f"{ADAPTER_BASE}-star-iter{iteration}"
        print(f"[iter {iteration}] Pushing to {new_adapter_repo}...", flush=True)
        train_model.push_to_hub(new_adapter_repo, private=True)
        tokenizer.push_to_hub(new_adapter_repo, private=True)

        illegal_rate = (stats["illegal"] + stats["parse_fail"]) / total * 100
        wandb.log({"pass_rate": pass_rate, "illegal_rate": illegal_rate,
                    "n_pass": stats["pass"], "n_fail": stats["fail"],
                    "n_illegal": stats["illegal"], "n_discarded": len(discarded),
                    "n_traces": len(traces), "train_loss": result.training_loss})
        wandb.finish()

        del train_model, trainer, dataset
        gc.collect()
        torch.cuda.empty_cache()

        current_adapter = new_adapter_repo
        iter_elapsed = time.time() - iter_start

        iter_result = {
            "iteration": iteration,
            "pass_rate": pass_rate,
            "stats": stats,
            "n_traces": len(traces),
            "train_loss": result.training_loss,
            "adapter": new_adapter_repo,
            "elapsed_s": round(iter_elapsed),
        }
        all_results.append(iter_result)
        print(f"\n[iter {iteration}] COMPLETE in {iter_elapsed:.0f}s — "
              f"pass rate {pass_rate:.0f}%, adapter: {new_adapter_repo}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("ALL ITERATIONS COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        print(f"  Iter {r['iteration']}: {r['pass_rate']:.0f}% pass, "
              f"loss={r['train_loss']:.4f}, {r['elapsed_s']}s", flush=True)

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main(
    narrations: str = "lem/data/narrations_train.jsonl",
    adapter: str = "jasonyandell/gemma-4-e2b-texas42-stage0",
    iterations: int = 3,
    limit: int = 0,
    lr: float = 1e-4,
    start_iteration: int = 0,
):
    """Run the STaR loop on B200."""
    import sys

    narrations_path = Path(narrations)
    if not narrations_path.exists():
        print(f"[error] Not found: {narrations_path}", file=sys.stderr)
        sys.exit(1)

    text = narrations_path.read_text()
    if limit > 0:
        lines = text.strip().split("\n")
        text = "\n".join(lines[:limit])
        print(f"[local] Limited to {limit} examples", file=sys.stderr)

    n = text.strip().count("\n") + 1
    print(f"[local] {n} examples, {iterations} iterations, GPU={GPU_TYPE}", file=sys.stderr)
    print(f"[local] Start adapter: {adapter}", file=sys.stderr)

    result_json = run_loop.remote(
        narrations_jsonl=text,
        start_adapter=adapter,
        n_iterations=iterations,
        lr=lr,
        start_iteration=start_iteration,
    )

    results = json.loads(result_json)
    print(f"\n{'='*60}", file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in results:
        print(f"  Iter {r['iteration']}: {r['pass_rate']:.0f}% pass, "
              f"loss={r['train_loss']:.4f}, {r['elapsed_s']}s, "
              f"adapter={r['adapter']}", file=sys.stderr)
