"""Single-GPU STaR loop: vLLM inference + LoRA training in one function.

One GPU, model loaded once. Each iteration:
  1. vLLM batch inference on narration prompts (~2 min on B200)
  2. Grade K1 (beat the bot) + identify failures (~instant)
  3. vLLM batch rationalization on failures (~1 min)
  4. Train LoRA on winning traces + rationalizations (~30s)
  5. Push adapter to HuggingFace
  6. Merge adapter, repeat from 1

Target: ~4 min per iteration on B200, ~8 min on H100.

Usage:
    # Run 3 iterations on H100
    modal run lem/gemma_star/star_loop.py --iterations 3 --gpu H100

    # Run 5 iterations on B200
    modal run lem/gemma_star/star_loop.py --iterations 5 --gpu B200

    # Quick test: 1 iteration, 20 examples
    modal run lem/gemma_star/star_loop.py --iterations 1 --limit 20 --gpu H100
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_BASE = "jasonyandell/gemma-4-e2b-texas42"

app = modal.App("lem-star-loop")

loop_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52",
        "accelerate>=1.2",
        "peft>=0.14",
        "trl>=0.15",
        "datasets>=3.0",
        "huggingface_hub>=0.27",
        "vllm>=0.8",
        "pillow",
        "wandb>=0.19",
    )
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


@app.function(
    image=loop_image,
    gpu="H100",
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
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    start_iteration: int = 0,
) -> str:
    """Run N STaR iterations on a single GPU."""
    import os
    import time

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = "/model-cache"

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

    # --- Load narrations ---
    examples = [json.loads(line) for line in narrations_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} narration prompts loaded")

    # --- Tokenizer (shared across phases) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    current_adapter = start_adapter
    all_results = []

    for iteration in range(start_iteration, start_iteration + n_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}")
        print(f"  Adapter: {current_adapter or '(base model)'}")
        print(f"{'='*60}")

        # =====================================================================
        # Phase 1: vLLM batch inference
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 1: vLLM batch inference...")
        t1 = time.time()

        # Build the merged model path for vLLM
        # vLLM can load adapters directly, but merging is simpler and avoids
        # compatibility issues with the ClippableLinear patch.
        if current_adapter:
            print(f"[iter {iteration}] Merging adapter for vLLM...")
            merge_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
            )
            merge_model = PeftModel.from_pretrained(merge_model, current_adapter)
            merge_model = merge_model.merge_and_unload()

            merge_path = f"/tmp/merged_iter{iteration}"
            merge_model.save_pretrained(merge_path)
            tokenizer.save_pretrained(merge_path)
            del merge_model
            torch.cuda.empty_cache()
            vllm_model_path = merge_path
        else:
            vllm_model_path = MODEL_ID

        # Initialize vLLM
        llm = LLM(
            model=vllm_model_path,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.7,  # leave room for training later
        )

        # Batch generate
        prompts = [ex["prompt"] for ex in examples]

        # Format as chat messages for vLLM
        chat_prompts = []
        for p in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            chat_prompts.append(formatted)

        outputs = llm.generate(chat_prompts, sampling_params)

        # Extract responses
        responses = [out.outputs[0].text for out in outputs]
        print(f"[iter {iteration}] Phase 1 done: {len(responses)} responses in {time.time()-t1:.0f}s")

        # Free vLLM GPU memory
        del llm
        torch.cuda.empty_cache()

        # =====================================================================
        # Phase 2: Grade K1
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 2: Grading...")
        stats = {"pass": 0, "fail": 0, "illegal": 0, "parse_fail": 0, "discarded": 0}
        graded = []

        for ex, response in zip(examples, responses):
            action = parse_play(response)
            grade = grade_k1(action, ex["bot_action"], ex["bot_eq"],
                             ex["all_eq"], ex["legal_actions"])
            stats[grade["grade"]] += 1
            graded.append({"example": ex, "response": response, "action": action, "grade": grade})

        total = sum(stats.values())
        pass_rate = stats["pass"] / total * 100
        print(f"[iter {iteration}] Grading: {stats['pass']}/{total} pass ({pass_rate:.0f}%)")
        print(f"  fail={stats['fail']} illegal={stats['illegal']} parse_fail={stats['parse_fail']}")

        # =====================================================================
        # Phase 3: Rationalize failures
        # =====================================================================
        # Only rationalize legal-but-wrong plays. Illegal/parse-fail traces
        # are poison — they contain reasoning about impossible states.
        # The illegality rate is a diagnostic for rules comprehension.
        legal_failures = [g for g in graded if g["grade"]["grade"] == "fail"]
        discarded = [g for g in graded if g["grade"]["grade"] in ("illegal", "parse_fail")]
        print(f"\n[iter {iteration}] Phase 3: Rationalizing {len(legal_failures)} legal failures "
              f"(discarding {len(discarded)} illegal/unparseable)...")
        t3 = time.time()

        if legal_failures:
            # Re-init vLLM for rationalization
            llm2 = LLM(
                model=vllm_model_path,
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.7,
            )

            rational_prompts = []
            for g in legal_failures:
                rp = (g["example"]["prompt"].rstrip()
                      + f"\n\nThe correct play here is {g['example']['best_action']}. "
                      f"Explain why {g['example']['best_action']} is the best choice.")
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": rp}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                rational_prompts.append(formatted)

            rational_outputs = llm2.generate(rational_prompts, sampling_params)
            rational_responses = [out.outputs[0].text for out in rational_outputs]

            del llm2
            torch.cuda.empty_cache()
            print(f"[iter {iteration}] Phase 3 done: {len(rational_responses)} rationalizations in {time.time()-t3:.0f}s")
        else:
            rational_responses = []

        # =====================================================================
        # Phase 4: Compile training traces
        # =====================================================================
        traces = []

        # Winning traces
        for g in graded:
            if g["grade"]["grade"] == "pass":
                traces.append({
                    "messages": [
                        {"role": "user", "content": g["example"]["prompt"]},
                        {"role": "assistant", "content": g["response"]},
                    ]
                })

        # Rationalizations (legal failures only — illegal traces discarded)
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
              f"{len(discarded)} discarded)")

        # =====================================================================
        # Phase 5: Train LoRA
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 5: Training LoRA...")
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
        print(f"[iter {iteration}] Training done: loss={result.training_loss:.4f} in {time.time()-t5:.0f}s")

        # Push adapter
        new_adapter_repo = f"{ADAPTER_BASE}-star-iter{iteration}"
        print(f"[iter {iteration}] Pushing to {new_adapter_repo}...")
        train_model.push_to_hub(new_adapter_repo, private=True)
        tokenizer.push_to_hub(new_adapter_repo, private=True)

        illegal_rate = (stats["illegal"] + stats["parse_fail"]) / total * 100
        wandb.log({"pass_rate": pass_rate, "illegal_rate": illegal_rate,
                    "n_pass": stats["pass"], "n_fail": stats["fail"],
                    "n_illegal": stats["illegal"], "n_discarded": len(discarded),
                    "n_traces": len(traces), "train_loss": result.training_loss})
        wandb.finish()

        # Cleanup for next iteration
        del train_model, trainer, dataset
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
              f"pass rate {pass_rate:.0f}%, adapter: {new_adapter_repo}")

    print(f"\n{'='*60}")
    print("ALL ITERATIONS COMPLETE")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Iter {r['iteration']}: {r['pass_rate']:.0f}% pass, "
              f"loss={r['train_loss']:.4f}, {r['elapsed_s']}s")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main(
    narrations: str = "lem/data/narrations_train.jsonl",
    adapter: str = "jasonyandell/gemma-4-e2b-texas42-stage0",
    iterations: int = 3,
    limit: int = 0,
    lr: float = 1e-4,
    gpu: str = "H100",
    start_iteration: int = 0,
):
    """Run the STaR loop."""
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
    print(f"[local] {n} examples, {iterations} iterations, GPU={gpu}", file=sys.stderr)
    print(f"[local] Start adapter: {adapter}", file=sys.stderr)

    # Override GPU type on the function
    # (Modal doesn't support runtime GPU override easily, so we just note it)
    if gpu != "H100":
        print(f"[local] NOTE: GPU override to {gpu} requires editing star_loop.py:gpu= line",
              file=sys.stderr)

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
# v2 - inlined grading functions
