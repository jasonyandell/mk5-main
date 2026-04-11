"""Single-GPU STaR loop: vLLM batch inference + LoRA training in one function.

One GPU, vLLM server for inference. Each iteration:
  1. vLLM concurrent inference on narration prompts
  2. Grade K1 (beat the bot) + identify failures (~instant)
  3. vLLM concurrent rationalization on failures
  4. Kill vLLM, train LoRA on winning traces + rationalizations (~30s)
  5. Push adapter to HuggingFace
  6. Restart vLLM with merged adapter, repeat from 1

Uses the official vLLM Gemma 4 recipe (vllm==0.19.0, transformers==5.5.0).
vLLM handles flash attention, paged KV cache, and continuous batching automatically.

Usage:
    # Run 3 iterations on B200
    modal run lem/gemma_star/star_loop.py --iterations 3

    # Quick test: 1 iteration, 20 examples
    modal run lem/gemma_star/star_loop.py --iterations 1 --limit 20

    # With H100 fallback (edit GPU_TYPE below)
    modal run lem/gemma_star/star_loop.py --iterations 3
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_BASE = "jasonyandell/gemma-4-e2b-texas42"
GPU_TYPE = "B200"  # B200 ($6.25/hr, 192GB) or "H100" ($3.95/hr, 80GB)
VLLM_PORT = 8000
MAX_MODEL_LEN = 8192  # Our prompts are ~2800 tokens, responses ~500-1500

app = modal.App("lem-star-loop")

loop_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        # vLLM for inference (official Gemma 4 recipe)
        "vllm==0.19.0",
        "transformers==5.5.0",
        # Training deps
        "peft>=0.14",
        "trl>=0.15",
        "datasets>=3.0",
        "huggingface_hub>=0.27",
        "wandb>=0.19",
        # HTTP client for vLLM API
        "aiohttp>=3.9",
        "openai>=1.0",
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


# --- vLLM server management ---

def _merge_adapter_to_disk(adapter_repo: str, output_dir: str):
    """Merge a LoRA adapter into the base model and save to disk for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_clippable_linear()

    print(f"[merge] Loading base model + adapter {adapter_repo}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_repo)
    model = model.merge_and_unload()

    print(f"[merge] Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(output_dir)
    del model
    print("[merge] Done.")


def _start_vllm(model_path: str) -> "subprocess.Popen":
    """Launch vLLM server as a subprocess, return the process handle."""
    import subprocess

    cmd = [
        "vllm", "serve", model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", "image=0,video=0,audio=0",
        "--async-scheduling",
        "--kv-cache-dtype", "fp8",
        "--dtype", "bfloat16",
        "--uvicorn-log-level", "warning",
    ]
    print(f"[vllm] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def _wait_for_vllm(timeout: int = 300):
    """Poll vLLM health endpoint until it's ready."""
    import time
    import urllib.request
    import urllib.error

    url = f"http://localhost:{VLLM_PORT}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"[vllm] Server ready in {elapsed:.0f}s")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM failed to start within {timeout}s")


def _stop_vllm(proc: "subprocess.Popen"):
    """Gracefully stop vLLM server."""
    import signal
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    print("[vllm] Server stopped.")


async def _batch_generate(prompts: list[str], max_tokens: int = 1024,
                          temperature: float = 0.6, concurrency: int = 16) -> list[str]:
    """Send prompts to vLLM concurrently via OpenAI API. Returns responses."""
    import asyncio
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"http://localhost:{VLLM_PORT}/v1",
        api_key="EMPTY",
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(prompt: str) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            msg = response.choices[0].message
            # Combine thinking + content
            parts = []
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if reasoning:
                parts.append(f"<think>\n{reasoning}\n</think>")
            if msg.content:
                parts.append(msg.content)
            return "\n".join(parts)

    results = await asyncio.gather(*[generate_one(p) for p in prompts])
    return list(results)


def _patch_clippable_linear():
    """Patch Gemma4ClippableLinear for PEFT compatibility."""
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
        print("[patch] ClippableLinear patched")
    except ImportError:
        pass


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
    """Run N STaR iterations on a single GPU with vLLM for inference."""
    import asyncio
    import os
    import time

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    os.environ["HF_HOME"] = "/model-cache"

    MERGED_DIR = "/tmp/merged-model"

    # --- Load narrations ---
    examples = [json.loads(line) for line in narrations_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} narration prompts loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    current_adapter = start_adapter
    all_results = []

    for iteration in range(start_iteration, start_iteration + n_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}")
        print(f"  Adapter: {current_adapter or '(base model)'}")
        print(f"  GPU: {GPU_TYPE}")
        print(f"{'='*60}")

        # =====================================================================
        # Prepare model for vLLM: merge adapter to disk if needed
        # =====================================================================
        if current_adapter:
            _merge_adapter_to_disk(current_adapter, MERGED_DIR)
            model_path = MERGED_DIR
        else:
            model_path = MODEL_ID

        # =====================================================================
        # Start vLLM server
        # =====================================================================
        print(f"\n[iter {iteration}] Starting vLLM server...")
        t_vllm = time.time()
        vllm_proc = _start_vllm(model_path)
        _wait_for_vllm(timeout=300)
        print(f"[iter {iteration}] vLLM ready in {time.time()-t_vllm:.0f}s")

        # =====================================================================
        # Phase 1: Concurrent inference via vLLM
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 1: Batch inference ({len(examples)} prompts)...")
        t1 = time.time()

        prompts = [ex["prompt"] for ex in examples]
        responses = asyncio.run(
            _batch_generate(prompts, max_tokens=max_new_tokens, temperature=temperature)
        )

        print(f"[iter {iteration}] Phase 1 done: {len(responses)} responses in {time.time()-t1:.0f}s "
              f"({len(responses)/(time.time()-t1):.1f} prompts/s)")

        # =====================================================================
        # Phase 2: Grade K1
        # =====================================================================
        print(f"\n[iter {iteration}] Phase 2: Grading...")
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
        print(f"[iter {iteration}] Grading: {stats['pass']}/{total} pass ({pass_rate:.0f}%)")
        print(f"  fail={stats['fail']} illegal={stats['illegal']} parse_fail={stats['parse_fail']}")

        # =====================================================================
        # Phase 3: Rationalize failures (reuse running vLLM server)
        # =====================================================================
        legal_failures = [g for g in graded if g["grade"]["grade"] == "fail"]
        discarded = [g for g in graded if g["grade"]["grade"] in ("illegal", "parse_fail")]
        print(f"\n[iter {iteration}] Phase 3: Rationalizing {len(legal_failures)} legal failures "
              f"(discarding {len(discarded)} illegal/unparseable)...")
        t3 = time.time()

        if legal_failures:
            rat_prompts = []
            for g in legal_failures:
                rp = (g["example"]["prompt"].rstrip()
                      + f"\n\nThe correct play here is {g['example']['best_action']}. "
                      f"Explain why {g['example']['best_action']} is the best choice.")
                rat_prompts.append(rp)

            rational_responses = asyncio.run(
                _batch_generate(rat_prompts, max_tokens=max_new_tokens, temperature=temperature)
            )
            print(f"[iter {iteration}] Phase 3 done: {len(rational_responses)} rationalizations "
                  f"in {time.time()-t3:.0f}s")
        else:
            rational_responses = []

        # =====================================================================
        # Stop vLLM to free GPU for training
        # =====================================================================
        _stop_vllm(vllm_proc)
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
              f"{len(discarded)} discarded)")

        # =====================================================================
        # Phase 5: Train LoRA
        # =====================================================================
        _patch_clippable_linear()
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
    start_iteration: int = 0,
):
    """Run the STaR loop on B200 with vLLM inference."""
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
