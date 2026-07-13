"""Eval Stage 0 (Qwen 3 1.7B): measure comprehension accuracy per question type.

Usage:
    modal run lem/gemma_star/eval_comprehension_qwen.py --eval-data lem/data/comprehension_eval_v5.jsonl
    modal run lem/gemma_star/eval_comprehension_qwen.py --adapter jasonyandell/qwen3-1.7b-texas42-stage0-v5 --limit 300
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "unsloth/Qwen3-1.7B"
ADAPTER_REPO = "jasonyandell/qwen3-1.7b-texas42-stage0-v5"

app = modal.App("lem-comprehension-eval-qwen")

eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "torch>=2.6",
        "transformers==5.5.0",
        "accelerate>=1.2",
        "peft>=0.14",
        "huggingface_hub>=0.27",
        "pillow",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def grade_response(example: dict, response: str, graders: dict) -> dict:
    """Grade a model response against ground truth.

    Dispatches through ``grade_offline.GRADERS`` so this file has a single
    source of truth for grading logic. If a new category is added without a
    corresponding grader, we deliberately return the honest failure mode
    rather than silently scoring it wrong.

    Returns: {correct: bool, details: str, category: str}
    """
    category = example["category"]
    grader = graders.get(category)
    if grader is None:
        return {"correct": False, "details": "unknown category", "category": category}

    result = grader(example["answer"], response)
    return {
        "correct": bool(result["correct"]),
        "details": result.get("detail", ""),
        "category": category,
    }


@app.function(
    image=eval_image,
    gpu="B200",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("qwen3-cache", create_if_missing=True)},
)
def run_eval(
    eval_jsonl: str,
    adapter_repo: str = ADAPTER_REPO,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    batch_size: int = 32,
) -> str:
    """Run eval on comprehension questions."""
    import os
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Import inside the function so Modal auto-serializes this local module
    # into the container image (matching the pattern used for torch/transformers).
    from lem.gemma_star.grade_offline import GRADERS

    os.environ["HF_HOME"] = "/model-cache"

    examples = [json.loads(line) for line in eval_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} eval examples", flush=True)

    print(f"[model] Loading {MODEL_ID}...", flush=True)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )

    if adapter_repo:
        print(f"[model] Loading adapter: {adapter_repo}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()

    model.eval()
    model = torch.compile(model, mode="reduce-overhead")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[model] Ready in {time.time()-t:.0f}s", flush=True)

    # --- Batch inference ---
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts
    all_messages = []
    for ex in examples:
        user_content = ex["prompt"].rstrip() + "\n\n" + ex["question"]
        all_messages.append([{"role": "user", "content": user_content}])

    formatted = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in all_messages
    ]

    responses = []
    t = time.time()
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]  # includes left-pad

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        for j, out in enumerate(outputs):
            generated = out[prompt_len:]
            # Keep special tokens so we can find <think>/<channel> markers
            text = tokenizer.decode(generated, skip_special_tokens=False)
            responses.append(text)

        done = min(i + batch_size, len(formatted))
        elapsed = time.time() - t
        print(f"  [{done}/{len(formatted)}] {elapsed:.0f}s", flush=True)

    # --- Dump raw responses for offline grading ---
    dump_lines = []
    for ex, resp in zip(examples, responses):
        dump_lines.append(json.dumps({
            "category": ex["category"],
            "question": ex["question"],
            "answer": ex["answer"],
            "response": resp,
            "seed": ex.get("seed"),
            "decl_name": ex.get("decl_name"),
        }))
    dump_text = "\n".join(dump_lines)
    print(f"\n[dump] {len(dump_lines)} raw responses saved", flush=True)

    # --- Grade ---
    results = {"total": 0, "correct": 0}
    by_category = {}
    failures = []

    for ex, resp in zip(examples, responses):
        grade = grade_response(ex, resp, GRADERS)
        cat = grade["category"]

        results["total"] += 1
        if grade["correct"]:
            results["correct"] += 1

        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0}
        by_category[cat]["total"] += 1
        if grade["correct"]:
            by_category[cat]["correct"] += 1
        else:
            if len(failures) < 20:  # save first 20 failures for debugging
                failures.append({
                    "category": cat,
                    "question": ex["question"],
                    "ground_truth": ex["answer"][:200],
                    "response": resp[:500],
                    "details": grade["details"],
                })

    # --- Report ---
    print(f"\n{'='*60}", flush=True)
    print(f"COMPREHENSION EVAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    overall_acc = results["correct"] / results["total"] * 100
    print(f"Overall: {results['correct']}/{results['total']} ({overall_acc:.1f}%)", flush=True)
    print(f"\nBy category:", flush=True)
    for cat in sorted(by_category):
        c = by_category[cat]
        acc = c["correct"] / c["total"] * 100
        print(f"  {cat:15s}: {c['correct']}/{c['total']} ({acc:.1f}%)", flush=True)

    if failures:
        print(f"\nSample failures ({len(failures)}):", flush=True)
        for f in failures[:5]:
            print(f"  [{f['category']}] Q: {f['question']}", flush=True)
            print(f"    GT: {f['ground_truth'][:100]}", flush=True)
            print(f"    Resp: {f['response'][:300]}", flush=True)
            print(f"    Detail: {f['details']}", flush=True)

    return json.dumps({
        "overall_accuracy": round(overall_acc, 2),
        "by_category": {k: round(v["correct"]/v["total"]*100, 2)
                        for k, v in by_category.items()},
        "total": results["total"],
        "correct": results["correct"],
        "failures": failures,
        "raw_responses": dump_text,
    }, indent=2)


@app.local_entrypoint()
def main(
    eval_data: str = "lem/data/comprehension_eval.jsonl",
    adapter: str = ADAPTER_REPO,
    limit: int = 0,
):
    """Run comprehension eval."""
    import sys

    eval_path = Path(eval_data)
    if not eval_path.exists():
        print(f"[error] Not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    text = eval_path.read_text()
    if limit > 0:
        lines = text.strip().split("\n")
        text = "\n".join(lines[:limit])
        print(f"[local] Limited to {limit} examples", file=sys.stderr)

    n = text.strip().count("\n") + 1
    print(f"[local] {n} eval examples from {eval_path}", file=sys.stderr)
    print(f"[local] Adapter: {adapter}", file=sys.stderr)

    result_json = run_eval.remote(
        eval_jsonl=text,
        adapter_repo=adapter,
    )

    result = json.loads(result_json)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Overall accuracy: {result['overall_accuracy']}%", file=sys.stderr)
    for cat, acc in result["by_category"].items():
        print(f"  {cat}: {acc}%", file=sys.stderr)

    # Save raw responses for offline grading
    if "raw_responses" in result:
        dump_path = Path("scratch/eval_responses.jsonl")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(result["raw_responses"])
        print(f"  Raw responses saved to {dump_path}", file=sys.stderr)
