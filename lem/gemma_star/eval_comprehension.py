"""Eval Stage 0 v4: measure comprehension accuracy per question type.

Runs inference on held-out comprehension eval set and grades answers.
Key metric: 0% illegal on legal_moves questions.

Usage:
    modal run lem/gemma_star/eval_comprehension.py
    modal run lem/gemma_star/eval_comprehension.py --adapter jasonyandell/gemma-4-e2b-texas42-stage0-v4
    modal run lem/gemma_star/eval_comprehension.py --limit 50  # quick test
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_REPO = "jasonyandell/gemma-4-e2b-texas42-stage0-v4"

app = modal.App("lem-comprehension-eval")

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


def _normalize_dom(d: str) -> str:
    """Normalize domino string to H-L format."""
    parts = d.split("-")
    if len(parts) != 2:
        return d
    a, b = int(parts[0]), int(parts[1])
    return f"{max(a,b)}-{min(a,b)}"


def _extract_legal_moves(response: str) -> set[str]:
    """Extract legal moves from a response. Looks for 'Legal moves: X, Y' pattern."""
    # Try explicit "Legal moves:" or "Legal move:"
    m = re.search(r"[Ll]egal moves?:\s*(.+?)(?:\.|$)", response)
    if m:
        doms = re.findall(r"\b(\d-\d)\b", m.group(1))
        return {_normalize_dom(d) for d in doms}
    # Fallback: "Play: X" pattern
    m = re.search(r"[Pp]lay:\s*(\d-\d)", response)
    if m:
        return {_normalize_dom(m.group(1))}
    return set()


def _extract_play(response: str) -> str | None:
    """Extract a single played domino from response."""
    # Try "Play: X"
    m = re.search(r"[Pp]lay:\s*(\d-\d)", response)
    if m:
        return _normalize_dom(m.group(1))
    # Try "Legal moves: X" (single move)
    moves = _extract_legal_moves(response)
    if len(moves) == 1:
        return next(iter(moves))
    return None


def _clean_response(response: str, question: str) -> str:
    """Strip thinking blocks, prompt echo, and formatting from model response.

    Gemma 4 generates <think>...</think> reasoning before the answer.
    We grade only on what comes after the thinking block.
    """
    import re

    # Strip Gemma 4 thinking block: <|channel>thought\n...<channel|>
    # Take everything AFTER the last <channel|> (end of thinking)
    eoc = "<channel|>"
    idx = response.rfind(eoc)
    if idx >= 0:
        response = response[idx + len(eoc):]
    else:
        # Also try </think> for other model variants
        idx = response.rfind("</think>")
        if idx >= 0:
            response = response[idx + len("</think>"):]

    # Strip prompt echo (if response contains the question, take everything after)
    q_lower = question.lower().strip().rstrip("?")
    resp_lower = response.lower()
    idx = resp_lower.rfind(q_lower)
    if idx >= 0:
        after = response[idx + len(q_lower):]
        after = after.lstrip("?\n\r \t")
        if after:
            response = after

    # Strip "model\n" prefix if present
    if response.lstrip().lower().startswith("model"):
        response = response.lstrip()
        response = response[5:].lstrip("\n\r \t")

    return response.strip()


def grade_response(example: dict, response: str) -> dict:
    """Grade a model response against ground truth.

    Returns: {correct: bool, details: str, category: str}
    """
    category = example["category"]
    ground_truth = example["answer"]
    response = _clean_response(response, example["question"])

    if category == "legal_moves":
        # Extract legal moves from both ground truth and response
        gt_moves = _extract_legal_moves(ground_truth)
        resp_moves = _extract_legal_moves(response)

        if not resp_moves:
            return {"correct": False, "details": "parse_fail: no legal moves found",
                    "category": category}

        if resp_moves == gt_moves:
            return {"correct": True, "details": "exact match",
                    "category": category}

        # Check if response moves are a subset (model might be too conservative)
        # or superset (model allowing illegal moves)
        extra = resp_moves - gt_moves
        missing = gt_moves - resp_moves
        details = []
        if extra:
            details.append(f"illegal_included: {extra}")
        if missing:
            details.append(f"legal_missed: {missing}")
        return {"correct": False, "details": "; ".join(details),
                "category": category}

    elif category == "is_trump":
        # Check yes/no matches
        gt_yes = ground_truth.lower().startswith("yes")
        resp_yes = response.lower().strip().startswith("yes")
        correct = gt_yes == resp_yes
        return {"correct": correct,
                "details": f"gt={'yes' if gt_yes else 'no'}, resp={'yes' if resp_yes else 'no'}",
                "category": category}

    elif category == "where_is":
        # Check key facts: "in your hand", "played by X on trick N", "not been played"
        gt_lower = ground_truth.lower()
        resp_lower = response.lower()
        if "in your hand" in gt_lower:
            correct = "in your hand" in resp_lower or "your hand" in resp_lower
        elif "was played by" in gt_lower:
            # Extract trick number from ground truth
            m = re.search(r"trick (\d)", gt_lower)
            gt_trick = m.group(1) if m else None
            correct = gt_trick is not None and f"trick {gt_trick}" in resp_lower
        elif "not been played" in gt_lower:
            correct = ("not" in resp_lower and "played" in resp_lower) or "unknown" in resp_lower
        else:
            correct = False
        return {"correct": correct, "details": "", "category": category}

    elif category == "count_status":
        gt_lower = ground_truth.lower()
        resp_lower = response.lower()
        # Key facts: captured by whom, still out, in your hand
        if "your team" in gt_lower:
            correct = "your team" in resp_lower or "you" in resp_lower
        elif "opponents" in gt_lower:
            correct = "opponent" in resp_lower
        elif "in your hand" in gt_lower:
            correct = "your hand" in resp_lower
        elif "still out" in gt_lower or "not been played" in gt_lower:
            correct = "not" in resp_lower or "still" in resp_lower or "out" in resp_lower
        else:
            correct = False
        return {"correct": correct, "details": "", "category": category}

    elif category == "what_beats":
        # Check if the key dominos mentioned in ground truth appear in response
        gt_doms = set(re.findall(r"\b(\d-\d)\b", ground_truth))
        resp_doms = set(re.findall(r"\b(\d-\d)\b", response))
        # Normalize
        gt_doms = {_normalize_dom(d) for d in gt_doms}
        resp_doms = {_normalize_dom(d) for d in resp_doms}
        # The response should mention the same beaters
        # Allow some flexibility — check if gt beaters are subset of resp
        if "nothing can beat" in ground_truth.lower():
            correct = "nothing" in response.lower() or "no" in response.lower() or "highest" in response.lower()
        else:
            # Check overlap — at least 80% of GT dominoes mentioned
            if gt_doms:
                overlap = len(gt_doms & resp_doms) / len(gt_doms)
                correct = overlap >= 0.8
            else:
                correct = True
        return {"correct": correct, "details": f"gt_doms={gt_doms}, resp_doms={resp_doms}",
                "category": category}

    return {"correct": False, "details": "unknown category", "category": category}


@app.function(
    image=eval_image,
    gpu="B200",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
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

    os.environ["HF_HOME"] = "/model-cache"

    examples = [json.loads(line) for line in eval_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} eval examples", flush=True)

    # --- Patch + load model ---
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
                eos_token_id=[1, 106],  # <eos> + <turn|> (end of model turn)
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
        grade = grade_response(ex, resp)
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
