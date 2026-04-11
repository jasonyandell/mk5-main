"""STaR harness for Stage 1: inference → grade → rationalize → training set.

Runs Gemma 4 E2B on narration prompts, grades responses against the E[Q] oracle's
actions (K1: beat the bot), rationalizes failures (R1: reveal correct action, ask
model to explain why), and outputs a LoRA training dataset.

Usage:
    # Run STaR iteration on Modal
    modal run lem/gemma_star/star_harness.py \
        --narrations lem/data/narrations_train.jsonl \
        --output lem/data/star_iter0.jsonl

    # With adapter from previous iteration
    modal run lem/gemma_star/star_harness.py \
        --narrations lem/data/narrations_train.jsonl \
        --adapter jasonyandell/gemma-4-e2b-texas42-stage0 \
        --output lem/data/star_iter1.jsonl

    # Limit to N examples (for testing)
    modal run lem/gemma_star/star_harness.py \
        --narrations lem/data/narrations_train.jsonl --limit 20 \
        --output scratch/star_test.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"

app = modal.App("lem-star-harness")

star_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52",
        "accelerate>=1.2",
        "peft>=0.14",
        "huggingface_hub>=0.27",
        "pillow",
    )
)


def parse_play(response_text: str) -> str | None:
    """Extract the domino the model chose to play from its response.

    Looks for patterns like "play 6-2", "Play: 5-5", "play the 3-1",
    or a bare domino "6-2" at the end. Returns the domino as "H-L" or None.
    """
    # Try explicit "play X-Y" patterns
    patterns = [
        r"[Pp]lay[:\s]+(?:the\s+)?(\d-\d)",
        r"[Aa]nswer[:\s]+(?:the\s+)?(\d-\d)",
        r"[Cc]hoice[:\s]+(?:the\s+)?(\d-\d)",
        r"I (?:would |will |should )?play (?:the )?(\d-\d)",
        r"\*\*(\d-\d)\*\*",  # bold domino
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)

    # Fallback: last domino-like pattern in the text
    all_doms = re.findall(r"\b(\d-\d)\b", response_text)
    if all_doms:
        return all_doms[-1]

    return None


def grade_k1(gemma_action: str | None, bot_action: str, bot_eq: float,
             all_eq: dict[str, float], legal_actions: list[str]) -> dict:
    """Grade a response using K1: did Gemma beat the bot?

    Returns a dict with grading results.
    """
    if gemma_action is None:
        return {"grade": "parse_fail", "gemma_action": None, "reason": "Could not parse action"}

    if gemma_action not in legal_actions:
        return {"grade": "illegal", "gemma_action": gemma_action,
                "reason": f"{gemma_action} is not a legal play (legal: {legal_actions})"}

    gemma_eq = all_eq.get(gemma_action, float("-inf"))

    if gemma_eq >= bot_eq:
        return {
            "grade": "pass",
            "gemma_action": gemma_action,
            "gemma_eq": gemma_eq,
            "bot_eq": bot_eq,
            "delta": round(gemma_eq - bot_eq, 3),
        }
    else:
        return {
            "grade": "fail",
            "gemma_action": gemma_action,
            "gemma_eq": gemma_eq,
            "bot_eq": bot_eq,
            "delta": round(gemma_eq - bot_eq, 3),
        }


@app.function(
    image=star_image,
    gpu="A10G",  # $1.10/hr — more memory bandwidth than L4 for generation
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def run_star_iteration(
    narrations_jsonl: str,
    adapter_repo: str = "",
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
) -> str:
    """Run one STaR iteration: inference + grading + rationalization.

    Args:
        narrations_jsonl: JSONL string with narration examples.
        adapter_repo: HF repo for LoRA adapter (empty = base model).
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        JSONL string of training examples (winning traces + rationalizations).
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/model-cache"

    # --- Patch ClippableLinear for PEFT ---
    if adapter_repo:
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
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
    )

    if adapter_repo:
        from peft import PeftModel
        print(f"[model] Loading adapter from {adapter_repo}...")
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()
        print("[model] Adapter merged")

    model.eval()

    def generate_response(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_tensors="pt", return_dict=True,
        )
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True,
            )
        generated = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=False)

    # --- Process examples ---
    examples = [json.loads(line) for line in narrations_jsonl.strip().split("\n")]
    print(f"[star] Processing {len(examples)} examples...")

    results = []
    stats = {"pass": 0, "fail": 0, "illegal": 0, "parse_fail": 0, "rationalized": 0}

    for i, ex in enumerate(examples):
        # Phase 1: Inference
        response = generate_response(ex["prompt"])
        gemma_action = parse_play(response)

        # Phase 2: Grade
        grade_result = grade_k1(
            gemma_action, ex["bot_action"], ex["bot_eq"],
            ex["all_eq"], ex["legal_actions"],
        )
        stats[grade_result["grade"]] = stats.get(grade_result["grade"], 0) + 1

        if grade_result["grade"] == "pass":
            # Keep winning trace as training example
            results.append({
                "type": "win",
                "seed": ex["seed"],
                "decl_id": ex["decl_id"],
                "narrator": ex["narrator"],
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": response},
                ],
                "gemma_action": gemma_action,
                "gemma_eq": grade_result.get("gemma_eq"),
                "bot_eq": ex["bot_eq"],
                "delta": grade_result.get("delta"),
            })

        elif grade_result["grade"] in ("fail", "illegal", "parse_fail"):
            # Phase 3: Rationalization — reveal correct action, ask why
            rationalization_prompt = (
                ex["prompt"].rstrip()
                + f"\n\nThe correct play here is {ex['best_action']}. "
                f"Explain why {ex['best_action']} is the best choice."
            )
            rational_response = generate_response(rationalization_prompt)

            results.append({
                "type": "rationalization",
                "seed": ex["seed"],
                "decl_id": ex["decl_id"],
                "narrator": ex["narrator"],
                "messages": [
                    {"role": "user", "content": rationalization_prompt},
                    {"role": "assistant", "content": rational_response},
                ],
                "original_grade": grade_result["grade"],
                "gemma_action": gemma_action,
                "correct_action": ex["best_action"],
                "best_eq": ex["best_eq"],
                "bot_eq": ex["bot_eq"],
            })
            stats["rationalized"] += 1

        if (i + 1) % 10 == 0:
            print(f"[star] {i+1}/{len(examples)} | "
                  f"pass={stats['pass']} fail={stats['fail']} "
                  f"illegal={stats['illegal']} parse_fail={stats['parse_fail']} "
                  f"rationalized={stats['rationalized']}")

    print(f"\n[star] Complete. {len(results)} training examples generated.")
    print(f"  Pass (kept traces): {stats['pass']}")
    print(f"  Fail → rationalized: {stats['fail']}")
    print(f"  Illegal → rationalized: {stats['illegal']}")
    print(f"  Parse fail → rationalized: {stats['parse_fail']}")

    # Return as JSONL
    return "\n".join(json.dumps(r) for r in results)


@app.local_entrypoint()
def main(
    narrations: str = "lem/data/narrations_train.jsonl",
    adapter: str = "",
    output: str = "lem/data/star_iter0.jsonl",
    limit: int = 0,
    temperature: float = 0.6,
):
    """Run one STaR iteration."""
    import sys

    narrations_path = Path(narrations)
    if not narrations_path.exists():
        print(f"[error] Narrations not found: {narrations_path}", file=sys.stderr)
        sys.exit(1)

    narrations_text = narrations_path.read_text()
    if limit > 0:
        lines = narrations_text.strip().split("\n")
        narrations_text = "\n".join(lines[:limit])
        print(f"[local] Limited to {limit} examples", file=sys.stderr)

    n_examples = narrations_text.count("\n") + 1
    label = "base" if not adapter else f"adapter={adapter}"
    print(f"[local] {n_examples} examples, model={label}", file=sys.stderr)

    result_jsonl = run_star_iteration.remote(
        narrations_jsonl=narrations_text,
        adapter_repo=adapter,
        temperature=temperature,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_jsonl)

    n_results = result_jsonl.count("\n") + 1 if result_jsonl.strip() else 0
    print(f"[local] Wrote {n_results} training examples to {output_path}", file=sys.stderr)
