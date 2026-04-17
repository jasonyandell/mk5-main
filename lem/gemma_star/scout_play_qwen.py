"""Scout: run the comprehension-trained Qwen on trick-6 decisions with thinking ON.

Purpose: see what reasoning the model produces when asked to make a play.
Does NOT grade, does NOT train — just dumps raw responses for a human to read.

Usage:
    modal run lem/gemma_star/scout_play_qwen.py
    modal run lem/gemma_star/scout_play_qwen.py --decisions lem/data/decisions_scout.jsonl --limit 10
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MODEL_ID = "unsloth/Qwen3-1.7B"
ADAPTER_REPO = "jasonyandell/qwen3-1.7b-texas42-stage0-v6"

app = modal.App("lem-scout-play-qwen")

scout_image = (
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
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=scout_image,
    gpu="B200",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("qwen3-cache", create_if_missing=True)},
)
def run_scout(
    decisions_jsonl: str,
    adapter_repo: str = ADAPTER_REPO,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    system_message: str = "",
    question_suffix: str = "",
    rationalize: bool = False,
    batch_size: int = 16,
) -> str:
    """Run Qwen on decision prompts with thinking mode and return responses.

    If rationalize=True, rewrites the question to reveal the bot's (expert) action
    and ask the model to justify it in the scratchpad. This elicits reasoning on
    confirmed-correct answers — bootstrapping STaR without hand-scripting reasoning.
    """
    import os
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/model-cache"

    examples = [json.loads(line) for line in decisions_jsonl.strip().split("\n") if line.strip()]
    print(f"[data] {len(examples)} decisions", flush=True)

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[model] Ready in {time.time()-t:.0f}s", flush=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts — compact state + question, with thinking ON
    all_messages = []
    for ex in examples:
        if rationalize:
            question = (
                f"A strong player chose to play the {ex['bot_action']} in this position. "
                f"In the scratchpad, reason through why that is a good choice. "
                f"Then end with: 'Play {ex['bot_action']}.'"
            )
        else:
            question = ex["question"]
            if question_suffix:
                question = question + " " + question_suffix
        user_content = ex["prompt"].rstrip() + "\n\n" + question
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_content})
        all_messages.append(msgs)

    formatted = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for msgs in all_messages
    ]

    responses = []
    t = time.time()
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        for out in outputs:
            generated = out[prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=False)
            responses.append(text)

        done = min(i + batch_size, len(formatted))
        print(f"  [{done}/{len(formatted)}] {time.time()-t:.0f}s", flush=True)

    # Pair responses with their grading context
    records = []
    for ex, resp in zip(examples, responses):
        records.append({
            "seed": ex["seed"],
            "decl_name": ex["decl_name"],
            "narrator": ex["narrator"],
            "prompt": ex["prompt"],
            "question": ex["question"],
            "legal_actions": ex["legal_actions"],
            "bot_action": ex["bot_action"],
            "bot_eq": ex["bot_eq"],
            "best_action": ex["best_action"],
            "best_eq": ex["best_eq"],
            "eq_gap": ex["eq_gap"],
            "all_eq": ex["all_eq"],
            "response": resp,
        })

    return "\n".join(json.dumps(r) for r in records)


@app.local_entrypoint()
def main(
    decisions: str = "lem/data/decisions_scout.jsonl",
    adapter: str = ADAPTER_REPO,
    limit: int = 10,
    output: str = "scratch/scout_play_responses.jsonl",
    thinking: bool = True,
    system_message: str = "",
    question_suffix: str = "",
    rationalize: bool = False,
):
    """Run a scout pass on Qwen v6 adapter."""
    import sys

    path = Path(decisions)
    if not path.exists():
        print(f"[error] Not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()
    if limit > 0:
        lines = text.strip().split("\n")
        text = "\n".join(lines[:limit])

    n = text.strip().count("\n") + 1
    print(f"[local] {n} decisions | adapter={adapter} | thinking={thinking}", file=sys.stderr)

    result = run_scout.remote(
        decisions_jsonl=text,
        adapter_repo=adapter,
        enable_thinking=thinking,
        system_message=system_message,
        question_suffix=question_suffix,
        rationalize=rationalize,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result)
    print(f"[local] Wrote {out_path}", file=sys.stderr)
