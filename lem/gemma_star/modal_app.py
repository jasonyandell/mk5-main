"""Modal app for Gemma 4 E2B inference with thinking mode.

First contact: does Gemma, given a Texas 42 narration prompt,
emit coherent reasoning in its thinking channel and pick a legal domino?

Usage:
    # Generate a prompt and run it through Gemma:
    python -m lem.narrate --seed 42 --narrator 3 --decl fives --bid 30 \
        --stop-at-narrator-turn 5 --with-primer \
        > scratch/prompt.txt

    modal run lem/gemma_star/modal_app.py --prompt-file scratch/prompt.txt
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"

app = modal.App("lem-gemma-first-contact")

gemma_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.52",
        "accelerate>=1.2",
        "huggingface_hub>=0.27",
        "peft>=0.14",
        "pillow",  # Gemma 4 E2B is multimodal; processor requires PIL even for text-only
    )
)


@app.function(
    image=gemma_image,
    gpu="L4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # Cache the model weights across invocations
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.6,
             adapter_repo: str = "") -> str:
    """Run Gemma 4 E2B inference with thinking mode enabled.

    Args:
        prompt: The full prompt text.
        max_tokens: Max new tokens to generate.
        temperature: Sampling temperature.
        adapter_repo: HuggingFace repo ID for a LoRA adapter to load on top
                      of the base model (e.g. "jasonyandell/gemma-4-e2b-texas42-stage0").
                      Empty string = no adapter (base model only).

    Returns the full decoded output including thinking channel tokens.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/model-cache"

    print(f"[gemma] Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    if adapter_repo:
        from peft import PeftModel
        print(f"[gemma] Loading LoRA adapter from {adapter_repo}...")
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()
        print("[gemma] Adapter merged.")

    model.eval()
    print(f"[gemma] Model loaded on {next(model.parameters()).device}")

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    input_len = input_ids.shape[-1]
    print(f"[gemma] Prompt: {input_len} tokens")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )

    generated = outputs[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    n_generated = len(generated)
    print(f"[gemma] Generated {n_generated} tokens")

    return text


@app.local_entrypoint()
def main(prompt_file: str = "", max_tokens: int = 2048, temperature: float = 0.6,
         adapter: str = ""):
    """Read a prompt from file (or stdin if not given) and run Gemma.

    Args:
        adapter: HF repo for LoRA adapter (e.g. "jasonyandell/gemma-4-e2b-texas42-stage0").
    """
    import sys

    if prompt_file:
        prompt = Path(prompt_file).read_text()
    else:
        print("[local] Reading prompt from stdin...", file=sys.stderr)
        prompt = sys.stdin.read()

    if not prompt.strip():
        print("[error] Empty prompt", file=sys.stderr)
        sys.exit(1)

    label = "base" if not adapter else f"adapter={adapter}"
    print(f"[local] Sending {len(prompt)} chars to Gemma ({label}) on Modal...", file=sys.stderr)
    result = generate.remote(prompt, max_tokens=max_tokens, temperature=temperature,
                             adapter_repo=adapter)

    print("=" * 60)
    print("GEMMA 4 E2B RESPONSE")
    print("=" * 60)
    print(result)
