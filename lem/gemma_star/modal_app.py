"""Modal app for Gemma 4 E2B inference via vLLM offline.

Single-prompt inference with thinking mode. Uses vLLM offline LLM class
for flash attention, paged KV cache — no HTTP server needed.

Usage:
    python -m lem.narrate --seed 42 --narrator 3 --decl fives --bid 30 \
        --stop-at-narrator-turn 5 --with-primer \
        > scratch/prompt.txt

    modal run lem/gemma_star/modal_app.py --prompt-file scratch/prompt.txt

    # With LoRA adapter:
    modal run lem/gemma_star/modal_app.py --prompt-file scratch/prompt.txt \
        --adapter jasonyandell/gemma-4-e2b-texas42-stage0
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"

app = modal.App("lem-gemma-first-contact")

gemma_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=gemma_image,
    gpu="L4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.6,
             adapter_repo: str = "") -> str:
    """Run Gemma 4 E2B inference via vLLM offline with thinking mode."""
    import os

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["HF_HOME"] = "/model-cache"

    kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )

    lora_request = None
    if adapter_repo:
        from vllm.lora.request import LoRARequest
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = 64

    print(f"[gemma] Loading vLLM engine...")
    llm = LLM(**kwargs)

    if adapter_repo:
        adapter_local = snapshot_download(adapter_repo)
        lora_request = LoRARequest("adapter", 1, adapter_local)
        print(f"[gemma] LoRA adapter: {adapter_repo}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    print(f"[gemma] Prompt: {len(formatted)} chars")
    outputs = llm.generate(
        [formatted],
        SamplingParams(temperature=temperature, max_tokens=max_tokens),
        lora_request=lora_request,
    )

    text = outputs[0].outputs[0].text
    print(f"[gemma] Generated {len(text)} chars")
    return text


@app.local_entrypoint()
def main(prompt_file: str = "", max_tokens: int = 1024, temperature: float = 0.6,
         adapter: str = ""):
    """Read a prompt from file (or stdin) and run Gemma."""
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
