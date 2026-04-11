"""Modal app for Gemma 4 E2B inference via vLLM.

Single-prompt inference with thinking mode. Uses vLLM for optimized generation
(flash attention, paged KV cache) instead of raw HF generate().

Usage:
    # Generate a prompt and run it through Gemma:
    python -m lem.narrate --seed 42 --narrator 3 --decl fives --bid 30 \
        --stop-at-narrator-turn 5 --with-primer \
        > scratch/prompt.txt

    modal run lem/gemma_star/modal_app.py --prompt-file scratch/prompt.txt

    # With LoRA adapter (merged into base before serving):
    modal run lem/gemma_star/modal_app.py --prompt-file scratch/prompt.txt \
        --adapter jasonyandell/gemma-4-e2b-texas42-stage0
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-E2B-it"
VLLM_PORT = 8000

app = modal.App("lem-gemma-first-contact")

gemma_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm==0.19.0",
        "transformers==5.5.0",
        "peft>=0.14",
        "openai>=1.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def _merge_adapter_to_disk(adapter_repo: str, output_dir: str):
    """Merge a LoRA adapter into the base model and save to disk for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_repo)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(output_dir)
    del model


@app.function(
    image=gemma_image,
    gpu="L4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/model-cache": modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)},
)
def generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.6,
             adapter_repo: str = "") -> str:
    """Run Gemma 4 E2B inference via vLLM with thinking mode enabled."""
    import os
    import subprocess
    import time
    import urllib.request
    import urllib.error

    from openai import OpenAI

    os.environ["HF_HOME"] = "/model-cache"

    MERGED_DIR = "/tmp/merged-model"

    # Prepare model path
    if adapter_repo:
        print(f"[gemma] Merging adapter {adapter_repo}...")
        _merge_adapter_to_disk(adapter_repo, MERGED_DIR)
        model_path = MERGED_DIR
    else:
        model_path = MODEL_ID

    # Start vLLM server
    cmd = [
        "vllm", "serve", model_path,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", "image=0,video=0,audio=0",
        "--dtype", "bfloat16",
        "--uvicorn-log-level", "warning",
    ]
    print(f"[vllm] Starting server...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for health
    start = time.time()
    while time.time() - start < 300:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if resp.status == 200:
                print(f"[vllm] Ready in {time.time()-start:.0f}s")
                break
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    else:
        proc.kill()
        raise TimeoutError("vLLM failed to start")

    # Generate via OpenAI API
    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")

    response = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    msg = response.choices[0].message
    parts = []
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if msg.content:
        parts.append(msg.content)
    text = "\n".join(parts)

    # Cleanup
    import signal
    proc.send_signal(signal.SIGTERM)

    print(f"[gemma] Generated {len(text)} chars")
    return text


@app.local_entrypoint()
def main(prompt_file: str = "", max_tokens: int = 1024, temperature: float = 0.6,
         adapter: str = ""):
    """Read a prompt from file (or stdin if not given) and run Gemma."""
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
