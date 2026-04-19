"""Modal vLLM endpoint for Gemma 4 E2B — Burl's agent-loop LLM.

Class-based serving pattern. The engine loads once in `@modal.enter()` and
vLLM's continuous batching fans parallel Burl rollouts across one L4 via
`@modal.concurrent(max_inputs=4)`.

Usage:
    modal run burl/modal/gemma_serve.py::warmup
    modal run burl/modal/gemma_serve.py::smoke

Harness import:
    from burl.modal.gemma_serve import GemmaServer
    server = GemmaServer()
    text = server.generate.remote("...", max_tokens=512, stop=["</tool>"])
"""

from __future__ import annotations

import modal

MODEL_ID = "google/gemma-4-E2B-it"
MODELS_VOL = "/model-cache"

app = modal.App("burl-gemma-serve")

gemma_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": MODELS_VOL})
)

vol = modal.Volume.from_name("gemma-e2b-cache", create_if_missing=True)


@app.cls(
    gpu="L4",
    image=gemma_image,
    volumes={MODELS_VOL: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
@modal.concurrent(max_inputs=4)
class GemmaServer:
    @modal.enter()
    def load(self) -> None:
        import threading
        import time

        from transformers import AutoTokenizer
        from vllm import LLM

        # vLLM V1 sync LLM.generate() cannot be called from multiple threads on
        # the same engine (ZMQ router asserts). Modal's @concurrent fans inbound
        # requests onto one container; we serialize access here and let the
        # decorator provide queue depth. Upgrading to AsyncLLMEngine would
        # unlock true continuous batching — deferred.
        self._gen_lock = threading.Lock()
        t0 = time.time()
        print(f"[gemma-serve] Loading vLLM engine for {MODEL_ID}...")
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"[gemma-serve] Ready in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        temperature: float = 0.6,
    ) -> str:
        import time

        from vllm import SamplingParams

        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False: Gemma 4's two-channel thinking mode injects a
        # literal "thought\n<prose>" preamble into responses, which our XML-tag
        # parser (<think>, <tool>, <commit>) cannot recover from. move3-grader's
        # debug-one confirmed the model reasons correctly but emits markdown
        # prose instead of XML under enable_thinking=True. If we want Gemma's
        # thinking channel later, route it to our <think> tags explicitly rather
        # than rely on the template's built-in format.
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

        t0 = time.time()
        with self._gen_lock:
            outputs = self.llm.generate([formatted], params)
        elapsed = time.time() - t0

        out = outputs[0].outputs[0]
        text = out.text
        n_tokens = len(out.token_ids)
        tok_s = n_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[gemma-serve] {n_tokens} tok in {elapsed:.2f}s "
            f"= {tok_s:.1f} tok/s"
        )
        return text


@app.local_entrypoint(name="warmup")
def warmup() -> None:
    """Single-shot verify: load + one generate. Reports cold-start + latency."""
    import time

    server = GemmaServer()
    t0 = time.time()
    print("[warmup] Dispatching first generate (triggers cold start)...")
    text = server.generate.remote(
        "Say 'hello from gemma' and nothing else.",
        max_tokens=32,
    )
    total = time.time() - t0
    print("=" * 60)
    print(f"[warmup] End-to-end: {total:.1f}s (includes cold start)")
    print(f"[warmup] Output: {text!r}")


@app.local_entrypoint(name="smoke")
def smoke() -> None:
    """Three prompts — verifies warm tok/s and concurrency via spawn fan-out."""
    import time

    prompts = [
        "Name one Texas county. One word.",
        "What is 17 + 25? Reply with just the number.",
        (
            "You have tools `is_legal(dom)` and `trump_declared()`. "
            "Briefly describe the first tool call you'd make if asked to play. "
            "Respond in one sentence."
        ),
    ]

    server = GemmaServer()
    t_all = time.time()
    handles = [server.generate.spawn(p, max_tokens=128) for p in prompts]
    results = [h.get() for h in handles]
    wall = time.time() - t_all

    print("=" * 60)
    print(f"[smoke] Wall clock for {len(prompts)} prompts: {wall:.1f}s")
    for i, (p, r) in enumerate(zip(prompts, results)):
        print("-" * 60)
        print(f"[{i}] PROMPT: {p}")
        print(f"[{i}] OUTPUT: {r.strip()}")
