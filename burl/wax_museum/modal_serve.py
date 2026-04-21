"""Modal L4 endpoint for wax_museum — 32k context, base Gemma 4 E2B, no LoRA.

Parallel to ``burl.modal.gemma_serve_native``. Distinct Modal app name
(``burl-wax-museum-serve``) so it can run alongside the production endpoint.

Why a separate serve file:
  - Bumps ``max_model_len`` from 8192 → 32768. The ergo probes (Practicality 1)
    ended at a 512-token wall; the production endpoint raised it to 2048 then
    8192. We want headroom for "let it think explicitly" on the hard-gated
    turns — 32k leaves room for verbose reasoning plus multi-turn history.
  - LoRA is disabled: this pilot runs on base Gemma only. No adapters needed.
  - ``enable_thinking`` stays False on the template (Practicality 1): the
    reasoning we want is on the answer channel where we can see it.

Usage:
    modal run burl/wax_museum/modal_serve.py::warmup
"""

from __future__ import annotations

import modal

MODEL_ID = "google/gemma-4-E2B-it"
MODELS_VOL = "/model-cache"

app = modal.App("burl-wax-museum-serve")

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
@modal.concurrent(max_inputs=2)
class WaxMuseumServer:
    @modal.enter()
    def load(self) -> None:
        import threading
        import time

        from transformers import AutoTokenizer
        from vllm import LLM

        self._gen_lock = threading.Lock()
        t0 = time.time()
        print(f"[wax-museum] Loading vLLM engine for {MODEL_ID} @ max_model_len=32768...")
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
            hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"[wax-museum] Ready in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        stop: list[str] | None = None,
        enable_thinking: bool = False,
    ) -> dict:
        """Single-shot generation matching GemmaServerNative.generate_native's shape."""
        import time

        from vllm import SamplingParams

        template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if tools is not None:
            template_kwargs["tools"] = tools

        formatted = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            skip_special_tokens=False,
        )

        t0 = time.time()
        with self._gen_lock:
            outputs = self.llm.generate([formatted], params)
        elapsed = time.time() - t0

        out = outputs[0].outputs[0]
        text = out.text
        n_tokens = len(out.token_ids)
        tok_s = n_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[wax-museum] {n_tokens} tok in {elapsed:.2f}s = {tok_s:.1f} tok/s")
        return {
            "text": text,
            "prompt_text": formatted,
            "n_tokens": int(n_tokens),
            "elapsed_s": round(elapsed, 2),
        }


@app.local_entrypoint(name="warmup")
def warmup() -> None:
    """Cold-start verify: load + one call with wax_museum's turn-1 tool surface."""
    import time

    server = WaxMuseumServer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "explore_game",
                "description": "Examine a candidate play and return its outcome distribution shape.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "play": {"type": "integer", "description": "domino_id 0..27"},
                    },
                    "required": ["play"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are Burl, a Texas 42 dominoes agent. On turn 1 you only "
                "have `explore_game`. Pick any domino_id in [14, 21, 27] and "
                "call explore_game(play=<that id>) to examine it. Reason "
                "briefly before calling."
            ),
        },
        {
            "role": "user",
            "content": "Your hand includes 14(5-2), 21(6-0), 27(6-6). Pick one to explore.",
        },
    ]

    t0 = time.time()
    print("[warmup] dispatching first generate (cold start + 32k model load)...")
    result = server.generate.remote(messages, tools=tools, max_tokens=1024)
    total = time.time() - t0
    print("=" * 60)
    print(f"[warmup] end-to-end {total:.1f}s  n_tokens={result['n_tokens']}")
    print("-" * 60)
    print(result["text"])
    print("-" * 60)
