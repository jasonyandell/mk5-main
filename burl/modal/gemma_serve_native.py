"""Modal vLLM endpoint for Gemma 4 E2B — native tool-use spike.

Parallel to `gemma_serve.py`. Distinct Modal app name so it can run alongside
the production endpoint without collision. The key difference: this server
accepts structured `messages` + `tools` and feeds them into
`tokenizer.apply_chat_template(..., tools=tools)`, so Gemma sees tool
declarations in the native shape it was post-trained on (`<|tool>...<tool|>`
block + Python signatures) instead of a prose tool menu in the user turn.

It also sets `skip_special_tokens=False` on the vLLM detokenizer so the
`<|tool_call>...<tool_call|>` special tokens survive into the returned text.
Our parser (`tool_loop_native.py`) keys off them.

Usage:
    modal run burl/modal/gemma_serve_native.py::warmup
    modal run burl/modal/gemma_serve_native.py::smoke

Harness import:
    from burl.modal.gemma_serve_native import GemmaServerNative
    server = GemmaServerNative()
    text = server.generate_native.remote(messages, tools=[...], max_tokens=2048)
"""

from __future__ import annotations

import modal

MODEL_ID = "google/gemma-4-E2B-it"
MODELS_VOL = "/model-cache"

# Distinct app name — safe to run concurrently with `burl-gemma-serve`.
app = modal.App("burl-gemma-serve-native")

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
class GemmaServerNative:
    @modal.enter()
    def load(self) -> None:
        import threading
        import time

        from transformers import AutoTokenizer
        from vllm import LLM

        self._gen_lock = threading.Lock()
        t0 = time.time()
        print(f"[gemma-native] Loading vLLM engine for {MODEL_ID}...")
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
        print(f"[gemma-native] Ready in {time.time() - t0:.1f}s")

    @modal.method()
    def generate_native(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        stop: list[str] | None = None,
        enable_thinking: bool = False,
    ) -> dict:
        """Native Gemma 4 tool-use generation.

        `messages` is an HF-shaped chat history (roles: system/user/assistant/tool).
        `tools` is a list of JSON-schema tool specs; when supplied, the chat
        template renders them as Gemma's native `<|tool>...<tool|>` block.

        Returns `{"text": str, "prompt_text": str, "n_tokens": int}`. The raw
        `prompt_text` is round-tripped so the client can log exactly what the
        server sent into vLLM.
        """
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

        # skip_special_tokens=False keeps <|tool_call> and friends in the output
        # so our parser can key off them.
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
        print(
            f"[gemma-native] {n_tokens} tok in {elapsed:.2f}s "
            f"= {tok_s:.1f} tok/s"
        )
        return {
            "text": text,
            "prompt_text": formatted,
            "n_tokens": int(n_tokens),
        }


@app.local_entrypoint(name="warmup")
def warmup() -> None:
    """Single-shot verify: load + one generate with tools. Reports shape."""
    import time

    server = GemmaServerNative()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Return the sum of two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "You have one tool: add(a, b). Call it to compute 17 + 25. "
                "Emit exactly one tool call, then stop."
            ),
        }
    ]

    t0 = time.time()
    print("[warmup] dispatching first generate (triggers cold start)...")
    result = server.generate_native.remote(
        messages, tools=tools, max_tokens=256,
    )
    total = time.time() - t0
    print("=" * 60)
    print(f"[warmup] End-to-end: {total:.1f}s (includes cold start)")
    print(f"[warmup] n_tokens: {result['n_tokens']}")
    print(f"[warmup] prompt_text ({len(result['prompt_text'])} chars):")
    print("-" * 60)
    print(result["prompt_text"])
    print("-" * 60)
    print(f"[warmup] raw completion ({len(result['text'])} chars):")
    print("-" * 60)
    print(result["text"])
    print("-" * 60)


@app.local_entrypoint(name="smoke")
def smoke() -> None:
    """Burl-shaped probe: give the model 3 tool declarations and see what it picks."""
    import time

    tools = [
        {
            "type": "function",
            "function": {
                "name": "is_legal",
                "description": "Return whether playing domino_id is legal for me right now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domino_id": {"type": "integer", "description": "0..27"},
                    },
                    "required": ["domino_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "trump_declared",
                "description": "Return the trump suit for this hand.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "eq_outcome_distribution",
                "description": (
                    "Return an expected-value outcome distribution for playing this domino_id."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "play": {"type": "integer", "description": "0..27"},
                        "n_samples": {"type": "integer", "default": 10},
                    },
                    "required": ["play"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "user",
            "content": (
                "I'm playing Texas 42. My hand is id=15 (5-0) and id=23 (6-2). "
                "I am seat 3, leading trick 6. Pick ONE tool call to make first "
                "to help you decide what to play. Then stop."
            ),
        }
    ]

    server = GemmaServerNative()
    t0 = time.time()
    result = server.generate_native.remote(
        messages, tools=tools, max_tokens=512, enable_thinking=False,
    )
    wall = time.time() - t0
    print("=" * 60)
    print(f"[smoke] wall={wall:.1f}s  n_tokens={result['n_tokens']}")
    print("-" * 60)
    print("prompt_text:")
    print(result["prompt_text"])
    print("-" * 60)
    print("completion:")
    print(result["text"])
    print("-" * 60)
