"""In-process MLX-LM-backed Gemma 4 E2B callable for local rollouts.

Parallel to ``burl/modal/gemma_serve_native.py`` but runs on an Apple-silicon
host via MLX-LM instead of Modal/vLLM. Exposes the same
``generate_native(messages, tools, ...)`` shape so the rollout harness can
swap backends with a one-line edit.

Two things the Modal server does for us that we have to reproduce by hand:

1. ``skip_special_tokens=False`` on detokenization. The downstream parser
   (``burl/harness/tool_loop_native.py``) keys off the literal
   ``<|tool_call>...<tool_call|>`` pair, so those tokens must survive into
   the returned string. MLX-LM's ``SPMStreamingDetokenizer`` already does
   this because its token map stores the raw UTF-8 bytes for every id
   (verified interactively on ``mlx-community/gemma-4-e2b-it-bf16``).
2. Turn-terminator stop. Gemma 4's chat template ends each assistant turn
   with ``<turn|>`` (id 106). MLX-LM's tokenizer loader already registers
   ``{1, 50, 106}`` as the EOS set for this model, so ``stream_generate``
   stops on ``<turn|>`` and ``<tool_response|>`` naturally — no extra
   scaffolding needed.

Usage as a library::

    from burl.modal.gemma_local import make_local_native_model
    model_fn = make_local_native_model()
    text = model_fn(messages, tools)

CLI smoke::

    python -m burl.modal.gemma_local --smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

log = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "mlx-community/gemma-4-e2b-it-bf16"

# NativeModelCallable = Callable[[list[dict], list[dict]], str]
# (kept as a forward-string to avoid importing burl.harness at module import)
NativeModelCallable = Callable[[list[dict], list[dict]], str]


class GemmaLocalNative:
    """Local MLX-LM replacement for ``GemmaServerNative.generate_native``.

    Loads the model+tokenizer once in ``__init__``. The returned
    ``generate_native`` signature matches the Modal method verbatim so the
    rollout script only has to change the adapter factory.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        adapter_path: str | None = None,
    ) -> None:
        t0 = time.time()
        log.info("[gemma-local] loading %s (adapter=%s)", model_repo, adapter_path)
        self.model_repo = model_repo
        self.adapter_path = adapter_path
        # ``mlx_lm.load`` already wires the right streaming detokenizer
        # (SPM with trim_space=False for Gemma 4) and registers EOS ids
        # {1, 50, 106} so <turn|> terminates generation.
        self.model, self.tokenizer = load(
            model_repo, adapter_path=adapter_path,
        )
        log.info("[gemma-local] ready in %.1fs", time.time() - t0)

    def generate_native(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        stop: list[str] | None = None,  # noqa: ARG002 — accepted for parity; MLX stops on EOS ids
        enable_thinking: bool = False,
        adapter_name: str | None = None,
    ) -> dict:
        """Generate one assistant turn.

        Returns ``{"text": str, "prompt_text": str, "n_tokens": int}`` — same
        shape as the Modal server so the call site is backend-agnostic.

        ``adapter_name`` is accepted for interface parity but logs a warning
        if used: local adapters are loaded once at ``__init__`` via
        ``adapter_path``, not swapped per-call.
        """
        if adapter_name is not None:
            log.warning(
                "[gemma-local] ignoring adapter_name=%r at call time; "
                "local adapters are bound at __init__ via adapter_path.",
                adapter_name,
            )

        template_kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if tools is not None:
            template_kwargs["tools"] = tools

        prompt_text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        assert isinstance(prompt_text, str), (
            "apply_chat_template with tokenize=False should return a string"
        )

        sampler = make_sampler(temp=float(temperature))

        text_parts: list[str] = []
        n_tokens = 0
        t0 = time.time()
        last_response = None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            text_parts.append(response.text)
            n_tokens = response.generation_tokens
            last_response = response
        elapsed = time.time() - t0

        text = "".join(text_parts)
        tok_s = n_tokens / elapsed if elapsed > 0 else 0.0
        finish = getattr(last_response, "finish_reason", None)
        log.info(
            "[gemma-local] %d tok in %.2fs = %.1f tok/s (finish=%s)",
            n_tokens, elapsed, tok_s, finish,
        )
        return {
            "text": text,
            "prompt_text": prompt_text,
            "n_tokens": int(n_tokens),
        }


def make_local_native_model(
    max_tokens: int = 2048,
    adapter_path: str | None = None,
    model_repo: str = DEFAULT_MODEL_REPO,
) -> NativeModelCallable:
    """Return a ``(messages, tools) -> str`` callable matching the signature
    ``burl/eval/run_move4_star_rollout.py`` expects for ``NativeModelCallable``.
    """
    server = GemmaLocalNative(model_repo=model_repo, adapter_path=adapter_path)

    def call(messages: list[dict], tools: list[dict]) -> str:
        result = server.generate_native(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.6,
            enable_thinking=False,
        )
        return result["text"]

    return call


# --------------------------------------------------------------------------- #
# CLI smoke                                                                    #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    """Mirror the ``smoke`` entrypoint in ``gemma_serve_native.py``.

    Three tool declarations, one Texas 42 prompt, 512 max tokens. Should
    finish in <60s on an M5 Max.
    """
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

    server = GemmaLocalNative()
    t0 = time.time()
    result = server.generate_native(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 E2B local (MLX-LM) driver.")
    parser.add_argument("--smoke", action="store_true", help="Run the Burl-shaped smoke probe.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
