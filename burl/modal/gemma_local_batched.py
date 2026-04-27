"""Batched MLX-LM Gemma 4 E2B wrapper for the Burl rollout harness.

Parallel to ``burl/modal/gemma_local.py`` — same model loader, same chat
template, same EOS handling. The difference is one step: instead of
``stream_generate`` per decision, this exposes a
``step_batch(active) -> list[str]`` that renders every active decision's
``(messages, tools)`` into a prompt and drives them through
``mlx_lm.batch_generate`` in one fused pass.

Numbers motivating this (from ``burl/experiments/batch_throughput_bench.md``
on M5 Max, bf16 Gemma 4 E2B):

    single-stream :  83 tok/s  aggregate (1.0x)
    batch=16      : 512 tok/s  aggregate (6.2x)
    batch=64      : 1206 tok/s aggregate (14.5x) <-- recommended default
    batch=128     : 1334 tok/s aggregate (16.1x, peak)

The tool-loop driver (``run_batch_decisions`` in
``burl/eval/run_move4_star_rollout_batched.py``) sits on top of this file.
This module deliberately stays "dumb": load the model once, expose a
step-a-batch-of-chat-states primitive. The tool dispatch and decision
state machine live in the eval file so the model wrapper doesn't have to
know what a ``BurlTrace`` is.

CLI smoke::

    PYTHONPATH=. python -u -m burl.modal.gemma_local_batched --smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

log = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "mlx-community/gemma-4-e2b-it-bf16"


class GemmaLocalNativeBatched:
    """Local batched MLX-LM driver for N-at-once Burl rollouts.

    Loads ``model, tokenizer = mlx_lm.load(...)`` once in ``__init__``
    (same as ``GemmaLocalNative``). Exposes ``step_batch(active)``: for
    every non-done entry in ``active``, render its ``messages`` + ``tools``
    through the chat template, tokenize, and drive the batch through
    ``batch_generate``. Returns a list of completion strings aligned to
    the order of the non-done entries (so the caller can zip them back
    onto their decision indices).

    ``step_batch`` is the only inference primitive this class exposes.
    The full tool-loop (parse -> dispatch tools -> append messages ->
    check commit -> retry on illegal) lives in the rollout driver so
    the model wrapper stays backend-focused.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        adapter_path: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
    ) -> None:
        t0 = time.time()
        log.info(
            "[gemma-local-batched] loading %s (adapter=%s)",
            model_repo, adapter_path,
        )
        self.model_repo = model_repo
        self.adapter_path = adapter_path
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.model, self.tokenizer = load(
            model_repo, adapter_path=adapter_path,
        )
        # Sampler identical to ``bench_batch_throughput`` and single-stream
        # ``gemma_local`` — temp=0.6, default top-k/p. Held stable so
        # batched-vs-sequential semantic comparison doesn't have a sampling
        # confounder.
        self._sampler = make_sampler(temp=self.temperature)
        log.info(
            "[gemma-local-batched] ready in %.1fs", time.time() - t0,
        )

    # --------------------------------------------------------------------- #
    # Prompt rendering                                                      #
    # --------------------------------------------------------------------- #

    def _render_prompt_ids(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> list[int]:
        """Run the chat template and tokenize to a list of ids.

        ``enable_thinking=False`` mirrors ``gemma_local.py`` / the bench.
        Gemma can still emit a ``<|channel>thought`` preamble — that's
        handled by the parser downstream, not here. If ``tools`` is None,
        we pass nothing so the template falls back to its no-tools
        rendering (harmless but we usually have tools).
        """
        template_kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if tools is not None:
            template_kwargs["tools"] = tools
        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        assert isinstance(text, str), (
            "apply_chat_template(tokenize=False) should return a string"
        )
        return self.tokenizer.encode(text)

    # --------------------------------------------------------------------- #
    # Batched step                                                          #
    # --------------------------------------------------------------------- #

    def step_batch(self, active: list[dict]) -> list[str]:
        """Advance all non-done decisions by one assistant turn.

        ``active`` is a list of dicts with at least::

            {
                "messages": list[dict],  # HF chat format
                "tools": list[dict],     # JSON tool schemas
                "done": bool,            # True -> skipped
                "max_tokens": int,       # OPTIONAL per-prompt cap;
                                         # falls back to self.max_tokens.
            }

        Returns a list of completion strings, one per non-done entry, in
        the same order as the non-done entries appear in ``active``. Done
        entries are NOT in the return list — the caller pairs results back
        to indices via its own not-done filter.

        Per-prompt ``max_tokens`` (Phase 1, Lever 1): MLX-LM's
        ``batch_generate`` accepts ``max_tokens: List[int]`` and stops each
        stream individually at its own budget. The harvest harness uses this
        to give early-turn streams a tighter cap (turn-aware policy in
        ``burl.wax_museum.schemas.max_tokens_for_state``). When the key is
        absent the entry falls back to ``self.max_tokens`` so existing
        callers keep working unchanged.

        Uses ``completion_batch_size=len(prompts)`` so MLX-LM's
        BatchGenerator runs the whole group concurrently. This matches the
        bench's measured-best shape for ~2400-token prompts at batch=64
        on M5 Max.
        """
        not_done = [e for e in active if not e.get("done", False)]
        if not not_done:
            return []

        prompts = [
            self._render_prompt_ids(e["messages"], e.get("tools"))
            for e in not_done
        ]
        per_prompt_max = [
            int(e.get("max_tokens", self.max_tokens)) for e in not_done
        ]
        resp = batch_generate(
            self.model,
            self.tokenizer,
            prompts=prompts,
            max_tokens=per_prompt_max,
            sampler=self._sampler,
            verbose=False,
            completion_batch_size=len(prompts),
        )
        # BatchResponse.texts is aligned to the input prompts order.
        assert len(resp.texts) == len(prompts), (
            f"batch_generate returned {len(resp.texts)} texts for "
            f"{len(prompts)} prompts"
        )
        return list(resp.texts)


# --------------------------------------------------------------------------- #
# CLI smoke                                                                    #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    """Batch-generate one step on 4 real iter-3-rules prompts.

    Mirrors the shape ``bench_batch_throughput.py`` uses but at a tiny
    batch and prints every completion so you can eyeball them. No tool
    dispatch happens here — the full loop is in the rollout driver.
    """
    # Deferred imports: the smoke is the only caller that needs decision
    # rendering, and we don't want to import the dataset module at top
    # level when this wrapper is used as a library.
    from burl.eval.decision_dataset import load_dataset
    from burl.harness.agent_runner import _current_player, _visible_history
    from burl.harness.agent_runner_native import (
        build_tool_schemas,
        render_native_messages,
    )

    dataset_path = "burl/eval/data/move4_decisions_n50.jsonl"
    decisions = load_dataset(dataset_path)[:4]
    tool_schemas = build_tool_schemas(enable_rules_tools=True)

    active: list[dict] = []
    for d in decisions:
        state = d.game_state
        me_abs = _current_player(state)
        hand = [x for x in state.hands[me_abs] if x not in state.played]
        history = _visible_history(state)
        system, user = render_native_messages(
            state, hand, history,
            enable_rules_tools=True,
            enable_primer=True,
        )
        active.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": tool_schemas,
            "done": False,
        })

    server = GemmaLocalNativeBatched(max_tokens=512, temperature=0.6)
    t0 = time.time()
    completions = server.step_batch(active)
    wall = time.time() - t0

    print("=" * 72)
    print(
        f"[smoke] batch={len(active)}  wall={wall:.1f}s  "
        f"completions={len(completions)}"
    )
    print("-" * 72)
    for i, (d, text) in enumerate(zip(decisions, completions)):
        preview = text if len(text) < 600 else text[:600] + " ...[truncated]"
        print(
            f"[{i}] seed={d.seed} decl={d.declaration} "
            f"seat={d.narrator_seat}  ({len(text)} chars)"
        )
        print("    " + preview.replace("\n", "\n    "))
        print()
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched Gemma 4 E2B MLX-LM driver (smoke only).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the 4-decision one-step smoke probe.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
