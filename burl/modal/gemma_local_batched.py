"""Batched MLX-LM Gemma 4 E2B wrapper for the Burl rollout harness.

Parallel to ``burl/modal/gemma_local.py`` — same model loader, same chat
template, same EOS handling. The difference is one step: instead of
``stream_generate`` per decision, this exposes a
``step_batch(active) -> list[str]`` that renders every active decision's
``(messages, tools)`` into a prompt and drives them through
``mlx_lm.batch_generate`` in one fused pass.

Numbers motivating this (from ``wiki/experiments/batch-throughput-bench.md``
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
from mlx_lm.models.cache import LRUPromptCache
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
        enable_prompt_cache: bool = False,
        prompt_cache_max_entries: int = 64,
        prune_lm_head: bool = False,
        prune_freq_tsv: str | None = None,
        prune_keep_n: int = 8192,
        log_argmax_winners: str | None = None,
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
        # Lever #16 phase 2: optional LM-head vocab prune. When enabled, slice
        # the tied output projection down to (top-N freq emit ids ∪ tokenizer
        # special ids) and wrap the sampler with a pruned-index →
        # original-vocab-id LUT. The input-side ``embed_tokens`` table is left
        # intact so prompts can still contain any vocab ID — only the output
        # matmul shrinks.
        self.prune_keep_ids: list[int] | None = None
        if prune_lm_head:
            from pathlib import Path as _P
            from burl.eval.lm_head_prune import (
                apply_lm_head_prune,
                load_keep_ids,
                wrap_sampler_with_lut,
            )
            if prune_freq_tsv is None:
                raise ValueError(
                    "prune_lm_head=True requires prune_freq_tsv path"
                )
            keep_ids = load_keep_ids(
                _P(prune_freq_tsv),
                int(prune_keep_n),
                self.tokenizer,
            )
            lut = apply_lm_head_prune(self.model, keep_ids)
            self._sampler = wrap_sampler_with_lut(self._sampler, lut)
            self.prune_keep_ids = keep_ids
            log.info(
                "[gemma-local-batched] lm_head pruned to %d kept ids",
                len(keep_ids),
            )
        # Lever #16 phase 2-redo (iter 30): calibration mode. Wrap sampler
        # with a shim that logs strict-argmax winners over the FULL-vocab
        # logprobs at every decode step. Run with prune_lm_head=False so the
        # logged ids are actual argmax winners (not LUT-mapped pruned-space
        # winners). The harvest in iter 28 used temp=0.6 sampling so it can
        # miss high-logit tokens that lose the sampling lottery; this log
        # captures what the bench's greedy temp=0 mode actually picks.
        if log_argmax_winners is not None:
            from pathlib import Path as _Pl
            from burl.eval.lm_head_prune import wrap_sampler_with_argmax_logger
            self._sampler = wrap_sampler_with_argmax_logger(
                self._sampler, _Pl(log_argmax_winners),
            )
            log.info(
                "[gemma-local-batched] argmax-winner log -> %s",
                log_argmax_winners,
            )
        # Phase 2 lever 1: prefix-aware prompt cache. When enabled, step_batch
        # threads previously-seen prefix KV through batch_generate so growing
        # message histories (turn N+1 = turn N + new tool messages) prefill
        # only the suffix. Disabled by default to preserve the existing
        # production behavior; bench/harvest enable it via the constructor.
        self._prompt_cache_enabled = bool(enable_prompt_cache)
        self._prompt_cache: LRUPromptCache | None = (
            LRUPromptCache(max_size=int(prompt_cache_max_entries))
            if self._prompt_cache_enabled else None
        )
        # Cache stats — exposed so the bench can record cache_hit_tokens per
        # ledger row.
        self.cache_hit_tokens_total = 0
        self.cache_processed_tokens_total = 0
        # mlx_lm's LRUPromptCache keys on a hashable "model" identifier;
        # nn.Module instances are not hashable, so we use a stable string per
        # wrapper instance.  All step_batch lookups within a single bench
        # run share this key.
        self._cache_model_key = (
            f"{model_repo}:{adapter_path or 'base'}:{id(self)}"
        )
        log.info(
            "[gemma-local-batched] ready in %.1fs prompt_cache=%s",
            time.time() - t0, self._prompt_cache_enabled,
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
            }

        Returns a list of completion strings, one per non-done entry, in
        the same order as the non-done entries appear in ``active``. Done
        entries are NOT in the return list — the caller pairs results back
        to indices via its own not-done filter.

        Uses ``completion_batch_size=len(prompts)`` so MLX-LM's
        BatchGenerator runs the whole group concurrently. This matches the
        bench's measured-best shape for ~2400-token prompts at batch=64
        on M5 Max.

        When the constructor flag ``enable_prompt_cache`` is True, the
        wrapper threads ``mlx_lm.models.cache.LRUPromptCache`` through
        ``batch_generate`` so growing message histories share their prefix
        with the prior turn's cache (turn N+1 prefills only the new
        suffix tokens added since turn N).
        """
        not_done = [e for e in active if not e.get("done", False)]
        if not not_done:
            return []

        prompts = [
            self._render_prompt_ids(e["messages"], e.get("tools"))
            for e in not_done
        ]

        if self._prompt_cache is None:
            resp = batch_generate(
                self.model,
                self.tokenizer,
                prompts=prompts,
                max_tokens=self.max_tokens,
                sampler=self._sampler,
                verbose=False,
                completion_batch_size=len(prompts),
            )
            assert len(resp.texts) == len(prompts), (
                f"batch_generate returned {len(resp.texts)} texts for "
                f"{len(prompts)} prompts"
            )
            return list(resp.texts)

        # Prefix-cache path: per-stream LRU lookup. We pass full prompts to
        # batch_generate alongside the trimmed-to-prefix caches; mlx-lm's
        # BatchGenerator processes only the suffix not covered by each cache.
        prompt_caches: list = []
        suffix_prompts: list[list[int]] = []
        all_tokens_per_stream: list[list[int]] = []
        cache_keys: list[list[int]] = [list(p) for p in prompts]
        for full_prompt in prompts:
            cache, rest = self._prompt_cache.fetch_nearest_cache(
                self._cache_model_key, full_prompt,
            )
            n_total = len(full_prompt)
            n_rest = len(rest)
            n_hit = n_total - n_rest
            self.cache_hit_tokens_total += n_hit
            self.cache_processed_tokens_total += n_rest
            # When fetch_nearest_cache returns None we still need a fresh cache
            # so batch_generate sees an aligned per-stream cache list.
            from mlx_lm.models.cache import make_prompt_cache as _mk_cache
            if cache is None:
                cache = _mk_cache(self.model)
            prompt_caches.append(cache)
            suffix_prompts.append(list(rest))
            all_tokens_per_stream.append(list(full_prompt[:n_hit]))

        # Defensive: BatchGenerator's insert() seq-splitting requires each
        # input segment to be non-empty. If any suffix is empty (the entire
        # prompt was a cache hit), we have to leave at least one token to
        # prefill so generation can step. mlx-lm handles this by appending a
        # single "split" token; we mirror by reverting to a fresh cache when
        # rest == [].
        for i, suf in enumerate(suffix_prompts):
            if not suf:
                from mlx_lm.models.cache import make_prompt_cache as _mk_cache
                prompt_caches[i] = _mk_cache(self.model)
                suffix_prompts[i] = list(prompts[i])
                all_tokens_per_stream[i] = []
                # The hit and processed counters were skewed by this row;
                # roll them back to "no hit, full processing".
                self.cache_hit_tokens_total -= len(prompts[i])
                self.cache_processed_tokens_total += len(prompts[i])

        resp = batch_generate(
            self.model,
            self.tokenizer,
            prompts=suffix_prompts,
            prompt_caches=prompt_caches,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
            return_prompt_caches=True,
            completion_batch_size=len(suffix_prompts),
        )
        assert len(resp.texts) == len(prompts), (
            f"batch_generate returned {len(resp.texts)} texts for "
            f"{len(prompts)} prompts"
        )

        # Insert post-decode caches keyed on (full_prompt + decoded_tokens) so
        # the next turn's fetch_nearest_cache finds the longest viable prefix.
        # We approximate the decoded tokens via the tokenizer encoding of the
        # response text; this can differ by at most the BPE rendering boundary
        # but the LRU's prefix-search degrades gracefully on partial matches.
        if resp.caches is not None:
            for full_prompt, text, cache in zip(prompts, resp.texts, resp.caches):
                if cache is None:
                    continue
                completion_ids = self.tokenizer.encode(text) if text else []
                key = list(full_prompt) + list(completion_ids)
                try:
                    self._prompt_cache.insert_cache(self._cache_model_key, key, cache)
                except Exception as exc:
                    log.warning(
                        "[gemma-local-batched] LRU insert_cache failed (%s); "
                        "skipping for this stream", exc,
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
