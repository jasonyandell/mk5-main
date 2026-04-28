"""LM-head vocab prune (lever #16 phase 2).

Gemma 4 E2B ties the LM head to ``embed_tokens`` (262,144 vocab × 1536 hidden
≈ 770 MB bf16). The Burl emit space — measured on the Apr-26 prod_2000_v2
harvest — is only 4,493 unique vocab IDs (1.71%). Keeping the full table for
*input* lookup but routing the *output* projection through a sliced copy
shrinks the lm_head matmul ~32× (262144 → 8192) without touching anything the
model already produces correctly.

Pipeline:

1. ``load_keep_ids(freq_tsv, keep_n, tokenizer)`` — read iter28's freq TSV,
   take the top-N vocab IDs, union with the tokenizer's ``all_special_ids``
   so BOS/EOS/pad/tool-protocol tokens can never be pruned-out by accident.
2. ``apply_lm_head_prune(model, keep_ids)`` — slice the embedding table to a
   new ``nn.Linear`` and stash it on ``model.language_model.lm_head``; flip
   ``tie_word_embeddings`` to False so ``Model.__call__`` takes the
   ``self.lm_head(out)`` branch.
3. ``wrap_sampler_with_lut(sampler, keep_ids)`` — return a new sampler that
   accepts pruned-space logits ``(B, keep_N)`` and returns *original-vocab*
   token IDs via the ``lut`` lookup. The KV cache + tokenizer downstream
   never know the model spoke a smaller vocabulary.

Tied-embedding subtlety: input embeddings still need the full table because
prompts can contain any token. Pruning ``embed_tokens.weight`` rows directly
would silently break input lookup. So we keep the full ``embed_tokens`` and
add a separate (sliced) ``lm_head`` — i.e. we *un-tie* the head at runtime.
The on-GPU footprint grows by exactly the size of the sliced lm_head
(8192 × 1536 bf16 ≈ 24 MB), not by the full embedding's 770 MB, because
``lm_head.weight`` is a *view* into ``embed_tokens.weight`` rows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)


def load_keep_ids(
    freq_tsv: Path,
    keep_n: int,
    tokenizer: Any,
) -> list[int]:
    """Build the keep-set: top-N from freq TSV ∪ tokenizer special ids.

    The freq TSV format is iter28's ``rank, vocab_id, count, cumulative_count,
    cumulative_frac`` — already sorted by frequency descending, so the top-N
    rank rows give us the top-N vocab IDs. We union with all special IDs the
    tokenizer knows about (BOS, EOS, pad, tool-protocol markers like
    ``<|tool_call>`` and ``<|channel>`` — Gemma 4's tokenizer reports 24 of
    them, all in the [0, 258884] range) so a special token can never be
    silently pruned even if it never appeared in the harvest.

    Returns sorted unique original vocab IDs. Stable order (sorted by ID,
    not by rank) makes the ``keep_idx -> original_id`` LUT trivial to read in
    a debugger.
    """
    keep: set[int] = set()
    n_taken = 0
    with open(freq_tsv) as f:
        header = f.readline()
        assert header.startswith("rank\tvocab_id\t"), (
            f"unexpected freq TSV header: {header!r}"
        )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            vocab_id = int(parts[1])
            keep.add(vocab_id)
            n_taken += 1
            if n_taken >= keep_n:
                break

    # Union special ids — tokenizer.all_special_ids should be a small list of
    # ints in vocab range. (HF/MLX tokenizer wrappers expose this.)
    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = {int(x) for x in tokenizer.all_special_ids}
    n_added = len(special_ids - keep)
    keep |= special_ids

    keep_sorted = sorted(keep)
    log.info(
        "[lm_head_prune] keep_set: top-%d freq + %d special "
        "(%d added beyond freq) -> %d total ids",
        n_taken, len(special_ids), n_added, len(keep_sorted),
    )
    return keep_sorted


def apply_lm_head_prune(model: Any, keep_ids: list[int]) -> mx.array:
    """Replace the model's tied-head output projection with a sliced copy.

    Mutates ``model.language_model`` in place:

    * Copies the rows of ``embed_tokens.weight[keep_ids]`` into a fresh
      ``nn.Linear(hidden, keep_N, bias=False)``.
    * Stashes that as ``language_model.lm_head``.
    * Flips ``language_model.tie_word_embeddings = False`` so the next forward
      pass takes the ``self.lm_head(out)`` branch instead of the
      ``embed_tokens.as_linear(out)`` branch.

    Input embedding lookup ``embed_tokens(inputs)`` is *unchanged* — the full
    table is still resident, so prompts containing any vocab ID still embed
    correctly. Only the output side shrinks.

    Returns the LUT array — shape ``(keep_N,)`` int32 — that maps a pruned
    logit index back to its original-vocab ID. Hand this to
    ``wrap_sampler_with_lut`` so downstream consumers (KV cache update,
    tokenizer.decode) keep seeing original vocab IDs.
    """
    lm = model.language_model
    if not getattr(lm, "tie_word_embeddings", False):
        raise RuntimeError(
            "apply_lm_head_prune expects tie_word_embeddings=True (Gemma 4 "
            f"E2B default); model reports tie_word_embeddings="
            f"{getattr(lm, 'tie_word_embeddings', None)!r}"
        )

    embed = lm.model.embed_tokens
    full_weight = embed.weight  # (V_full, H) bfloat16
    v_full, hidden = int(full_weight.shape[0]), int(full_weight.shape[1])

    keep_arr = mx.array(keep_ids, dtype=mx.int32)
    if int(keep_arr.max()) >= v_full or int(keep_arr.min()) < 0:
        raise ValueError(
            f"keep_ids out of range: min={int(keep_arr.min())} "
            f"max={int(keep_arr.max())} vocab_size={v_full}"
        )

    keep_n = int(keep_arr.shape[0])
    sliced = full_weight[keep_arr]  # (keep_N, H), same dtype
    mx.eval(sliced)
    log.info(
        "[lm_head_prune] sliced lm_head weight: shape=%s dtype=%s "
        "(was V=%d, now V_keep=%d, %.1fx reduction)",
        tuple(sliced.shape), sliced.dtype, v_full, keep_n, v_full / max(keep_n, 1),
    )

    head = nn.Linear(hidden, keep_n, bias=False)
    head.weight = sliced
    lm.lm_head = head
    lm.tie_word_embeddings = False

    # Sanity: forward path will now route through lm_head; embed_tokens still
    # answers input lookups. The args struct still says vocab_size=262144;
    # we don't update it because some MLX-LM internals (sanitize, cache
    # construction) read it and expect the input-side vocab.

    return keep_arr


def wrap_sampler_with_lut(
    base_sampler: Callable[[mx.array], mx.array],
    lut: mx.array,
) -> Callable[[mx.array], mx.array]:
    """Wrap a sampler to remap pruned-space → original-vocab token IDs.

    The MLX-LM BatchGenerator's contract: sampler receives ``(B, V_logits)``
    logprobs and returns ``(B,)`` token IDs. With the lm_head pruned, the
    model emits logits over ``V_keep`` rows, the sampler picks an index in
    ``[0, V_keep)``, and the returned ID is fed into ``embed_tokens(inputs)``
    on the next decode step + into ``tokenizer.decode`` on stop. Both
    expect *original* vocab IDs. So we LUT-remap on the way out.
    """
    def remapped(logprobs: mx.array) -> mx.array:
        sampled_pruned = base_sampler(logprobs)
        return lut[sampled_pruned]
    return remapped
