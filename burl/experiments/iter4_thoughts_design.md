# iter-4-thoughts — make Gemma's own reasoning train-visible

**Author**: corpus-chef (T17)
**Date**: 2026-04-19
**Scope**: infrastructure. Bypass Gemma 4's `strip_thinking()` at SFT time so `<|channel>thought` regions inside assistant content reach the loss. No training launched; this is the foundation for iter-4 bootstrap.

## TL;DR

**iter-0, iter-1, iter-2, iter-3-v2, and iter-3-rules all trained on corpora where Gemma's own thought prose was silently dropped before tokenization.** On a representative iter-3-rules row, 532 tokens of thought content — the reasoning that produced the winning commit — never reached the SFT loss function. This is a real signal we've been throwing away for five adapter generations.

T17 adds a `preserve_thoughts=True` kwarg to `train_iter0` that swaps the chat-template path for a `formatting_func` that bypasses `strip_thinking()`. When enabled, the user turn is still rendered via `apply_chat_template(add_generation_prompt=True)` (so `<bos><|turn>user\n...<turn|>\n<|turn>model\n` stays canonical), but the assistant span is appended verbatim (thoughts intact) followed by `<turn|>\n`. The default path is unchanged — iter-0/1/2/3 reproducibility is preserved. 12 new tests pin the behavior; the full burl suite (now 117 tests) stays green.

**No training launched.** `burl/train/star_iter4_thoughts.py` is the downstream launcher; it defaults to the iter-3-rules corpus so iter-4-thoughts is a clean A/B against iter-3-rules with `preserve_thoughts` as the only variable. Team-lead greenlights that run separately.

## The problem — evidence from Gemma 4 E2B's chat template

Gemma 4's `chat_template.jinja` contains a `strip_thinking()` macro invoked on any `role == 'model'` content:

```jinja
{# from chat_template.jinja, ~line 148 #}
{% macro strip_thinking(content) %}
  {# removes every <|channel>thought ... <channel|> region #}
  ...
{% endmacro %}
```

SFTTrainer with `processing_class=tokenizer` (the iter-0/1/2/3 path) calls `apply_chat_template` per row, which invokes this macro. The regression test `test_chat_template_strips_thought_blocks` at `burl/train/test_iter2_loader.py:167` pins the behavior: rendered text never contains `<|channel>thought` prose.

**Empirical demonstration** (from `burl/data/star_iter3_rules_corpus.jsonl` row 0):

| path | output chars | output tokens | `<|channel>thought` survives? |
|---|---:|---:|:---:|
| `apply_chat_template` (iter-0/1/2/3 path) | 5,023 | **1,714** | ✗ stripped |
| `formatting_func` (iter-4 path) | 7,025 | **2,246** | ✓ preserved |
| **delta** | +2,002 | **+532 tokens** | — |

**That delta is the reasoning signal we've never trained on.** On this representative row, 532 tokens (~24% of the total) are thought content. For iter-3-rules's ~50-row corpus at 3 epochs, that's ~80,000 reasoning tokens per epoch that the trainer has been blind to. Multiply across five adapter generations: we've accumulated roughly half a million tokens of Gemma's own winning reasoning that never contributed a gradient.

## Why this unblocks the bootstrap philosophy

User has been explicit: we do not want to inject Haiku/external reasoning examples into the corpus. The bootstrap plan is reinforcement of **Gemma's own reasoning shape** when the K1 filter agrees it worked. Today that plan is impossible to execute — even when K1 keeps a trace, the trainer sees only the tool-call chain and commit. The reasoning that led to a K1-winning play is effectively training-side invisible.

With `preserve_thoughts=True`:

1. **Rollout stays unchanged.** Gemma 4 produces `<|channel>thought` blocks natively via its thinking reflex; the rollout orchestrator doesn't need to opt in.
2. **K1 filter stays unchanged.** `burl_eq >= bot_eq` traces flow into the corpus as before; the `<|channel>thought` blocks are already in the assistant content (see `run_move4_star_rollout::compose_assistant_content`).
3. **Only SFT changes.** The formatting_func exposes those already-captured thought tokens to the loss. Next-token prediction on reasoning becomes a gradient signal the same way it already is on tool calls.

That's the entire bootstrap enabler. No new data collection, no prompt-shape changes, no corpus blending.

## What changed

### 1. `burl/train/star.py` — gated path

- New constant `GEMMA4_TURN_TERMINATOR = "<turn|>\n"` (pinned atomic by the new test).
- New helper `build_preserve_thoughts_formatting_func(tokenizer)` that returns a per-row `fmt(example) -> str` callable.
- `train_iter0` gains `preserve_thoughts: bool = False`. When True, SFTTrainer is instantiated with `formatting_func=...`; otherwise the existing `processing_class=tokenizer` path is unchanged.
- Module docstring expanded with the full rationale + invariants.

### 2. `burl/train/star_iter4_thoughts.py` — new launcher

Mirrors `star_iter3_rules.py`'s thin-wrapper pattern. Defaults:
- corpus: `burl/data/star_iter3_rules_corpus.jsonl` (T12's output)
- adapter: `jasonyandell/gemma-4-e2b-texas42-burl-iter4-thoughts`
- recipe: identical to iter-1/iter-2/iter-3-v2/iter-3-rules (rank 16, 3 epochs, lr 1e-4, batch 2 × grad_accum 4, bf16, sdpa) — the **only** variable under test is `preserve_thoughts=True`.

### 3. `burl/train/test_formatting_func.py` — 12 tests pinning the behavior

| test | purpose |
|---|---|
| `test_formatting_func_rejects_malformed_rows` | Schema contract on the callable. No tokenizer needed. |
| `test_turn_boundary_is_atomic[<|turn>]` etc. (6 params) | If `<|turn>` / `<turn|>` / `<|channel>` / `<channel|>` / `<|tool_call>` / `<tool_call|>` ever fragment, the formatting_func assumption breaks. |
| `test_turn_terminator_constant_matches_tokenizer` | `GEMMA4_TURN_TERMINATOR` is what `apply_chat_template` emits; we haven't diverged from tokenizer-canonical. |
| `test_formatting_func_preserves_thought_prose` | Core pin — both raw string and tokenizer round-trip preserve thought prose and `commit_play`. |
| `test_formatting_func_preserves_user_turn_boundary` | User-side structure (`<bos>`, `<|turn>user`, `<|turn>model`) is canonical; training and inference prompts match. |
| `test_formatting_func_vs_chat_template_divergence` | Direct A/B on the same row: formatting_func preserves, chat_template strips. Documents the intended behavioral divergence. |
| `test_labels_include_thought_token_positions` | Thought tokens appear as a contiguous subsequence in the encoded ids AND none of them equal `pad_id` — so label-mask with pad→-100 leaves every thought position trainable. |

The tokenizer-dependent tests skip cleanly when the HF cache is absent (for CI environments without the model); the pure-string tests always run.

### 4. `scratch/burl_p5_iter2_prep/iter4_thoughts_design.md` — this doc

## Out of scope (not changed)

- **No rollout changes.** Nothing in `burl/eval/run_move4_star_rollout.py`, `agent_runner_native.py`, `gemma_serve_native.py`, or the K1/EQ-gate pipeline changes. Thoughts are already produced and already flow into the corpus; only what the trainer sees changes.
- **No default-path behavior changes.** `preserve_thoughts=False` is the default; iter-0/1/2/3 launchers continue to train exactly as before. This is a purely additive infra change.
- **No training launched.** T17 is foundation; the iter-4-thoughts vs iter-3-rules A/B is a downstream task (T18 or similar).
- **No corpus rebuilding.** iter-4-thoughts uses the existing iter-3-rules corpus so the variable under test is solely `preserve_thoughts`.

## Risk register (what could go wrong when iter-4-thoughts actually fires)

1. **Loss spike / divergence.** Thought tokens are ~24% of sequence; the loss landscape is materially different. Mitigation: smoke run at `--smoke` first (max_steps=10, 5 rows) before committing a full epoch.
2. **Gemma's thought style doesn't match bot-winning decisions.** If rollout thoughts are "structural-reasoning that hallucinates rules" (the iter-0/1 failure mode), reinforcing them is negative transfer. Mitigation: the K1 filter only keeps traces where the *commit* was a bot-match; the thought that produced the commit is coupled to the good outcome. Not a guarantee, but a strong prior.
3. **Inference-time thought blocks blow up token costs.** If iter-4 generates longer thoughts at eval time, cost per decision could 2-3×. Mitigation: this is a measurable outcome for the downstream A/B, not a T17 concern.
4. **Formatting_func EOS handling.** TRL's SFTTrainer with `formatting_func` may or may not append EOS automatically; our terminator is `<turn|>\n`, which is what Gemma expects. The `test_turn_terminator_constant_matches_tokenizer` test pins parity with `apply_chat_template` output so we don't silently drift.

## What to watch when iter-4-thoughts fires

- **Loss trajectory.** Should descend comparably to iter-3-rules (a ~50-row 3-epoch run was ~80s on B200 last cycle). If training wall time grows substantially, it's because the formatting_func emits longer sequences (expected +~30%); cost scales accordingly.
- **Eval-time output shape.** Does iter-4-thoughts generate thought blocks at inference? If yes, the bootstrap reinforcement worked — reasoning is now part of the policy. If no, we know the K1-filtered corpus wasn't thought-dense enough to move the generation distribution.
- **Bot-match rate vs iter-3-rules.** This is the main eval metric. If thought-reinforcement helps decision quality, iter-4-thoughts > iter-3-rules. If not, we've learned that the tool-call chain was carrying all the useful signal.

## Budget footprint (T17 only)

- Modal: **$0.00**. No GPU, no rollouts, no training.
- Test run cost: negligible (local pytest, 117 tests in 13.6s).

## Artifacts

- `burl/train/star.py` — modified: new helper, kwarg, gated trainer instantiation.
- `burl/train/star_iter4_thoughts.py` — new launcher (iter-3-rules corpus + `preserve_thoughts=True`).
- `burl/train/test_formatting_func.py` — 12 new tests (all pass, full burl suite 117/117 green).
- `scratch/burl_p5_iter2_prep/iter4_thoughts_design.md` — this doc.
- (Unchanged, referenced for context): `burl/train/test_iter2_loader.py::test_chat_template_strips_thought_blocks` — the regression test that identified the problem in T6.
