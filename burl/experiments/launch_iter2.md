# How to fire iter-2 training

**Author**: corpus-chef (T6)
**Date**: 2026-04-19
**Status**: launcher + local sanity checks landed. Training NOT run — team-lead fires when T1 + T3 have their corpus contributions merged into the blended input.

---

## Schema surprise — READ FIRST

While wiring the local sanity check I pinned a behavior of the Gemma 4 E2B chat template that changes what the corpus verbosity blend actually does at training time.

**Finding**: `chat_template.jinja` (~line 148 of the cached template, macro `strip_thinking()`; invoked on every `role == 'model'` string content at line ~307) removes every `<|channel>thought ... <channel|>` region from assistant messages before they reach the tokenizer. `SFTTrainer(processing_class=tokenizer)` calls `apply_chat_template(messages, tokenize=True)` per row, so the trainer never sees the thought prose.

Minimal reproduction:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
msgs = [
  {"role":"user","content":"test"},
  {"role":"assistant","content":
    "<|channel>thought\nSECRET_REASONING\n<channel|>"
    "<|tool_call>call:commit_play{domino_id:1}<tool_call|>"},
]
print(tok.apply_chat_template(msgs, tokenize=False))
# → '<bos><|turn>user\ntest<turn|>\n<|turn>model\n<|tool_call>call:commit_play{domino_id:1}<tool_call|><turn|>\n'
# SECRET_REASONING is gone. Only the tool call survived.
```

Pinned as a regression test at `burl/train/test_iter2_loader.py::test_chat_template_strips_thought_blocks`.

### Implications for the T4 blend

1. **The 33% short blend does not change the training signal much.** From the trainer's perspective, every long row is already effectively short — the chat template strips the thought prose. The tool-call chain + `commit_play` is what's left, and that's what the shortener preserves anyway. So training iter-2 on the 118-row blended preview will produce nearly-identical gradients to training on the 79-row un-blended subset.

2. **The observed iter-0/iter-1 rambly thought outputs are base-model behavior, not trained-in behavior.** Gemma 4 E2B's post-training baked the thinking channel as a default reflex; SFT at rank 16 over 30-50 rows isn't enough capacity to overwrite it. The adapter emits thoughts because the base model emits thoughts — not because the training data did.

3. **To actually compress inference-time thought output, we need one of:**
   - (a) Bypass the chat template and feed the raw verbatim assistant content to SFTTrainer via a `formatting_func` (not `messages` / `apply_chat_template`). Requires editing `burl/train/star.py` — out of scope for T6.
   - (b) Move thought content into the structured `reasoning` field on the assistant message (lines 236-238 of the template render `message.get('reasoning')` between `<|channel>thought\n...\n<channel|>` markers). The template only does this when `message.get('tool_calls')` is truthy, so the corpus would also need to migrate to the structured `tool_calls` schema. A corpus-rewrite task; also out of scope for T6.
   - (c) Prompt-side only: add "keep reasoning internal, emit at most one short plan line" to the iter-2 system prompt. Cheapest lever and likely first to try.

4. **The blend is still worth shipping**, for three reasons:
   - **More rows** (118 vs. 79) = more coverage of commit-discipline exemplars. iter-1's N=30 showed commit discipline is fragile at small N.
   - **Deterministic data augmentation** — the synthetic shorts share tool call IDs/domino IDs with their long source, so the trainer sees each decision's canonical tool chain twice. That's a meaningful regularization signal even if the thought prose is invisible.
   - **It doesn't cost anything more to train on.** Same B200, same recipe, same walltime roughly.

5. **Recommended team-lead reaction**:
   - Fire iter-2 on the blend as-is (below). Treat it as a larger-corpus iter-1 clone, not a verbosity fix.
   - If iter-2's inference output still rambles, the next lever is (c) (prompt) then (a) (bypass template).
   - Rewrite the T4 design doc's claim that the blend fixes verbosity at train time. Team-lead already signed off on 0.33 — the blend's benefit is now re-framed as "coverage + regularization," not "verbosity reduction in the training signal."

---

## Fire commands

### Smoke (first thing, 5 rows, max_steps=10, pushes to `-burl-smoke`)

```bash
modal run burl/train/star_iter2.py::main_iter2 --smoke
```

Expected: ~2-3 min end-to-end. Confirms the Modal app, the Gemma 4 cache, the B200 container, the LoRA push, and the tokenizer all work before the real run.

### Full iter-2 run (all 118 rows, 3 epochs, LR 1e-4, rank 16)

```bash
modal run burl/train/star_iter2.py::main_iter2
```

This uses every default documented at `burl/train/star_iter2.py:34-40`. Adapter will push to `jasonyandell/gemma-4-e2b-texas42-burl-iter2`.

### Point at the real iter-2 corpus (once T1/T3 land their contributions)

When T1 (rules-as-tools) and T3 (Haiku reference traces) deliver their corpus rows and team-lead merges them into a refreshed iter-2 raw corpus, re-run the blender first then train:

```bash
# Re-blend with real iter-2 rollouts (example names)
python -m burl.corpus.blender \
  --in burl/data/star_iter2_rollout.jsonl burl/data/star_iter2_haiku_refs.jsonl \
  --out burl/data/star_iter2_blended.jsonl \
  --target-short-ratio 0.33 --seed 42

# Re-run the corpus sanity check against the new path (one-line edit to
# burl/train/test_iter2_loader.py::PREVIEW_CORPUS, or copy the file in place).
python -m pytest burl/train/test_iter2_loader.py -v

# Then fire training at the real corpus
modal run burl/train/star_iter2.py::main_iter2 \
  --corpus burl/data/star_iter2_blended.jsonl
```

### Hyperparameter sweeps

All params are CLI-flagged (see `burl/train/star_iter2.py::main_iter2`). Examples:

```bash
# Higher rank for capacity ablation
modal run burl/train/star_iter2.py::main_iter2 --rank 32

# Warmer LR, fewer epochs
modal run burl/train/star_iter2.py::main_iter2 --lr 2e-4 --epochs 2

# Subsample 50/118 rows (matches iter-0's N for apples-to-apples)
modal run burl/train/star_iter2.py::main_iter2 --n-examples 50

# Custom adapter name (e.g. ablation branding)
modal run burl/train/star_iter2.py::main_iter2 \
  --adapter-name gemma-4-e2b-texas42-burl-iter2-longonly
```

---

## What the launcher does NOT do

- **Does not re-run T4's blender.** It expects an already-blended `{messages: [...]}` JSONL at `--corpus`. The preview at `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl` is the default.
- **Does not edit `burl/train/star.py`.** It imports `train_iter0` from that module verbatim and supplies iter-2 defaults via a second `@app.local_entrypoint()` named `main_iter2`. iter-0 and iter-1 can still be launched via the original `main` entrypoint on the same `app`.
- **Does not touch the eval pipeline.** Phase 4 iter-2 eval is a separate re-run of `burl/eval/run_move4_spike.py --adapter jasonyandell/gemma-4-e2b-texas42-burl-iter2` (already done once per-adapter in iter-0/iter-1).
- **Does not write to `main`, does not push, does not open a PR.** All of that is team-lead's.

---

## Files landed by T6

| path | role |
|---|---|
| `burl/train/star_iter2.py` | local_entrypoint wrapper over `train_iter0` |
| `burl/train/test_iter2_loader.py` | 17 pytest sanity checks on the blended corpus |
| `scratch/burl_p5_iter2_prep/launch_iter2.md` | this doc |

Testing (local, zero Modal spend):

```bash
python -m pytest burl/corpus/test_blender.py burl/train/test_iter2_loader.py -v
# 38 passed in ~7.5s
```
