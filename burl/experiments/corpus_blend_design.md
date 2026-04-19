# iter-2 corpus verbosity blend — design

**Author**: corpus-chef (burl-iter2-prep team)
**Date**: 2026-04-19
**Scope**: `burl/corpus/*` plus a sample preview corpus.
No training kicked off from this document — team-lead decides when/how.

## The problem we are trying to fix

`SPIKE_REPORT.md` (Phase 4, iter-0; iter-1 follow-up) documents the same
corpus pathology twice:

- **iter-0** (trained on Layer-1 corpus): `mean_tokens_out ≈ 5.6 KB/decision`.
  Sampled traces show `~3 KB <|channel>thought` blocks re-deriving game state
  that was already in the user message. `eq_outcome_distribution` fired
  2 times across 10 decisions. `is_legal` fired 16 times. **Bot-match 60%,
  −10 pp vs. the rollout base.**
- **iter-1** (trimmed primer, retrained): bot-match on completed decisions
  recovered to 80%, but commit discipline collapsed (5/10 retry-exhausted).
  The long thought blocks persisted in the corpus — the trim lived in the
  prompt, not in the training rows.

In both adapters Gemma inherited the Layer-1 prose substrate and treated
thought blocks as a *load-bearing* preamble to every commit. Spike v2, trained
on nothing, wins at 88.9% with 1.5 KB average output — it already knows how
to emit a terse tool-call chain.

The shape we want iter-2 to learn: **tool-grounded commits with short, minimal
narrative**. The shape we currently train on: 4-5 KB of re-derived state.

## The fix: LS-Mixture style corpus blend

arxiv 2505.03469 ("LS-Mixture SFT") reports that mixing structure-preserved
short traces with full-reasoning traces cuts response length ~47.6% and nudges
accuracy up ~+2.3 pp at equal training budget. The structure-preserved-short
trace keeps the final answer and *the load-bearing intermediate structure*,
dropping the narrative prose around it.

That fits Burl cleanly. The load-bearing structure is the tool-call chain
(`is_legal` → `eq_outcome_distribution` → `commit_play`); the narrative is
the `<|channel>thought` prose around it. Stripping thought, keeping the
tool envelopes, gives us a direct analogue.

Unsloth's guidance ("keep ≥75% reasoning when mixing") also points the same
way — and in our case the tool calls *are* reasoning (the architectural
premise of Burl). A short trace is not a non-reasoning trace; it is reasoning
expressed entirely through tool grounding rather than prose.

## Short-trace definition

A row is **short** iff all of:

1. `len(assistant_content) ≤ 1500` chars (total cap — safety).
2. `thought_chars(assistant_content) ≤ 200` chars (summed across all
   `<|channel>thought ... <channel|>` blocks).
3. The assistant content contains at least one
   `<|tool_call>call:commit_play{...}<tool_call|>` envelope (i.e. the row
   actually represents a completed decision).

Anything else is **long**.

Rationale for thresholds:

- The one naturally-short row across the 80-row existing corpus is 921 chars
  with 0 thought chars. A 1500 char cap is generous enough to admit future
  short rationalizations with a one-sentence plan, tight enough to exclude
  the full thought blocks (observed median ~4 KB).
- 200 thought chars is sufficient for a "Partner has 2s led, I follow with
  21" one-line plan without admitting the 2-4 KB re-derivations that drive
  the regression.
- The `commit_play` check eliminates mid-trace fragments if the shortener
  ever malfunctions.

## Source of short traces: synthesize, don't wait

Of the 80 existing rows (50 iter-0 + 30 iter-1), exactly **1** classifies as
short. Extraction alone gives us nothing useful. Therefore we **synthesize**
short variants from the long rows.

### Shortener spec

Input: one long assistant message.
Output: the same tool-call and tool-response envelopes, in the same order,
with:

1. All well-formed `<|channel>thought ... <channel|>` blocks removed.
2. Orphan `<|channel>thought` headers and `<channel|>` closers removed (Gemma
   occasionally emits imbalanced ones — row 2 of the iter-0 corpus has
   2 closers and 1 opener).
3. Runs of 3+ blank lines collapsed to 2. Leading/trailing whitespace trimmed.

Invariants (tested in `burl/corpus/test_blender.py`):

- Idempotent: `shorten(shorten(x)) == shorten(x)`.
- Preserves every `<|tool_call>...<tool_call|>` verbatim, including the
  terminal `commit_play`.
- Preserves every `<|tool_response>...<tool_response|>` verbatim.
- `len(output) < len(input)` on any row that contained a thought block.

Synthetic rows are marked in the row dict:

```json
{"verbosity": "short_synthetic", "synthetic": true, "source_verbosity": "long"}
```

The trainer (`burl/train/star.py:131`) reads `{"messages":[...]}` only and
ignores the extra keys, so this metadata rides along into the data file
without touching the training code. It becomes useful if we later want to
upsample / downsample synthetics, stratify by source verbosity, or filter
out synthetics for an ablation.

## Blend ratio recommendation: 33% short / 67% long

**Recommended target_short_ratio = 0.33.** Rationale:

- LS-Mixture reports the 25-50% band as the sweet spot; 33% is a natural
  midpoint that most implementations default to.
- Unsloth's "≥75% reasoning" floor is satisfied because every short trace
  still contains ≥1 tool-call reasoning chain (the mechanical reasoning
  Burl was designed to learn). We are not diluting with pure-answer rows.
- Our failure mode is *over-verbosity*, not under-reasoning, so erring
  toward the upper end of LS-Mixture's band is safer than erring toward
  the lower end. 33% is one step up from LS-Mixture's 25% starting point
  without pushing past the 50% ceiling where accuracy drops start to show
  up in the paper.
- Practical: with `n_long = 79` and a 33% target the blender produces
  39 short rows for a 118-row blended corpus. That's ~2× iter-1's N=30
  corpus — enough additional signal to robustly generalize commit
  discipline (which 30 rows underserved, per iter-1 post-mortem).

If team-lead wants to sweep, 0.25 → 0.33 → 0.50 on the same 79 longs is a
one-line CLI change each. Costs nothing to ablate at train time.

### What this looks like in practice

Preview corpus: `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl`
(118 rows). Produced by:

```bash
python -u -m burl.corpus.blender \
  --in burl/data/star_iter0_corpus.jsonl burl/data/star_iter1_corpus.jsonl \
  --out scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl \
  --target-short-ratio 0.33 --seed 42
```

Stats written alongside (`.stats.json`):

| metric | value |
|---|---|
| raw rows | 80 |
| natural long | 79 |
| natural short | 1 |
| synthetic short produced | 79 |
| target short ratio | 0.33 |
| actual short ratio | **0.3305** |
| blended rows total | 118 |
| blended long | 79 |
| blended short (1 natural + 38 synthetic) | 39 |

Length distribution in the blended corpus:

| cohort | n | mean assistant chars | median |
|---|---|---|---|
| long | 79 | 5078 | 4387 |
| short_synthetic | 38 | 767 | 495 |

Mean short / mean long = **15%** — a −85% reduction on the rows that got
shortened, comfortably exceeding LS-Mixture's reported −47.6%. The overall
corpus mean (weighted by ratio) is ~3.6 KB/row vs. ~5.1 KB/row unblended,
so the corpus-wide mean drops ~30% even before the trainer sees it.

## What this does NOT claim to fix

- **The tool-use suppression** (`eq_outcome_distribution` fired 2/10 in
  iter-0 vs. 15/10 in spike v2) lives in the rollout side, not the verbosity
  side. Shortening traces preserves whatever tool mix the rollout produced;
  it does not add eq-distribution calls that were not there. Re-harvesting
  from the spike-v2 prompt shape (team-lead's plan) is the right lever for
  that.
- **Commit discipline** is a function of corpus coverage, not verbosity. A
  larger short pool helps only insofar as it exposes more "and here's where
  I commit" exemplars on diverse decision shapes. iter-1 showed commit
  discipline is fragile at N=30; the 118-row blend goes back above iter-0's
  50 natural rows.
- **The substantive rules errors** (e.g. iter-0 asserting "doubles are
  trump under twos") are comprehension, not verbosity. Shortening will not
  delete the wrong claim; it will delete the long rationalization *around*
  the wrong claim. That's probably still a win — less wrong prose to
  memorize — but the fix for the comprehension error itself is corpus
  selection (K1 pass) or rules-as-tools (T1 track).

## Open knobs the team-lead might tune

| knob | default | what changes |
|---|---|---|
| `--target-short-ratio` | 0.33 | Drop to 0.25 for "gentle LS-Mixture"; raise to 0.50 if iter-3 still shows thought-block regressions. |
| `--seed` | 42 | Only affects which short rows are sampled when the pool exceeds target. Deterministic. |
| `--no-shorten` | off | Disables synthesis. Uses only natural shorts. With 1 natural short across 80 rows this produces a near-no-op blend; useful as a sanity baseline if we later collect natural short traces from a different prompt. |
| `SHORT_THOUGHT_CHARS_MAX` | 200 | Raise if we want to admit one-paragraph plans as "short". Lower for a stricter tool-only cut. |
| `SHORT_ASST_CHARS_MAX` | 1500 | Same idea — cap on total output. |

## What happens next (not in scope for this task)

1. Team-lead eyeballs the preview jsonl, sanity-checks a handful of
   synthetic shorts against their long sources.
2. If the shape looks right, regenerate when iter-2's raw corpus arrives
   (team-lead may re-harvest from the spike-v2 prompt shape first — orthogonal
   to this module).
3. Train iter-2 on the blended output. Eval same N=10 held-out set.
4. Compare iter-2 assistant output length vs. iter-0/iter-1. If `mean_tokens_out`
   drops toward spike-v2 territory (1.5 KB), the blend is working.

## Files landed by this task

- `burl/corpus/__init__.py` — module stub
- `burl/corpus/blender.py` — loader, classifier, shortener, blender, CLI
- `burl/corpus/test_blender.py` — 21 pytest cases (all passing locally)
- `scratch/burl_p5_iter2_prep/corpus_blend_design.md` — this doc
- `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl` — 118-row
  sample output for team-lead review
- `scratch/burl_p5_iter2_prep/star_iter2_blended_preview.stats.json` — the
  stats dump from the CLI run
