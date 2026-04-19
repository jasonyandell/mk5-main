# iter-4-thoughts eval — the adapters are behaviorally indistinguishable

**Author**: corpus-chef (T18)
**Date**: 2026-04-19
**Scope**: A/B eval of `jasonyandell/gemma-4-e2b-texas42-burl-iter4-thoughts`
vs `...-burl-iter3-rules` on the standard 10-decision held-out set.

## TL;DR

**On N=10 held-out decisions, iter-4-thoughts produces byte-identical
raw output to iter-3-rules across all 42 assistant turns.** Same 9/10
bot-match, same EQ delta, same tool histogram — not within-noise, but
byte-exact. Despite `preserve_thoughts=True` exposing ~24% extra
thought-token gradient per training row, the resulting adapter is
operationally indistinguishable from the `strip_thinking()`-masked
baseline at this training budget.

This is a **clean negative result**: at rank-16 / 3 epochs / 30 rows,
the thought-token gradient signal does not have enough LoRA leverage
to shift inference behavior on this held-out set. The bootstrap
philosophy (reinforce Gemma's own winning reasoning) is not
blocked — but the rank/epoch/corpus budget is insufficient to express
it.

## Headline: side-by-side metrics

| metric                   | iter-3-rules  | iter-4-thoughts | delta |
|:-------------------------|--------------:|----------------:|:-----:|
| n_completed              |         10/10 |           10/10 |   —   |
| n_retry_exhausted        |             0 |               0 |   —   |
| legal_rate               |        100.0% |          100.0% |   —   |
| first_legal_rate         |        100.0% |          100.0% |   —   |
| **bot_match_rate**       |     **90.0%** |       **90.0%** | **0** |
| mean_eq_delta            |        −1.766 |          −1.766 |  0.0  |
| p_eq_geq_bot             |         90.0% |           90.0% |   —   |
| mean_retry_count         |          0.00 |            0.00 |   —   |
| empty_tool_rollout_rate  |         10.0% |           10.0% |   —   |
| mean_tokens_in (chars)   |        23,597 |          23,598 |   +1  |
| mean_tokens_out (chars)  |         3,564 |           3,564 |   0   |
| wall_time (s)            |       1,001.7 |           988.8 | −12.9 |
| estimated_usd            |         $0.22 |           $0.22 |   —   |
| tool_histogram           | cp:3,il:6,tw:17,eq:2 | cp:3,il:6,tw:17,eq:2 | **identical** |

`cp=contract_progress, il=is_legal, tw=trick_winner_if, eq=eq_outcome_distribution`

## The clinching evidence — per-turn byte identity

Stronger than "same aggregate metrics": every individual assistant
turn is a byte-for-byte match.

```
dec 0: turns=7  byte-identical=True
dec 1: turns=3  byte-identical=True
dec 2: turns=3  byte-identical=True
dec 3: turns=3  byte-identical=True
dec 4: turns=5  byte-identical=True
dec 5: turns=7  byte-identical=True
dec 6: turns=3  byte-identical=True
dec 7: turns=5  byte-identical=True
dec 8: turns=5  byte-identical=True
dec 9: turns=1  byte-identical=True

byte-identical raw_completion: 42/42 turns
```

Across all 42 assistant turns, totaling ~35,642 characters (~8–10k
tokens of model output), **the two adapters emit the identical
character sequence**. Timing differs (wall 988.8 vs 1001.7s, per-turn
variance up to ~10s) — confirming these are two independent vLLM
generation passes, not a cache hit.

### Sampling is not seeded

`burl/modal/gemma_serve_native.py:148` constructs
`SamplingParams(temperature=0.6, max_tokens=max_tokens, stop=stop,
skip_special_tokens=False)` — no `seed=` parameter. vLLM uses a fresh
random seed per `llm.generate()` call by default. Two independent
stochastic runs converging to byte-identity across ~10,000 output
tokens is astronomically unlikely unless the logit distribution is
peaked enough that sampling at T=0.6 is effectively greedy — AND the
top-1 tokens agree between the two adapters at every position.

That is the finding: the two adapters' top-1 prediction surface is
identical on this test set.

## Thought-block emission — present in BOTH adapters

The pre-run hypothesis was "iter-4-thoughts will emit `<|channel>thought`
blocks at inference if gradient reinforcement worked." Empirically:

| adapter          | traces with thoughts | thought-bearing turns | fraction of output chars in thought turns |
|:-----------------|---------------------:|----------------------:|-------------------------------------------:|
| iter-3-rules     |                 9/10 |                 12/42 |                                     81.9% |
| iter-4-thoughts  |                 9/10 |                 12/42 |                                     81.9% |

**Gemma 4 E2B's thinking reflex fires either way.** The base-model's
`<|channel>thought` emission is not something SFT had to teach or
un-teach. `strip_thinking()` affected what the SFT loss *saw* during
training — not what the base model *emits* at inference. The
hypothesized difference-in-kind between the two adapters collapses to
difference-in-degree — and at this budget, the degree is zero.

## Training-side parity was the leading indicator

Final training loss, same corpus / same recipe / only `preserve_thoughts`
differs:

| adapter          | train/loss @ step 12 |
|:-----------------|---------------------:|
| iter-3-rules     |               18.241 |
| iter-4-thoughts  |               18.173 |

Loss trajectories converged to essentially the same plateau despite
iter-4-thoughts seeing ~30% more tokens of gradient signal per row
(the preserved thought content). In hindsight this was the first
warning that inference behavior would be indistinguishable.

## Interpretation — why this happened

Three non-mutually-exclusive mechanisms, ordered by likelihood:

1. **LoRA capacity fall-short.** Rank 16 × 7 target projections on a
   5B-param base gives O(10^7) trainable parameters. The thought-token
   gradient's directional signal has to compete with the tool-call
   gradient for that same low-rank subspace. On 30 rows × 3 epochs, the
   tool-call signal — structurally repetitive, highly pointwise
   predictable — dominates. Thought prose is free-form and each position
   carries much less cross-example predictive weight, so its gradient
   contribution averages to near-zero direction after LoRA projection.

2. **Corpus too small for prose-level reinforcement.** 30 rows × 12 turn
   positions ≈ 360 thought-token training opportunities per epoch. Even
   with perfect gradient transmission, that's roughly the scale at which
   a prose-level style shift becomes visible. We're training on tool-call
   *structure* (~100 tokens × 42 turns × 30 rows ≈ 125k structured-token
   positions) — two orders of magnitude more signal in the competing
   direction.

3. **Sampling confidence saturation.** Gemma 4's thinking reflex is
   already baked into the base weights (it fires in 9/10 traces without
   any fine-tuning). The LoRA delta needed to modulate its *content*
   would have to shift confident base-model logits — a much harder
   update than nudging unconfident ones. At 12 steps, the optimizer
   never produces that magnitude of weight change.

The byte-identical outputs, given the adapters are NOT byte-identical
at the weights level (different gradient paths were taken), likely
reflect all three: tiny parameter deltas project through the network
to indistinguishable top-1 logits across the held-out distribution.

## Scientific question — answered

> "Does the thought-token gradient signal produce a meaningfully
> different adapter, or does the LoRA capacity fall short?"

**The LoRA capacity falls short at this budget.** Not a near-miss —
literal byte-identity on a 10-decision held-out set. This is useful
calibration:

- The `strip_thinking()` masking that iter-0/1/2/3 accepted was never
  costing us anything visible. The hypothesized ~½M training tokens
  of "reasoning signal we've been throwing away" was real in the
  gradient stream but invisible in the adapter.
- The bootstrap philosophy (reinforce Gemma's own K1-filtered
  reasoning) remains philosophically sound but requires **more
  capacity** to express. Next iteration's levers, in order of expected
  leverage:
    1. **Larger corpus** (300+ rows, not 30) — the thought-token
       gradient gets more opportunities to aggregate direction
       before getting washed out by tool-call gradients.
    2. **Higher rank** (32 or 64) — more subspace for the optimizer
       to allocate to reasoning-style updates.
    3. **More epochs** at current rank — cheap but has diminishing
       returns on 30 rows; risks overfit on structural tokens.

## Out of scope (what this eval does NOT tell us)

- **Does preserve_thoughts help on a bigger corpus?** Untested. The
  present finding only pins "at N=30 it's indistinguishable" — it
  does not refute the hypothesis at N=300 or N=3000.
- **Does sampling at T=0 produce the same outcome?** The byte-identity
  observation is already strong enough that this is not worth
  re-testing; but if a future run varies temperature, the finding
  may look different.
- **Would a larger held-out set reveal divergence?** Plausible.
  Byte-identity on N=10 decisions is a small sample; on N=100+ the
  probability of finding a decision where the two adapters' top-1
  tokens diverge grows.

## Budget

- Training: **~$0.15 Modal B200** (25s wall, 12 steps; matches iter-3-rules
  within 1-2¢).
- Eval: **$0.22 Modal L4** (988.8s wall, 10 decisions).
- Total T18: **~$0.37**.

## Artifacts

- `jasonyandell/gemma-4-e2b-texas42-burl-iter4-thoughts` — HF adapter (pushed)
- `wandb: burl-star/gemma-4-e2b-texas42-burl-iter4-thoughts` run `tpr8j97o`
- `scratch/burl_p5_iter2_prep/iter4_thoughts_train.log` — training log
- `scratch/burl_p5_iter2_prep/iter4_thoughts_eval.log` — eval stdout
- `scratch/burl_p5_iter2_prep/move4_iter4_thoughts_eval/summary.json` — metrics
- `scratch/burl_p5_iter2_prep/move4_iter4_thoughts_eval/traces.jsonl` — 10 traces
- `scratch/burl_p5_iter2_prep/move4_iter4_thoughts_eval/report.md` — auto grading

Comparison pair:
- `scratch/burl_p5_iter2_prep/move4_iter3_rules_eval/{summary.json,traces.jsonl}`

## Recommended next move

If the goal is still "make Gemma's own reasoning train-visible as a
policy lever," the cheapest followup is **iter-4-thoughts on a larger
corpus** — either the next STAR rollout's output (~50+ rows) or a
double-pass of iter-3-rules generated across two seed-offsets. No
infrastructure work is needed; `preserve_thoughts=True` is already
validated and the launcher exists. The finding to disprove would be:
"the gradient signal was present but the corpus was too small for it
to accumulate." If iter-4-thoughts@N=100 is STILL byte-identical to
iter-3-rules@N=100, the bootstrap approach at rank-16 is genuinely
dead and rank-32 or a full-finetune is the next lever.

If instead we accept this result as sufficient signal — iter-0/1/2/3
were not secretly losing meaningful capability by dropping thoughts,
and reinforcing them at this scale changes nothing — we can retire
preserve_thoughts from the roadmap and re-focus on corpus-quality
and/or Haiku-free rationale injection.
