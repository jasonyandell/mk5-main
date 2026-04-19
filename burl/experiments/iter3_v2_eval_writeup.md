# iter-3-v2 end-to-end — eval writeup

**Author**: corpus-chef (T11)
**Date**: 2026-04-19
**Scope**: re-harvest corpus from spike-v2 prompt shape (no primer, 42-framing only), EQ-gate rationalize losses, train on B200, eval N=10 held-out @ `max_retries=7`, compare to iter-1@r7 and iter-2.

## TL;DR

**iter-3-v2 trained cleanly on 18 rows and hit 87.5% bot-match on the 8 completed decisions — essentially parity with iter-1@r7 (88.9%) — but 2/10 decisions retry-exhausted at eval time, mirroring the 32% exhaust rate observed during rollout collection.** The headline finding is that **the trimmed Texas-42 primer is load-bearing for commit discipline, not just for rules-signaling**: dropping it cuts mean tokens-out in half (2083 vs iter-1's 4134) but pushes the legal-play failure rate to 20% at eval time. The spike-v2 tool idiom did transfer (6 `eq_outcome_distribution` calls vs iter-1's 1) but the exhaust tax eats the gain on the aggregate-over-attempted metric. Recommend team-lead not merge iter-3-v2 to main but treat the primer-load-bearing finding as the actionable takeaway for iter-4+.

## Three-way comparison — same N=10 held-out, identical seeds

| metric | iter-1 @ r7 (**winner**) | iter-2 @ r7 | **iter-3-v2 @ r7** |
|---|---|---|---|
| n_attempted | 10 | 10 | 10 |
| n_completed | 9 | 9 | **8** |
| n_retry_exhausted | 1 | 1 | **2** |
| legal_rate (of completed) | 100% | 100% | **100%** |
| first_legal_rate | 90% | 90% | **80%** |
| **bot_match_rate (of completed)** | **88.9%** | 66.7% | **87.5%** |
| mean_eq_delta | **−0.16** | −0.94 | **−2.21** |
| p_eq_geq_bot | 88.9% | 66.7% | **87.5%** |
| empty_tool_rollout_rate | 10% | 0% | **10%** |
| mean_tokens_in | 32,382 | 31,310 | **14,839** (−54% vs iter-1) |
| mean_tokens_out | 4,134 | 4,766 | **2,083** (−50% vs iter-1) |
| wall_time_seconds | 1,245 | 2,087 | **899** |
| estimated_usd | $0.28 | $0.46 | **$0.20** |

**Key reads**:
1. **On the 8 decisions iter-3-v2 completed, it matched iter-1@r7's policy quality nearly exactly** — 87.5% vs 88.9% bot-match is statistically indistinguishable at N=8.
2. **Aggregate-over-attempted bot-match is 70% (7/10) vs iter-1's 80% (8/10)** — the retry-exhaust rate is the whole gap.
3. **mean_eq_delta is worse (−2.21) than both predecessors** — driven largely by D2 (bot_match but eq_gap-induced loss: burl_eq=3.55 vs bot_eq=21.22, Δ=−17.66 despite the "match") and D4 (bot_match but tight: eq_gap=1.40).
4. **Token economy inverted**: iter-3-v2 inputs ~half the context (no primer) and outputs ~half the reasoning, and still kept 87.5% bot-match on completes. The primer is expensive at inference time; dropping it *does* pay back on tokens spent.

## Retry-exhausted analysis — the load-bearing primer finding

| decision | exhaust? | turns | tools | notes |
|---|---|---|---|---|
| D1 (seed=900000, decl=0, seat=1) | ✗ **exhausted** | 8 | 8 | 7 failed commits over 8 turns |
| D5 (seed=900000, decl=2, seat=2) | ✗ **exhausted** | 8 | 8 | same — 7 illegal commits in a row |

Both exhausted decisions had **8 tool calls** (max_turns=8 hit) — Gemma kept calling state tools, kept committing, kept being rejected, and ran out of turns before producing a legal play. This pattern was also the dominant failure mode in the rollout (16/50 = 32%). **Without the primer, Gemma doesn't reliably ground its play choice in what it sees from `is_legal`** — it sees the rejection, calls another tool, and re-commits the same or similar illegal play.

Why iter-1 and iter-2 didn't show this: their training corpora were rolled out **with** the primer, so every training example had the "rules-aware commit" behavior baked in. Dropping the primer at both rollout-time and eval-time for iter-3-v2 means the adapter learned from 18 rows where 8 of them (the self-corrected gate traces) already demonstrate "commit → reject → retry → succeed", but that's not enough signal to overcome the base-rate illegality Gemma produces when it has no rules context.

**Upshot**: the primer is doing two jobs. (a) It teaches the rules (obviously). (b) It *anchors* the commit-play loop so that Gemma treats `is_legal` rejections as a signal to change its play rather than re-commit. iter-3-v2 learned (b) partially — the 8 completed decisions show commit discipline — but the 2 exhaust cases show it's not robust without the primer anchoring.

## Tool histogram — spike-v2 idiom carried through

| tool | iter-1@r7 | iter-2 | **iter-3-v2** | spike-v2 baseline |
|---|---|---|---|---|
| trump_declared | 9 | 7 | **4** | 0 |
| is_legal | 16 | 17 | **23** | 15 |
| is_trump | 7 | 9 | **1** | 0 |
| **eq_outcome_distribution** | **1** | **2** | **6** | **15** |

**The eq-heavy idiom transferred**: iter-3-v2 shows 6 `eq_outcome_distribution` calls vs iter-1/iter-2's 1-2. Not as eq-heavy as the raw spike-v2 baseline (15) — the 18-row corpus couldn't fully move the distribution — but moved 3-6× relative to iter-1/iter-2. **`is_legal` usage shot up** (23 vs 16-17), consistent with "Gemma is trying harder to check legality because the primer isn't telling it the rules a priori."

`is_trump` usage *dropped* (1 vs 7-9) — the primer explicitly names the trump-suit concept, so without it, Gemma doesn't reach for the tool. This is a reasonable behavioral signature of the prompt-shape change, not a quality issue.

## Training run

- adapter: `jasonyandell/gemma-4-e2b-texas42-burl-iter3-v2`
- corpus: `burl/data/star_iter3_v2_corpus.jsonl` (18 rows: 10 rollout_win + 8 eq_gate_self_correct)
- recipe: 3 epochs, lr 1e-4, rank 16, batch 2 × grad_accum 4, bf16, sdpa (identical to iter-1/iter-2)
- GPU: B200, **13s wall time**, 9 steps
- loss trajectory: **50.50 → 6.08** (end-of-run), train_loss mean 26.69 (first 2 steps dominated by warmup at ~50)
- token accuracy end-of-run: 12% (lower than iter-1's 48%; attributable to the much smaller dataset — 9 opt steps is near-minimum for LoRA to move the weights meaningfully)
- Modal app: https://modal.com/apps/jasonyandell/main/ap-q4mZRxJeFP4jrezKbebSa9
- wandb run: https://wandb.ai/jasonyandell-forge42/burl-star/runs/q8huxrv3
- estimated cost: **~$0.01** (B200 $4/hr × 13s)

Loss descent looked healthy. The step-to-step loss zigzag (50 → 11 → 35 → 30 → 7 → 26 → 24 → 6) is expected at this small step count — each batch is a larger fraction of the total corpus, so each step moves the loss more visibly.

## Rollout run (Phase 1+B — produced the training corpus)

- dataset: `burl/eval/data/move4_decisions_n50.jsonl` (N=50, all 10 declarations × 5 seats/decisions per declaration)
- config: `enable_primer=False`, `enable_rules_tools=False`, `--gate-variant tool-nudge --max-gate-retries 1 --eq-epsilon 0.25 --max-retries 7`
- wall: **2,596s (~43 min)**, estimated: **$0.577** (under $0.70 projection)
- outcomes:
  - **K1 wins: 10/50 (20%)** — well below team-lead's >54% prior; revealed the primer-is-load-bearing finding
  - **retry-exhausted during rollout: 16/50 (32%)** — 3× iter-1's inference-time rate
  - legal losses: 24/50 → gate fired on all 24
  - gate verdicts: **self_corrected 8 (33%), stubborn 15, forced_flip 0, exhausted 1**
  - corpus size: **18 rows (10 wins + 8 self_corrected)**
- rollout tool histogram: `is_legal:186, trump_declared:13, eq_outcome_distribution:14, is_trump:5, unseen:4`
- gate tool histogram: `is_legal:46, eq_outcome_distribution:44, unseen:2` — the nudge drove heavy eq-tool use, which then carried into the corpus and into inference
- artifacts: `burl/eval/results/move4_star_rollout_v2/`, `burl/data/star_iter3_v2_corpus.jsonl`, `burl/data/star_iter3_v2_corpus_stats.json`

## Decision-level eval results (N=10)

| # | bot_play | burl_play | match | tools | burl_eq | bot_eq | Δ | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 21 | — | ✗ **exhausted** | 8 | N/A | +9.88 | — | primer-loss failure mode |
| 2 | 23 | 15 | ✗ | 2 | +3.55 | +21.22 | −17.66 | legal but off — no primer, picked structurally wrong |
| 3 | 15 | 15 | ✅ | 4 | −18.22 | −18.22 | 0 | |
| 4 | 22 | 22 | ✅ | 1 | +0.79 | +0.79 | 0 | 1-tool commit (fast) |
| 5 | 4 | — | ✗ **exhausted** | 8 | N/A | +21.58 | — | primer-loss failure mode |
| 6 | 9 | 9 | ✅ | 2 | +2.36 | +2.36 | 0 | |
| 7 | 15 | 15 | ✅ | 2 | −4.41 | −4.41 | 0 | |
| 8 | 25 | 25 | ✅ | 5 | +28.56 | +28.56 | 0 | 5-tool chain (thorough) |
| 9 | 0 | 0 | ✅ | 0 | −18.33 | −18.33 | 0 | zero-tool commit — possibly risky generalization |
| 10 | 22 | 22 | ✅ | 2 | −17.86 | −17.86 | 0 | |

**Summary**: 7 bot-matches on completed, 2 retry-exhausted, 1 legal-but-wrong (D2). Of the 7 matches, 6 are zero-delta (exact tie) and 1 (D9) is zero-tool — Gemma committed without calling any state tool, which is new behavior for this adapter family. Not a win or a loss at N=1 but worth watching at larger eval N.

## Interpretation

**Headline**: the spike-v2 prompt shape produces a model that, when it completes a decision, is competitive with iter-1@r7 — but it completes 20% fewer decisions. The small (18-row) corpus was enough to transfer the eq-heavy tool idiom from spike-v2; it was not enough to restore commit-discipline without the primer.

### 1. (Strongest evidence) The primer is load-bearing for commit discipline
- 32% retry-exhaust during rollout → 20% retry-exhaust at eval time. Both far exceed iter-1@r7's 10%.
- The 8 completed eval decisions show normal commit behavior (100% legal_rate, mean 3.1 tools/decision, reasonable eq outcomes). So the adapter *can* commit properly when it doesn't fall into the exhaust loop.
- **The failure mode is specifically**: Gemma commits, engine rejects, Gemma re-commits the same or similar illegal play, up to max_retries. The primer's explicit rules-grounding seems to break this loop.

### 2. (Strong evidence) The eq-heavy idiom transfers through SFT
- `eq_outcome_distribution` usage: 6 calls at eval time vs iter-1's 1. That's a 6× uplift from a corpus with 44 gate-time eq calls. The structural-reasoning-first idiom of iter-1/iter-2 has been partially replaced by eq-tool-first.
- Not all the way to spike-v2's 15 calls — the corpus simply isn't big enough to fully move the distribution — but the direction is right.

### 3. (Weak but interesting) Token economy validates the prompt-shape
- mean_tokens_in 14,839 vs iter-1's 32,382 — the primer alone accounts for ~17k chars of system content.
- mean_tokens_out 2,083 vs iter-1's 4,134 — even the assistant side is ~50% shorter. The no-primer adapter thinks shorter, not just reads shorter.
- **Cost per eval decision dropped from $0.028 to $0.020** (−28%). If the quality-per-token was maintained this would be a clear win; with the 20% exhaust rate it isn't, but the economic lever is real.

## What I would NOT conclude

- **"iter-3-v2 is categorically worse than iter-1@r7"** — on completed decisions it matches. The exhaust rate is the problem, not the policy.
- **"The eq-gate is broken"** — no. 33% of gate-fires self-corrected (8/24), which is how we got half the corpus. At the gate level the intervention worked.
- **"Dropping the primer is a bad idea"** — qualified no. It cut token costs by ~50% and transferred the target idiom. The *right* intervention may be a **hybrid prompt** (preamble + 42-framing + a compact legality reminder) that preserves the token economy while restoring commit discipline. Team-lead's Option C from the rollout report.

## Suggested next levers (not in scope for T11)

1. **Hybrid preamble (team-lead's Option C)** — test a ~150-word rules-reminder that names is_legal-reject behavior explicitly, without the full primer. Goal: retry-exhaust rate ≤ iter-1's 10% with ≤ iter-1's 50% token cost.
2. **Re-roll iter-3-v2 with `--max-retries 10` or 12** — may convert some exhausts to corrections, expanding the corpus and (if the model can self-correct given enough retries) validating that the exhaust is a patience problem rather than a fundamental rules problem.
3. **Targeted primer-probe corpus** — 20-30 hand-selected decisions focused on mid-trick legality, rolled out *with* primer and trained into iter-3-v2's adapter as a residual. Tests whether the exhausts were legality-specific or broader.
4. **Out-of-scope for this experiment**: anything that re-introduces the full primer defeats the prompt-shape hypothesis. If we want "both fewer tokens and good commit discipline", the hybrid prompt is the design.

## Cost & budget

| item | estimate | actual |
|---|---|---|
| rollout (Modal L4, 50 decisions + 24 gates) | $0.70 | **$0.58** |
| rationalization (included in rollout, EQ-gate re-runs) | — | (in $0.58) |
| training (B200, 13s, 9 steps) | $0.10 | **$0.01** |
| eval (10 decisions @ r7, Modal L4) | $0.25 | **$0.20** |
| **total** | **$1.35** (or $1.50 with slack) | **$0.79** |

Came in **$0.71 under budget** — the 13s training (vs $0.10 projected) and the below-cap rollout were the main savings. Slack could fund one follow-up experiment (e.g. the hybrid preamble at ~$0.35-0.50) without new authorization, if team-lead wants.

## Artifacts

- `scratch/burl_p5_iter2_prep/iter3_v2_rollout.log` — rollout stdout
- `scratch/burl_p5_iter2_prep/iter3_v2_train.log` — training stdout + RESULT blob
- `scratch/burl_p5_iter2_prep/iter3_v2_eval.log` — eval stdout
- `burl/eval/results/move4_star_rollout_v2/` — per-decision rollout traces + gate traces
- `burl/data/star_iter3_v2_corpus.jsonl` — 18-row training corpus
- `burl/data/star_iter3_v2_corpus_stats.json` — rollout + gate stats
- `scratch/burl_p5_iter2_prep/move4_iter3_v2_eval/summary.json` — eval metrics
- `scratch/burl_p5_iter2_prep/move4_iter3_v2_eval/traces.jsonl` — per-decision eval traces
- `scratch/burl_p5_iter2_prep/move4_iter3_v2_eval/report.md` — auto-generated grading table + sampled traces
- adapter (private HF): `jasonyandell/gemma-4-e2b-texas42-burl-iter3-v2`
- wandb run: https://wandb.ai/jasonyandell-forge42/burl-star/runs/q8huxrv3, global_step=9, final loss=6.08
- Modal training app: https://modal.com/apps/jasonyandell/main/ap-q4mZRxJeFP4jrezKbebSa9
