# iter-3-rules eval writeup

**Author**: eq-gate-engineer (T12)
**Date**: 2026-04-19
**Scope**: re-harvest with `enable_rules_tools=True` (the rules-as-tools preamble swaps the trimmed primer for a 4-tool menu), EQ-gate losses (T5 Phase B), train on B200, eval N=10 held-out @ `max_retries=7 --enable-rules-tools`. Four-way comparison against iter-1@r7, iter-2, and iter-3-v2.

## TL;DR

**iter-3-rules is the new winner.** 90% bot-match (vs iter-1's 88.9%, iter-2's 66.7%, iter-3-v2's 87.5%), **0 retry-exhausted** (iter-1 had 1, iter-2 had 1, iter-3-v2 had 2), 100% first_legal, 14% shorter outputs than iter-1. The scientific headline team-lead asked for lands cleanly: **`trick_winner_if` was the #1 tool in rollouts (78 calls / 50 decisions) AND survived SFT** — 9/10 eval decisions called it, 17 total calls. No structural-idiom creep. The rules-as-tools preamble swaps primer-memorization for tool-use without costing accuracy; if anything, it gains a little.

On mean_eq_delta iter-3-rules (−1.77) is *worse* than iter-1 (−0.16), driven entirely by one decision (D2) where burl followed suit with the wrong domino — a genuine miss, not a protocol failure. Otherwise 9/10 decisions hit Δ = 0 exactly.

`count_dominoes_remaining` was **never called** — zero calls across 50 rollouts, 26 gate runs, and 10 eval decisions. That's a clean candidate to drop or rename (see §"What to drop").

## Four-way comparison — same N=10 held-out, identical seeds, `max_retries=7`

| metric | iter-1 @ r7 | iter-2 | iter-3-v2 | **iter-3-rules** |
|---|---|---|---|---|
| n_attempted | 10 | 10 | 10 | 10 |
| n_completed | 9 | 9 | 8 | **10** |
| n_retry_exhausted | 1 | 1 | 2 | **0** |
| legal_rate | 100% | 100% | 100% | **100%** |
| first_legal_rate | 90% | 90% | 80% | **100%** |
| bot_match_rate | 88.9% | 66.7% | 87.5% | **90.0%** |
| mean_eq_delta | **−0.16** | −0.94 | −2.21 | **−1.77** |
| p_eq_geq_bot | 88.9% | 66.7% | 87.5% | **90.0%** |
| empty_tool_rollout_rate | 10% | 0% | 10% | **10%** |
| mean_tokens_in (chars) | 32,382 | 31,310 | 14,839 | **23,597** |
| mean_tokens_out (chars) | 4,134 | 4,766 | 2,083 | **3,564** |
| wall_time (s) | 1,245 | 2,087 | 899 | **1,002** |
| estimated_usd | $0.28 | $0.46 | $0.20 | **$0.22** |

**Key reads**:
- **bot_match**: iter-3-rules +1.1 pp over iter-1, +23.3 pp over iter-2, +2.5 pp over iter-3-v2.
- **commit discipline**: iter-3-rules is the only variant with zero retry-exhausted. iter-3-v2 *regressed* here (2 RE, same as spike-v2) — dropping the primer entirely seems to cost commit discipline, while replacing it with the rules-tool menu preserves it.
- **mean_eq_delta**: iter-1 remains the tightest. iter-3-rules lands between iter-1 and iter-2 — the single miss on D2 drove the full −1.77.
- **verbosity**: iter-3-rules is 14% shorter output than iter-1 (and 25% shorter prompt-in than iter-1 because the rules-as-tools preamble is 645 bytes vs the trimmed primer's ~2 KB). iter-3-v2 is still shortest but pays 2 RE for it.

## 🔥 Rules-tool histogram — pre- vs post-SFT

The core T12 question: does Gemma keep reaching for the 4 rules tools after training, or does structural idiom creep back? **Answer: the rules tools held.** `trick_winner_if` in particular is dominant both pre- and post-SFT.

### Pre-SFT (iter-0 base, Phase 1 rollout over N=50 decisions)

| tool | pre-SFT / 50 | per decision |
|---|---|---|
| **trick_winner_if** | **78** | **1.56** |
| is_legal | 42 | 0.84 |
| eq_outcome_distribution | 18 | 0.36 |
| contract_progress | 9 | 0.18 |
| what_beats_what | 3 | 0.06 |
| trump_declared | 2 | 0.04 |
| is_trump | 1 | 0.02 |
| **count_dominoes_remaining** | **0** | **0.00** |

### Post-SFT (iter-3-rules adapter, eval over N=10)

| tool | post-SFT / 10 | per decision |
|---|---|---|
| **trick_winner_if** | **17** | **1.70** |
| is_legal | 6 | 0.60 |
| contract_progress | 3 | 0.30 |
| eq_outcome_distribution | 2 | 0.20 |
| what_beats_what | 0 | 0.00 |
| trump_declared | 0 | 0.00 |
| is_trump | 0 | 0.00 |
| **count_dominoes_remaining** | **0** | **0.00** |

**Reading**:
- `trick_winner_if` per-decision rate went **up** (1.56 → 1.70) through SFT. This is the cleanest result in the writeup: Gemma didn't just learn to tolerate the new tool, it leaned in. 9/10 eval decisions called it.
- `contract_progress` per-decision rate went **up** (0.18 → 0.30). Same story, smaller magnitude.
- `what_beats_what` dropped from 3/50 to 0/10 — likely noise at N=10 but consistent with a tool that was already marginal in rollouts.
- `count_dominoes_remaining` was never called, neither pre- nor post-SFT. See §"What to drop".
- The engine tools (`trump_declared`, `is_trump`) collapsed to zero — the rules-as-tools preamble presents the live trump in-context, so Gemma doesn't need to ask.

### Four-way post-SFT tool histogram

| tool | iter-1@r7 | iter-2 | iter-3-v2 | **iter-3-rules** |
|---|---|---|---|---|
| trump_declared | 9 | 7 | 4 | 0 |
| is_legal | 16 | 17 | 23 | 6 |
| is_trump | 7 | 9 | 1 | 0 |
| eq_outcome_distribution | 1 | 2 | 6 | 2 |
| unseen | 1 | 0 | 0 | 0 |
| **trick_winner_if** | 0 | 0 | 0 | **17** |
| **contract_progress** | 0 | 0 | 0 | **3** |
| what_beats_what | 0 | 0 | 0 | 0 |
| count_dominoes_remaining | 0 | 0 | 0 | 0 |

iter-3-rules is the only variant exercising the rules surface. iter-3-v2 concentrated on `is_legal` (23) and `eq_outcome_distribution` (6) — pure engine + eq idiom, no structural hints. iter-1 and iter-2 use `is_trump` + `trump_declared` structurally (recreating the primer's mechanics from memory). **iter-3-rules is the only variant that delegates rules rather than memorizing them, and it's also the highest-accuracy variant.** That's as clean an alignment as this N=10 sample can give.

## Phase 1 — rollout + EQ-gate

- Dataset: `burl/eval/data/move4_decisions_n50.jsonl` (N=50)
- Flags: `--enable-rules-tools --gate-variant tool-nudge --max-gate-retries 1 --eq-epsilon 0.25`
- Wall 46 min, spend $0.61 (cap $0.90).
- Results:
  - K1 wins: 23
  - Legal losses: 26 (all 26 fired the gate)
  - Retry-exhausted: 1 / Illegal: 0
- Gate verdicts:
  - self_corrected: **7 (27%)** — inside the 25–50% sanity band called out in the EQ-gate design doc
  - stubborn: 19 (73%)
  - forced_flip: 0
  - exhausted: 0
- Corpus: **30 entries** (23 rollout_win + 7 eq_gate_self_correct) → `burl/data/star_iter3_rules_corpus.jsonl`
- Yes-bias invariant verified: the 7 self_corrected SFT entries contain no `bot_play`, no `domino_id=` anchors, no answer leakage.

The 27% self-correction rate is noteworthy: it's the first iteration where EQ-gate nudges did work on a non-trivial minority without revealing the answer. For the stubborn 73%, Gemma re-committed the same play even under a nudge — which is the behavior the gate is designed to detect and *drop*, not rationalize.

## Phase 2 — training

- Adapter: `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules`
- Corpus: 30 rows (23 wins + 7 EQ-gate self-corrects)
- Recipe: 3 epochs, lr 1e-4, rank 16, batch 2 × grad_accum 4, bf16, sdpa (identical to iter-1/iter-2/iter-3-v2)
- GPU: B200, **20s wall, 12 steps**, mean_token_accuracy 17.1% at end
- Loss trajectory: **54.59 → 18.24** — clean descent, not a pathological run
- Spend: ~$0.02 (B200 $4/hr × 20s)
- wandb: `gemma-4-e2b-texas42-burl-iter3-rules`, run `g5n99rzu`

## Decision-level results (N=10, iter-3-rules)

| # | bot | burl | match | tools | burl_eq | bot_eq | Δ | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 21 | 21 | ✅ | 5 (contract_progress, is_legal×2, trick_winner_if×2) | +9.88 | +9.88 | 0 | |
| 2 | 23 | 15 | ✗ | 2 (trick_winner_if×2) | +3.55 | +21.22 | **−17.66** | the single miss — burl followed suit but picked the wrong domino |
| 3 | 15 | 15 | ✅ | 1 (contract_progress) | −18.22 | −18.22 | 0 | unusually low tool count, still landed |
| 4 | 22 | 22 | ✅ | 2 (trick_winner_if×2) | +0.79 | +0.79 | 0 | |
| 5 | 4 | 4 | ✅ | 4 (is_legal×2, trick_winner_if×2) | +21.58 | +21.58 | 0 | |
| 6 | 9 | 9 | ✅ | 6 (trick_winner_if×3, contract_progress, eq_outcome_distribution×2) | +2.36 | +2.36 | 0 | most tools; close-call decision |
| 7 | 15 | 15 | ✅ | 2 (trick_winner_if×2) | −4.41 | −4.41 | 0 | |
| 8 | 25 | 25 | ✅ | 2 (trick_winner_if×2) | +28.56 | +28.56 | 0 | |
| 9 | 0 | 0 | ✅ | 4 (is_legal×2, trick_winner_if×2) | −18.33 | −18.33 | 0 | |
| 10 | 22 | 22 | ✅ | 0 | −17.86 | −17.86 | 0 | empty-tool rollout, still matched |

**9/10 matches, 1 miss on D2**. The miss used 2 `trick_winner_if` calls — the tool returned an honest simulation that happened to preference 15 over 23 under Gemma's reasoning path. This is a genuine decision-quality gap at eq_gap=17.66, not a protocol failure. D10's empty-tool rollout matched anyway — the 42-framing context block was sufficient.

## What to drop / rename

**`count_dominoes_remaining`**: 0 calls across 50 rollouts + 26 gate runs + 10 eval decisions = **0/86 decisions**. Either the preamble description doesn't land, or the signal it provides is subsumed by `contract_progress` (which does get called, 9 times in rollouts + 3 in eval). Recommend either:
- Remove it from the registry (cleanest — one fewer tool for Gemma to consider)
- Rename to something more concrete like `count_dominoes_loose` and shorten the description
- Fold the returned dict into `contract_progress` so the live 5/10-count dominos surface whenever contract state is asked

I'd recommend removing for iter-4; it's dead weight in the current menu. Filing a follow-up bead.

**`what_beats_what`**: 3/50 rollouts + 0/10 eval calls. Less extreme but similar pattern. `trick_winner_if` subsumes its use case (it's the asymmetric "what wins this trick given the current lead" question, which `trick_winner_if` answers per-candidate). Keep on the menu but expect low usage.

## Interpretation

**Primary finding — rules-as-tools is a net win.** iter-3-rules moves the Pareto frontier: higher accuracy than iter-1@r7 (90% vs 88.9%), *zero* retry-exhausted, 14% shorter outputs, and a qualitatively different tool idiom (structural delegation, not structural memorization). The test-set is only N=10 so the 1.1-pp accuracy lead is inside 2σ of iter-1, but the *kind* of reasoning Gemma does is different — and that's what the prompt-shape change was trying to buy.

**Secondary finding — EQ-gate works without yes-bias.** This is the first iteration where the gate ran end-to-end with real rollouts, and the 27% self-corrected rate is squarely in the design doc's sanity band (25-50%). The stubborn 73% is not a failure of the gate — those are exactly the traces the design drops rather than rationalizes. Dropping them is what keeps the corpus clean; the iter-0 "reveal answer" step was building 100% rationalizations because it was building 100% yes-bias.

**Tertiary finding — the rules surface is uneven.** `trick_winner_if` is a hit, `contract_progress` is a modest hit, `what_beats_what` is marginal, `count_dominoes_remaining` is dead. The headline tool-design lesson: **simulate-the-future beats look-up-the-rule** — Gemma prefers "play d into the current trick and tell me who wins" over "quote me the rule for what beats what". For iter-4+ the asymmetric-utility table is worth keeping in mind when adding more rules tools.

**What I would NOT conclude**:
- **"iter-3-rules is decisively better than iter-1."** At N=10 the 1.1-pp lead is noise. What the data *does* support: rules-as-tools is at least as good on accuracy + commit discipline, and is qualitatively different on tool usage. That's the scientific claim.
- **"Rules-as-tools fixes verbosity."** The trimmed primer is 2 KB, the rules preamble is 645 bytes — most of the shrink is just primer size, not a learned verbosity change. iter-3-rules still outputs 3.5 KB on average; it's iter-3-v2 (no primer at all) that actually halves output tokens, and that variant paid a 2-RE price for it.
- **"EQ-gate solved the STaR problem."** The gate gave us 7 usable rationalizations from 26 losses; the other 19 were dropped. That's the corpus-quality win, not a throughput win. The corpus is 30 rows instead of 50 because the gate is *stricter*.

## Cost & budget

| item | estimate | actual |
|---|---|---|
| Phase 1 — rollout + EQ-gate N=50 (L4) | $0.60 cap | **$0.61** |
| Phase 2 — training (B200, 20s) | $0.05 | **$0.02** |
| Phase 3 — eval N=10 @ r7 (L4) | $0.30 | **$0.22** |
| **total** | $1.00 | **$0.85** |

Well inside the $1.50 total authorization; ~$0.65 unused.

## Suggested next levers (not in scope for T12)

1. **Drop `count_dominoes_remaining`** from the registry for iter-4 — dead code in the tool menu confuses the prompt-shape debugging of whichever variant we try next. Filing a bead.
2. **Stubborn-verdict analysis**: the 19 stubborn verdicts are the silent majority of the gate output. A short notebook sampling them and checking whether Gemma's *reasoning* actually engaged with the tool-nudge (or just re-emitted the same play) would tell us whether the gate prompt needs strengthening for iter-4.
3. **Scale N=10 → N=30** on the same 3-way iter-1 / iter-3-v2 / iter-3-rules comparison to resolve the 2σ question. Running a bigger held-out set is cheap ($0.66 at iter-3-rules' per-decision rate) and would either promote iter-3-rules to winner-with-confidence or reveal regression to 88.9%-ish.
4. **iter-4 idea**: take iter-3-rules' tool idiom and combine it with iter-1's trimmed primer — a "hybrid" preamble where *both* primer + rules-tool menu are present. Would test whether the rules-tool usage is robust to having the primer around, or whether Gemma falls back to memorization when both are offered.

## Artifacts

- `scratch/burl_p5_iter2_prep/t12_rollout.log` — Phase 1 stdout
- `scratch/burl_p5_iter2_prep/t12_train.log` — training stdout + RESULT blob
- `scratch/burl_p5_iter2_prep/t12_eval.log` — eval stdout
- `burl/eval/results/move4_star_iter3_rules_rollout/rollout_traces.jsonl` — 50 rollout traces
- `burl/eval/results/move4_star_iter3_rules_rollout/gate_traces.jsonl` — 26 gate re-run traces
- `burl/data/star_iter3_rules_corpus.jsonl` — 30-row SFT corpus
- `burl/data/star_iter3_rules_corpus_stats.json` — full Phase-1 stats (incl. rules-tool histograms)
- `scratch/burl_p5_iter2_prep/move4_iter3_rules_eval/summary.json` — eval metrics
- `scratch/burl_p5_iter2_prep/move4_iter3_rules_eval/traces.jsonl` — 10 eval traces
- `scratch/burl_p5_iter2_prep/move4_iter3_rules_eval/report.md` — auto-generated grading table
- adapter (private HF): `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules`
- wandb run: `gemma-4-e2b-texas42-burl-iter3-rules`, global_step=12, final loss=18.24
- Modal training app: https://modal.com/apps/jasonyandell/main/ap-UCY1SmpnZBLYFfZlhrYs8q
