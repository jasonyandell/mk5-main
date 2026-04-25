# Gus — Practicalities

> Running receipts log. Appends what we actually learned while building the
> LAMIR-ready distillation pipeline. Written in the spirit of
> `burl/PRACTICALITIES.md`.

Read alongside `OVERVIEW.md` (vision), `BUILD_PLAN.md` (architecture + plan),
and the three `MORNING*_STATUS.md` snapshots (point-in-time results).

---

## 1. Regret is vs the oracle teacher, not the theoretical optimum

`regret = oracle_best_eq - eq[student_action]` — the distance between the
student's pick and the oracle's best legal action, measured in the oracle's
own E[Q] vector. Two implications:

- **A perfect student has regret 0**, which is in principle reachable — we're
  distilling, not playing a new game.
- **The oracle itself is noisy.** Stage-1 Q-network MAE is ~0.94, plus SEM=0.5
  from the per-decision adaptive sampling. The practical floor for student
  regret is ~0.5-1 Q-pt; below that we're chasing teacher noise.

So 1.39 mean regret at 3000g = ~2× the teacher-noise floor. Room to improve,
but don't expect zero.

**Do not** chase regret against an unknown optimum — that's a different project
(RL, self-play, game-theoretic solving).

## 2. The Q-MAE ≈ 10 number is not what it looks like

Q-head MAE ~10 Q-pts on a ±42 scale *sounds* bad. It isn't — it's measuring
per-world Q predictions against a target whose intrinsic variance is σ ≈ 20
(decision 0 empirically). Predicting per-world outcomes of a hidden-info game
from partial observations has an irreducible component that no amount of model
capacity can predict away.

Contrast:
- **Perfect-info oracle Q-MAE: 0.94.** Target is deterministic given full deal.
- **Imperfect-info Q-MAE: ~10.** Target is stochastic given visible state, and
  the randomness is what we're modeling through the belief head.

**V-MAE (~5 for the 3000g big) is the right "how good is the student's
positional understanding" number.** It predicts the marginal (averaged-over-
worlds) E[Q], which is a much smoother target. 5 Q-pt V-MAE on a 42 scale
is usable but has headroom.

**π_me regret (1.39 Q-pt) is the right "how well does the student play"
number.** Everything else is infrastructure.

## 3. Single-step PIMC underperforms direct π_me — strategy fusion

PIMC-via-Q_head (sample K worlds from belief, compute Q_head per world,
average, argmax) loses to the π_me head by 3-4 percentage points of bot-match
regardless of K (tested K=50, K=200, K=500). All saturate at ~62%.

Why: **strategy fusion.** PIMC averages over per-world "best actions" as if
the policy could condition on the world. Real policies have to pick one
action without knowing which world is true. Averaging per-world Qs gives a
score that no single committed policy can realize.

π_me is trained directly on the oracle's argmax(marginal E[Q]), which already
internalizes the anti-strategy-fusion correction. PIMC at inference can't add
it back — it literally undoes what π_me learned.

**Rule of thumb**: if you have a policy trained on marginalized-E[Q]
supervision, don't re-marginalize at inference. Just use the policy.

**Multi-step LAMIR is different** because each level of the search tree
commits to an action — no strategy fusion. But that requires a π_opp head
(not yet trained) and engine plumbing (deferred).

## 4. Data scaling dominates capacity at these corpus sizes

The clean result of the session:

| corpus | 1.2M params | 3.4M params | 7.4M params |
|---|---|---|---|
| 100g | 2.48 | — | — |
| 1000g | 2.16 | 2.10 | — |
| 2000g | — | 1.60 | 1.65 (XL flat) |
| 3000g | — | **1.39** | 1.54 (XL regressed) |

Reading:
- Doubling data reduced regret every time (100g → 1000g: −0.32; 1000g → 2000g:
  −0.56; 2000g → 3000g: −0.21).
- Going from 1.2M to 3.4M helped marginally at 1000g (−0.06 regret).
- Going from 3.4M to 7.4M actively hurt regret at 2000g and 3000g, despite
  composite score improving.

The 7.4M XL on 3000g gave the highest composite score (0.219 vs 0.208) but
WORSE regret (1.54 vs 1.39). The heavier regularization (dropout 0.15, WD
0.03) improved the auxiliary head MAEs that the composite rewards but
*blunted* the policy head.

**Lesson**: composite metrics can disagree with the metric that matters.
Regret > composite > bot-match > individual head accuracy.

## 5. The regret distribution is bimodal — 73% perfect, 6% blunder

For the 3000g_big adapter on 560 held-out decisions:

| regret bucket | count | % |
|---|---|---|
| optimal / dead-tie (<0.1 Q-pt) | 410 | 73.2% |
| near-tie (0.1-0.5) | 23 | 4.1% |
| minor slip (0.5-1) | 13 | 2.3% |
| modest miss (1-2) | 23 | 4.1% |
| real miss (2-4) | 30 | 5.4% |
| big miss (4-8) | 27 | 4.8% |
| **BLUNDER (8+)** | **34** | **6.1%** |

Percentiles: p50=**0** (median is perfect!), p90=4.4, p95=9.1, p99=20.1, max=29.1.

**1.39 mean regret is entirely driven by the 6% blunder tail.** If we could
halve the blunder rate without hurting the perfect rate, mean regret drops to
~0.9.

This is consistent with 42's strategic structure: most decisions are trivial
or near-tied, a few are make-or-break, and the student occasionally walks off
the cliff. Future work on blunder reduction should probably target these
specific decisions (identify via regret, augment training with additional
examples, or use LAMIR look-ahead as a safety net on high-entropy positions).

## 6. Explicit void features barely help — transformer infers voids from plays

Adding engine-computed void features (24-dim: 3 opponents × 8 suits, fed to
CLS position) moved belief top-1 from 37.2% → 38.6% at 1000g and left π_me
flat. The transformer was already learning void inference attentionally from
play tokens.

**Corollary**: explicit features are worth it when they encode information
the network CANNOT derive from inputs. If the derivation is straightforward
and the architecture is attention-based, the explicit version is cosmetic.

## 7. GPU contention on MPS is real but linear-ish

Running 10k-game gen concurrently with XL training on one M5 Max:
- Gen chunk time: 7 min (alone) → ~9 min (contended)
- XL epoch time: 22s (alone) → ~65s (contended)

Both cost ~30% wall overhead. Not free, but parallel scheduling still beats
sequential when total compute > 1 Apple unit. **Run them simultaneously when
wall-time pressure is low**; sequentialize when a single run needs to finish
fast.

## 8. Data-to-oracle training-set comparison

The Stage-1 Q oracle was trained on 11.24M tokenized states (1,124 seeds × 10
declarations, ~215 GB).

Gus's current corpus (3,800 games):
- **360M (state, world) training examples** — ~32× more in raw count
- **106K distinct info-states** — 100× FEWER than the oracle
- **3,800 seed/decl combinations** — 3× fewer than the oracle

We get more data per state (3,400 sampled worlds per decision) but much less
state diversity. At 10k games: 280K distinct states — still way below the
oracle's 11M.

**Unseen lever**: generate 10 declarations per seed instead of 1 (the oracle's
recipe). Would 10× state diversity at the same gen cost. Worth trying if
further data scaling plateaus.

## 9. The joint-world tensor is the pipeline's keystone

Saving `(world_hands, q_per_world)` per decision was a substantive
infrastructure lift (edits to `forge/eq/generate/{types,actions,pipeline,
adaptive,cli}.py`). Without it, the full 4-head student wouldn't be
possible — the Q_head's supervision signal lives entirely in that tensor.

**Storage is manageable**: ~6 MB per game at SEM<0.5, so 10k games ≈ 60 GB.
Gitignored, regeneratable.

**Opt-in**: `--save-joint-worlds` flag keeps the tensor off normal gen runs.

## 10. Timeline calibration

On M5 Max (MPS, solo run):
- 100-game adaptive gen at SEM<0.5: ~7 min
- 1000-game training (3.4M params, 60 epochs): ~10 min
- 3000-game training (3.4M params, 60 epochs): ~25 min
- 3000-game training (7.4M params, 80 epochs): ~70 min alone, ~90 min contended

The gen generator is the long pole for scaling experiments. Budget 7-10 min
per 100-game chunk of corpus expansion.

## Open questions (that came up, didn't block)

- **Does a better teacher exist?** Stage-1 oracle Q-MAE is 0.94. A higher-
  accuracy oracle (more solver samples, bigger Q network) would move the
  student's teacher-noise floor down.
- **Does distillation with the raw PDF help?** We distill on per-world Q
  scalars, not the 85-bin outcome PDF. The PDF is richer; might help Q-head
  calibration if targeted.
- **Is there a fundamental mid-game hardness floor?** Decisions 4, 8, 11, 12,
  16 always have high regret. Is that the oracle's own noise concentrating
  there, or real strategic ambiguity no student can resolve?

## 11. Game-level play costs ~18pp vs the oracle baseline (arena)

Decision-level regret of 1.39 Q-pt sounds small; game-level it compounds
to real outcomes. Arena at n=150 (15 seeds × 10 decls), student at seat 0
vs all-E[Q]-bot baseline on the same seeds:

- Student team wins: **78/150 = 52%**
- All-bot baseline: **105/150 = 70%**
- Student bidder-made rate: 22% (bot: 39%)
- Avg bidder points: 20.9 (student) vs 25.7 (bot), Δ −4.8 per hand

1.39 Q-pt regret per decision × 28 decisions = ~39 Q-pt team-level shift,
which is ~18 pp of team-level win rate. Decision regret compounds about as
expected.

This is the "so what does this student actually play like" number. It's
weaker than the oracle but not random — above 50% team-win rate on sharp
games. Detect-and-route (receipt 13) is the lever to close most of this
gap without retraining.

## 12. Ensembling by naïve averaging hurts regret (while boosting bot-match)

Tested 4 ensemble strategies over 8 trained adapters at `gus/adapters/`:

| strategy | bot-match | regret | blunders |
|---|---|---|---|
| best single (v2_voids_3000g_big) | 67.9% | **1.39** | 34 |
| majority vote | 69.1% | 1.55 | 40 (worse) |
| softmax avg | 71.3% | 1.46 | 36 (worse) |
| v-weighted avg | 71.3% | 1.43 | 35 (worse) |
| **oracle-per-decision (ceiling)** | **89.6%** | **0.36** | **8** |

Bot-match goes up, regret goes up: ensembles win near-ties (where any
legal play is ~equivalent) but lose on sharp decisions by diluting the
confident-right adapter with less-confident ones. The oracle ceiling
shows massive latent diversity — 58% of decisions have adapter
disagreement, 24% have 3+ distinct actions picked — but simple voting
strategies can't exploit it. **The right move is a router, not an
ensemble.**

## 13. Blunder detect-and-route is the practical ceiling (detector AUC 0.84+)

Small gradient-boosted classifier that predicts "will the student
blunder at this decision" from state-only features:

- **Oracle-feature detector** (uses oracle E[Q] summary stats): ROC-AUC
  **0.926**. At 15% flag rate: 80% recall of blunders.
- **Student-feature detector** (inference-deployable — uses student's π
  confidence, Q_head stats across K=20 sampled worlds, V_head,
  belief entropy): ROC-AUC **0.839**. At 20% flag rate: regret drops
  1.13 → 0.49 (57% reduction).

Top student-feature importances:

1. `pi_peak` (0.19) — student's policy confidence
2. `pi_entropy` (0.07)
3. `v_minus_polexpq` (0.07) — V/π consistency gap
4. `q_mean_std_legal` (0.06) — student's analog of oracle_spread
5. `v_minus_qmean_chosen` (0.06)

**Finding**: student's Q_head spread is a weaker proxy for oracle
spread than hoped (0.045 importance vs oracle's 0.390). Q_head was
trained on one random world per forward pass — spread signal is noisy.
Two fixes queued for v4-class experiments:
- Multi-world variance regularization during Q_head training
- K=50+ worlds at inference (cheap, batched)

**Upshot**: detect-and-route with inference-only features is viable.
Project regret ~0.5 Q-pt (near teacher-noise floor of 0.5-1) and game-
level arena should recover ~60% of the −18 pp gap. That's the practical
ceiling for vanilla distillation; beating it requires multi-step LAMIR
or a cleaner teacher.

## Emerging architecture: detect → route → fallback

The shape of a deployable Gus v1.0:

```
   student π_me (fast path, 1.39 regret, ~ms)
          │
          ▼
   blunder_detector (tiny GBM, ms)  ── flags ~20%
          │                         │
          │ ok                      │ flag
          ▼                         ▼
   commit action             fallback: oracle argmax (has budget)
                             OR PIMC-Q-K50 (doesn't work YET,
                                            needs Q_head variance reg)
```

**Router PoC (receipt 14) confirms**: oracle fallback at 20-25% flag hits
the projected 0.49-0.56 regret. But PIMC-Q-K50 as fallback HURTS (1.47 vs
1.39 baseline) because the student's Q_head is too noisy. Next-best-
adapter fallback is even worse (1.55). **Without an oracle budget, the
detect-and-route architecture is blocked on the Q-head-variance fix**
(multi-world training regularizer or K=100+ at inference).

## 15. Distribution-target belief calibration — works in isolation, doesn't propagate

Frozen-trunk fine-tune of just the belief head with a **soft** target —
the empirical marginal `P(domino ∈ seat)` over the M sampled worlds
already saved in `world_hands`, instead of the one-hot truth. Script
at `scratch/train_belief_distribution.py`.

Isolated:

| metric                       | before (truth) | after (dist.) |
|------------------------------|---------------:|--------------:|
| truth top-1 accuracy         | 38.4%          | 38.3%         |
| KL vs world-marginal (held-out) | 0.078       | **0.062**     |
| (uniform-belief baseline KL) | 0.081          | —             |

The calibration is genuinely better — distribution training closes
about 47% of the gap between uniform-prior and a hypothetical perfect
belief. Truth-trained belief was optimizing for the mode, not the shape.

Downstream:

| test                   | before | after  |
|------------------------|-------:|-------:|
| pimc-belief K=50       | 66.4%  | **65.4%** (regressed) |
| blunder detector ROC-AUC | 0.792 | 0.793 (flat) |
| blunder detector PR-AUC  | 0.175 | 0.133 (regressed) |

Better belief did NOT help the student play better. Likely cause: the
Q_head was co-trained with mode-sharp world samples at training time;
softening the belief distribution at inference creates distribution
shift the Q_head wasn't prepared for. Improvements don't stack on top
of a frozen ecosystem.

**Implication**: the right follow-up is CO-TRAINING {belief, world_encoder,
Q_head} together with distribution-belief + regular Q loss. Stacking
calibration fixes on a frozen architecture doesn't work — the integrated
approach is necessary.

**Not promoted** (agent's call): ablation result, good diagnostic, not a
ship-able adapter.

## 14. Router PoC reality-check: oracle fallback works, student fallbacks don't

End-to-end validation of the detect-and-route architecture on 560 held-out
decisions (`gus/eval/router.py`):

|           flag% | oracle | PIMC-Q-K50 | v1_full_1000g |
|----------------:|-------:|-----------:|--------------:|
|            none | 1.39   | 1.39       | 1.39          |
|              5% | 1.15   | 1.39       | 1.48 (worse)  |
|             10% | 0.97   | 1.44       | 1.49          |
|             15% | 0.82   | 1.45       | 1.49          |
|         **20%** | **0.56** | 1.47 (worse) | 1.55       |
|             25% | **0.49** | 1.48     | 1.69          |

- **Oracle fallback** hits the projected ~0.5 Q-pt regret at 20-25%
  flag rate — the architecture works as designed if we can afford
  oracle calls at inference.
- **PIMC-Q-K50 fallback is actively worse than no routing**. At 20%
  flag: catches 23 of 34 blunders via detector, but PIMC-Q only
  fixes 7 of them and introduces 6 new blunders on non-blunder
  decisions the detector flagged. Q_head trained on one world per
  forward is too noisy for belief-sampled averaging to rescue.
- **Next-best-adapter fallback** (v1_full_1000g) is worst — smaller
  weaker model is wrong on the same hard decisions.

Implication: to ship a no-oracle-inference student, the prerequisite
is a Q_head that survives multi-world averaging. Two approaches:
multi-world variance regularization during training (change train_v2_voids
loss to query Q_head on K worlds and penalize cross-world variance) OR
K=100+ at inference (cheap but only 2× improvement over K=50 at best).

## 16. Eager dataset OOMs at 10k games — the lazy IterableDataset is the fix

Baked into the pipeline since day one, `JointWorldFullDataset` loaded
every chunk's games into a list at `__init__`. Fine for the 100-1000-game
corpora (≤11 GB) and even for 3000 games (~33 GB). Breaks hard at 10k.

**Observed** (2026-04-21 15:41 CDT): v3 consistency training on the full
100-chunk 10k-game corpus was killed by macOS SIGKILL (OOM) before the
first forward pass. The eager load hit ~110 GB of unified memory and the
OS wouldn't tolerate it alongside the Python runtime.

**Wrong instinct**: just subsample. Works as a patch but doesn't scale,
and the Vast-fleet plan targets 100k+ games / multi-TB corpora where
even subsampling the eager path fails.

**Right fix**: `JointWorldFullIterable(IterableDataset)` in the same
module. Streams chunks in random order; each chunk is loaded, its items
are shuffled and yielded through an 8192-item shuffle buffer, then the
chunk is released. Memory footprint is O(one chunk + buffer),
independent of corpus size.

Smoke-test receipts (5 chunks × 2 epochs, CPU):
- Peak RSS: **3.4 GB**, flat across epochs (no growth leaks).
- Throughput: ~4500 items/s (CPU-bound on item-building, not disk-bound).
- __len__ scan: ~0.5s per chunk one-time; cached to disk via
  `--length-cache` so subsequent runs skip it.

Tradeoffs accepted:
- **Approximate shuffle.** Each batch draws from an 8192-item buffer
  dominated by the currently-loading chunk, so mini-batches have slight
  chunk-locality bias. WebDataset uses the same pattern at much larger
  scale; in our training it was imperceptible (see §X below once v3-10k
  vs v2-10k-big numbers land). If shuffle quality ever becomes suspect,
  bump buffer to 32K items (~200 MB, still trivial).
- **No multi-worker support yet.** `DataLoader(num_workers>0)` raises
  explicitly rather than silently misbehave. Our scripts use 0 anyway.

Landed in `gus/model/dataset_seq_world.py` and behind `--lazy` flags in
`train_v2_voids.py` and `train_v3_consistency.py`. Eager class retained
for eval corpora (small, random-access friendly). Two tools for two
jobs — not legacy; eager is the right tool on ≤1 GB inputs.

**Implication**: the distillation pipeline is now memory-bounded by
chunk size, not corpus size. Vast-fleet scale-out is no longer blocked
on infrastructure.

## 17. Oracle optimizes p_make, not E[Q] — utility is cliff-shaped

While investigating how the teacher picks its "best move," we traced
`forge/eq/generate/actions.py` line 126:

```python
score = p_make_masked + 1e-6 * e_q_normalized
actions = score.argmax(dim=1)
```

`p_make` is the probability of making the contract (P(Q ≥ 18) for
offense at bid 30). `e_q_normalized` is E[Q] scaled to [0, 1] over
legal actions. The 1e-6 coefficient is smaller than float32 precision
on most p_make gaps — **the oracle is effectively optimizing pure
p_make**.

This is the correct utility for 42 scoring. A mark is a mark; 30 and 42
yield the same mark. 29 and -42 yield the same mark (to opponents).
Margin has no durable value. The threshold cliff is the whole thing.

What we confirmed:
- "29 is never 30" is preserved — p_make=0 (cert. loss) loses to any
  p_make>0 at argmax. ✓
- Above threshold, all winning actions tie. Oracle picks arbitrarily.
  Acceptable: margin above threshold buys nothing.
- Below threshold, all losing actions tie. Oracle picks arbitrarily.
  Acceptable: a loss is a loss.

What we **did not** confirm:
- Hardcoded bid=30 threshold. `decl_id` (0-9) encodes trump only, not
  bid amount. The `[:, :, 60:]` (offense) and `[:, :, 25:]` (defense)
  bin offsets are literally P(Q ≥ 18) and P(Q ≥ -17), which are
  correct only for bid=30. For bid=36 the offense threshold shifts to
  Q ≥ 30 (bin 72+). Teacher is systematically overconfident on
  higher-bid hands. **Pre-Vast-launch blocker, tracked in GEN_FLEET.md.**

**The utility-vs-dynamics distinction.** The user's intuition that
"going for it wins more" is real — but it's a *dynamics* claim, not a
utility one. Margin at trick 3 compounds into safety at trick 7
through trump-depletion, partner signaling, and buffer against
unexpected opponent plays. A single-decision utility has no way to see
this — it's a property of sequences. The answer is multi-step planning
(LAMIR), not a richer utility.

Decomposition that stays clean:
- **Utility**: `U = p_make`. Pure threshold. Trivial to compute,
  trivial to explain.
- **Dynamics**: LAMIR rollouts re-score each action by rolled-out
  future p_make. Naturally values "margin that buys safety later."
- **These stack cleanly** — LAMIR uses the utility at the rollout
  leaves and aggregates via expectation. No need to corrupt the
  utility to imitate what planning provides.

**Earlier wrong turn worth remembering**: proposed `U = E[Q] + C·p_make`
as a "cliff-preserving, margin-rewarding" utility. Fails on the user's
own test case: A(p_make=1e-8, E[Q]=-42) vs B(p_make=0, E[Q]=+16) picks
B — the guaranteed loss — because E[Q] dominates when p_make×100 is
small. Real preference is pure p_make; margin belongs in dynamics.

**Implication**: no oracle-side utility rewrite needed. The cliff
behavior is correct as-is. The action items on the oracle side are
(1) bid_value plumbing for the threshold, (2) per-seat oracle softmax
for π_opp training targets. Neither touches the utility shape.

## 18. qMAE is data-scaling resistant — the laggard head needs a training-recipe fix

With 3.3× more data (3k → 10k games, same architecture, matched hparams),
most metrics compounded cleanly. qMAE did not:

| metric                       | 3k     | 10k    | relative improvement |
|------------------------------|-------:|-------:|---------------------:|
| regret (Q-pts)               | 1.35   | 0.55   | **−59%**             |
| bot-match (pi_me)            | 66.4%  | 76.1%  | **+9.7 pp**          |
| V-MAE                        |  4.5   |  3.6   | **−20%**             |
| **qMAE**                     | **9.4**| **8.7**| **−7%**              |
| belief top-1                 | ~38%   | ~39%   | roughly flat         |

qMAE reduction was an order of magnitude smaller than regret reduction.
v3's consistency loss helped marginally (8.25–8.69 range vs v2's 8.72) —
exactly the V/π decoupling it was designed to fix — but the magnitude is
faint relative to the structural problem.

**Why qMAE is structurally harder than V-MAE** (see also §2):

- Q_head predicts **per-world** expected Q. The target is noisy because
  it depends on *which world* was sampled, plus downstream play
  uncertainty within that world. V_head predicts the **marginal** E[Q]
  across worlds — a much smoother target.
- Current training pipeline samples **one random world per forward
  pass** (see `JointWorldFullDataset.__getitem__` / _build_item). The
  gradient is noisy because each forward sees a different world per
  decision, and Q_head's cross-world consistency is never explicitly
  rewarded.

**V-MAE of 3.6 is near the Bayes floor** for its target (marginal E[Q]
at visible state). **qMAE of 8.7 has real headroom**.

**Known fixes, not yet shipped** (either would be ~1 day of work):

1. **Multi-world variance regularization.** During training, query
   Q_head on K worlds for each decision; penalize high cross-world
   variance on `Q[action]` for the action_taken. Forces Q to be
   smoother — predicts the conditional mean with less world-dependence.
2. **Joint co-training of {belief, world_encoder, Q_head}** with a
   distribution-belief loss (§15 was the isolated ablation — it didn't
   help because Q_head wasn't co-trained; this is the integrated
   follow-up).

Either fix is a training-recipe change, not architecture or data.

**Why this doesn't block LAMIR-1:**

LAMIR-1's rollout uses V_head at the leaf (MAE 3.6, near floor).
Q_head is currently used for PIMC-via-world-averaging, which we
already know underperforms π_me direct (§3). LAMIR-1 doesn't invoke
Q_head at all. So the qMAE plateau is a separable concern from the
north-star roadmap — tackle it when/if the Q_head becomes critical
(router with inference-only features, or richer rollout-leaf models).

## 19. Interpretability probes — Gus has internalized real 42 strategy

Six probes on v3-10k against a deliberately hard eval position (seed
900000, declaration blanks, P0 leading). This hand is "straight
garbage" in the colloquial sense — P0 is the bidder holding one low
trump (3-0) against opponents with 5 of 7 trumps including the boss
(0-0). In real play you'd never bid this; the corpus forces the bid,
so it's a stress test. The key meta-finding: even on this nightmare
hand, Gus matches the ground-truth oracle within 0.5 Q-pts on V.

### Probe 1 — domino embedding similarity

Extracted `tok_emb.weight[:28]` (the per-domino identity embedding)
from v3-10k, computed pairwise cosine similarity.

- **Doubles cluster**: double↔double avg cos = +0.047 vs
  double↔non-double = −0.022 → Gus learned "doubleness"
- **Counts cluster**: count↔count = +0.037 vs count↔non-count = −0.021
  → Gus learned "countness" (the 5-pip-sum or 10-pip-sum rule)
- **Pip-family tightening by magnitude**: pip=0 intra-family avg cos =
  +0.007, pip=6 = +0.042 → higher-pip dominoes cluster more strongly,
  consistent with "they matter more so get more distinct representation"

Magnitudes are small (top neighbors ~+0.17) but signs and orderings
are clear. Categorical concepts (doubleness, countness, high-pip) live
in the raw embedding. **Relational concepts ("6-6 protects 6-4") do
NOT live here** — they're contextual, encoded downstream in attention.

### Probe 2 — attention pattern evolution

Captured per-head attention weights across all 6 encoder layers on
game 0 decision 0. For the CLS token (pooled state), attention
targets over layers:

| layer | dominant target | interpretation |
|---|---|---|
| 0 | DECL (weight 0.65) | anchor on trump declaration |
| 1-2 | MINE[6] = 6-4 (~0.22) | survey the big non-trump, see its vulnerability |
| 3-4 | shift toward MINE[0] = 1-1 (~0.17) | reconsider toward safer lead |
| 5 | MINE[0] = 1-1 (0.26) | concentrate on the chosen action |

π_me's output: 0.91 probability on 1-1 — matching last-layer CLS
attention concentration. **This looks like genuine multi-step
reasoning**, not one-shot argmax: declaration context → survey → risk
assessment → commit.

### Probe 3 — counterfactual hand swaps (leave-one-out)

For P0's decision 0, edited MINE tokens one at a time (swapping in
different dominoes), measured ΔV. Two headline findings:

- **Swap 1-1 → 0-0** (top trump of blanks): V jumps +11.66 Q-pts. Gus
  knows 0-0 dominates blanks.
- **Swap 1-1 → 6-6** (intuition-check from user — 6-6 looks like an
  upgrade): V DROPS 5.34 Q-pts. Initially read as suspicious.

V_head does not consume world_assignment (checked by source
inspection), so the swap is in-distribution for V. This is a real
model claim, not an input-validity artifact.

### Probe 4 — oracle ground-truth verification

Ran the actual 3.3M-param oracle (`domino-qval-large-3.3M.ckpt`) on
both counterfactual deals, adaptive SEM<0.5, 18.9s on MPS. Full
per-action E[Q] extracted for both.

|                            | original (1-1) | counterfactual (6-6) | Δ     |
|----------------------------|---------------:|---------------------:|------:|
| Gus V_head                 |         −11.54 |               −16.89 | −5.34 |
| Oracle best E[Q]           |         −11.07 |               −16.91 | −5.83 |

**Agreement within 0.5 Q-pts.** The counterfactual IS worse. Gus is
not biased — the oracle also says swapping in 6-6 hurts P0.

And the oracle's reasoning becomes legible: in the counterfactual, it
picks **5-2 as the best lead** (E[Q] = −16.91) and ranks **6-6 as the
WORST legal lead** (E[Q] = −24.41). The hand is in defensive territory;
the oracle plays low/safe; 6-6 is a trap you lead when you shouldn't.

**Game-theoretic explanation of the negative ΔV**: the swap is
bilateral. P0 gains 6-6 (redundant — 6-suit already secure with
6-1/6-3/6-4 against only 2 non-trump 6s outside). The opponent (seat 3
in this deal) gains 1-1 — the TOP of the 1-suit, which includes the
count dominoes 5-1 and 4-1 in that seat's hand. The opponent's gain
exceeds P0's gain by ~6 Q-pts. Matches the observed delta exactly.

**Important correction**: I initially attributed this to "strategy
fusion" in the PIMC training. The user correctly pushed back — with
adaptive sampling to SEM<0.5 we have thousands of world samples and
aggregation is tight. Strategy fusion in its classic form doesn't
apply. The real cause is the bilateral-swap asymmetry above, plus a
depth-vs-breadth saturation effect: concentrating high cards in one
suit has diminishing returns because opponents void out quickly and
trump your subsequent leads in that suit.

### Probe 5 — per-domino impact atlas for 6-6

Across 20 eval games × bilateral swaps = 238 measurements:

| declaration | n | mean ΔV | \|ΔV\|_mean | verdict |
|---|---:|---:|---:|---|
| sixes (trump) | 14 | +17.72 | 17.72 | always helpful |
| doubles (doubles are trump) | 42 | +21.99 | 22.09 | always helpful (41/42) |
| follow-me-8 | 28 | +16.96 | 16.96 | always helpful |
| fives | 28 | +4.32 | 7.94 | mostly helpful; ONE catastrophe |
| threes | 28 | +2.78 | 5.63 | mixed |
| twos | 28 | +3.06 | 6.68 | mixed |
| ones | 14 | +2.73 | 6.76 | mixed |
| blanks | 14 | +2.53 | 3.44 | modestly helpful |
| fours | 28 | −0.09 | 4.52 | essentially neutral |

The catastrophes on fives/fours/ones/twos/threes are all the same
pattern: swapping 6-6 IN by displacing the declaration's trump boss
(5-5, 4-4, 1-1, 2-2, 3-3). Worst case: fives decl, swap 6-6→5-5,
ΔV = −27.67.

**The takeaway**: 6-6's value to Gus is entirely contextual on
trumpness of the declaration and which card it displaces.
**Gus has not learned "big card = good"** — it has learned the
context-dependent value function that 42 actually has.

### Probe 6 — hand-level threats and boons

Using the stored joint-world tensor (4000 sampled worlds with per-world
Q values) we grouped Q by where each non-P0 domino lives in each world.
No forward passes needed — pure conditioning on the oracle's own
training data. For game 0 decision 0 (leading 1-1), baseline E[Q] = −11.15.

**Top BOONS** (largest uplift when partner holds the domino):

| domino | E[Q\|partner] | uplift vs baseline |
|---|---:|---:|
| 0-0 (top trump) | +5.44 | **+16.6** |
| 6-0 (high trump) | +1.33 | +12.5 |
| 5-0 (mid-high trump) | −0.96 | +10.2 |
| 4-0 (mid trump) | −6.03 | +5.1 |
| 2-0 (low trump) | −6.79 | +4.4 |

**Top THREATS** (largest drop when specific opp holds the domino):

| threat | Q if held there | drop vs baseline |
|---|---:|---:|
| R-opp holds 0-0 | −20.52 | **−9.4** — "if R-opp has the boss we're sunk" |
| L-opp holds 0-0 | −17.58 | −6.4 |
| L-opp holds 6-0 | −17.55 | −6.4 |
| R-opp holds 6-0 | −17.31 | −6.2 |
| L-opp holds 5-0 | −16.35 | −5.2 |

**All five threats AND all five boons are trumps.** The hand's outcome
is determined by trump distribution before any card is played. 0-0
alone contributes a 26-Q-pt swing based on location. This is the
right answer to the user's original question "what dominoes are most
involved in outcomes, for good or bad" — for this specific hand,
it's the five trumps, and specifically the interaction between trump
location and the game's defensive posture.

**Subtle note**: dominoes like 2-1, 5-1, 3-1 show high-magnitude
NEGATIVE bias_us (E[Q|partner] ~−18 vs E[Q|opp] ~−7). This is
**not** a strategic threat — it's a correlation artifact. If partner
is randomly assigned a weak card, partner's other 6 cards are drawn
from a slightly stronger residual pool; conditioning flips this to
"opp has 2-1 = opp team weaker on average." Real strategic threats
are only the positive-bias trumps.

### Meta-conclusions

1. **Gus has internalized categorical game features** (doubles,
   counts, high-pip magnitude) in the raw embedding, and
   **contextual/relational knowledge** (declaration-dependent value,
   trump-vs-non-trump, catching relationships) in the transformer
   layers.
2. **Gus's V agrees with the ground-truth oracle** even on a
   nightmare-grade losing hand (ΔV error 0.49 Q-pts on a 5.83-point
   delta). The distillation captures the oracle's full strategic
   understanding, not just its argmax.
3. **Counterfactual sensitivity is a usable interpretability tool.**
   Per-domino impact atlas and per-hand threat/boon ranking can be
   generated cheaply (seconds on CPU for forward passes; pure lookup
   from the joint-world tensor for conditioning-on-location queries).
4. **Game-theoretic facts we surfaced experimentally**: depth-vs-breadth
   saturation in non-trump suits; 6-6's trumpness-gated value; offensive
   cards as liabilities in defensive positions; the game being mostly
   decided by trump distribution in bad-hand states.

Probe scripts and output logs live in `scratch/probe_*` for reference.
These are candidates for promotion to `gus/eval/` if the atlas or
threat-boon analysis becomes a recurring tool (e.g., pre-game hand
evaluator, post-hand "what if" explorer).

## 20. LAMIR-1 attempt: depth-1 look-ahead does not beat direct π_me with distilled heads

Full LAMIR-1 harness built (`gus/eval/lamir1.py`), bug-hunted, and run. Eight
inference variants evaluated on 560 held-out decisions. None beat the 0.551
direct π_me baseline. Best look-ahead result: q-bootstrap at 0.679 (+23%
regret, no full rollout). Full rollout variants range from 1.645 (v-bootstrap)
to 2.350 (lamir1-piopp + Fix 6) — all substantially worse than direct.

**Two bugs confirmed and fixed:**
- *Bug 5* (`e4e6862`): opp tokens built from real deal hands instead of
  world-hypothetical hands. Clean fix: `_world_game_hands()` substitutes
  per-world opp hands before every token build.
- *Sign-flip*: V_head output is in leaf player's team frame; negate when
  leaf player is on opp team. Fixed pos=2 regret from 1.692→0.868 in isolation.
- *Fix 6 attempted* (`2c380a6`): zeroing played-domino rows from
  world_assign before the leaf Q_head call. Regret ticked UP across all
  modes. Reason: the training convention preserves the original
  world_assignment as the played_mask advances. Our zeroing broke an
  invariant the model relied on.

**Root cause of the ceiling**: Kubíček & Lisý explicitly warn that a value
function trained by distillation (like our V_head) cannot be used for
look-ahead reasoning. The scalar noise of the distilled V/Q is enough to
flip argmax at decision boundaries, while π_me trained on argmax directly
preserves ordering. The paper's fix is the T×T multi-valued-states matrix
value function — months of work.

**Side product**: π_opp head at 68.57% oracle top-1 accuracy. Even at this
quality, using π_opp as the rollout model doesn't help — confirming the
bottleneck is the leaf evaluator, not the opp simulation quality.

**V_head is architecturally world-blind**: std=0.000 across 200 world samples
for the same decision. Cannot differentiate world hypotheses.

## 21. Belief is at the Bayes ceiling — top-1 is a solved axis

Ran a diagnostic (`gus/eval/belief_ceiling.py`) that computes the
theoretical best top-1 achievable on the eval corpus: for each unseen
domino, take `argmax_seat P(seat | oracle sampled worlds)`. Since the
oracle's samples ARE the posterior, this argmax IS Bayes-optimal.

**Result on `corpus_eval_20.pt` (20 games, 560 decisions, ~2914 worlds/decision)**:

```
Bayes-optimal top-1:  39.184%
Gus v3 belief head:   ~38-39% (receipts §6, probe data)
```

**Gus is at ceiling.** Supervised-to-truth belief training produced a model
that matches the Bayes-optimal limit of the information available on this
corpus. Zeb's 39% wasn't a plateau — it was the information ceiling.

### Per-decision-idx: where the information lives

| d_idx range | Bayes top-1 | interpretation |
|---|---:|---|
| 0-5 (pre-first-trick) | ~33% | pure prior — 1-in-3 over 3 opp seats, no play info yet |
| 6-15 (mid-hand) | ~36-40% | voids + lead signaling start to sharpen |
| 18-25 (endgame) | ~50-75% | many played dominoes narrow the field |
| 26 (penultimate) | 100% | only one domino unseen |

Early-hand belief is essentially "I see my hand, so each unseen domino is
1-in-3." No architecture can exceed that — the information isn't available.
Late-hand accuracy rises sharply because each play rules out possibilities
deterministically (follow-suit → void inference → direct exclusion).

### Implication: pivot from accuracy to distribution shape

**Top-1 accuracy is a dead lever**. Any belief head that exceeds ~39.2% on
this corpus is overfitting; any lower is underfitting. Gus is tuned.

**The real unfinished work is posterior shape (calibration)**. §15 already
showed that distribution-target training lowered KL-vs-truth 0.078 → 0.062
(closed 47% of the gap from uniform prior to perfect belief). That was the
right direction but stalled: downstream play didn't improve because the
other heads weren't co-trained.

**The experiment that closes §15's open loop**: train belief + world_encoder
+ Q_head **jointly** with the distribution-belief target, rather than
stacking a belief-calibration fix on a frozen ecosystem. This is the single
most informative belief experiment we could run — tests whether a
better-calibrated belief translates to better look-ahead value estimates
when the consuming heads are trained to use its shape.

### Why this matters for look-ahead

PIMC / BMCS / LAMIR all consume the full `P(seat | domino)` distribution,
not argmax. Gus's belief might be right at the mode but miscalibrated in
the tails — and tail mass is what determines how much weight rollouts put
on unusual worlds. Better tail calibration → better-weighted world samples
→ better look-ahead value aggregation, independent of any leaf-evaluator
fix.

### What this unlocks (or doesn't)

- **Top-1-chasing architectures** (per-opponent memory tokens, longer
  context, auxiliary losses) are lower-priority. They might help marginally
  on late-hand voids-plus-signaling but you're already harvesting most of
  the signal. Diminishing returns.
- **LAMIR's "no explicit belief" elegance** is more tempting than it first
  sounded. If most of the belief information is captured by a small number
  of discrete strategic buckets (abstract infosets), collapsing to bucket
  membership loses very little. One training objective replaces three.
- **Corpus-specific finding**: 39.2% is the ceiling *on this corpus*.
  Schema v2 / diverse-seed corpora may have slightly different ceilings
  because of the sampling budget. Worth re-running the diagnostic on any
  new eval corpus before claiming "belief works / doesn't work."

**Side benefit**: this diagnostic (~60 lines) now exists as a permanent
sanity check. Before any future belief architecture claim, run the ceiling
first — the "is belief broken?" question reduces to "is it at ceiling?"

### Follow-up experiment — joint co-train falsified, but something else found

Ran the joint co-train experiment that §21 proposed
(`gus/train/train_belief_q_joint.py`, 15 epochs, 1000-game corpus, frozen
trunk + π_me + V_head, unfrozen belief + world_encoder + Q_head, α=1 β=1).
Result: `gus/adapters/v3_belief_q_joint.pt`.

**Belief KL dropped 20%**: 0.0840 → 0.0672 (epoch 11). Top-1 stayed at
~38% (ceiling). So the calibration fix itself worked — the target changed
from truth-mode to distribution-shape and the head followed.

**But the downstream regret did NOT improve — it got slightly worse.**
On 560 held-out decisions:

| adapter | q-bootstrap (corpus worlds) | q-bootstrap-belief (belief-sampled) |
|---|---:|---:|
| original v3_consistency_10000g | 0.685 | **0.655** |
| joint-co-trained | 0.718 | 0.679 |

Both paths got slightly worse with joint training. **The hypothesis that
"co-training heads lets calibration propagate" is falsified on our setup.**

Diagnosis: Q_head was trained with the ORIGINAL belief-head's output
distribution as implicit context. Jointly retraining everything shifts the
state_emb → Q_head mapping. The calibration win doesn't compensate; we
just moved Q_head off its sweet spot. This is worth remembering: in a
distillation pipeline, "better belief" isn't a strictly additive
improvement when downstream heads were already trained against the old
belief's shape.

### The unexpected win — belief-sampled worlds beat corpus worlds

Introducing `q-bootstrap-belief` (worlds sampled from the belief head at
inference rather than read from the oracle's saved world_hands) as a
required A/B for the co-train experiment **surfaced an independent
finding**: on the ORIGINAL adapter (no co-train), belief-sampled worlds
give regret 0.655 vs corpus worlds 0.685 — a **4.4% relative
improvement** just from changing the world source.

This is the closest any look-ahead variant has come to the 0.551 direct
baseline (gap 0.104 Q-pts, ~19%).

Likely mechanism (unverified): oracle adaptive sampling runs to SEM<0.5,
which can overconcentrate probability on a few high-posterior worlds at
near-consensus decisions. Belief-head softmax sampling is smoother and
more aligned with the world-distribution Q_head saw during training.
Either way, **sampling from the learned belief head is apparently a
better world source for look-ahead than the oracle corpus** — an actual
usable technique for any future PIMC/BMCS/LAMIR work.

**Artifact**: `gus/eval/lamir1.py --mode q-bootstrap-belief`. Reuses
`gus/model/sample_worlds.py` (the file symmetry-checker wrote off-plan
early in the overnight session — it turned out to be exactly what this
test needed).

**Next experiment (not run)**: vary K (number of sampled worlds). If
K→∞ gives the asymptote, we know whether the sampling diversity matters
or just the sampling distribution. At K=200 we already match or beat
corpus-at-M~3000. That suggests it's the distribution that's better, not
the diversity.

## 22. "What you do past belief" — the heart of 42 is already in our data

Framing credit: the user, on recognizing that belief is at ceiling
(§21). The §21 finding reframed the problem: Gus's belief head is
correct; the remaining 42-craft isn't about *knowing better*, it's
about *acting well given unresolvable uncertainty*. Human experts call
this "reading the table," "signaling," "playing the percentages." The
AI literature has almost no vocabulary for it.

### The extractable analytics (not yet run)

Every decision in the corpus has a `q_per_world` tensor: [M, 7] of
oracle Q values, one per sampled world × legal action. This tensor
already tells us *where the strategic uncertainty lives*. Three derived
quantities, all computable with pure indexing (no new model training):

1. **Outcome-variance per decision**: `q_per_world[:, action_taken].std()`
   Tells you how much the outcome of the chosen action depends on which
   hidden world is true. Low → "this play is robust, all worlds look
   similar." High → "knife's edge; hand-shape decides." The high-variance
   decisions are the ones worth writing about.

2. **Action-choice fragility**: for each decision, compute the per-world
   argmax action (`q_per_world.argmax(dim=-1)`, shape [M]). Count how
   many DIFFERENT actions the oracle would pick across the M worlds.
   If it's 1, the decision is cleanly determined. If it's 3+, the
   decision is a fog-of-war call — the optimal play depends on which
   world you're in, and you can't know.

3. **Belief-limited high-impact decisions**: join (1) and (2) with
   §21's per-decision belief-sharpness. Decisions where belief is ~33%
   (pure prior, no signal) AND action-choice fragility is high (3+ argmax
   actions) AND outcome-variance is high are the **defining moments of
   42** — everything is on the line AND you genuinely don't know. These
   are the situations a book would be built around.

Estimated effort to extract: **~1 afternoon**. Uses existing corpus,
existing oracle data. Output: a ranked list of (game, decision) pairs
sorted by "drama" (all three quantities high), plus a per-decision-idx
distribution showing where in the trick these moments concentrate.
Trick-pos distribution alone tells us something real about 42 — is the
drama in the lead, the second-to-last play, the mid-hand strategic
choice?

### Beyond analytics — the research question

Gus's π_me is trained on `argmax(marginal E[Q])`. At belief-limited
decisions, the marginal E[Q] averages over worlds that favor
DIFFERENT actions. The marginal argmax is then *"play for the mode of
the action distribution,"* not *"play for any specific world."* This is
one specific meta-strategy among several that a real player might use:

- **Mode**: play the action that's best on average (what π_me learns)
- **Signal**: play an action whose choice *informs partner* about your
  hand shape, even if its own E[Q] is slightly worse
- **Hedge**: play an action that has the least downside across worlds,
  not the highest mean (minimax under belief)
- **Gamble**: play an action that's catastrophic in most worlds but
  wins big in a specific one (bid-dependent; "only way I make 42 is
  if partner has the 6-6")

None of these are distinguishable from π_me's single-output training.
A richer student could output *multiple meta-strategies* and let the
player (or a learned selector) pick one based on match context (are we
ahead or behind on score, is partner likely to punish a missed signal,
etc.). That's a real, paper-worthy direction that doesn't need LAMIR's
CFR+ machinery — it just needs the oracle's per-world data (which we
have), split across meta-strategy targets, and a small head that outputs
a distribution over meta-strategies given the state.

**This is what ("what you do past belief") names.** It's the gap
between "oracle-optimal action assuming future play is optimal" (π_me's
training target) and "human-optimal action given real partners who
read signals, real opponents who make mistakes, real score context." The
oracle doesn't care about any of that; a strong player does.

### Connection to §19 (probes)

§19 established that Gus has internalized categorical 42 concepts
(doubleness, countness, trump-dependent value) and contextual reasoning
(6-6 as liability in defensive positions). The natural extension of
that probe series is to look at Gus's behavior on the belief-limited
decisions surfaced by the analytics above — does Gus's π_me pick
robust actions or mode-of-distribution actions? Knowing which one
would tell us whether the model has implicitly learned a meta-strategy
choice or just follows "play for the mode."

### Status

Noted as an option for future work. Not blocking anything. No code
written. Could be next-session or next-month or never — it's in the
log now so it doesn't get lost.

## 23. Belief-sampled Q-mean is a viable fallback signal, not a replacement

Follow-up archaeology on the untracked `gus/eval/eval_lamir1.py` surfaced a
result that was easy to miss: the script's verdict only evaluates
`lamir1-argmax`, while the `lamir1-K` mode can beat direct π on regret.

On `v3_consistency_10000g.pt` over `corpus_eval_20.pt`:

| policy | mean regret | bot-match |
|---|---:|---:|
| direct π_me | 0.551 | 76.1% |
| Q-mean, K=100, seed 42 | 0.505 | 75.5% |
| Q-mean, K=200, corrected RNG, seed sweep mean | 0.506 | ~75.6% |
| Q-mean, K=500, corrected RNG, seed 42 | 0.500 | 75.7% |

The important slice is the tail:

| slice | n | direct regret | Q-mean K=200 regret |
|---|---:|---:|---:|
| direct blunders >=8 | 8 | 11.42 | 5.94 mean (5.28-8.55 range) |
| direct big misses >=4 | 29 | 7.11 | 3.87 mean (3.64-4.49 range) |

Pure Q-mean is not a clean replacement. It sometimes damages easy direct-π
wins. But it is a strong second opinion: on disagreements, when Q-mean is
right it often fixes a large mistake, and when it is wrong it usually breaks a
near-perfect direct pick.

The best exploratory shape was a tiny router:

```text
use direct π by default
if direct π disagrees with Q-mean and pi_peak < 0.5, use Q-mean
```

After fixing `sample_worlds()` so the caller's RNG state advances across
batches, the router still holds up across five sampled-world RNG seeds at
K=200: it routes only ~6.25% of decisions and lands at 0.424 mean regret
(range 0.405-0.460), versus 0.551 for direct π.

A train-to-eval learned router makes the result more interesting. Trained on
`corpus_train_100.pt` with target "direct blunder fixed by Q-mean", then
evaluated on `corpus_eval_20.pt` at K=100 across five sampled-world seeds:

| policy | regret | blunders/seed | big misses/seed | routed |
|---|---:|---:|---:|---:|
| direct π_me | 0.551 | 8.00 | 29.00 | - |
| pure Q-mean | 0.489 | 7.60 | 22.60 | - |
| learned route top 5% | 0.434 | 4.40 | 20.80 | 5.00% |
| learned route top 7% | 0.440 | 4.20 | 21.20 | 6.96% |

The 5% learned cutoff introduced zero new blunders across the five seeds. The
7% cutoff fixed four direct blunders per seed on average and introduced only
0.2 new blunders/seed.

There is also more headroom in the sampled-world family. Mean-Q is not the only
aggregator: quantiles, lower-confidence bounds, and per-world argmax votes
sometimes fix a different direct blunder. An oracle over direct π plus these
candidate aggregators gets to ~2.6 blunders/seed at K=100 and K=200, but the
first learned multi-candidate selector still bottoms out around 4 blunders.
So candidate generation is not exhausted; candidate selection is now the hard
part.

The hand gate is eval-tuned and the learned result still measures only 8 eval
blunders, so this needs larger validation before promotion. But it changes the
working conclusion:

> Direct π remains the fast path, but belief-sampled Q-mean is a useful
> fallback signal for uncertain direct-π decisions and materially cuts the
> blunder tail. No oracle fallback required for this first version.

Detailed writeup and reproduction notes:
`gus/analysis/qmean_router_findings.md`.
