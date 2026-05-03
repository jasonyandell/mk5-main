---
title: Gus (sibling project — neural player + belief + value)
kind: entity
first_seen: d858781
last_updated: local-2026-04-30
status: active
---

## What it is

Gus is a third sibling project alongside [[lem]] and [[burl]]: a multi-task transformer
that provides neural policy, belief, and value heads for [[texas-42]]. It **skips the
reasoning channel** — instead of rationalization or tool orchestration, it trains directly
against [[forge]]'s E[Q] oracle as a variance-free reward signal.
(commit messages @ 42a7535, 31e10ef)

## Why this shape (from the kickoff doc)

- **PG beats CFR** (ICLR 2025): policy-gradient distillation from a strong oracle outperforms
  counterfactual regret minimization at this game scale.
- **LAMIR** (Oct 2025): look-ahead through a belief-weighted tree using cached per-world Q
  values avoids re-querying the oracle at inference — the efficiency breakthrough that makes
  deep look-ahead cheap.
- Grounded also in Bridge AI (BMCS) and PRR-TM teammate modeling.

## Five-head LAMIR-ready architecture (gus/BUILD_PLAN.md @ 31e10ef)

All five heads share one state encoder (~256-dim MLP):

| Head | Input → Output | Target | Status |
|---|---|---|---|
| `belief_head` | `state → P(dom ∈ seat)` [84 out] | True deal (all hands) | v0+ |
| `V_head` | `state → scalar E[Q]` | Oracle mean over sampled worlds | v0+ |
| `π_me_head` | `state → action softmax` [7] | Oracle argmax E[Q] | v0+ |
| `world_encoder + Q_head` | `(state, world_layout) → Q` [7] | Per-world oracle Q | **v1 — LAMIR-critical** |
| `π_opp_head` | `(state, opp_seat) → action softmax` | Oracle argmax from opponent's seat | **v2 — deferred** |

**Why world-conditioned Q is LAMIR-critical**: a marginal student gives `E[Q | current_belief]`
but can't re-condition as beliefs update. A world-conditioned student caches `Q_m` for each
sampled world; after an opponent play, belief update is a cheap
`Σ_m w_m(new_belief) · Q_m` re-weight over the cache — no new oracle calls.
See [[joint-world-tensor]]. (gus/BUILD_PLAN.md @ 31e10ef)

## Training staging

- **v0**: belief + V + π_me heads. State features ~200-400 dims; total ~3-5M params (Zeb-class).
- **v1**: add world-conditioned Q head. Requires [[joint-world-tensor]] corpus.
- **v2**: add π_opp head. Requires corpus re-gen with opponent-view oracle queries.
- (Later revisions introduce v3 with consistency regularizer — see later ingests.)

Data: seeds 0–99 for training, seeds 900000–900099 for eval (same convention as LEM and Burl).

## Product positioning

Gus is the **fast player**. [[lem]] is the downstream commentary/explanation head — once
Gus's player works, LEM can be a low-cost interpretation layer on top. [[burl]] takes a
different path (tool orchestration + reasoning). All three siblings share [[forge]]
infrastructure (solver, E[Q] framework, engine, [[zeb]]).

## Current frontier: explicit strategy tags probe (2026-04-30)

The Winning 42 strategy-book harvest produced a new Gus experiment: can cheap,
human-legible public-state/action tags help a tiny policy student over raw public
state alone?

Result: yes, materially, though not enough to beat `E[Q] N=10`. On 28k early-decision
examples, a tiny `tokens + voids` probe scored **2.012 regret**, while the same probe
with 68 global strategy features plus 7×32 action-local features scored **1.181 regret**.
`E[Q] N=10` remained far ahead at **0.167 regret** on the same 560-decision eval slice.

The experiment is now promoted out of scratch:

- `gus/model/strategy_features.py`
- `gus/eval/strategy_probe.py`
- [[experiments/gus-strategy-tags-probe]]

## Progress: v0 → v1 → v2 scaffolding (2026-04-20/21, commits c04bda3–3c02d10)

### v0 — MLP belief head (c04bda3)

Sanity check: `StateEncoder` (MLP) + `BeliefHead` on 183-dim flat features (decl + player +
decision_idx + 6×28 domino masks). ~200K params. 100g corpus: peak eval **34.6%** (chance
33.3%), severe overfit (train → 100%). Architecture is correct; data is the bottleneck.
(commit message @ c04bda3)

### v1 — Transformer belief student (8dbf7f3)

Replaces MLP with a transformer over tokenized play sequences: CLS + DECL + 7 MINE + 24
PLAY = 33 tokens, five channels per position (token/type/trick/pos/player_rel). Enables
attention-based void inference. 100g corpus:

| Decision idx | Unseen dominoes | Belief top-1 |
|---|---|---|
| 0 (first play) | 21 | 33% (chance floor) |
| 10 (mid-game) | 13 | 42% |
| 22 (late-game) | 4 | 45% |
| 26 (last play) | 1 | **75%** |

Overall peak eval **37.5%**. Late-game belief IS learning; the floor at decision 0
mathematically caps the aggregate. Ceiling is data, not architecture. (commit message @ 8dbf7f3)

### Full 4-head student — belief + V + π_me + Q (da21f52)

All four heads on shared transformer encoder. `WorldEncoder` fuses a [28,3] seat-one-hot
layout into `state_emb` for the Q head. 100g corpus, 20 epochs, MPS:

| Metric | Value |
|---|---|
| π_me bot-match (held-out) | **57.9%** (chance ~25%) |
| Belief top-1 (held-out) | 34.5% |
| V MAE | 11.9 (on [-42,+42] scale) |
| Q MAE (legal actions) | 18.4 |

Train and eval π_me track within 1–2 pts. **Dense Q supervision (3400× signal per decision)
regularizes the encoder** — no overfit, unlike belief-alone v0. Per-decision π_me:
decision 0 = 20% (chance), decision 13 = 60%, decisions 24-27 = 100%.
(commit message @ da21f52)

### 1000g scaling + v2 void features (2e4f586, 3c02d10)

Glob-expand fix (2e4f586) allows chunked corpus paths so `n_games=1000` fits within MPS
INT_MAX limits. `VoidsEncoder` (v2, 3c02d10) projects a [24]-dim (3 opps × 8 suits) void
indicator into `d_model` and adds to pooled `state_emb` before heads. 1000g corpus, d=192,
4 layers, 40 epochs:

| Metric | v1 | v2 (+ explicit voids) |
|---|---|---|
| π_me | 66.1% | 65.2% (~flat) |
| Belief | 37.2% | **38.6%** (+1.4pp) |
| V MAE | 7.6 | 7.7 (flat) |
| Q MAE | 12.3 | 12.2 (flat) |

Explicit voids are marginal — the transformer already inferred them attentionally from play
tokens. Real next lever: more data + bigger model. (commit messages @ 2e4f586, 3c02d10)

### Scaling ladder + eval metrics (2026-04-21, commits 5a4c9b9–fdcd654)

**Full adapter ladder** (560 held-out decisions):

| Corpus | Params | Bot-match | Regret (Q-pts) | Near-ties |
|---|---|---|---|---|
| 100g | 0.4M | 59.3% | 2.48 | 68.9% |
| 1000g | 1.2M | 65.4% | 2.16 | 73.4% |
| 1000g | 3.4M | 62.7% | 2.10 | 70.5% |
| **2000g** | **3.4M** | **67.3%** | **1.60** | **75.4%** |
| 2000g | 7.4M | 65.7% | 1.65 | 75.2% |
| 2000g | 3.4M (120ep) | 66.8% | 1.64 | — |

**Best adapter at this frontier: `v2_voids_big_2000g`** (d=256, 6 layers, 3.4M params,
60 epochs). Mean regret 1.60 Q-pts on ±42 scale = ~1.9% of Q-range lost per decision.
75% of bot-mismatches are near-tie alternative choices, not strategic blunders. End-game
(decisions 24-27): 100% match, 0 regret. 120-epoch run confirms ceiling at same
corpus/architecture. (commit messages @ 0472125, 5cdec8a, fdcd654)

**Scaling lessons:**
1. Data dominates capacity — 100g→1000g→2000g shows clean regret reduction; 3.4M→7.4M
   on same data is flat.
2. Explicit voids marginal once data and model compound — transformer infers voids
   attentionally from play tokens.
3. 60 epochs vs 120 epochs on same corpus/arch: tied. Ceiling confirmed.

**Inference modes** (5a4c9b9, 560 decisions on 1000g):

| Mode | Bot-match |
|---|---|
| Direct π_me argmax | 65.4% |
| PIMC-Q (Q_head on 1 oracle-sampled world) | 62.1% |
| PIMC-belief (Q_head on 50 belief-sampled worlds) | 61.8% |

Direct π_me wins because it IS the marginalized policy — trained on oracle's argmax(E[Q])
which already averages over thousands of consistent worlds. Adding K student-sampled worlds
introduces variance without new single-step information. LAMIR's value is **multi-step**
look-ahead with mid-tree belief updates, which requires a π_opp head (not yet trained).
See [[topics/pimc]] and [[topics/lamir1]]. (commit message @ 5a4c9b9)

**Regret metric** (2a09050): `regret = oracle_best_eq − student_chosen_eq`. Better than
bot-match for a game full of near-ties. See [[topics/regret-eval]]. (commit message @ 2a09050)

**Decision-hardness analyzer** (a50c9ef): high-regret decisions correlate with high oracle
E[Q] spread (max − min over legal actions). Student makes honest mistakes on strategically
consequential positions (decision 0 spread = 13.2 Q-pts, student regret = 4.0 vs random
baseline 6.6), not on easy near-ties. (commit message @ a50c9ef)

**Open directions at this frontier**: π_opp head (needs corpus regen with opponent-view
oracle Q), multi-step LAMIR tree search (needs π_opp), 5000g–10000g scaling.

### 3000g + arena + v3 consistency (2026-04-21, commits 286eb23–b007cf3)

**3000g NEW BEST** (286eb23): same 3.4M architecture, 60 epochs.

**Regret distribution** (from PRACTICALITIES receipt #5): bimodal — 73% of decisions are
perfect, 6% blunder tail drives the 1.39 mean. Most "bot-mismatches" are near-tie
alternatives, not strategic errors. (commit message @ f0139a3)

**Arena game-level eval** (1a2f67f): full-game simulation — student at seat 0, E[Q] bot at
other seats. 20-game pilot (seeds 900020-900021, 10 declarations each):
- Student: 10/20 contracts made (50%), avg bidder points 21.2
- All-bot baseline: 16/20 contracts made (80%), avg bidder points 29.6
- Gap: −8.4 points/hand, **~30pp contract-made drop from 1.39 Q-pt regret**

The game-level gap is larger than decision-level regret suggested — errors compound.
(commit message @ 1a2f67f)

**V/π decoupling** (surfaced by blunder forensics from 1a2f67f): V correctly predicts the
oracle's best legal E[Q] (e.g., +26) while π_me concentrates probability on a losing action
(e.g., worth −0.4). The two heads share an encoder but their training objectives don't
directly couple them. See [[topics/v-pi-decoupling]].

**v3 consistency regularizer** (b007cf3): addresses V/π decoupling via:

```
L_consistency = (V_head.detach() − Σ_legal softmax(π_me) · e_q)²
```

V is detached — gradient only reshapes π. Forces policy-expected-Q to match V's prediction,
concentrating π on actions that agree with V's assessment. Warmup from 0 → w_consistency
(default 0.3) over first 10 epochs so V converges before consistency pressure kicks in.
(commit message @ b007cf3)

**[[gen-fleet]]** (a8bc35a): Vast.ai distributed corpus generation plan. ~$15 for 8-worker
7.5-hour run covering 10k diverse-seed games. Prerequisite: `--n-decl-per-seed` flag
(current: 1 decl/seed vs oracle's 10 → 100× fewer distinct states per unit compute).
Pre-launch fix list landed at a14200f — see [[gen-fleet]] for all five blockers.
(commit message @ a8bc35a)

### 10k scale + v3 consistency verdict (2026-04-21, commits b4040c5–31f0ec3)

**Full adapter ladder** (560 held-out decisions):

| Adapter | Corpus | Bot-match | Regret (Q-pts) | Near-ties |
|---|---|---|---|---|
| v1_full | 100g | 59.3% | 2.48 | 68.9% |
| v1_full | 1000g | 65.4% | 2.16 | 73.4% |
| v2_voids_big | 1000g | 62.7% | 2.10 | 70.5% |
| v2_voids_big | 2000g | 67.3% | 1.60 | 75.4% |
| v2_voids_big | 3000g | 67.9% | 1.39 | 77.3% |
| v3_consistency | 3000g | 66.4% | 1.346 | 77.5% |
| v2_voids_big | **10000g** | 73.21% | 0.818 | 81.8% |
| **v3_consistency** | **10000g** | **76.07%** | **0.551** | **85.5%** |

**v3_consistency_10000g is the new best**: regret 0.551 Q-pts — first sub-1.0 regret in
Gus, below the teacher-noise floor for the first time. (commit messages @ b4040c5, 31f0ec3)

**Regret decomposition** (v2_voids_3000g → v3_consistency_10000g):
- Data scaling alone (v2: 3000g → 10000g): **−41%** (1.391 → 0.818)
- Consistency loss at 10k (v2 → v3 at same corpus): **additional −33%** (0.818 → 0.551)
- Stacked total: **−60%** regret (1.391 → 0.551)

At 3000g, v3 vs v2 was a wash (1.346 vs 1.391 — within noise). At 10000g, v3 pulls
decisively ahead. **The consistency regularizer scales better than plain distillation** —
V_head becomes a more trusted anchor as data grows, making the "force π to pick actions V
endorses" pressure more effective. Decision: v3 consistency rides forward into LAMIR and
schema v2 re-gen. (commit message @ 31f0ec3)

**qMAE scaling plateau** (PRACTICALITIES receipt 18, commit 41fdb3c): with 3.3× more data
(3000g → 10000g), most metrics compounded cleanly but qMAE did not:

| Metric | 3k | 10k | Relative Δ |
|---|---:|---:|---:|
| Regret (Q-pts) | 1.35 | 0.55 | −59% |
| Bot-match | 66.4% | 76.1% | +9.7pp |
| V-MAE | 4.5 | 3.6 | −20% |
| **qMAE** | **9.4** | **8.7** | **−7%** |

Root cause: Q_head trains on **one random world per forward pass** — the gradient is noisy
because no cross-world consistency is ever explicitly rewarded. V_head predicts the smooth
marginal E[Q]; qMAE predicts a noisy per-world target with intrinsic variance.

**Known fixes, not yet shipped**:
1. Multi-world variance regularization: query Q_head on K worlds per decision, penalize
   cross-world variance on the taken action.
2. Joint co-training of `{belief_head, world_encoder, Q_head}` with a distribution-belief
   target (standalone calibration failed because Q_head wasn't co-trained — §15 of PRACTICALITIES).

**This does not block LAMIR-1**: LAMIR-1's rollout uses V_head at the leaf (MAE 3.6, near
Bayes floor). Q_head is only needed for PIMC-via-world-averaging, which already underperforms
π_me direct. (commit message @ 41fdb3c)

### Blunder detector + detect-and-route (2026-04-21, commits f90682c–a09ef43)

**Ensemble analysis** (f90682c): 8 adapters under four strategies on 560 held-out decisions:

| Strategy | Bot-match | Regret |
|---|---|---|
| Best single (v2_voids_3000g_big) | 67.9% | 1.39 |
| Majority vote | 69.1% | 1.55 ← WORSE |
| Softmax avg | 71.3% | 1.46 ← WORSE |
| V-weighted | 71.3% | 1.43 ← WORSE |
| Oracle-per-decision (ceiling) | 89.6% | 0.36 |

Averaging boosts bot-match but hurts regret — confident-right adapter diluted by less-certain
ones on sharp decisions. Oracle-per-decision ceiling (0.36 regret, 74% reduction) confirms
large latent adapter diversity. **A router is the right shape, not an averaging ensemble.**
(commit message @ f90682c)

**Blunder detector v1** (f90682c) — oracle-feature GBM classifier (not deployable as-is):
- ROC-AUC 0.926; at 15% flag rate: 80% recall; at 25%: 99% recall
- Dominant features: `oracle_spread` and `oracle_eq_std` = 77% of importance
- Projected regret with oracle-argmax fallback at 20% flag: 1.23 → 0.46

**Blunder detector v2** (5373223) — student-feature only (deployable at inference):
- 28 features all derivable from student's own outputs (π_me entropy/peak/margin, V_head,
  Q_head stats across K=20 sampled worlds, consistency gaps, decision metadata, belief stats)
- ROC-AUC 0.839 (vs v1's 0.926); PR-AUC 0.15 (vs 0.29)
- At 20% flag rate with oracle-argmax fallback: regret 1.13 → 0.49 (57% reduction)
- Top feature: `pi_peak` (0.19) — when the student's policy is uncertain, trigger fallback
- Q_head spread underperforms as blunder proxy (Q trained on one world per forward pass —
  noisy). Fix: multi-world variance regularization during training, or K=50+ at inference.

(commit message @ 5373223)

**Detect-and-route wrapper** (eba5103): inference wrapper with three fallback policies.
Reality-check on 560 held-out decisions (baseline 1.39 regret):

| Flag % | Oracle fallback | PIMC-Q K=50 | Next-best adapter |
|---|---|---|---|
| 5% | 1.15 | 1.39 | 1.48 |
| 20% | 0.56 | 1.47 ← HURTS | 1.55 ← HURTS |
| 25% | **0.49** | 1.48 | 1.69 |

Findings (commit messages @ eba5103, a09ef43):
1. **Oracle routing works** — 0.49 regret at 25% flag matches projection.
2. **PIMC-Q-K50 HURTS**: fixes 7 blunders but introduces 6 new errors on non-blunder
   decisions the detector incorrectly flags. Q_head trained on one world is too noisy.
3. **Next-best-adapter is worst** — smaller adapters also wrong on the hard decisions where
   the primary needs help.

**Practical implication**: detect-and-route needs oracle calls to ship without a degradation
risk. The prerequisite for oracle-free deployment is a Q_head that survives multi-world
averaging (multi-world variance regularization during training, or K=50+ at inference).
Router benefit concentrates on mid-game decisions (dec 0-12); end-game (24-27) correctly
never flagged. (PRACTICALITIES receipts 11-14)

**Emerging architecture for Gus v1.0**: fast path (student π_me) + blunder detector gate +
oracle fallback on flagged decisions. Projected regret ~0.49 Q-pt — near the 0.5-1.0 Q-pt
teacher-noise floor for vanilla distillation.

### LAMIR-1 — depth-1 look-ahead harness + ceiling (2026-04-22, commits 581bf1f–b42669a)

LAMIR-1 (Look-Ahead Monte-Carlo with Info Re-weighting, 1-ply) is the first concrete
implementation of multi-step look-ahead for Gus. Key premise: query π_me from a
rotation-equivariant view as a free stand-in for π_opp (no separate π_opp head needed),
then score the resulting leaf state with V_head or Q_head averaged over M sampled worlds.
Implementation lives in `gus/eval/lamir1.py`. See also [[topics/lamir1]].

#### Six rollout modes

| Mode | What it does | Leaf evaluator |
|---|---|---|
| `direct` | Argmax over π_me_logits (baseline) | — |
| `v-bootstrap` | Depth-1 V_head immediately after candidate action, no opp rollout | V_head, world-blind |
| `lamir1` | Full 1-ply rollout: simulate remaining trick via rotated π_me as π_opp | V_head averaged over M worlds |
| `q-bootstrap` | Depth-1 Q_head — rotate world to next actor's POV, max over legal Q, sign-flip if opp team, average over M worlds | Q_head, world-conditioned |
| `lamir1-qleaf` | Same opp sim as `lamir1`, Q_head leaf from trick-winner's POV | Q_head, world-conditioned |
| `lamir1-piopp` | Same as `lamir1-qleaf` but opp steps use trained `PiOppHead` instead of rotated π_me | Q_head, world-conditioned |

#### pi_opp head (commits 93859a0, dcd9365, 1a1a324, b4e8ecd)

`PiOppHead` is a dedicated opponent policy head: frozen v3_consistency trunk + small head
with a 3-way seat embedding (L-opp / partner / R-opp relative to the decision player).
Loss: legal-masked cross-entropy against `oracle_softmax_per_seat[rel_seat]` from the
Schema v2 corpus. Each batch item generates 3 training pairs (one per opponent seat).
Trained on 1000-game v2 corpus in 20 epochs; 1,879 params; saved as
`gus/adapters/v3_10k_piopp.pt`. **Result: 68.57% oracle top-1 accuracy** — substantially
better than rotated π_me (~55%). (commit messages @ 93859a0, dcd9365, 1a1a324)

Schema v2 loader (`dcd9365`): `JointWorldFullDataset` and `JointWorldFullIterable` now
expose three additional keys when the corpus has v2 fields:
`oracle_softmax_per_seat [4,7]`, `legal_mask_per_seat [4,7]`, `voids_per_seat [4,24]`.
Backwards compatible — v1 corpora load unchanged.

**NaN loss bug** (b4e8ecd): illegal slot log_probs are −∞ after log_softmax masking. The
dot-product with the target distribution produced IEEE `0 × (−∞) = NaN`. Fix: zero
illegal slot log_probs before the dot-product.

#### Full 8-mode inference ladder (560 held-out decisions, MORNING4_STATUS / PRACTICALITIES §20)

| Mode | Regret | Bot-match |
|---|---:|---:|
| **direct π_me** | **0.551** | **76.07%** |
| q-bootstrap | 0.679 | 72.50% |
| v-bootstrap | 1.645 | 66.96% |
| lamir1-qleaf | 2.006 | 64.29% |
| lamir1 | 2.094 | 60.36% |
| lamir1-piopp | 2.268 | 62.10% |
| lamir1-piopp + Fix 6 | 2.350 | 62.50% |

**No look-ahead mode beats direct π_me.** q-bootstrap (depth-1, no rollout) is the best
look-ahead result at 0.679 (+23% regret). Full rollouts are uniformly worse — compounding
errors across 1-3 opp steps overwhelm any leaf signal. Damage concentrates at trick_pos
0-2; trick_pos 3 (last player) is unchanged. (commit messages @ 581bf1f, 7d2af99, 8106f01, b42669a)

#### Bugs found and fixed

**Bug 5** (e4e6862): opp token builds used the real deal hands rather than per-world
hypothetical hands. When π_me chose a slot against a world-specific hand, the slot→domino
lookup resolved to the wrong domino for worlds where the hand differed from reality. Fix:
`_world_game_hands()` precomputes per-world 4-player hands, passed into all
`_build_tokens_voids` calls inside `lamir1_decision`.

**Sign-flip bug** (566bc4d): V_head output is in the leaf current_player's team frame.
When the leaf player is on the opponent team relative to the original decision-player P,
the value must be negated before argmax. Fixed pos=2 regret from 1.692 → 0.868 in
isolation.

**Bug 6** (2c380a6): after 1-3 opp rollout plays, the `world_assign` tensor passed to the
leaf Q_head still reflected the pre-rollout hand layout — played dominoes were still marked
present in opp hands, giving Q_head stale beliefs. Fix: during the rollout loop, record
the domino ID played by each opp per world; after the loop, zero those rows in
`world_assign_leaf` before the Q_head call. Applied to both `lamir1_qleaf_decision` and
`lamir1_piopp_decision`. **Result: Fix 6 made every mode worse** — the training convention
preserves the original world_assignment as the played_mask advances; zeroing broke that
invariant. The "fix" was confirmed as a misread of the training convention.

#### Root cause of the ceiling

Kubíček & Lisý explicitly warn that a value function trained by distillation cannot be used
for look-ahead reasoning: scalar noise in the distilled V/Q is enough to flip argmax at
decision boundaries, while π_me trained on oracle argmax preserves ordering. The paper's
fix is the T×T multi-valued-states matrix value function — months of work.

V_head is **architecturally world-blind** (std=0.000 across 200 world samples for the same
decision — it takes only `state_emb` with no `world_encoder` input). Q_head is
world-conditioned but too noisy from single-world training. Neither is a reliable leaf
evaluator. The bottleneck is the leaf evaluator, not opp simulation quality: π_opp at
68.57% oracle accuracy also failed to improve rollout results.

#### LAMIR-1 pivot options (PRACTICALITIES §20, MORNING4_STATUS)

| Option | Effort | Notes |
|---|---|---|
| Accept depth-1 ceiling; ship q-bootstrap as second-opinion in router | Small | Use when π_me entropy is high |
| Train look-ahead-compatible V-head (V on expected Q under sampled opp play) | Medium | Closer to paper recipe without CFR+ |
| Implement LAMIR faithfully (multi-valued states + CFR+) | Large/research | Only if expert-human competition is explicit goal |
| Bridge-AI / BMCS recipe (PPO self-play using π_opp + Q_head as raw materials) | Medium | Gus doesn't have to be LAMIR |

**Path (a) post-session experiment — Q_head depletion augmentation**: fine-tuned Q_head
with random partial depletion (zero k random assigned rows, k ~ Uniform(1,3), p=0.5 per
item) to address OOD at rollout leaves. Result: q_mae improved 8.277 → 8.169 but
lamir1-qleaf regret got slightly worse (2.156 → 2.216). OOD augmentation fixes a
measurement artifact but not the fundamental ordering problem — the leaf evaluator needs
end-to-end rollout training, not just input-distribution augmentation. Path (a) closed.
(MORNING4_STATUS §Path a)

### Probe analysis — interpretability confirms real game structure (2026-04-21, commit 245918d)

Six probes on `v3_consistency_10000g` against a deliberately hard eval position (seed
900000, declaration blanks — user's description: "straight garbage," P0 has one low trump
against opponents holding 5 of 7 trumps including the boss 0-0). Despite the nightmare
hand, Gus's V_head matches the ground-truth oracle to within 0.5 Q-pts. (PRACTICALITIES
receipt 19)

**Probe 1 — domino embeddings**: doubles cluster (avg cos +0.047 vs non-doubles −0.022),
counts cluster (+0.037 vs −0.021), high-pip families are tighter. Categorical concepts
(doubleness, countness, magnitude) live in the raw embedding. Relational concepts
("6-6 protects 6-4") are contextual — encoded in the transformer layers, not the embeddings.

**Probe 2 — attention evolution**: CLS attention across 6 encoder layers follows a readable
reasoning chain: layer 0 anchors on the declaration (weight 0.65), layers 1-2 survey the
big non-trump 6-4, layers 3-4 reconsider toward 1-1, layer 5 concentrates on 1-1 at 0.26.
Final π_me probability on 1-1: **0.91**. Multi-step reasoning, not one-shot argmax lookup.

**Probe 3 — counterfactual hand swaps**: swapping 1-1 → 0-0 raises V +11.7 Q-pts (top trump
is gold). Swapping 1-1 → 6-6 **drops V −5.3 Q-pts** — superficially counter-intuitive
since 6-6 is a "bigger" card.

**Probe 4 — oracle verification**: ran the 3.3M-param oracle on both counterfactual deals
(18.9s on MPS). Oracle ΔE[Q_max] = −5.83; Gus ΔV = −5.34. **Agreement within 0.5 Q-pts.**
The counterfactual is correct game theory, not a model artifact: in a defensive hand, 6-6
displaces 1-1 (which the opponent then holds), and the opponent's gain exceeds P0's gain.
Initial "strategy-fusion leakage" diagnosis was wrong; the bilateral-swap asymmetry plus
depth-vs-breadth saturation in the 6-suit fully explains the delta. Oracle confirmed.

**Probe 5 — per-domino impact atlas for 6-6** (238 bilateral swaps across 20 eval games):

| Declaration | Mean ΔV | Verdict |
|---|---:|---|
| Sixes / doubles / follow-me-8 (trump) | +17 to +22 | Always helpful |
| Fives / threes / twos / ones | +2 to +4 | Mixed; catastrophe if displaces trump boss |
| Fours | −0.09 | Essentially neutral |
| Worst case: 6-6 displaces 5-5 in fives | −27.67 | Trump-boss displacement |

6-6's value is entirely determined by trumpness of the declaration and which card it
displaces. **Gus has not learned "big card = good" — it learned the context-dependent
value function 42 actually has.**

**Probe 6 — hand-level threats and boons** (conditioned on stored joint-world tensor, no
new forward passes needed). For game 0 decision 0, baseline E[Q] = −11.15:

| Top boon (partner holds) | Q uplift |
|---|---:|
| 0-0 (top trump) | +16.6 |
| 6-0 (high trump) | +12.5 |

| Top threat (opp holds) | Q drop |
|---|---:|
| R-opp holds 0-0 | −9.4 — "if R-opp has the boss we're sunk" (literal) |

0-0 alone contributes a **26-Q-pt swing** based on location. All five top threats and all
five top boons are trumps. The hand's outcome is determined by trump distribution before
the first card hits the table.

**Meta-conclusion**: Gus has internalized categorical features in the embedding, contextual
strategy in the transformer layers, and distilled the oracle's strategic understanding
faithfully enough to match it within 0.5 Q-pts even on a losing hand. The counterfactual
sensitivity tool is promotable to `gus/eval/` if threat-boon analysis becomes a recurring
feature (e.g., pre-game hand evaluator). (commit message @ 245918d)

### Shine analysis — where the student is ceiling-tight (2026-04-22, commit 7a9c720)

Post-G5 forensics on 410 perfectly-decided positions from `v2_voids_3000g_big`:

| Category | Share | Criterion |
|---|---|---|
| Dead-tie | 59% | Oracle spread < 0.5 Q-pts (any choice wins) |
| Moderate | 15.6% | Oracle spread 0.5–5 Q-pts, student chosen correctly |
| Sharp-and-perfect | 25.4% | Oracle spread > 5 Q-pts, student chosen correctly |

The student shines most on structurally forced positions (e.g., `legal_count ≤ 2`) and
late-game positions (`decision_idx ≥ 22`), where spread is intrinsically low. These regions
account for 450/560 held-out decisions and show ≤ 2% blunder rate.

**Zero-inference routing heuristic**: trust the student when `legal_count ≤ 2 OR
decision_idx ≥ 22`. Reserve oracle or detector overhead for mid-game, multi-action positions
where spread is high and the student's 1.39 Q-pt mean regret is concentrated.
(commit message @ 7a9c720)

### Belief propagation gap (2026-04-22, commit 137a8e7)

PRACTICALITIES receipt 15: a frozen-trunk belief fine-tune improved KL divergence versus
world-marginal (0.078 → 0.062, −47%), confirming the belief head IS absorbing new signal.
Yet downstream metrics regressed:

- PIMC-belief bot-match: **−1pp** vs unfine-tuned checkpoint
- Blunder detector PR-AUC: regressed

Root cause: the Q_head was co-trained with mode-sharp belief during main training. Softening
the belief distribution creates a **distribution shift** that the Q_head has never seen, so
Q estimates over belief-sampled worlds degrade. The belief head is better calibrated; the
Q_head can no longer use it properly.

**Fix**: co-train `{belief_head, world_encoder, Q_head}` jointly rather than fine-tuning the
belief head with a frozen trunk. Full retrain with 10k+ corpus is the natural next opportunity.
(commit message @ 137a8e7)

### Lazy IterableDataset (2026-04-22, commit f138069)

`JointWorldFullIterable` is a streaming PyTorch `IterableDataset` that reads corpus chunks
through a shuffle buffer, never materializing the full dataset in RAM. Motivation: a 10k-game
corpus is ~110 GB — a `MapDataset` would OOM before training starts.

RSS ceiling in practice: **~3.4 GB regardless of corpus size**, measured at 5 chunks. The
`--lazy` flag enables this path in all training scripts. (commit message @ f138069)

**Mirror (2026-05-03)**: the full v1 + v2 + eval corpus (118.68 GB across 230 files) is
published as a public HF dataset:
[jasonyandell/texas-42-joint-world-corpus](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus).
`MANIFEST.json` carries sha256 + paired generation log for every chunk so re-pulled
copies can be verified against the local original.

### Explanation sketcher (2026-04-22, commit 695f2ef)

Template-filled natural-language rationalization from student head outputs — no new ML. Slots:

| Slot | Source |
|---|---|
| `pi_peak`, `pi_entropy` | π_me head |
| Top-2 actions | π_me argmax |
| Belief top-2 dominoes | belief_head |
| V/π disagreement flag | V_head vs softmax(π)·Q_head |
| Q-spread risk language | Q_head spread across sampled worlds |
| Pattern labels | `forced` / `end_game` / `blunder_hotspot` |

Designed as a play-visualizer component — fills the slot where LEM was originally positioned
as a commentary head (see kickoff doc at 42a7535). Implemented in pure template logic,
interpretable without invoking any language model. (commit message @ 695f2ef)

### Q_head augmentation postmortem (2026-04-22, commits a9fa0c6, 5f390fb) — path closed

`train_q_head_augmented.py` implements the depletion-augmentation fine-tune: frozen trunk,
encoder, belief_head, V_head, and π_me; only `q_head + world_encoder` trainable (160,007
params). With probability `aug_prob`, zeros k random assigned-domino rows from `world_assign`
before the forward pass (k ~ Uniform(1, 3)). Target `q_per_world` is unchanged — the
hypothesis is that Q_head should be invariant to whether played dominoes remain in the
assignment tensor.

**Training** (15 epochs, lr=5e-5): best eval qMAE improved 8.277 → 8.169 (epoch 10).
Adapter saved as `gus/adapters/q_head_aug.pt`.

**Eval** (`lamir1-qleaf + Bug 6 + augmented Q_head`):

| Mode | Regret | Bot-match |
|---|---:|---:|
| lamir1-qleaf + Bug 6 (baseline) | 2.156 | 63.2% |
| lamir1-qleaf + Bug 6 + aug Q_head | 2.216 | 63.8% |

Slightly worse. The 0.1-pt qMAE improvement did not translate to better rollout decisions.
The 0.06 regret regression is noise, not signal either way.

**Conclusion (path closed)**: OOD augmentation fixes a measurement artifact — qMAE on
depleted inputs — but not the fundamental ordering problem. Scalar noise from distillation
is large relative to the action-value gap at decision boundaries. The leaf evaluator would
need end-to-end training in rollout context to fix this; augmenting the input distribution
alone is insufficient. (commit messages @ a9fa0c6, 5f390fb; MORNING4_STATUS §Path a)

### Belief at Bayes ceiling (2026-04-22, commit 548d32a)

Diagnostic `gus/eval/belief_ceiling.py` (~60 lines) computes the theoretically best
top-1 achievable from the oracle's own sampled worlds: for each unseen domino, take
`argmax_seat P(seat | oracle sampled worlds)`. Since the oracle's samples ARE the
posterior, this argmax is Bayes-optimal.

**Result on `corpus_eval_20.pt` (20 games, 560 decisions, ~2914 worlds/decision)**:

```
Bayes-optimal belief top-1:  39.184%
Gus v3 belief head:          ~38-39%
```

**Gus is at ceiling.** The "belief is hard" intuition was correct — but the cause is that
the information isn't there (especially at d_idx 0-5, where early-hand belief is ~33% =
pure prior with no play info yet), not architecture or capacity.

| d_idx range | Bayes top-1 | Interpretation |
|---|---:|---|
| 0-5 | ~33% | Pure prior — 1-in-3 over 3 opp seats |
| 6-15 | ~36-40% | Voids + lead signaling start to sharpen |
| 18-25 | ~50-75% | Played dominoes narrow the field |
| 26 | 100% | Only one domino unseen |

**Implication**: top-1 accuracy is a dead lever. Any belief head exceeding ~39.2% on this
corpus is overfitting; any lower is underfitting. Gus is tuned. The real unfinished work is
posterior **shape** (calibration) — §15 already showed distribution-target training closes
47% of the KL gap (0.078 → 0.062). The open loop is co-training belief + world_encoder +
Q_head jointly so that better-calibrated belief propagates to better look-ahead value
estimates. (commit message @ 548d32a; PRACTICALITIES §21)

### Belief co-train experiment + q-bootstrap-belief mode (2026-04-22, commit cf8ff79)

`gus/train/train_belief_q_joint.py` runs the §21 proposed experiment: frozen trunk +
π_me + V_head; unfrozen `{belief_head, world_encoder, Q_head}` (α=1, β=1). Trained on
1000-game corpus, 15 epochs.

**Belief calibration improved** as designed: KL vs world-marginal dropped 0.084 → 0.067
(−20%). Truth top-1 stayed at ~38% (ceiling).

**Downstream regret did NOT improve — slightly worse**:

| Adapter | q-bootstrap (corpus worlds) | q-bootstrap-belief (belief-sampled) |
|---|---:|---:|
| Original v3_consistency_10000g | 0.685 | **0.655** |
| Joint co-trained | 0.718 | 0.679 |

**Hypothesis falsified**: co-training heads lets calibration propagate — not in this setup.
Q_head was already sitting at a sweet spot for the original belief's output distribution;
moving the belief head moved Q_head off it. Calibration improvement doesn't stack additively
in a distillation pipeline where downstream heads were trained against the old belief's shape.

**Unexpected win — q-bootstrap-belief**: sampling worlds from the belief head at inference
(rather than reading from the oracle corpus) gives regret **0.655** on the original
adapter — the closest any look-ahead variant has come to the 0.551 direct baseline (gap
0.104, ~19%). Likely mechanism: oracle adaptive sampling can overconcentrate on a few
high-posterior worlds at near-consensus decisions; belief-head softmax sampling is smoother
and more aligned with the distribution Q_head saw during training. Available as
`--mode q-bootstrap-belief` in `gus/eval/lamir1.py`; reuses `gus/model/sample_worlds.py`.
(commit message @ cf8ff79; PRACTICALITIES §21)

### Future direction: "what you do past belief" (2026-04-22, commit 94d8646)

The §21 finding (belief at Bayes ceiling) reframes the remaining 42-craft problem: the
unresolved skill is not **knowing better** about the hidden deal, it is **acting well given
unresolvable uncertainty**. The user described this as "the heart of the game."
(PRACTICALITIES §22; commit message @ 94d8646)

**Extractable analytics** — all computable from the existing `q_per_world` tensor with no
new training, estimated effort ~1 afternoon:

1. **Outcome-variance per decision**: `q_per_world[:, action_taken].std()` — how much the
   outcome of the chosen action depends on which hidden world is true.
2. **Action-choice fragility**: count distinct `q_per_world.argmax(dim=-1)` actions across M
   worlds — if 3+, the decision is a fog-of-war call where optimal play depends on unseen
   information.
3. **Belief-limited high-impact decisions**: join (1) and (2) with per-decision belief
   sharpness — the defining moments of 42 where belief is ~33% AND fragility is high AND
   outcome-variance is high.

**Research direction**: π_me is trained on `argmax(marginal E[Q])`, committing to one
meta-strategy ("play for the mode"). Human experts use several — mode, signal, hedge,
gamble — and select based on match context (score, partner expectations, opp mistakes). A
richer student could output a meta-strategy distribution trained from the oracle's per-world
tensor (training data already available); no LAMIR or CFR+ required.

Status: noted as a future direction; no code written; not blocking anything.

## Role for Burl (prior to G1)

Burl's [[belief-trajectory]] tool uses Gus's `v3_consistency_10000g` belief adapter,
providing per-domino posterior, shift-since-last, V, and CLS attention. This predates the
formal Gus replay trail — Burl consumed an existing Gus checkpoint before Gus's own
development was documented here. (commit message @ d858781)
