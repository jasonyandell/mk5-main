---
title: The Gus Line (Apr 2026 — neural student, belief, and the look-ahead ceiling)
kind: trail
first_seen: 2026-07-13
last_updated: 2026-07-13
status: complete
---

## What this line is

[[gus]] is the sibling project that skipped the reasoning channel: distill [[forge]]'s
E[Q] oracle into a multi-head transformer (belief + V + π_me + world-conditioned Q +
π_opp) and play from the heads directly ([[student-distillation]]). The line ran
roughly 2026-04-20 to 2026-04-30, produced the project's best single fast player
(`v3_consistency_10000g`, 0.551 Q-pt regret), proved that belief accuracy is
information-limited rather than architecture-limited, and closed the look-ahead
question by measuring it to death. Its belief head is the posterior engine the
[[jud]] line now builds on. This trail walks the whole arc; each
experiment below carries its finding on its own page.

## The arc

### 1. Substrate: the joint-world tensor

The [[joint-world-tensor]] — per-decision `(world_hands, q_per_world)` saved by the
oracle — is what makes a variance-free student possible.

- [[gus-joint-world-tire-kick]] — single-game validation that the belief-Q signal is
  real (converged r≈0.2-0.4) and M≥100 samples are required; MPS generation is practical.

### 2. From belief-only to four heads

- [[gus-v0-v1-belief]] — MLP belief head overfits severely; the v1 transformer gets the
  right per-decision shape (33% at decision 0 → 75% at decision 26); ceiling is data,
  not architecture.
- [[gus-4head-baseline]] — all four heads on 100g: [[dense-q-supervision]] (~3400×
  denser signal per decision) eliminates the overfit; π_me 57.9%.
- [[gus-v2-voids-1000g]] — explicit engine-computed voids buy only +1.4pp belief; the
  transformer already infers voids attentionally from play tokens.

### 3. Scaling and the eval vocabulary

- [[gus-scaling-ladder]] — the full 100g→10000g adapter ladder; data dominates
  capacity; [[regret-eval]] replaces bot-match as the primary metric.
- [[gus-lamir-primitive-eval]] — direct π_me beats single-step PIMC at any K; π_me
  already IS the marginalized policy.

### 4. Game level, V/π decoupling, and v3

- [[gus-arena-pilot]] — 1.39 Q-pt decision regret compounds to a ~30pp contract-made
  gap over full games; blunder forensics surfaces [[v-pi-decoupling]] (V knows +26,
  π picks −0.4).
- [[gus-v3-consistency-full-run]] — the [[consistency-regularizer]] fix scales better
  than plain distillation: first sub-1.0 regret (0.551 at 10k, −60% vs v2-3k). The
  [[qmae-plateau]] (qMAE −7% while regret −59%) marks Q_head as the weak head.

### 5. Detect-and-route

- [[gus-blunder-detector]] — ensembles hurt regret, a router is the right shape;
  student-feature GBM reaches ROC-AUC 0.839 with `pi_peak` as top feature.
- [[gus-shine-analysis]] — 73% of decisions are already perfect; `legal_count ≤ 2 OR
  decision_idx ≥ 22` covers 80% of decisions at ≤2% blunder rate.
- [[gus-router-pilot]] — oracle fallback works (0.49 regret at 25% flag); PIMC-Q and
  next-best-adapter fallbacks both hurt. Concept page: [[blunder-detector]],
  architecture: [[detect-and-route]].
- [[gus-qmean-router]] (2026-04-25, after the LAMIR ceiling) — the no-oracle router
  that works: a belief-sampled Q-mean *second opinion* on ~5-7% of decisions cuts
  regret 0.551 → ~0.42-0.43 with zero new blunders. Standing conclusion: candidate
  selection, not candidate generation, is the hard problem.

### 6. Interpretability

- [[gus-probe]] — six probes on v3-10k: counterfactual V sensitivity matches oracle
  E[Q] deltas within 0.5 Q-pts; the student learned 42's context-dependent value
  function, not "big card = good." A template-only explanation sketcher over the same
  head outputs followed ([695f2ef](../sources/695f2ef.md)).

### 7. The LAMIR ladder and its ceiling

The line's central negative result: look-ahead built from distilled heads loses to the
policy head it was meant to improve ([[lamir1]], [[lamir1-ceiling]]).

- [[gus-lamir1-pilot]] — first end-to-end rollout actively hurts; damage concentrates
  at trick_pos 0-2.
- [[gus-lamir1-mode-comparison]] — v-bootstrap/q-bootstrap/qleaf modes plus two silent
  bugs; depth-1 V_head is as broken as full rollout, so the leaf evaluator, not
  opponent simulation, is the bottleneck.
- [[gus-pi-opp-training]] — dedicated seat-conditioned opponent head ([[pi-opp-head]])
  reaches 68.6% oracle top-1 (vs ~55% rotated π_me); a NaN masking bug fixed en route.
- [[gus-lamir1-piopp]] — the complete 8-mode ladder: nothing beats direct π_me (best
  look-ahead q-bootstrap at 0.679 vs 0.551); Kubíček & Lisý's warning about distilled
  value functions confirmed; four pivot options documented.
- [[gus-q-head-augmentation]] — pivot path (a) falsified: depletion augmentation fixes
  the qMAE measurement artifact, not the argmax-flipping noise ([[q-head-augmentation]]).

The project took pivot option 4 — self-play, no CFR+ — which became [[w42-jud-v1]] and
the [[jud]] line.

### 8. Belief endgame: ceiling, co-train, and what lies past belief

- [[gus-belief-calibration-diagnostic]] — distribution-target fine-tune improves
  calibration but downstream play regresses: the [[belief-propagation-gap]].
- [[gus-belief-co-train]] — belief top-1 is at the Bayes ceiling (39.184%,
  [[belief-bayes-ceiling]]); joint co-training ([[belief-co-train]]) is falsified; the
  surprise win is **q-bootstrap-belief** — belief-sampled worlds beat corpus worlds
  (0.655 vs 0.685), the closest look-ahead to the direct baseline.
- [[gus-drama-atlas]] — the [[past-belief-future-direction]] analytics, run on the full
  corpus: 26.2% of decisions are genuine fog-of-war drama, and 62% of drama is the
  opening lead. The craft of 42 is front-loaded and belief-blind.

### 9. Side branch: strategy tags → w42

- [[gus-strategy-tags-probe]] — book-derived public-state tags cut a tiny model's
  regret 2.012 → 1.181 (E[Q] N=10 still boss at 0.167). Its "Next" list was executed
  the following day as the [[w42]] workstream — this probe is w42's origin.

## What the line concluded

1. **The fast player is real and ceiling-tight.** `v3_consistency_10000g` (0.551
   regret, 76.07% bot-match) remains the best-known single adapter; ~0.49 with routing.
2. **Belief accuracy is solved — by information limits.** Top-1 is at the Bayes
   ceiling; the remaining lever was calibration shape, and even that doesn't propagate
   additively through a distillation pipeline.
3. **Look-ahead over distilled heads is a dead end** at this game scale; the leaf
   evaluator's scalar noise flips argmax at decision boundaries.
4. **Routing beats ensembling**, and a belief-sampled second opinion beats both — but
   candidate selection is the residual hard problem.

## Where the value went

The line's assets moved to the [[jud]] frontier rather than dying with it.
The belief head became the champion's posterior engine — auction-conditioned belief
(rung #24) measured +2.59pp accuracy ([[w42-champion-auction-belief]]); belief-weighted
*play* sampling measured dead (rung #25), routing belief's value to bidding/defense via
self-play ([[belief-conditioned-self-play]]). [[belief-co-train]]'s q-bootstrap-belief
result is the standing evidence for that wiring. `gus/bidding/` (Gus as the simulating
player for contract evaluation) is the substrate for the champion's auction work, and
[[burl]]'s production [[belief-trajectory]] tool serves the same
`v3_consistency_10000g` belief head. The [[gus-strategy-tags-probe]] branch grew into
[[w42]]. Zeb-era's open "how do beliefs feed policy?" question ([[belief-feeding-policy]])
found its answer in this line's architecture.

## Links

[[gus]] · [[jud]] · [[w42]] · [[forge]] · [[joint-world-tensor]] · [[student-distillation]]
