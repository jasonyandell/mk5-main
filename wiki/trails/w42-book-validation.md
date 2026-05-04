---
title: W42 Book Validation
kind: trail
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

This trail routes the Winning 42 book-validation cluster. It keeps [[w42]] small
while preserving the evidence pages that make the work useful.

## 1. What W42 is doing

[[w42]] treats Winning 42 as a hypothesis source, not as rule truth. The project
turns book advice into measurable detector rows, paired oracle contrasts, model
features, report buckets, and bounded claim statuses.

The current charter is [[w42-promote-or-retire]]: keep W42 as active research,
promote durable code and reports under `w42/`, and do not promote research models
into [[gus]] or [[burl]] without an explicit later decision.

## 2. Book and chapter inventory

The chapter inventory lives as leaf pages under `experiments/` because each page
is a detector inventory and evidence receipt, not a headline topic.

Core index:

- [[winning42-strategy-measurement]] — chapter inventory and measurement frame.
- [[winning42-ch01-in-a-nutshell]] through [[winning42-ch16-statistical-odds]] —
  chapter-level detector and evidence pages.
- [[at-risk-points]] — Roberson's bidding framework used as a voice and concept
  anchor for bidding and post-commit Q&A.

## 3. Phase ladder

The W42 validation ladder is best read as phases:

- Phase 1 / survey: [[w42-final-empirical-strategy-report]],
  [[w42-strategy-tags-v0]], [[w42-strategy-tags-v1-map]],
  [[w42-v0-strategy-tags-baseline]], [[w42-rich-tag-many-signal-probe]],
  [[w42-strategy-tag-family-ablations]].
- Phase 2 / claim analysis: [[w42-phase2-statistics-claims-ledger]],
  [[w42-phase2-claim-analysis-matrix]], [[w42-phase2-claim-analysis-harness]],
  [[w42-phase2-decision-table]], [[w42-claim-analysis-synthesis-report]].
- Phase 3 / generated counterfactuals and joined rows:
  [[w42-phase3-auction-bid-discipline-corpus]],
  [[w42-phase3-84-seed-mining-corpus]],
  [[w42-phase3-sequence-seat-counterfactuals]],
  [[w42-phase3-joined-claim-row-model-table]].
- Phase 4 / 64-claim closure sweep: [[w42-phase4-sequence-handshape-tests]],
  [[w42-phase4-84-dynamic-seed-tests]],
  [[w42-phase4-doubles-notrump-regime-tests]],
  [[w42-phase4-laydown-rule-accounting]],
  [[w42-phase4-scoring-objective-tests]],
  [[w42-phase4-claim-completion-board]],
  [[w42-phase4-bidding-count-exposure-tests]],
  [[w42-phase4-final-claim-audit]].
- Book validation v1 / wave 1: [[w42-book-validation-campaign]],
  [[w42-bookval-v1-wave1-independent-audit]],
  [[w42-bookval-v1-wave1-distribution-lens-reranker]],
  [[w42-bookval-v1-wave1-mark-utility-transform]],
  [[w42-bookval-v1-wave1-hidden-threat-impact-ranker]],
  [[w42-bookval-v1-wave1-cross-ai-agreement]].
- Book validation v1 / wave 2 paired probes:
  [[w42-bookval-v1-wave2-infra-design]],
  [[w42-bookval-v1-wave2-bid-aware-atlas]],
  [[w42-bookval-v1-wave2-reentry-v2]],
  [[w42-bookval-v1-wave2-low-trump-trap]],
  [[w42-bookval-v1-wave2-pounce-window-bid30]],
  [[w42-bookval-v1-wave2-void-creation]],
  [[w42-bookval-v1-wave2-void-creation-follow]],
  [[w42-bookval-v1-wave2-ch02-multistep]],
  [[w42-bookval-v1-wave2-ch10-action-level]],
  [[w42-bookval-v1-wave2-pounce-high-bid]].
- Utility and planning frontier: [[w42-bookval-v2-utility-lens-synthesis]],
  [[w42-bookval-v3-utility-argmax-divergence]],
  [[w42-lens-v1-utility-head-to-head]], and [[book-strategy-player]].

## 4. Evidence surfaces

Use these leaves by question shape:

| question | pages |
|---|---|
| What is the claim ledger state? | [[w42-phase2-statistics-claims-ledger]], [[w42-phase4-claim-completion-board]], [[w42-phase4-final-claim-audit]] |
| Which book claims have generated evidence? | [[w42-phase3-auction-bid-discipline-corpus]], [[w42-phase4-sequence-handshape-tests]], [[w42-phase4-84-dynamic-seed-tests]], [[w42-phase4-doubles-notrump-regime-tests]], [[w42-phase4-scoring-objective-tests]] |
| Which labels help a model? | [[w42-claim-tag-model-probe]], [[w42-phase3-joined-claim-row-model-table]], [[w42-rich-tag-many-signal-probe]], [[w42-strategy-tag-family-ablations]] |
| Where does scalar EV hide risk? | [[w42-phase2-distribution-aware-ev-report]], [[w42-hidden-threat-legacy-mining]], [[w42-bookval-v1-wave1-distribution-lens-reranker]] |
| What hidden tiles drive branch shape? | [[w42-phase2-hidden-domino-threat-attribution]], [[w42-powered-branch-atlas-v1]], [[w42-branch-atlas-scaled-v0]], [[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] |
| Which utility wins at one-step play? | [[w42-bookval-v2-utility-lens-synthesis]], [[w42-bookval-v3-utility-argmax-divergence]], [[w42-lens-v1-utility-head-to-head]] |
| Which book claims need plans, not one-step probes? | [[w42-book-claim-synthesis-and-ai-directions]], [[book-strategy-player]] |
| What must the planning framework obey? | [[book-strategy-player]]'s amended Phase 1 contract, nine algebras, ten laws, and property-test checklist |
| What data exists? | [[w42-claim-data-inventory]], [[w42-dataset-manifest]], [[w42-wandb-run-comparison-dashboard]], [[w42-wandb-series-logging-standard]] |

Older claim-validation leaves remain useful as evidence, but should not be
loaded first: [[w42-bidder-sequencing-claim-validation]],
[[w42-partner-support-claim-validation]], [[w42-setter-defense-claim-validation]],
[[w42-84-claim-validation]], [[w42-doubles-no-trump-claim-validation]],
[[w42-scoring-objective-drift-claim-validation]], and
[[w42-odds-ruleset-claim-validation]].

## 5. Current frontier

[[w42-book-validation-campaign]] is the live campaign surface. The baseline
state after [[w42-phase4-final-claim-audit]] is that every one of the 64 book
claims has evidence and/or explicit bounded blockers; no row is ownerless.

Wave 1 added independent audit, utility-lens disagreement, mark-utility,
hidden-threat ranking, and cross-AI agreement. Wave 2 then produced the first
hard promotions and demotions under paired contrasts: bid-only-enough is
supported, void creation is position-dependent, and high-bid pounce is
contradicted once tested at the action contrast rather than aggregate proxy
level.

Wave 3 and Wave 4 narrowed the utility-lens lesson. The original "many claims
may be p_make-optimized" theory collapsed to one concrete split, but the EV and
p_make argmax policies still disagree sharply on the ch05 void-creation-follow
corpus. [[w42-lens-v1-utility-head-to-head]] then showed EV wins the one-step
utility contest.

The current frontier is planning-aware validation. [[w42-book-claim-synthesis-and-ai-directions]]
names the single-decision blind spot: many book claims are multi-step plans.
[[book-strategy-player]] is the route for testing those claims without loading
the entire W42 leaf pile. Its amended Phase 1 contract is now build-ready:
fresh recognition is split from commitment, only the winning fresh strategy
commits, shared facts are lawful wrappers, fallback dry-runs are pure, decision
records are replayable, and the nine algebras / ten laws become the property-test
checklist.

## Related pages

[[w42]] · [[winning42-strategy-measurement]] · [[w42-book-validation-campaign]] ·
[[w42-book-claim-synthesis-and-ai-directions]] · [[w42-lens-v1-utility-head-to-head]]
· [[book-strategy-player]] · [[w42-promote-or-retire]] · [[gus]] · [[burl]] ·
[[forge]]
