---
title: w42 Book Claim Synthesis And AI Directions
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] now has enough phase-2, phase-3, and phase-4 evidence to say what
Winning 42 taught the project, what the current machinery could confirm, what
remains unconfirmed, and how the result should change future AI work.

The core lesson is not "the book is right" or "the book is wrong." The book is a
good hypothesis generator. Its best claims become measurable detectors,
counterfactual slices, distribution-shape reports, and model features. The broad
folk sayings usually need gates: command, closure, current control, live count,
off exposure, high/top double control, score mode, and whether a claim depends
on private hand state.

The strongest project-level result is that the book vocabulary carries signal.
[[w42-phase3-joined-claim-row-model-table]] improves a public-safe legal-action
row model from `1.3598` to `1.1257` mean regret and from `64.464%` to `68.107%`
best-mean match. Sequence/seat is the dominant family; bidding and 84 add
smaller positive signal; hidden public proxy is diagnostic but not yet a model
feature.

## Confirmed Or Strongly Supported

### Rule and scoring substrate

[[w42-phase4-laydown-rule-accounting]] confirms the low-level game accounting:
count identity, 42-point hand total, suit membership, trump exclusivity,
follow-suit masks, trick winner, lead control, count capture, and contract
scoring. It also makes laydown correctness executable for tiny full-information
fixtures: a claim is valid only if every legal continuation wins the rest.

[[w42-phase4-scoring-objective-tests]] confirms that marks change the objective,
not just the scoreboard. Marks create early terminal states, erase defender
partial points in made ordinary contracts, compress ordinary set severity, and
can disagree with point-score match winners in generated proxies.

### Tactical play

[[w42-phase4-sequence-handshape-tests]] confirms the gated version of several
folk rules:

- commanding called doubles are good bidder leads;
- blanket non-double called-suit leads are bad against off alternatives;
- partner closure count donation is strong;
- broad partner count donation before closure is weak;
- partner count-liability leads are bad;
- setter count-calling leads and pounce take-count-now are strong;
- reckless setter count into bidder control is bad.

This is exactly the right scientific shape: the folk rule survives after being
made narrower.

### 84 and endgame preservation

[[w42-phase4-84-dynamic-seed-tests]] supports 84 preservation as a real action
surface. On reached natural seed states, preserving live weapons/protectors beats
spending them, dead-asset release has positive value, and bidder trump-pull
before final-off is strongly positive.

### Doubles and no-trump

[[w42-phase4-doubles-notrump-regime-tests]] confirms the Chapter 9 caution
against "many doubles means doubles-trump." In generated same-hand paired
regime tests, no-trump often beats doubles-trump for four-plus-double hands.
High/top double control is the near-flat pro-doubles slice; missing top doubles
and no-trump support doubles favor no-trump.

### Bidding and risk

[[w42-phase4-bidding-count-exposure-tests]] supports the static/generated
contract versions of Chapter 2:

- three-plus trumps are a strong biddability prior;
- risk <=12 is better but only modestly, so it is not a standalone rule;
- four/five off exposure is bad;
- double-ahead protection is side-specific, not global;
- partner two-plus-double prior is about `58%`;
- natural max-profitable bid buckets exist, but are not a full auction story.

[[w42-phase3-auction-bid-discipline-corpus]] gives the cleanest bid-only-enough
slice: no positive-margin bid improved over the minimum winning bid in the
generated auction-pressure table.

## Not Confirmed Yet

The unconfirmed claims are mostly not failures of the book; they are failures of
the current state representation or generator.

Still needing better tech:

- exact private hand-shape causality: reentry preservation, void creation, low
  trump traps, count-protection throwaways, trump-rich setter recognition;
- high-bid 35/36 pressure and high-bid off-pounce claims, because the main row
  table is bid `30`;
- arbitrary laydown/state snapshots and Burl false-claim audits;
- straight-off 84 set rates by "good players";
- score 42-vs-84 terminal match counterfactuals;
- full throwaway ladders and final set attribution for 84;
- real auction policy, opponent response, and partner-bid intelligence;
- human wall-clock tournament speed and real tournament advancement/tiebreakers;
- reputation, table style, and repeated-player adaptation.

[[w42-phase4-final-claim-audit]] records these as bounded future work, not
ownerless gaps.

## What This Suggests For E[Q]

Scalar expected value is useful, but it hides exactly the things the book is
good at naming. Many folk rules are not mean-value rules; they are distribution,
threshold, and plan-shape rules.

The next E[Q]-adjacent surface should keep the per-world outcome distribution
available and let policies choose through several utilities:

- `p_make`: probability of crossing the contract threshold;
- `p_set`: probability of falling short;
- `threshold_mass`: mass just above or below make/set thresholds;
- `tail_loss` or CVaR: lower-tail disaster risk;
- `mark_ev`: match/mark utility under Chapter 10 scoring;
- `bid_margin_survival`: utility that values making the current bid before extra
  point margin;
- `count_tail_risk`: explicit penalty for live count dumps in vulnerable states;
- `robust_value`: conservative lower quantile over sampled worlds;
- `explanation_tags`: which book detector explains why the utility differs from
  scalar EV.

In practice, E[Q] should not be replaced by one new scalar. The better move is a
utility lens:

```text
world outcomes -> distribution features -> utility family -> selected action
```

The same saved `q_per_world` can support EV, make-probability, mark utility,
tail-risk, or lexicographic "make first, then improve margin" objectives. The
project can then ask which objective agrees with the book, with Gus, with Burl,
or with future human play.

## Model Experiments To Run

1. **Distribution-lens action ranking.**
   Re-rank existing branch-atlas and Gus-corpus actions under EV, `p_make`,
   mark utility, CVaR, and threshold-mass utilities. Measure disagreement,
   regret, set-tail reduction, and which book detectors explain the differences.

2. **Claim-aware row model v2.**
   Train a legal-action model with public features plus phase-4 detector tags,
   but add auxiliary heads for `p_make`, lower-tail mass, set risk, and
   detector-family labels. The goal is not just lower regret; it is a model that
   can say which folk concept changed the action.

3. **State-injection generator.**
   Build a generator that creates exact late states for reentry, void creation,
   low-trump traps, 35/36 pounce pressure, 84 throwaway ladders, and laydown
   proofs. This is the main missing technical tool for the unconfirmed claims.

4. **Utility-conditioned Gus or w42 model.**
   Train the same public state/action representation against multiple utility
   labels: EV, mark EV, `p_make`, CVaR, and bid-margin survival. Test whether a
   single trunk can support objective-conditioned play better than the current
   scalar target.

5. **Burl trace faithfulness bench.**
   Use the proof checker and detector rows as prompts: does Burl ask for the
   right facts before claiming a laydown, donating count, choosing no-trump, or
   preserving an 84 weapon? The book gives a natural rubric for reasoning, not
   just move quality.

6. **Belief and hidden-threat calibration.**
   Use hidden-threat shelves from [[w42-hidden-threat-legacy-mining]] to train or
   audit belief features for specific unseen dominoes. The book repeatedly talks
   about one tile changing the plan; the model should learn which hidden tile is
   load-bearing.

## Practical AI Improvements

The immediate improvement is knowledge: use phase-4 detector tags as reporting
and explanation features in Gus/Burl evals. They should be especially useful for
tail-risk debugging, not just mean-regret scoring.

The next code improvement should be a reusable W42 state-injection harness. It
would make the remaining book claims testable and provide adversarial evals for
existing AIs. The highest-value generated buckets are:

- bidder reentry and off timing;
- setter void creation and high-bid pounce;
- 84 final-two-trick preservation and throwaway ladders;
- no-trump support-double late-off plans;
- exact laydown claims from arbitrary saved snapshots.

The next model improvement should be objective conditioning. A model that can
rank an action by EV, mark utility, make probability, and lower-tail risk may be
more useful than a slightly stronger scalar-EV student, because the project can
choose the utility that matches the game context.

## Wave 1 Findings (Book Validation v1)

The first wave of the book-validation campaign ran five offline analyses on
already-existing artifacts. Five new wiki pages capture the details:
[[w42-bookval-v1-wave1-distribution-lens-reranker]],
[[w42-bookval-v1-wave1-mark-utility-transform]],
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]],
[[w42-bookval-v1-wave1-cross-ai-agreement]], and
[[w42-bookval-v1-wave1-independent-audit]].

The wave's structural findings:

- **Scalar EV "lies" in 64.9% of decisions on the seed-9430 branch atlas**, but
  the alternative-utility pick never beats EV on EV terms. Median EV cost when
  alternative wins is +0.18-0.26 points. Disagreement is pure risk-framing.
- **CVaR_10 and robust_q25 agree with EV at 82-83%; p_make and threshold_mass
  only at 59%.** Tail-aware utilities track the oracle. Make-rate framing does
  not. The book's "make first" framing is less aligned with optimal play than
  its "manage tail risk" framing.
- **No-trump shows 90% EV-lying rate**, doubles-trump 39% CVaR disagreement.
  Clean detector signals for "scalar EV is the wrong target here."
- **mark_ev is algebraically identical to p_make at bid=30 with one-mark
  multiplier.** The Ch 10 multiplier (`ch10-special-bid-mark-multiplier`) is
  unreachable on the fixed-bid corpus. A bid-aware E[Q] generator is now a
  hard prerequisite for further mark-objective work.
- **Genuine mark-vs-point flips are only 3.6% of decisions** (10/280); 81.5% of
  nominal flips are surface-flattening artifacts of the binary {-1, 0, +1}
  mark transform near the 30-point make threshold.
- **The "one tile decides the plan" rhetoric is real but rare**: 1.85% of
  decisions show >=60% impact concentration on a single hidden tile.
- **Trump-count tiles are 100% directionally helpful as load-bearing tiles** -
  the easiest belief-attention target to train against.
- **5-5 in twos and 4-4 in no-trump are the highest-impact non-trump
  load-bearing tiles** - low-pip declarations promote the highest off-suit
  double into the strategic role usually held by trump.
- **Setter seats are asymmetric**: right-setter load-bearing tiles are ~30%
  trump doubles; left-setter is ~37% plain tiles. The book treats setters
  symmetrically; the data does not.
- **Detector hygiene**: `ch05_reckless_count` mean regret 9.18 over 2,300
  cases (overfires beyond its qualifying window); `ch03_called_non_double` is
  wrong as an absolute action endorsement 86% of the time (within-pair
  contrast is fine, action-level endorsement is not);
  `ch05_setter_pressure_regime` is a regime label, not an action label.
- **Divisive decisions concentrate on setter seats**: of the 100 most
  divisive decisions in the cross-AI spot-check pack, 83 are setter seats
  (45 left, 38 right). This compounds the setter-seat-asymmetry finding
  from hidden-threat ranking: setters are simultaneously where the detectors
  most over-fire, where book advice diverges from oracle play, AND where
  hidden-tile attribution is most asymmetric. Setter-defense is the single
  richest tactical surface for future probes.
- **Worst detector misfires are all `ch05_reckless_count_to_bidder_control`
  on trick-0 setter closures**: detector picks regret 38-43 in the worst
  three cases; EV consistently picks a non-count tile. The detector cannot
  distinguish the qualifying sub-condition from the broader regime. Tracked
  as `t42-v0m5`.

### Ledger absorptions

Two Ch 10 rows moved to `context-limited` after the independent audit found
phase-4 worker evidence the prior audit did not absorb:

- `ch10-point-system-skill-signal`: `underpowered` -> `context-limited`
  (4 heuristic policy pairs, point/mark separation observed; bounded by small
  N and no oracle).
- `ch10-timed-marks-advancement-objective`: `not-yet-tested` ->
  `context-limited` (160 synthetic timed trials, advancement disagreement rate
  0.15; bounded by trick-budget proxy, no real clock or bracket).

The non-vocabulary status string `supported-for-generated-trace-proxy` in
`w42/phase4_scoring_objective_tests/claim_summary.csv` for
`ch10-tournament-speed-tradeoff` was normalized to `context-limited`.

Status counts after Wave 1: supported 23, context-limited 14, underpowered
20, not-yet-tested 5, contradicted 2.

## Current Bottom Line

The book taught the project a vocabulary for action reasons. The confirmed
parts are mostly gated and tactical, not slogans. The unconfirmed parts are
now precisely technical: state injection, high-bid generation, real auction
policy, population play, and objective-conditioned utility.

That is a good outcome. The book did not become rules; it became a test suite,
a feature vocabulary, and a roadmap for better agents.

Wave 1 sharpened the path: tail-aware utilities (CVaR, robust_q25) are the
near-free upgrade over scalar EV; mark-objective work is blocked on bid-aware
generation; hidden-threat attribution has surfaced concrete training targets
(trump-count, 5-5-in-twos, 4-4-in-no-trump, setter-seat asymmetry); and
several detectors need refinement before they can serve as direct training
labels.

## Wave 2 Findings (Book Validation v1) — Infra Builds

Wave 2 landed the two infra pieces flagged by Wave 1 as prerequisite for
further claim movement. Two new wiki pages capture the details:
[[w42-bookval-v1-wave2-reentry-preservation]] (state-injection harness +
first probe) and [[w42-bookval-v1-wave2-bid-aware-atlas]] (bid-aware
E[Q] driver + smoke sweep).

**State-injection harness** ([[w42-bookval-v1-wave2-reentry-preservation]]):
`GameStateTensor.from_snapshot` and `generate_eq_from_snapshots` now let
the forge generator start from arbitrary mid-game snapshots. 121 forge
tests pass with the new code, 0 regressions. The integration probe on
200 reentry-shape snapshots returned `underpowered` because the snapshots
were extracted from random-play trajectories rather than oracle-greedy
play - a context bias the agent flagged in its own write-up.

**Bid-aware E[Q] driver** ([[w42-bookval-v1-wave2-bid-aware-atlas]]):
the W42-side wrapper sweeps {30,32,35,36,39,42,84} on the same hands
and recomputes `mark_ev` with the correct multiplier per bid. Validation
at bid=30 against [[w42-branch-atlas-scaled-v0]] passes 10/10 decl_id
pairs within sampling noise.

**Headline cross-bid finding**: the Wave 1.2 algebraic identity
`mark_ev == p_make` holds within each bid value but **breaks across bids**.
Mean absolute divergence between `mark_ev` and `threshold_mass` rises
monotonically with bid: `0.398` at bid=30, `0.640` at bid=36, `0.897` at
bid=42, `3.774` at bid=84. Cross-bid mark_ev change rate is 56-63% at
bids 32-42 vs bid=30, and 100% at bid=84. This is the first corpus where
`ch10-special-bid-mark-multiplier` is meaningfully testable.

The remaining six Wave 2 probes (`t42-26j8`, `t42-jysl`, `t42-ntbe`,
`t42-wikw`, `t42-ey88`, `t42-8na4`) are now unblocked. The first
production run should be a 50-seed × 1000-sample CUDA sweep so paired
contrasts at bids 32 / 35 / 36 / 42 / 84 carry enough statistical power
to move ledger rows (rather than just demonstrating divergence).

## Wave 2 Probes (in flight)

[[w42-bookval-v1-wave2-reentry-v2]] — re-runs the original Wave 2.A
reentry probe on oracle-greedy snapshots from
[[w42-claim-data-inventory]]. **Phase-conditional verdict**: overall EV
delta is `-1.23` (CI `[-2.62, +0.16]`, consume direction); the late-game
slice (tricks 5-6, n=60) is statistically `contradicted` against the
book at CI `[-6.76, -1.17]` (consume strictly better); mid-game (n=162)
is underpowered (CI `[-1.79, +1.36]`); no early-game data in the corpus.
Status: `context-limited`. The book's reentry-preservation advice does
not survive paired counterfactual testing in the phase where it is most
pointed.

[[w42-bookval-v1-wave2-low-trump-trap]] — paired contrast on 257
oracle-greedy snapshots. Overall delta is `-1.95` (CI `[-2.94, -1.07]`),
but the aggregate is dominated by positions where hoarding the dominant
trump is broadly correct, not the specific trap the book warns about.
**In the count-bearing subgroup (n=86, the actual book scenario)**,
mean delta is `-0.02` with CI `[-2.16, +2.15]` — symmetric, no signal.
The trap fires in 42.4% of cases (severe in 7.4%) but is offset by the
larger correct-hoard class. Status: `context-limited`. The probe
correctly distinguishes the book's narrow claim from the broader
contrast pulled by the detector vocabulary.

[[w42-bookval-v1-wave2-pounce-window-bid30]] — paired contrast on 52
oracle-greedy snapshots (500 → 52 after filtering 1-legal-move and
setter-led-trick positions). **Major methodological finding**: the
book's setter-pounce advice is right for `p_make` but wrong for scalar
EV at bid=30. The oracle (which optimizes `p_make` at the contract
threshold) chose pounce in `59.6%` of paired contrasts. Scalar EV said
decline was better in `65.4%`. The split is sharpest in 10-point count
cases (n=5): EV delta is `+15.68` with CI `[+1.60, +29.76]`
(decline strictly better), yet the oracle still pounced `80%` of the
time. Status: `context-limited`. The book's pounce instruction encodes
a `p_make` objective at the contract threshold; this matters for any
agent that evaluates pounce decisions under a non-`p_make` utility.

At the time this wave landed, this looked like the campaign's first
clear strategic-claim-level p_make/EV split — not just an action-ranking
disagreement (Wave 1.4 cross-AI matrix at 59%) but a verdict reversal.
**Wave 3.0 (utility-lens meta-analysis) later narrowed this read.** The
pounce-bid30 result reproduces in the re-analysis (oracle pounce 60%,
EV span-zero), but the broader hypothesis that "many book claims encode
implicit p_make reasoning" did not generalize to the closed probe set.
See the Wave 3.0 reconciliation section below for the corrected
formulation; ch05-void-creation-follow is now the cleaner case study.

[[w42-bookval-v1-wave2-void-creation]] (lead, n=276) and
[[w42-bookval-v1-wave2-void-creation-follow]] (follow, n=500) together
yield the campaign's clearest **position-dependent reversal**:

- Lead-position void creation is **contradicted**: EV delta `-2.63`
  (CI `[-3.42, -1.84]`) — voiding by leading a singleton makes the
  setter worse off.
- Follow-position void creation is **`context-limited` in book
  direction**: EV delta `+0.77` (CI `[+0.12, +1.42]`) — voiding by
  discarding a singleton when forced off-suit is marginally better.

The mechanism is intuitive in retrospect: leading announces the void
at a cost; following only chooses which suit to deplete. **The book's
canonical scenario (follow) survives. The adjacent lead scenario,
which sounds similar, fails.** This is the campaign's strongest case
study for why scope precision matters: a single ch05 advice paragraph
contains two operationalizable scenarios with opposite empirical
verdicts.

A future synthesis pass should classify every `context-limited` and
`underpowered` row by which sub-scenarios it survives — many book
claims likely have similar position-dependent or phase-dependent
splits.

## Wave 2.B.2 Full Bid-Aware Sweep — Power Analysis

[[w42-bookval-v1-wave2-bid-aware-atlas]] now contains 259,618 action
rows from a 50-seed × 10-decl × 7-bid sweep on M5 MPS. Validation at
bid=30 vs [[w42-branch-atlas-scaled-v0]] passes 10/10. Power analysis
verdicts:

- `ch10-special-bid-mark-multiplier`: **sufficient** at all bids
  (n=14,000 paired decisions per bid, CI half-widths < 0.008). Mark_ev
  changes in 65-100% of decisions as bid rises, with 100% at bid=84.
  Adds action-level evidence to a row already `supported` via
  deterministic transform.
- `ch02-bid-only-enough`: **sufficient** for `mark_ev` and `p_make`
  paired deltas (bid=32 minus bid=30: `mark_ev` `-0.076` CI
  `[-0.085, -0.067]`; `p_make` `-0.038` CI `[-0.042, -0.034]`). Both
  CIs exclude zero in book direction (overbidding hurts). Borderline
  for `threshold_mass`. **Ledger row promoted: `not-yet-tested` ->
  `context-limited`** (paired counterfactual evidence on the
  same-contract one-step bid-margin slice; auction-policy and
  multi-step bid-margin scopes remain untested).
- `ch12-setter-pounce-high-bid-off`: **sufficient** at bid=39
  (delta `+0.376`, CI `[+0.010, +0.751]`) and bid=42 (delta `+0.410`,
  CI `[+0.068, +0.765]`); borderline at bid=35/36/84. The pounce-Q
  signal flips from negative at bid=30 (Wave 2.E) to positive at
  bid >= 39, consistent with the book's "pounce harder at high bids"
  framing. **Ledger row promoted: `underpowered` ->
  `context-limited`** (rough proxy on aggregate Q delta, not yet
  snapshot-level paired contrast). Wave 2.E.2 (bead `t42-8kbh`) will
  produce snapshot-level evidence.

Status counts after Wave 2.B.2: supported 23, context-limited 16,
underpowered 19, not-yet-tested 4, contradicted 2. Two rows promoted
into the active-evidence pool; the campaign now has its first
non-trivial movement out of `not-yet-tested` and `underpowered` based
on paired counterfactual evidence.

## Wave 2.G — First `supported` Promotion

[[w42-bookval-v1-wave2-ch02-multistep]] extended the bid=32-vs-bid=30
result to all 5 adjacent step pairs in {30, 32, 35, 36, 39, 42}.
Headline:

- All 5 step pairs show overbid penalty in mark_ev with 95% CIs
  excluding zero (deltas `+0.07` to `+0.15`, n=8,168-10,052 per
  step).
- Cohen d grows monotonically `0.16` → `0.47`.
- All **85 of 85** slice cells (decl × seat × phase × step) support
  book direction.
- Transitive 30→42 cumulative matches sum-of-steps within 0.81%
  (additive).

**Ledger row promoted: `ch02-bid-only-enough` `context-limited` →
`supported`.** This is the campaign's first promotion above
`context-limited` from a non-supported start. Status counts after
Wave 2.G: supported 24, context-limited 15, underpowered 19,
not-yet-tested 4, contradicted 2.

The remaining caveat is intentional: cross-contract bid choice
(different declarations at different bids) and full auction-policy
response remain Wave 3 work.

## Wave 2.E.2 — High-Bid Pounce CONTRADICTED (with Methodology Lesson)

[[w42-bookval-v1-wave2-pounce-high-bid]] tested
`ch12-setter-pounce-high-bid-off` at the snapshot level on 1,140
high-bid pounce-eligible positions (bids 35/36/39/42). Result:

- Pooled EV delta `-10.42` (CI `[-11.25, -9.59]`); pooled p_set delta
  `-0.047` (CI `[-0.056, -0.038]`).
- All 4 bid buckets contradict: pounce-better fraction 21.6%-26.6%
  per bid.
- Effect is 3x larger in magnitude than at bid=30 (Wave 2.E).

**Ledger row demoted: `ch12-setter-pounce-high-bid-off`
`context-limited` -> `contradicted`** (the earlier Wave 2.B.2
promotion was based on an aggregate per-team Q-delta proxy and is
reversed by paired snapshot-level evidence).

**Methodology lesson** worth pulling forward: the Wave 2.B.2 promotion
relied on aggregate Q-delta showing setter teams perform better at
high bids in book direction. The Wave 2.B.2 agent flagged this as a
"rough proxy" requiring snapshot-level confirmation; the orchestrator
promoted anyway because the CI was clean. Wave 2.E.2's snapshot-level
paired evidence reverses the verdict completely. The aggregate Q-delta
and the paired pounce-vs-decline contrast are fundamentally different
objects:

- Aggregate: setter-team mean Q across all chosen actions in games at
  bid X. Reflects general game dynamics (high bids over-commit
  bidders, helping setters in expectation regardless of pounce
  decisions).
- Paired: same-snapshot pounce vs decline. Tests the local action
  question.

Going forward, **aggregate proxies should never trigger ledger
promotion**; only paired same-snapshot or same-decision contrasts on
the relevant action shape qualify. This is added to the agent contract
in `w42/book_validation_v1/AGENTS.md` for future waves.

Status counts after Wave 2.E.2: supported 24, context-limited 14,
underpowered 19, not-yet-tested 4, contradicted 3.

## Wave 2.H — Mark-Multiplier Threshold Insight

[[w42-bookval-v1-wave2-ch10-action-level]] reframes the
mark-multiplier story. Key findings:

- **71% of decisions at bid=42** have a different `mark_ev` top-1
  action than at bid=30 (n=14,000 paired decisions). Action-flip rate
  grows monotonically through bids 32/35/36/39/42 then **plateaus at
  bid=84**.
- **The plateau exists because both bid=42 and bid=84 share
  `threshold_q=42`**. The 2× mark multiplier at bid=84 cannot change
  argmax (positive affine transform); the strategic effect operates
  through `threshold_q` recomputation, not the multiplier scalar.
- **`mark_ev ≡ p_make` on top-1 at every bid** — Wave 1.2's algebraic
  identity confirmed structurally and across the bid-aware corpus.

This is a refinement of the book's framing: the mark-multiplier story
is really a **threshold-q story** in disguise. Several other
"mark-aware" chapter claims (set severity compression, special-bid
multipliers, defender erasure under marks) likely operate through
threshold-q recomputation rather than the multiplier scalar; future
analyses should test that hypothesis.

The row stays `supported` (already there); the evidence base is now
wider. No status counts change from this finding.

## Wave 3.0 — Utility-Lens Meta-Analysis (Reconciliation)

[[w42-bookval-v2-utility-lens-synthesis]] re-processed all seven closed
Wave 2 probes through 5 utility lenses (EV, p_make, mark_ev, CVaR_10,
robust_q25). The result **substantially narrows the p_make/EV split
hypothesis** that Waves 2.E and 1.4 had projected onto the broader
campaign:

**Headline:** of seven claims tested across multiple utilities, only
**one** shows a true objective-dependent verdict relevant to model
training signal:

| Claim | EV | p_make | Other | Type |
|-------|----|--------|-------|------|
| ch05-void-creation-follow | **supported** | spans_zero | mark_ev/CVaR_10 spans_zero | **Soft flip** (model-relevant) |
| ch05-void-creation-lead | contradicted | contradicted | CVaR_10 spans_zero | Soft flip (CVaR only — noise) |
| ch12-setter-pounce-bid30 | spans_zero | spans_zero | — | No flip (claim unresolved at all utilities) |
| ch12-setter-pounce-high-bid | contradicted | contradicted | mark_ev/CVaR_10 contradicted | **Unanimous contradicted** |
| ch02-bid-only-enough | (missing) | supported | mark_ev supported | Unanimous supported |
| ch03-reentry-preservation | spans_zero | spans_zero | mark_ev/CVaR_10 spans_zero | Unresolved at all utilities |
| ch04-low-trump-trap | contradicted | (missing) | (missing) | Single-utility evidence |

No claim flips between `supported` and `contradicted`. All "flips" are
between a verdict and `spans_zero` — i.e. one utility detects a signal
and another doesn't.

**Updated read on the p_make/EV thread:** the campaign's earlier
formulation ("the book may encode p_make-optimized advice at the contract
threshold, with EV-optimal play differing in some sharp-threshold
positions") was right *at the claim level* (Wave 2.E setter-pounce-bid30
showed it; Wave 1.4 cross-AI matrix flagged it) but **does not generalize**
to the closed probe set. The cleanest expression of the split is now
ch05-void-creation-follow, where EV-greedy void creation buys the setter
+0.77 EV but doesn't move P(make the set).

The earlier strong claim that "high-bid pounce is a p_make/EV split"
(Wave 2.E.2) is **superseded**: the Wave 3.0 re-analysis shows pounce-
high-bid is contradicted under all 4 available utilities (`-10.42` EV,
`-0.047` p_make, `-0.047` mark_ev, `+4.40` CVaR_10 — all CIs exclude
zero in the bidder-helps direction). The book is wrong here irrespective
of objective. This was hidden in Wave 2.E.2 because that probe only
reported EV.

**Caveat (algebraic identity reminder):** for high-bid probes where
mark_multiplier > 1 (bid=84), `mark_ev` should differ from `p_make` by
the multiplier scalar. Wave 3.0 reports them as identical for several
high-bid claims; this is consistent with the dataset being dominated by
mm=1 cases (bids 30-42 where mm=1) but is worth verifying when the
ledger schema is populated. The argmax-equivalence still holds (positive
affine), so verdicts are correct; only effect *sizes* would shift.

### Schema decision

The Wave 3.0 agent recommended adopting per-utility status columns
(`ledger_status_ev`, `..._p_make`, `..._mark_ev`, `..._cvar_10`,
`..._robust_q25`, `utility_flip_flags`).

**Orchestrator decision: ADOPT-DEFERRED.** Schema is the right shape but
populating it requires probes to record p_make / mark_ev / CVaR /
robust_q25 alongside EV. Five of seven closed probes are missing two or
more utilities. Adoption now would create a sparse ledger; adoption
later (after the next probe wave records all five utilities by default)
yields a complete ledger. Action: add a probe-output contract
amendment to `AGENTS.md` requiring all five utilities in
`paired_contrasts.csv`, and revisit schema after the next 3-5 probes
land with full coverage.

### Implications for model design (initial reading — see Wave 4.0 correction below)

The Wave 3.0 reconciliation initially read the evidence as: "ch05-void-
creation-follow is the only situation in the validated claim set where
an EV head and a p_make head would disagree on what to learn."

That read was correct at the *paired-contrast / ledger-verdict* level
but **misleading at the policy-action level**, as Wave 4.0 immediately
demonstrated.

Status counts are unchanged by Wave 3.0 (it is an analysis of existing
verdicts, not a new probe): supported 24, context-limited 14,
underpowered 19, not-yet-tested 4, contradicted 3.

## Wave 4.0 — Utility-Argmax Divergence (Architecture-Decision Gate)

[[w42-bookval-v3-utility-argmax-divergence]] (`t42-hmjr`) measured
argmax-under-utility for ALL legal actions on the same 500 ch05-void-
creation-follow snapshots that drove Wave 3.0. The motivating question:
do EV-greedy and p_make-greedy policies actually pick different actions
at these snapshots, or only differ in contrast-magnitude?

**Result: EV-argmax and p_make-argmax disagree on 206/500 snapshots
(41.2%, CI [36.8%, 45.6%])** — an order of magnitude above the 5%
gate. The Wave 3.0 paired-contrast was *not* a magnitude-only
artefact; the policy-action divergence is real and substantial.

Full pairwise disagreement matrix (n=500):

| pair | disagree rate | 95% CI |
|------|---:|---|
| EV vs p_make | 41.2% | [36.8%, 45.6%] |
| EV vs mark_ev | 41.2% | [36.8%, 45.6%] (= p_make at bid=30) |
| EV vs CVaR_10 | 43.0% | [38.6%, 47.4%] |
| EV vs robust_q25 | 29.6% | [25.8%, 33.8%] |
| p_make vs CVaR_10 | 44.0% | [39.8%, 48.4%] |
| p_make vs robust_q25 | 38.4% | [34.2%, 42.8%] |
| CVaR_10 vs robust_q25 | 30.8% | [26.8%, 35.0%] |
| **p_make vs mark_ev** | **0.0%** | [0.0%, 0.0%] |

The p_make ≡ mark_ev identity at bid=30 (Wave 1.2 / Wave 2.H affine
algebra) is **empirically confirmed at the argmax level** — every
single one of 500 snapshots has identical p_make and mark_ev
selections, as the positive-affine algebra demands.

### Framing inversion (important)

The void/preserve confusion table inverts the intuitive Wave 3.0
framing ("EV likes void, p_make is neutral"):

| utility | argmax = void | argmax = preserve | argmax = neither |
|---|---:|---:|---:|
| EV | 29.4% | 33.0% | **37.6%** |
| p_make | 38.6% | 31.0% | 30.4% |
| mark_ev | 38.6% | 31.0% | 30.4% |
| CVaR_10 | **42.6%** | 27.8% | 29.6% |
| robust_q25 | 40.6% | 30.6% | 28.8% |

p_make picks the void slot *more often* than EV does (38.6% vs 29.4%).
The risk-aware utilities (CVaR_10 at 42.6%, robust_q25 at 40.6%) pick
void most aggressively of all. **EV is the outlier** — it more often
selects a third action that is neither void nor preserve (37.6%, the
highest "neither" rate of any utility).

The literal "EV→void AND p_make→preserve" canonical pattern accounts
for only 5.8% (29 / 500) of snapshots — meaningful, but only ~14% of
total EV/p_make disagreement. The remaining ~36% is composed of other
slot-pair disagreements.

**Corrected mechanistic read:** the book's void-creation advice (in
follow position) aligns with **risk-aware utilities** (p_make / CVaR /
robust_q25) more than with **mean-EV**. EV-greedy strategy on this
corpus often selects a third-option discard that is even better than
void in EV terms but worse in tail / threshold terms. Voiding gives
*optionality* (you can trump that suit later), which improves tail
outcomes more than mean — a sensible structural reason for the split.

This refines the Wave 3.0 reconciliation: the multi-objective story is
NOT narrowed to one claim. It is **broadened to a 30-44% policy-action
divergence between EV and risk-aware utilities** on a corpus where the
book's advice has been validated. Wave 3.0 missed this because paired
contrasts measure magnitudes on specific action pairs, not what each
utility's argmax actually picks.

### Architecture decision gate: TRIPPED

The Wave 4.0 gate said: ≥5% disagreement with a coherent pattern →
scope rung-2 (utility-tunable searcher). The actual disagreement is
~10× the gate (41% vs 5%) and the pattern is coherent (EV is the
outlier; risk-aware utilities cluster). The gate is decisively
tripped. Whether to actually build rung-2 is now a separate
prioritization decision (held for orchestrator/user review), but the
evidence base is in place.

## Lens v1 — Head-to-Head Game Outcomes (Who Actually Wins?)

[[w42-lens-v1-utility-head-to-head]] (`t42-4ouu`) is the cheap
1-step-greedy player that lets us play actual games with each utility
and count points. The build was much smaller than the rung-2 MCTS we
were originally scoping (~150 LOC + reuse of the Zeb eq-vs-eq batched
path), and the round-robin completed in 7 minutes wall on M5 Max
(parallel-hand K=500, N=10 per Q-query, fp32 — fp16 sanity passed but
MPS doesn't autocast inside the model forward).

Round-robin: {ev, p_make, cvar_10, robust_q25} × 6 pairings × 1000
hands paired-seed (mark_ev excluded — affine-identical to p_make at
bid=30; sanity matchup confirmed). Bid forced to 30 to stay aligned
with the Wave 4.0 corpus.

| Team A | Team B | mean margin / hand | 95% CI | A win rate |
|---|---|---:|---|---:|
| **ev** | p_make | **+5.42** | [+4.03, +6.81] | 59.5% |
| **ev** | cvar_10 | **+3.98** | [+2.55, +5.45] | 55.2% |
| **ev** | robust_q25 | **+2.49** | [+1.09, +3.98] | 56.2% |
| p_make | cvar_10 | −1.91 | [−3.35, −0.45] | 44.3% |
| p_make | robust_q25 | −2.97 | [−4.52, −1.51] | 44.3% |
| cvar_10 | robust_q25 | −1.71 | [−3.15, −0.27] | 47.4% |

**Every CI excludes zero. Total ordering: ev > robust_q25 ≳ cvar_10
> p_make.** The Wave 3.0 paired-contrast +0.77 EV gain on void-vs-
preserve translates into a +5.42 pts/hand head-to-head EV advantage at
the *full game* level — about 7× larger in normalized magnitude.

### The inversion that matters

This **inverts the natural reading of Wave 4.0**. Wave 4.0 said: "the
book + p_make / CVaR / robust_q25 cluster on void; EV is the outlier
preferring third-option discards." Lens v1 says: **EV's third-option
discards are not noise — they win games.** Every risk-aware utility
loses to mean-EV head-to-head, on every paired comparison, with CIs
that exclude zero.

The book aligns with the *worst-scoring* utility on this corpus
(p_make, −5.42 vs ev) on the specific scenario where Wave 3.0 / Wave
4.0 measured. Two interpretations both consistent with the data:

1. **The book's void-creation advice is locally correct but globally
   suboptimal in EV terms** — the action it endorses is +0.77 EV vs
   the named alternative (Wave 3.0), but EV-greedy finds an action
   that's even better than both, ~5 pts/hand cumulatively across a
   game.
2. **p_make-greedy plays globally suboptimally because it's the wrong
   meta-objective for this corpus** — bid=30 contracts are
   structurally ~always-make, so optimizing for "did I make it?"
   collapses to nearly indifferent action selection over a wide range
   of legal plays. Optimizing for mean score (EV) breaks the tie in a
   point-rewarding direction.

The state-conditioned utility hypothesis ([[w42|t42-nwuu]] filed
2026-05-03) becomes more interesting under this finding: p_make may
not be globally worse, just worse on the bid=30 setter-defense corpus
that dominates these games. The book may encode state-dependent
utility selection that no single fixed utility captures — ev wins on
average, but a state-conditioned policy that switches utilities by
game state could in principle beat fixed-ev.

### Sample-sweep confirmation (N=10 is the right operating point)

ev vs p_make at N ∈ {10, 50, 100}, 250 hands each: margins +6.18,
+4.54, +3.13 — all CIs exclude zero, ranking preserved. Magnitude
drops with N (more sample noise at low N inflates margins) but
direction is robust. **N=10 is the right default for head-to-head
ranking work.** This empirically extends the Zeb-era N=10 ≈ N=100
finding from `forge/zeb/OVERVIEW.md` to Lens-vs-Lens matchups, where
both sides are Q-driven and noise might correlate.

### Production-code follow-up

Non-obvious: **`forge.eq.generate.actions.select_actions` is essentially
Lens(p_make)** (it picks p_make-argmax with EV as tie-break). The
production E[Q] action selector is therefore using the **worst** of
the four utilities tested by Lens v1. Switching it to ev-argmax is a
one-line change. Hypothesis: this would lift the existing E[Q]-N=100
vs Zeb-Large win rate (currently 55.7%, see `forge/zeb/OVERVIEW.md`)
toward whatever the +5.42 pts/hand head-to-head advantage translates
to in Bradley-Terry Elo. Filed as a separate follow-up bead so it
doesn't get lost.

### What rung-2 (MCTS over forge) would now answer

Lens v1 already produced the headline answer ("ev wins"), so rung-2
moves from "is utility-conditioning worth building?" to a more
focused question: **does deeper search amplify or attenuate the
EV-over-p_make gap?** Plausible answers:

- **Amplify:** EV's "third-option" picks are good *because* they set
  up better future-tree positions; deeper search would find this
  endogenously and the gap widens.
- **Attenuate:** EV's wins on this corpus are because forge's
  one-step Q already captures most of the relevant information;
  deeper search converges all utilities toward minimax and the gap
  shrinks.

Either way is a finding. Worth building only if there's appetite for
the next layer of evidence — Lens v1 already answered the build-or-
kill question for the multi-utility *architecture* thread.

### Disaster utility — confirms EV is the ceiling for fixed pointwise utilities

A user-suggested utility was added and tested: `disaster` clips every
sub-threshold sample's Q value to −42, then takes the expectation
under the PDF (so it matches EV above threshold and floors everything
below). Head-to-head 1000 paired-seed hands, N=10, fp32:

- disaster vs **ev**: −1.55 pts/hand, CI [−3.07, +0.05] (just barely
  loses; CI grazes zero)
- disaster vs **p_make**: **+2.74**, CI [+1.24, +4.37] (clearly wins)
- disaster vs **robust_q25**: +0.32, CI [−1.13, +1.81] (tied)

Implied ordering: `ev ≳ disaster ≳ robust_q25 ≳ cvar_10 > p_make`.
**Disaster keeps the part of EV that matters most (continuous reward
above threshold) and only loses the damage-control information on
losing hands. Cost vs EV: ~1.5 pts/hand.** No fixed *pointwise* utility
distinguishably beats EV at one-step lookahead — EV is by construction
the maximum-information summary of the per-action outcome distribution,
and any other pointwise utility either throws information away (hard
cliff, p_make) or imposes a fixed reward shape that EV can already
represent (soft cliff = EV minus a bias term on a region; same argmax
modulo a constant). Artifacts: `w42/lens_v1/results/disaster_head_to_head.csv`,
`w42/lens_v1/run_disaster.py`. Detail: [[w42-lens-v1-utility-head-to-head]].

## Methodology insight — the single-decision blind spot

The deepest finding of this campaign is not in any single probe; it
is a property of the campaign's measurement shape itself.

**Most book claims are multi-step plans, but most probes are single-
decision contrasts.** "Lead a singleton on trick 1 to set up a void
by trick 3." "Hold trump 4 until trick 5 to catch the queen." "84-
throwaway-ladder across the last 3 tricks." "Reentry preservation
across the next four tricks." "Pounce-window timing." These are
sequences. The decision-quality of each individual move within a
plan is often *worse* than the locally-greedy alternative — that is
the whole point of a setup move. The plan pays off downstream.

Single-decision EV (and Lens(EV) by extension) literally cannot see
this. EV at one decision asks "given everyone plays default future
moves, what is the average outcome of action X?" The "default future
moves" inside the forge oracle do not include the bidder's own plan.
So EV evaluates each move as if no one (including itself) is plotting
two tricks ahead.

This reframes a lot of campaign findings retroactively:

- **Wave 4.0's 41% EV vs p_make argmax disagreement on void-creation-
  follow** isn't "EV picks weird things." It might be "EV picks the
  locally-best move; the book picks the move that *sets up the next
  two tricks*; one-step Q can't tell the difference between those
  because the rollout assumes default future play by both sides."
- **The 19 underpowered + 14 context-limited claims in the ledger**
  are mostly multi-step strategy claims tested at single-decision
  granularity. They may be stuck not because the book is wrong but
  because the methodology is too zoomed-in for the claims' real shape.
- **Lens v1's "ev wins by +5.42 vs p_make"** measured EV's *individual
  moves* against p_make's *individual moves*. Of course EV won that
  contest — it's by construction the best per-decision thing. The book
  was not in that contest at all; the book plays plans, not moves.

So "EV wins" is a finding bounded by the test's abstraction level. To
fairly test the book's actual strength, **the campaign needs planning-
aware probes** (probes whose contrast unit is a multi-decision
sequence, not a single action). Three architectures could supply this:

1. **MCTS over forge** — generic K-deep planner, finds any sequence
   the simulator supports. Heavyweight build (~1-2 days). Answers:
   does any planning beat one-step EV? Almost certainly yes; the
   interesting number is by how much.
2. **Lookahead-Lens** — cheap K=2 or K=3 step lookahead in the same
   Lens framework. ~300 LOC, half a day. Captures most of the planning
   benefit if book plans are short (which most are: 2-3 tricks).
3. **Book-strategy player** — for each named book strategy, hand-code
   the multi-step policy and play it head-to-head against EV-greedy.
   Cost: low per strategy (~1-2h each), but you have to know which
   strategies to encode. Tests "is THIS specific book strategy
   point-positive?" — claim-by-claim, the actual question the campaign
   has been trying to answer.

**For the book validation campaign specifically, #3 is the most directly
useful.** #1 or #2 would prove planning generically beats EV (almost
certainly true) but would not tell us *which book claims* are right.
#3 tests the book directly. The deferred Wave 2.F (84-throwaway,
[[w42|t42-wikw]]) is exactly this shape and now has a sharper
motivation than when it was deferred — it's the campaign's first
concrete multi-step strategy to encode and test.

For broader model-design questions, #2 is the cheaper general-purpose
tool.

This insight is the natural Wave 5 frontier: stop testing book claims
at the single-decision granularity for the strategy-shaped ones, start
encoding the strategies and testing the *plan*.

### The architectural payoff: strategies are not just probes, they are training data

Detailed in [[book-strategy-player]]. Brief version: every decision the
strategy-player makes is logged with `(game_state, plan_state,
applicable_strategies, priorities, chosen_strategy, chosen_action,
fallback_action, hand_outcome)`. The `fallback_action` is a *free
counterfactual* — what Lens(ev) would have done at the same state — so
each decision is naturally a paired sample. Multiplied across ~28
decisions per hand × thousands of hands, this is a structured labeled
dataset for training a learned strategy selector (Model A) or even an
end-to-end policy (Model C), without any extra simulation cost.

Three models become trainable from the same recorded data:

- **Model A — strategy selector.** Replaces hand-crafted `priority(gs, ps)`
  with a learned head. Same framework, learned arbitration. [[burl]]
  becomes a candidate (small structured action space + narrative
  reasoning is exactly Burl's shape); [[gus]] becomes the input encoder
  (this is what Gus was designed for — feeding decision-time models).
- **Model B — plan-success predictor.** P(plan completes) and E[points
  if it does], per (state, strategy). Useful as input to A or as an
  early-bail signal.
- **Model C — end-to-end policy distillation.** Bypasses the library at
  inference. Strategies become *training scaffolding*. Discovery
  side-effect: clusters of fallback-invoked decisions where the trained
  selector deviates from hand-crafted priorities are candidate new
  strategies nobody wrote down.

This is the cleanest path past the EV ceiling that the campaign has
identified. Multi-utility heads are dead (Lens v1). Soft-cliff utilities
are dead (disaster). Generic MCTS is expensive and doesn't directly use
book wisdom. Lookahead-Lens is impractical (4-player branching). The
strategy-selector path uses the book's encoded plan-shaped wisdom as a
structured action space, trains on cheap self-play with free
counterfactuals, and the deployed model has planning capability without
paying MCTS's branching cost.

[[book-strategy-player]] has the full architecture: Strategy protocol,
five composition modes (state-conditioned, chaining, hierarchical,
opponent-aware, cross-hand), DecisionRecord format, phased build plan,
and the role-resurrection of [[burl]] / [[gus]] / [[zeb]].

## Links

### Caveats (carried forward)

- All 500 snapshots are bid=30 (mm=1, mark_ev ≡ p_make by construction).
  The cleanest test of mark_ev divergence requires a bid=84 snapshot
  corpus. Wave 2.F (84-throwaway) is the open path.
- All to-act players are setters at bid=30, where the make-threshold
  (Q ≥ −17) is loose; CVaR_10 / robust_q25 do more discrimination here
  than they would on offense. A bid-aware mixed-position corpus is the
  natural follow-up scope.
- 100 worlds per snapshot leaves sample noise on near-tied slots.
  Bootstrap CIs quantify snapshot-level variance, not per-snapshot
  resampling variance.
- Read-only on the claim ledger; no row moves.

## Links

[[w42]] | [[w42-phase4-final-claim-audit]] |
[[w42-phase3-joined-claim-row-model-table]] |
[[w42-phase4-sequence-handshape-tests]] |
[[w42-phase4-84-dynamic-seed-tests]] |
[[w42-phase4-doubles-notrump-regime-tests]] |
[[w42-phase4-bidding-count-exposure-tests]] |
[[w42-phase4-scoring-objective-tests]] |
[[w42-phase4-laydown-rule-accounting]] |
[[w42-bookval-v1-wave1-distribution-lens-reranker]] |
[[w42-bookval-v1-wave1-mark-utility-transform]] |
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] |
[[w42-bookval-v1-wave1-cross-ai-agreement]] |
[[w42-bookval-v1-wave1-independent-audit]] |
[[w42-bookval-v2-utility-lens-synthesis]] |
[[w42-bookval-v3-utility-argmax-divergence]]
