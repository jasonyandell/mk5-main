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
[[w42-bookval-v1-wave1-independent-audit]]
