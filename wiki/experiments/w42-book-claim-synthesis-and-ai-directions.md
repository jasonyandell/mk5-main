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

## Current Bottom Line

The book taught the project a vocabulary for action reasons. The confirmed
parts are mostly gated and tactical, not slogans. The unconfirmed parts are now
precisely technical: state injection, high-bid generation, real auction policy,
population play, and objective-conditioned utility.

That is a good outcome. The book did not become rules; it became a test suite,
a feature vocabulary, and a roadmap for better agents.

## Links

[[w42]] | [[w42-phase4-final-claim-audit]] |
[[w42-phase3-joined-claim-row-model-table]] |
[[w42-phase4-sequence-handshape-tests]] |
[[w42-phase4-84-dynamic-seed-tests]] |
[[w42-phase4-doubles-notrump-regime-tests]] |
[[w42-phase4-bidding-count-exposure-tests]] |
[[w42-phase4-scoring-objective-tests]] |
[[w42-phase4-laydown-rule-accounting]]
