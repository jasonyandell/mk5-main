---
title: w42 Book Claim Synthesis And AI Directions
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-10-12
status: complete
---

**This is the canonical rollup of the Winning 42 book-validation work** — the
one page to read for what the campaign found. The closure baseline is
[[w42-phase4-final-claim-audit]]; the wave-by-wave record is
[[w42-book-validation-campaign]] (dormant since Wave 5); the curated path is
[[w42-book-validation]]. Which clock is this: `phase2/3/4` below is the
2026-05-02→03 claim-ledger sweep; `Wave 1–5` and bookval `v1`/`v2`/`v3` are the
2026-05-03 campaign (v2/v3 are utility-lens passes, not versions of the v1
corpus); neither numbering relates to the 2026-06+ [[champion-ladder]] rungs.

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

No strategic claim from the book was established. The `supported` rows are
arithmetic, rule fixtures, scoring definitions, two near-tautological negative
controls, one tautology and one null; the evidence is assessed in
[What the evidence is worth](#what-the-evidence-is-worth) below. Adding book
detector tags to a public-safe legal-action row model
([[w42-phase3-joined-claim-row-model-table]]) moved best-mean match from
`64.464%` to `68.107%` (mean regret `1.3598` → `1.1257`) on one split with no
CI; random legal play scores 56.5% on that split because 34.6% of decisions are
forced, and on non-forced decisions the tagged model is about 51% against 34%
random. The tags are weak features, not validated book strategy.

## What the evidence is worth

**Instrument.** Every oracle-era verdict came from E[Q] labels produced by a
learned Q network (`domino-qval-large-3.3M`, qMAE 0.94 points) averaged over
50–4800 sampled worlds, not from an exact solver. Every tactical delta is in
points (Q on the −42..+42 scale), not marks. Every tactical corpus is bid 30
with the bidder forced to seat 0, played by p_make-greedy E[Q] at n=10, 100
deals (the eval set: 20 deals); every bootstrap resamples decisions, never
deals, so the CIs are too narrow. The April Gus v2 corpus behind the phase-3/4
contrasts was later found 27–67% malformed ([[otis-phase-r]]); no w42 contrast
was rerun on the regenerated corpus.

**"Supported" means arithmetic.** Of the 24 `supported` ledger rows, 20 are
arithmetic, rule fixtures or scoring definitions. Two (unsafe partner donation,
reckless count to the bidder) are near-tautological negative controls: the
cost equals the count given away (5-count −6.0, 10-count −13.8 Q).
`ch02-bid-only-enough` is a tautology (below). `ch05-pounce-count-before-certainty`
is null on its literal label, +0.44 [−0.20, +1.10] Q at n=525
([[w42-phase4-sequence-handshape-tests]]); the +4.10 it was promoted on pools
100 last-seat pairs where the win is already certain. No strategic claim from
the book was established by a test that could have failed and that measured
the book's act. Two of the arithmetic rows are wrong (odds, below).

**Bid-only-enough cannot fail.** `forge/bidding/estimator.py:72-100` scores
every bid from the same points sample against a rising threshold, so
P(points ≥ bid) can only fall as the bid rises, and under one-mark scoring
`mark_ev = 2·p_make − 1` falls with it. "85 of 85 slice cells", "0/14,976
positive-margin rows" and the Wave 2.G promotion restate that arithmetic. The
book's actual auction claim (outbid by the minimum; the 30/31/35/36 lattice;
what a raise tells the table) was never tested; the ledger row's own result
field still says no auction-counterfactual artifact exists.

**Setter-side probes withdrawn.** The void-creation lead and follow probes and
both pounce probes (bid 30, high bid) treat E[Q] as Team 0's view and negate
it for the setter; forge documents E[Q] from the acting player's view
(`forge/eq/generate/actions.py:113,160`). The sign is unverified on data.
Their verdicts, including "high-bid pounce contradicted", are withdrawn
pending a sign check (#122).

**The other "contradicted" rows are not findings.** The virtual boss
(`ch04-effective-double-highest-remaining`) is one of 12 uncorrected proxies,
n=21 from 20 deals: inconclusive. `ch08-either-matching-double-not-two-to-one`
refutes a reading the book does not make: not a test.

**Agreement rates have a high chance floor.** On the 28,000-decision cross-AI
corpus 35.7% of decisions are forced; a random legal move matches the EV
argmax 57.3% of the time overall and 40.1% on detector-active decisions.
Detector–EV agreement (43.7%; ch04 48.7%) is chance level. The raw
public-state baseline's 56.6% is the 56.2% random rate.

**Odds.** Ch16's four-trump 37% and 52% assume 27 equiprobable placements of
the missing trumps; they are not equiprobable. The exact figures are
33/95 = 34.7% and 47/95 = 49.5% ([[book-study-2026-10-12]]).

**Coverage.** 64 ledger rows against at least 277 advice sentences in the
book. Chapters 1, 6, 11, 13, 14 and 15 got zero rows; Ch12 (Advanced Bidding &
Playing, about 30% of the strategy text) got 2; Ch10 (one advice sentence)
got 9. The book's worked example hands were never used as test positions (#123).
Results at n=526 (lead-away, −3.43 Q) and n=1866 (count-calling, +3.03 Q)
sit on the chapter pages and never reached their ledger rows, which still read
`underpowered` at n=22 and n=51. `claims.csv` itself is hand-edited, its
generator would revert it, and its summaries disagree with it
([[w42-phase2-statistics-claims-ledger]], #125).

**Diagnosis.** A perfect-information evaluator plans perfectly within each
sampled world. What it lacks is information-set consistency: it fuses
strategies across worlds the player cannot tell apart. That is strategy
fusion ([[the-wall]]), not the "single-decision blind spot" this page's
methodology section names. The book was restudied with an
imperfect-information player in [[book-study-2026-10-12]].

## What The Phase-2/3/4 Pages Recorded

Read every point-delta below through [What the evidence is worth](#what-the-evidence-is-worth):
E[Q] network labels, points, bid 30, unclustered CIs, the malformed April corpus.

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

[[w42-phase4-sequence-handshape-tests]] measured the gated version of several
folk rules, in oracle points:

- commanding called doubles are good bidder leads;
- blanket non-double called-suit leads are bad against off alternatives;
- partner closure count donation is strong;
- broad partner count donation before closure is weak;
- partner count-liability leads are bad;
- setter count-calling leads and pounce take-count-now score positive, but the
  literal pounce-before-certainty label is null (+0.44 [−0.20, +1.10], n=525);
- reckless setter count into bidder control is bad (a near-tautological
  negative control).

None of these is a measurement of the book's act under uncertainty; most are
class-vs-class contrasts on positions the oracle reached.

### 84 and endgame preservation

[[w42-phase4-84-dynamic-seed-tests]] found that preserving live
weapons/protectors scores more points than spending them on reached seed states
(+1.946 Q, no CI). An 84 is all-or-nothing, and the threshold-mass deltas are
−0.0004, +0.003 and 0.0: the effect on make/set is zero.

### Doubles and no-trump

[[w42-phase4-doubles-notrump-regime-tests]] (greedy simulation) agrees with the Chapter 9 caution
against "many doubles means doubles-trump." In generated same-hand paired
regime tests, no-trump often beats doubles-trump for four-plus-double hands.
High/top double control is the near-flat pro-doubles slice; missing top doubles
and no-trump support doubles favor no-trump.

### Bidding and risk

[[w42-phase4-bidding-count-exposure-tests]] measured static/generated
contract versions of Chapter 2 (greedy Monte Carlo, bidder's hand fixed):

- three-plus trumps raise simulated p_make (+0.293, no CI, confounded with
  declaration choice; untested causally);
- risk <=12 is better but only modestly, so it is not a standalone rule;
- four/five off exposure is prevalent; its outcome effect is underpowered;
- double-ahead protection is side-specific, not global;
- partner two-plus-double prior is about `58%`;
- natural max-profitable bid buckets exist, but are not a full auction story.

[[w42-phase3-auction-bid-discipline-corpus]] found no positive-margin bid
improving over the minimum winning bid. That cannot fail: every bid is scored
from the same points sample against a rising threshold. The auction claim was
never tested.

## Not Confirmed Yet

Nothing strategic is confirmed. The untested claims below were blocked by the
corpus (bid 30 only, no auctions, no 84 games) and the generator:

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

[[w42-phase4-final-claim-audit]] records these as bounded future work. Closure
meant "has an owner or a stated blocker", not "was tested".

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

- **Scalar EV and another utility pick differently in 64.9% of decisions on
  the seed-9430 branch atlas**; the alternative pick never beats EV on EV
  terms, which is true of any argmax by definition. Median EV cost when
  alternative wins is +0.18-0.26 points. Disagreement is pure risk-framing.
- **CVaR_10 and robust_q25 agree with EV at 82-83% on one deal (n=280); p_make
  and threshold_mass at 59%.** The reranker page's own figures are 74.4% /
  70.1% and 61.1%. Forced moves count as agreement and no chance baseline was
  computed (random legal play matches the EV argmax 57.3% of the time on the
  cross-AI corpus); one in four or five decisions disagree. This does not show
  that tail-aware utilities track the oracle, and says nothing about the book.
- **No-trump shows 90% EV-lying rate**, doubles-trump 39% CVaR disagreement.
  Clean detector signals for "scalar EV is the wrong target here."
- **mark_ev is an affine function of p_make at bid=30 with one-mark
  multiplier** (by construction). The Ch 10 multiplier (`ch10-special-bid-mark-multiplier`) is
  unreachable on the fixed-bid corpus. A bid-aware E[Q] generator is now a
  hard prerequisite for further mark-objective work.
- **Genuine mark-vs-point flips are 4.3% of decisions** (12/280, one deal); 81.5% of
  nominal flips are surface-flattening artifacts of the binary {-1, 0, +1}
  mark transform near the 30-point make threshold.
- **The "one tile decides the plan" rhetoric is real but rare**: 1.85% of
  decisions show >=60% impact concentration on a single hidden tile.
- **Trump-count tiles were "100% directionally helpful"** on 20 cases from 3
  hands, every one a tile held by the actor's partner (knowing partner holds
  trump raises Q); over the top-5 rows the split is 39 helpful vs 35 harmful.
- **5-5 in twos and 4-4 in no-trump are the highest-impact non-trump
  load-bearing tiles** - low-pip declarations promote the highest off-suit
  double into the strategic role usually held by trump.
- **Setter seats are asymmetric**: right-setter load-bearing tiles are ~30%
  trump doubles; left-setter is ~37% plain tiles. The book states the
  asymmetry itself (p.51).
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

The book gave the project a vocabulary for action reasons and a set of
label-free detectors. It did not give a single strategic claim that this
campaign established: see [What the evidence is worth](#what-the-evidence-is-worth).
The untested parts needed state injection, high-bid generation, real auction
policy, population play and, above all, an evaluator without strategy fusion.

Wave 1's claim that tail-aware utilities (CVaR, robust_q25) are a near-free
upgrade over scalar EV did not survive Lens v1 (below); mark-objective work is blocked on bid-aware
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

## Wave 2 Probes (closed)

[[w42-bookval-v1-wave2-reentry-v2]] — re-runs the original Wave 2.A
reentry probe on oracle-greedy snapshots from
[[w42-claim-data-inventory]]. **Phase-conditional verdict**: overall EV
delta is `-1.23` (CI `[-2.62, +0.16]`, consume direction); the late-game
slice (tricks 5-6, n=60) is statistically `contradicted` against the
book at CI `[-6.76, -1.17]` (consume strictly better); mid-game (n=162)
is underpowered (CI `[-1.79, +1.36]`); no early-game data in the corpus.
Overall the result is null. The late slice was chosen post hoc with no
multiplicity correction, and the book's main context (the early game) has
n=0: the claim was not tested where the book makes it.

[[w42-bookval-v1-wave2-low-trump-trap]] — paired contrast on 257
oracle-greedy snapshots. Overall delta is `-1.95` (CI `[-2.94, -1.07]`),
but the aggregate is dominated by positions where hoarding the dominant
trump is broadly correct, not the specific trap the book warns about.
**In the count-bearing subgroup (n=86, the actual book scenario)**,
mean delta is `-0.02` with CI `[-2.16, +2.15]`: null, and powered only for
effects of 3 points or more. The ledger reads `underpowered`.

[[w42-bookval-v1-wave2-pounce-window-bid30]] (n=52),
[[w42-bookval-v1-wave2-void-creation]] (lead, n=276) and
[[w42-bookval-v1-wave2-void-creation-follow]] (follow, n=500) are setter-side
probes whose scripts negate E[Q] for the setter on the assumption that it is
Team 0's view; forge documents it from the acting player's view. **Their
verdicts are withdrawn pending a sign check (#122).** The readings once drawn
from them (a bid-30 "p_make/EV split" on pounce; "lead-void contradicted,
follow-void survives" as a position-dependent reversal) are not findings.
Independent of the sign, the bid-30 pounce contrast is null (CI
`[-0.57, +6.75]`, 10-count slice n=5), and the follow-void effect (+0.77 Q)
is below the ≈0.93-point minimum detectable effect and null on the clean
subset (+0.08 [−0.37, +0.53]).

## Wave 2.B.2 Full Bid-Aware Sweep — Power Analysis

[[w42-bookval-v1-wave2-bid-aware-atlas]] now contains 259,618 action
rows from a 50-seed × 10-decl × 7-bid sweep on M5 MPS. Validation at
bid=30 vs [[w42-branch-atlas-scaled-v0]] passes 10/10. Power analysis
verdicts:

- `ch10-special-bid-mark-multiplier`: n=14,000 paired decisions per bid.
  The row is a definition of marks scoring; the action-level "flip" rate is
  an artifact (Wave 2.H below).
- `ch02-bid-only-enough`: bid=32 minus bid=30 gives `mark_ev` `-0.076` and
  `p_make` `-0.038`. A higher threshold on the same points distribution
  lowers p_make mechanically, so this could not have come out the other
  way. It was promoted `not-yet-tested` → `context-limited`, then to
  `supported` by Wave 2.G; neither move rests on a test.
- `ch12-setter-pounce-high-bid-off`: aggregate setter-team Q delta of
  `+0.376` at bid=39 and `+0.410` at bid=42 was taken as a proxy and the row
  promoted to `context-limited`. An aggregate over all chosen actions is not
  a pounce contrast.

The "power analysis" applied post-hoc labels ("sufficient" if n ≥ 100 and the
CI is narrow); no minimum detectable effect was computed.

## Wave 2.G — Bid-Only-Enough Multi-Step (a tautology)

[[w42-bookval-v1-wave2-ch02-multistep]] extended the bid=32-vs-bid=30
result to all 5 adjacent step pairs in {30, 32, 35, 36, 39, 42}: every step
lowers mark_ev (n=8,168-10,052 per step), "85 of 85" slice cells agree, and
the steps add up. `mark_ev = 2·p_make − 1` at one mark, and P(points ≥ bid)
cannot rise with the bid, so every cell had to agree. The pairing on
(seed, decl, decision_idx) also joins trajectories that diverge across bid
runs. The row was promoted to `supported` (status counts after Wave 2.G:
supported 24, context-limited 15, underpowered 19, not-yet-tested 4,
contradicted 2); the promotion rests on arithmetic, not on the book. The
book's auction claim was never tested.

## Wave 2.E.2 — High-Bid Pounce (withdrawn pending sign check)

[[w42-bookval-v1-wave2-pounce-high-bid]] compared two hand-picked actions
("lowest-cost winning play" vs "lowest-cost non-winning play") on 1,140
high-bid pounce-eligible positions (bids 35/36/39/42): pooled setter EV delta
`-10.42` (CI `[-11.25, -9.59]`). The row was demoted to `contradicted`
against the probe's own proposal of `context-limited` overall. The script
negates E[Q] for the setter on the assumption that it is Team 0's view;
forge documents it from the acting player's view. If the sign is inverted
the effect favours the pounce by the same margin. **The verdict is withdrawn
pending a sign check (#122).** Independently, the probe tests a different act
from the book's and does so with a perfect-information evaluator
([[w42-book-second-pass]]).

The rule drawn from this episode, that aggregate proxies should not trigger
ledger promotion, is sound on its own terms (an aggregate setter-team Q over
all chosen actions is not a pounce contrast), but the high-bid pounce
reversal is not evidence for it until the sign is checked.

Status counts after Wave 2.E.2: supported 24, context-limited 14,
underpowered 19, not-yet-tested 4, contradicted 3.

## Wave 2.H — Mark-Multiplier Threshold

[[w42-bookval-v1-wave2-ch10-action-level]] reported that **71% of
decisions at bid=42** have a different `mark_ev` top-1 action than at
bid=30 (n=14,000 paired decisions). That number is an artifact. At bids
42/84 all actions tie on mark_ev, so "top-1" is the first row; 36.5% of
decisions are single-legal and 2,162 forced decisions count as flips
because the states differ across bid runs; the bid-independent EV argmax
"flips" 24–29% on the same pairs. The bid-attributable change is roughly
6–17 percentage points.

- Bid=42 and bid=84 share `threshold_q=42`, so the 2× multiplier cannot
  change argmax (positive affine transform).
- `mark_ev ≡ p_make` on top-1 at every bid holds by construction.

The row stays `supported` because it is a definition of marks scoring.

## Wave 3.0 — Utility-Lens Meta-Analysis (Reconciliation)

[[w42-bookval-v2-utility-lens-synthesis]] re-processed all seven closed
Wave 2 probes through 5 utility lenses (EV, p_make, mark_ev, CVaR_10,
robust_q25). Four of the seven are setter-side probes withdrawn pending the
sign check (#122); one is the bid-only-enough tautology. The table records
what the lenses reported:

| Claim | EV | p_make | Other | Reading |
|-------|----|--------|-------|------|
| ch05-void-creation-follow | supported | spans_zero | mark_ev/CVaR_10 spans_zero | withdrawn (#122) |
| ch05-void-creation-lead | contradicted | contradicted | CVaR_10 spans_zero | withdrawn (#122) |
| ch12-setter-pounce-bid30 | spans_zero | spans_zero | — | withdrawn (#122); null either way |
| ch12-setter-pounce-high-bid | contradicted | contradicted | mark_ev/CVaR_10 contradicted | withdrawn (#122) |
| ch02-bid-only-enough | (missing) | supported | mark_ev supported | tautology |
| ch03-reentry-preservation | spans_zero | spans_zero | mark_ev/CVaR_10 spans_zero | null |
| ch04-low-trump-trap | contradicted | (missing) | (missing) | null in the book's scenario |

The p_set / threshold_mass columns in the wave-2 probes use Q ≥ 30 on the
±42 differential scale, where the make line is Q ≥ 18; 35–58% of p_set
values are saturated, so the "spans zero" readings on those columns rest on a
broken metric. "High-bid pounce is contradicted under all 4 utilities; the
book is wrong here irrespective of objective" is withdrawn with the probe.

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

The Wave 3.0 reconciliation read ch05-void-creation-follow as the one
claim where an EV head and a p_make head would disagree on what to learn.
That probe is withdrawn pending the sign check (#122), and paired contrasts
on two actions do not show what either utility's argmax picks; Wave 4.0
measured that directly.

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
(41.2%, CI [36.8%, 45.6%])**, above an ad hoc 5% gate. Two random legal
pickers disagree 70.4% of the time on the same snapshots, and 44% of
snapshots have top p_make = 1.0, broken by lowest slot index; tie counts
were not reported. Wave 4.0 maximizes raw e_q from the actor's view, so it
does not share the setter-probe sign question.

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

p_make and mark_ev select identically on all 500 snapshots, as they must:
at bid=30 mark_ev is a positive affine function of p_make by construction.

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

EV's argmax is the void tile on 29.4% of snapshots: the book's follow-void
advice is not what the EV oracle plays. The 30–44% divergence between EV and
risk-aware utilities is a property of the utilities on this bid-30 setter
corpus; the book's advice was never validated on it.

### Architecture decision gate: TRIPPED

The Wave 4.0 gate said: ≥5% disagreement with a coherent pattern →
scope rung-2 (utility-tunable searcher). The actual disagreement is
~10× the gate (41% vs 5%), but the gate was chosen ad hoc and no chance
baseline was set against it. Whether to actually build rung-2 is now a separate
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

Every CI excludes zero **in points**. Every hand was forced to bid 30
regardless of the cards. Recomputed in marks from
`w42/lens_v1/results/per_hand_margins.csv` (make = bidder ≥ 30): EV vs
p_make +0.110 marks/hand [0.048, 0.168], EV vs CVaR +0.106, EV vs robust
+0.072 [0.006, 0.134]; p_make vs CVaR −0.012 [−0.076, 0.050], p_make vs
robust −0.052 [−0.114, 0.008], CVaR vs robust −0.06 [−0.122, 0.000]. EV beats
the others in marks; the "total ordering" does not survive. A snapshot
contrast on two actions (Wave 3.0's +0.77) does not translate into a
whole-game matchup margin.

### The inversion that matters

This **inverts the natural reading of Wave 4.0**. Wave 4.0 said: "the
book + p_make / CVaR / robust_q25 cluster on void; EV is the outlier
preferring third-option discards." Lens v1 says: **EV's third-option
discards are not noise — they score points.** Every risk-aware utility
loses to mean-EV head-to-head in points; in marks EV's edge holds and the
other pairings are null.

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

## Methodology insight — strategy fusion, not a single-decision blind spot

Most book claims are multi-step plans ("lead a singleton on trick 1 to set
up a void by trick 3", the 84 throwaway ladder, reentry preservation), and
most probes were single-decision contrasts. The campaign concluded that
single-decision EV cannot see a plan because "the default future moves inside
the forge oracle do not include the bidder's own plan."

That diagnosis is wrong. A perfect-information evaluator plays optimal
multi-step continuations inside every sampled world; it plans perfectly
within each world. Its defect is that each world gets its own plan, chosen
with knowledge of all four hands, and the plans are averaged: the value is
not consistent with any strategy a player who cannot see the hands could
follow. That is strategy fusion ([[the-wall]]). A setup move that pays off
only because the player does not know where the tiles are is exactly what
an averaged perfect-information value cannot credit, however deep its search.

So the remedy was never planning-aware probes on the same oracle; it was an
imperfect-information evaluator. The project built one later and restudied
the book with it ([[book-study-2026-10-12]]). Lens v1's "ev wins" is a
comparison among one-step utilities of the same fused value and says nothing
about the book either way.

The campaign proposed three planning architectures, all over the forge
oracle and so all subject to the same fusion:

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

The campaign preferred #3 because it tests claims one by one. The deferred
Wave 2.F (84-throwaway, [[w42|t42-wikw]]) had this shape; it never produced
output.

For broader model-design questions, #2 is the cheaper general-purpose
tool.

None was built; the one Wave 5 probe that ran was
[[w42-champion-teaching-battery]].

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

## Caveats (carried forward)

- All 500 snapshots are bid=30 (mm=1, mark_ev ≡ p_make by construction).
  The cleanest test of mark_ev divergence requires a bid=84 snapshot
  corpus. Wave 2.F (84-throwaway) never ran.
- All to-act players are setters at bid=30, where the make-threshold
  (Q ≥ −17) is loose; CVaR_10 / robust_q25 do more discrimination here
  than they would on offense. A bid-aware mixed-position corpus is the
  natural follow-up scope.
- 100 worlds per snapshot leaves sample noise on near-tied slots, and the
  label network's own error (qMAE 0.94) is larger than several reported
  effects. Bootstrap CIs resample snapshots, not deals, and so understate
  the uncertainty.
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
