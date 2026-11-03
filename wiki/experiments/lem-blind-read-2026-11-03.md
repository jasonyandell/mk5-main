---
title: "A blind weight read of the shipped field net, graded on real rows (Lem)"
kind: experiment
first_seen: 2026-11-03
last_updated: 2026-11-03
status: registered
---

Jason's commission (2026-11-03, via session code-fa): a frontier model reads the shipped field net's raw
weights with no data, as in "Sparks of Weight-Space Programming" (Takehi et al., 2026-11-02,
experiments 1–3). We then test its predictions on the 434,535 held-out rows, starting with count
smearing and the lead/follow switch. The reader's own write-up and all 192 unit names are in
`scratch/desks/lem/blind-read-2026-11-03/` (README, `round1-report.md`, `round2.md`, `neurons.csv`),
mirrored to the Hub with the evidence.

## The read

**The net.** `zeb/net/margin_d250k_h128x64.fnet`, the field-v2 net behind the published player
(128 × 64, integer, ReLU after shifts). It was exported as `net.npz` with arrays only, into a private
repo that held nothing else.

**The reader.** A cloud Claude session (Opus 5.5) with no memory, no CLAUDE.md and no web. It reports
that nothing in its context hinted at the domain. It ran Python on `net.npz` only, for about $3.90. An
earlier local read was contaminated (the memory index named texas-42) and is not used.

- **Round 1** (arithmetic only). It named Texas 42, the exact tile indexing, the five count tiles,
  the relabel of trump to sixes, and the trick blocks by seat. It missed `gone` (read as "my played
  tiles"), `void`, `count`, and the 4 + 3 split of `open`/`winning`.
- **Round 2** (we revealed the `scenario/field.py` block layout and the training contract). It wrote
  its own 42 simulator, had the net play all four seats (84,000 states, 20% random plays, every bid
  30), correlated every unit with about 110 features of its own choosing, and named all 192 units.

What it predicted is graded below against **the net's behaviour on real rows**, not only against our
audit's labels. Our audit is a weak referee: it read an all-rules hand net below a random conjunction
net and called an unrelated-OR decoy clean ([[lem-hand-written-field-net-2026-11-02]], #193 comment
6596308458).

## What this tests, and what it does not

**This grades the reader and our instruments. It does not test the already-clean thesis.** The thesis
is about why things share a unit: grown by generalization, or packed because they don't interfere.
Its own tests are distant-concept controls and watching units form. Every cell here is reader ×
simulator × net.
- **A held prediction** says this reader's explanation survived real rows. It does not say the unit
  has one job.
- **A missed prediction** does not say the unit has several. Its first suspects are the simulator
  (every bid 30, one play in five random, voids guessed) and the reader's words (rung 3, then rung 2).
- Neither moves the thesis.
- **Circuits 1–5 are the reader's own pick of its most striking findings**, not a sample of the 192.
  Transfer to the other units needs a seeded random sample (parked, see Follow-ups).
- **The reader's vocabulary is its own.** It correlated each unit with about 110 features it chose,
  which is a probe: it answers questions the reader brought (the Tokyo guardrail). Its 27 "unclear"
  units and its "packs two things" list are vocabulary and simulator gaps before they are
  superposition. P8 reads that list with both poles.
- The nearest check in the essay is the blind maximum, where an agent writes inputs from an
  explanation alone. This is weaker: the names are scored against a fixed corpus. The essay trusts
  tests like that only up to a point, since cats-and-cars would pass many of them on the V neuron.

Some predictions grade the net, not the reader. Those are labelled **fidelity**: the net against
Walt's own move (P2b's Walt clause, P6.6, P7).

## Rows, instrument, floors

- **Rows.** The audit's 434,535 held-out rows (`scratch/lem/audit-2026-10-15/sample.npz`: the five
  training pours' `deal % 10 == 0`), 25,000 deals. Teacher values: Walt's `q`
  (`scratch/lem/fidelity-2026-10-16/q`). Walt's own move is `chosen`.
- **Forward pass.** `lem.probe.net.forward`: z1, a1, z2, a2, scores, bit-identical to the referee.
  Ablating a layer-1 unit sets its a1 to 0 and recomputes layer 2 and the scores. Ablating a block
  zeroes those input bits, except the count block, which is replaced as the reader did (below).
- **Scripts.** `facts.py` (cells, sha256 5388d62b3edd…) and `grade.py` (every statistic below, one part
  per prediction, sha256 8501f4823519…), in `scratch/lem/blind-read-2026-11-03/`, to the Hub with the results.
- **Intervals.** 95%, by bootstrap over deals (2,000 resamples, seed 0).
- **Untrained floor.** Three untrained 128 × 64 field-v2 twins: the initial nets at zero epochs, seeds
  0–2, `scratch/lem/dropout-2026-10-17/floor/field_h128x64_s{0,1,2}.npz`
  ([[lem-dropout-order-2026-10-17]]). Each behavioural statistic is computed on the twins too.
  A unit-level statistic uses the twin's best unit for the same question, because unit numbers mean
  nothing across nets.
- **Chance.** Random legal play, wherever a statistic is a move rate.
- **Which signal.** Every correlation uses the pre-activation z (the reader's choice); every on rate
  uses a > 0 (after shift and ReLU). Where a label is read on a pole, both tails are reported: the top
  and bottom 1% of z.
- **Ablation.** Zeroing a unit that is on in most rows takes the net off its distribution, so each
  single-unit ablation also reports a mean ablation (a set to its mean over all rows).
- **Rank among the layer.** A causal claim about one unit is ranked against every unit of its layer,
  each ablated on the same rows (128 for layer 1, 64 for layer 2). The dose is reported beside it:
  the unit's on rate and mean activation on those rows.

**Cells** (from `facts.py`, viewer information only; sizes are rows and deals):

| cell | definition | rows | deals |
|---|---|---:|---:|
| last, partner winning | I play fourth, partner holds the trick, I have a legal count tile and a legal non-count tile | 10,204 | 8,711 |
| last, opponent winning | the same with left or right holding the trick | 36,893 | 19,276 |
| lead, trump out | I lead, hold a trump and a non-trump, and at least 4 trumps are gone | 34,753 | 17,737 |
| holds 6-5, 6-6 out | 6-5 (canonical 26) in hand; 6-6 neither in hand nor gone | 25,994 | 13,216 |
| holds 6-5, 6-6 gone | 6-5 in hand; 6-6 gone | 32,217 | 12,429 |
| lead, off double | I lead, hold an off double and some other tile | 104,993 | 24,965 |
| second hand | I play second, every legal tile follows the led suit, at least two legal tiles, no count among them | 20,797 | 14,903 |
| 6-6 legal | the trump double is legal with at least one other tile (lead: 17,529) | 46,755 | 21,538 |

"Count" is the five count tiles under the real declaration (5-5 and 6-4 for ten, 5-0, 4-1 and 3-2 for
five). "Decided" means the bidders need fewer than 2 more points or the defenders do (the `need` or
`slack` thermometer is empty).

## What is already seen (settled before any run)

The audit ([[lem-field-net-audit-2026-10-15]], `poles.md`) reports each pole's top correlate on these
same rows. Some of the reader's claims are settled by it, or by arithmetic:

- **P1a holds.** r(L1 53, lead) = −0.92, so +0.92 with following. r(L1 57, lead) = +0.89. The lead heads
  L2 45, 63, 57, 15 and 54 have r +0.77 to +0.88 with `lead`; the follow heads L2 40, 6, 3 have r −0.69,
  −0.78, −0.68. L2 40 is on in only 1.6% of rows.
- **P3a, mostly seen, and against "near-copies".** L1+060 reads `partner-out-of-trump` (0.95), top r
  0.57 with gone 6-6, on in 14.2% of rows. L1+013 reads gone 6-5 (top r 0.68), on in 38.6%. L1+125
  reads `lead` (1.00), on in 23.5%. Different on rates and a lead-gated pole make "copies" unlikely.
  L1 60's real pole may be a void reading that the reader's simulator only guessed. Its r with the
  number of trumps gone, and the three units' pairwise r, are unseen.
- **P4, partly seen.** L1+039 reads holding 6-5 (precision 1.00; r +0.72); L1−039 reads gone 6-5.
  L1 20: r +0.51 with holding 6-6, its + pole `holds-master` (0.991).
- **L1 82's + pole is `second`** (precision 1.00).
- **P5c misses.** The void block, zeroed, changes 8.6% of all moves (`void_ablate.json`, 2026-11-02),
  2.15× the reader's 4%, outside a factor of 2. (The README's 12.8% is a different denominator, the
  rows where a void is shown.)
- **P6.4 holds by arithmetic.** The audit's 12 dead layer-2 units are among the reader's 13, and L2 52
  is on in 0.1% of rows, so pruning all 13 changes at most 0.1% of moves.
- **The weights are facts of the file**, checked equal: L1 53 → L2 45 is −7,415; L1 57 → L2 40 is −9,522;
  L1 60 → L2 40 is −13,146. L1 53 and 57 have no lateral connection. They are two complementary gates
  onto layer 2, not mutual inhibition.

Everything below marked **unseen** has not been computed on any row of the shipped net. (A smoke run of
the grading script used an untrained twin in its place.)

## Predictions (the reader's, made testable)

Each is the reader's claim, our operational form, and the bar. "Held" needs every unseen part;
"partial" means the direction held and the bar did not.

**P1. The lead/follow switch** (round 2, circuit 1). P1a is settled (above).
- P1b, unseen: on follow rows, ablating the follow units {L1 53, L1 70} turns the lead heads
  {45, 63, 57, 15} on. **Bar:** their on rate (any of the four, a > 0) at least doubles, and the move
  changes on ≥ 5% of follow rows. L1 53 alone is reported beside it (zeroed and mean-ablated), with its
  rank among the 128 by the rise in the heads' summed activation (top 5 expected). The rank is close to
  fixed by |w2| × mean a1, so it is secondary.
- P1c, unseen: on lead rows, ablating L1 57 turns the follow heads {40, 6, 3} on. The same bars.
- Also reported: regret before and after each ablation.

**P2. Count smearing** (circuit 2; graded first, with P1).
- P2a, unseen: score gap = the mean score of my legal count tiles minus the mean score of my legal
  non-count tiles, averaged over rows. **Bar:** positive in "last, partner winning", negative in
  "last, opponent winning", each within a factor of 3 of the reader's +9.6M and −9.5M.
- P2b, unseen: the net's count-play rate, partner winning minus opponent winning. **Bar:** ≥ 0.40.
  Beside it: random legal play, Walt's `chosen`, the three twins. The net's difference should be
  within 0.10 of Walt's.
- P2c, unseen (the reader's test 2, the threshold): in "last, partner winning", the gap is smaller
  where the contract is decided (2,355 rows) than where it is not (7,849). **Bar:** at least 25%
  smaller, its interval excluding 0. Count on the trick against the gap is reported, no bar.
- The clause "within 0.10 of Walt's" in P2b is **fidelity**: it grades the net, not the reader.

**P3. A trump-exhaustion detector** (circuit 3).
- P3a, unseen: r(z of L1 60, trumps gone) ≥ 0.85; r(L1 60, L1 13) ≥ 0.8 and r(L1 60, L1 125) ≥ 0.8.
  The seen facts above make this a likely miss. Both tails of L1 60's z are reported against trumps
  gone, partner void in trump, and "I hold no trump".
- P3b, unseen: the reader names three near-copies, so the test ablates them together. In "lead, trump
  out", ablating {L1 13, 60, 125} raises the net's trump-lead rate. **Bar:** a rise whose interval
  excludes 0, above the 95th percentile of 200 random triples of layer-1 units (seed 0). L1 60 alone,
  zeroed and mean-ablated, with its rank among the 128, is secondary. Each unit's dose there is
  reported first.

**P4. Second-best-trump awareness** (circuit 4). What the unit is for is its effect on play.
- P4a, unseen: L1 39 is on (a > 0) in "holds 6-5, 6-6 out" and silenced in "holds 6-5, 6-6 gone".
  **Bar:** on rate ≥ 0.5 in the first, ≤ 0.2 in the second.
- P4b, unseen: L1 39 moves the 6-5. In "holds 6-5, 6-6 out", ablating it changes the net's 6-5 play
  rate. **Bar:** an interval excluding 0, top 10 of 128 by |change|. Steering: in "holds 6-5, 6-6 gone",
  forcing it on at its mean dose from the first cell moves the 6-5 rate the other way.
- L1 20 as the 6-6 holder: settled (r +0.51, above).

**P5. The block-ablation ranking** (circuit 5). The reader's simulator gave: count 26%, winning 21%,
gone 15%, need 8%, slack 7%, void 4%, side 1%.
- Procedure as the reader's: the count block is replaced, per row, by the layout of another pip
  trump drawn at random (seed 0); every other block is zeroed. The share of all rows whose move changes.
- P5a, unseen: count has the largest effect. P5b, unseen: Spearman ρ between the reader's seven and ours
  ≥ 0.8. P5c: settled miss (void, above); the other six are reported against a factor of 2.
- Floor: the twins' shares under the same procedure.

**P6. The reader's own test list** (round 2, "What I'd test").
- 6.1, voids: no new bar (P5 and 2026-11-02 cover it).
- 6.2, the smearing threshold: P2c.
- 6.3, unseen: L2 54 (lead off doubles) fires mainly after trumps are pulled. **Bar:** on lead rows,
  its on rate with ≥ 5 trumps gone is at least twice its rate with ≤ 2 gone. The reader's own
  simulator correlate was "I hold no trump", so that cell gets the same bar beside it. Ablating it in
  "lead, off double" lowers the off-double lead rate, rank top 5 of 64.
- 6.4: settled (above).
- 6.5, unseen: with the count block replaced (P5), the changed moves land on count slots more often
  than chance. **Bar:** the share of changed moves whose old or new pick is on a count slot of either
  layout is ≥ 0.8 and at least 0.2 above the same share for a random legal tile drawn in place of the
  new pick.
- 6.6, unseen, **fidelity**, second hand low: in "second hand", the net plays the lowest-ranked legal tile of the led
  suit at least 0.20 above random legal play, and within 0.05 of Walt's rate.

**P7, fidelity. "The net is reluctant to play 6-6"** (round 1). In "6-6 legal", the net plays the 6-6 at least
0.05 less often than Walt does. Reported for all such rows and for leads.

**P8. The reader's "units that pack two things"** (round 2): L1 14, 1, 3, 5, 106. Our own
prediction, from the essay's practice of reading a unit as a gate with two poles: where the reader
itself wrote the second thing as negative (14, 1, 3), it is enriched in the bottom 1% of z and not in
the top 1%. **Bar:** for each of the three, the second thing's rate in the bottom 1% is at least twice
its base rate, and in the top 1% at most its base rate. For L1 5 and 106 both things are reported in
both tails, with no bar; L1+106 already reads 6-4 as count (precision 1.00). Nothing here can headline
"mixed": a unit the reader's words do not fit is not named in its list.

## Follow-ups (parked, not graded here)

- A blind maximum for the P1–P4 units: an agent that sees only the reader's name writes positions
  that should drive the unit, near misses and distant ones, scored by the net. That is the cell that
  bears on "one main job".
- A seeded random sample of about 20 of the reader's 192 names, each graded on real rows.

## The scope clause

This net, these inputs (one position, uniform beliefs, no history), this vocabulary (the reader's
110 features and our list v2). What we learned from this model, not all of 42, yet. A reader's miss
is evidence about its simulator or its words before it is evidence about the net.

## Review before sealing

A fresh-context Opus subagent read the essay whole from the local copy, then this draft, the README,
round 1, round 2 and `neurons.csv`, and checked the audit's `poles.md`. It found:
- the draft never said that this grades the reader, not the thesis;
- P3b's single-unit ablation contradicted the reader's own "near-copies" claim;
- several predictions marked unseen were settled (P1a, P3a, P4, P5c, P6.4);
- P6.5 had no chance baseline;
- the statistics did not say z or a;
- the reader's "packs two things" list, the only claims near the thesis, was missing.

All of these were adopted, together with:
- joint ablation for P1;
- a play bar and a steering clause for P4;
- the void-in-trump cell for P6.3;
- mean-ablation companions;
- fidelity labels;
- the over-read sentences.

Two of its suggestions are parked as follow-ups: the blind maximum and the random sample.
