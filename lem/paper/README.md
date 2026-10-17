<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Interpreting a distilled imperfect-information player against an inspectable teacher

*Jason Yandell. Living draft, started 2026-10-16. CC0.*

> **The thesis under test (#193).** Cammarata's "already clean" [1]: a network's own neurons are
> mostly clean, and squeezed layers grow coarser, not mixed. On the field net, 24 neurons are cleaner
> than 192 (0.83 vs 0.61) and carry 0.684 of the play's gap to the teacher (§7). The result looks
> impossible to a superposition prior. The paper tests it until it breaks or holds: #194–#200.
> Every result is scoped to this net, these inputs (one position, uniform beliefs, no
> player-behaviour history) and this vocabulary (Silas concept list v1 or v2). A unit we cannot
> name is "not named in list vN", a vocabulary or data gap before it is superposition (#202). We
> can say what we learned from this model, not all of 42, yet.

Every claim below links to the experiment page that measured it and to the evidence on
Hugging Face. Numbers carry their interval and their chance baseline. Sections marked
**[gap]** name a measurement that does not exist yet; the Gaps list at the end is the work
order (the Lem desk's, [`wiki/topics/lem-the-teacher.md`](../../wiki/topics/lem-the-teacher.md)).

## Abstract (draft)

Interpretability work on game-playing networks usually lacks ground truth for what the network
*should* represent. We study a small network distilled from a teacher whose computation is
explicit: Walt, an imperfect-information planner for Texas 42 that samples the worlds a seat
cannot tell apart, holds exact beliefs over them, and chooses one plan per information set. Its
vocabulary is executable: Silas, a typed relational language over those worlds, names the
game's concepts (the master, the boss trump, a seat's shown void, count on the table) and
computes exact posteriors. Every feature found in the student can therefore be checked against
what the teacher computes. On the shipped 192-neuron field net, three poles in five have one
main job in the teacher's vocabulary (0.597 [0.581, 0.608] of live poles; shuffled chance 0.000),
a reader with only the viewer's cards names most of them (0.758 detected; shifted null 0.124),
and the play nonetheless needs 165 of 192 neurons to match itself 95% of the time: clean units,
dense decision. Across nets trained on 5k, 50k and 250k deals the nameable share rises
(0.297, 0.519, 0.597) while the declaration detectors are present from the start and the play
stays dense. Retrained from scratch at 24 neurons, the net is cleaner, not mixed (0.83 against
0.61 at 192), and still closes 0.684 of the gap from random play to the teacher; its units
broaden within one concept family, keep the family they formed in (0.82 against a 0.28 null),
and match their own family in a wide net at 2.4–2.9 times the random rate. We set out the belief probes, checkpoint studies and causal tests that an answer
key makes possible.

## 1. The lab

- **The game.** Texas 42: 28 dominoes, four seats in two partnerships, seven tricks, a bid,
  a declaration (seven pip trumps, doubles trump, no trump). The suit algebra separates suit
  membership, following and trick order; count is five tiles (5-5, 6-4 worth ten; 5-0, 4-1,
  3-2 worth five). Rules: [`wiki/topics/rules-of-42.md`](../../wiki/topics/rules-of-42.md);
  algebra: [`wiki/topics/suit-algebra-spec.md`](../../wiki/topics/suit-algebra-spec.md).
- **Why it is a laboratory.** Small enough to be exact (a seat's fiber of hidden worlds is
  enumerable late in the hand; Silas counts 399,072,960 deals a bidder cannot tell apart at the
  auction and answers exact odds over them), rich enough to carry beliefs, partnership and the
  modeling of other seats. [`wiki/entities/silas.md`](../../wiki/entities/silas.md).
- **The fence.** No clairvoyant value, no perfect-information solve, enters a label, a target or
  a player. The teacher solves information states; a double-dummy value per world averaged over
  worlds is strategy fusion and is fenced by a repository-wide test.
  [`wiki/topics/strategy-fusion-fence.md`](../../wiki/topics/strategy-fusion-fence.md).

## 2. The inspectable teacher

Walt ([`wiki/entities/scenario-player.md`](../../wiki/entities/scenario-player.md)):

    Walt = R_live[ R_model[ U ] ]
    V(C) = max_a Σ_o V(C_{a,o})   at the seat's own turn
         = Σ_o V(C_o)             at another seat's turn

A seat holds a weighted set of scenarios, chooses one action across every scenario it cannot
tell apart, and lets its plan branch only where an observation separates them. The other seats
are the same construction one rung down; the bottom rung is a field (random legal play, or a
learned field net). Everything the teacher computes is inspectable: the sampled worlds and their
weights, the per-world outcome of every legal move (the *splat*), the chosen plan. Walt beats the
oracle's expected value E[Q] at play, +0.74 marks per game (2026-10-10), without consulting it.

Silas supplies the vocabulary and exact beliefs: a query returns the answer tuples that hold in
each hidden world consistent with a seat's information; a belief is an exact measure over those
worlds; a policy may read the viewer's information only (a rule that reads a hidden hand does not
compile). The book's concepts (Roberson, *Winning 42*) are written in it
([`silas/lib/book.silas`](../../silas/lib/book.silas)). The book is the authority on reasons
([`wiki/topics/book-authority.md`](../../wiki/topics/book-authority.md)).

## 3. The students

- **Now: the field net.** `zeb/net/margin_d250k_h128x64.fnet`, a 343-input integer MLP with
  hidden layers of 128 and 64 (192 neurons, 384 poles), trained on 250,000 poured deals to
  reproduce the teacher's move distribution at the modeled seats (softmax of the teacher's
  per-move advantage, τ = 2). Its inputs are the viewer's information only: hand, tiles gone, the
  open trick by relative seat, shown voids, where the count sits, side, position in the trick,
  points needed. It plays the modeled seats inside Zeb, the shipped player.
  Training lane: `scenario/field_train.py`; data: [texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k).
- **Later: Lem's nets.** A transformer over the information state predicting every seat's
  next play (the model of others) and the teacher's value, trained from a league of mixed tables
  (#178, #182). **[gap]**

## 4. Concept audit against the teacher's vocabulary

Pre-registered before any pole was scored; [`wiki/experiments/lem-field-net-audit-2026-10-15.md`](../../wiki/experiments/lem-field-net-audit-2026-10-15.md);
evidence [`lem/field-net-audit-2026-10-15`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0982f55bd74a92d6819cf29e2fe7f309e998dab5/lem/field-net-audit-2026-10-15).

- Sample: the trainer's own held-out rows, 434,535 positions the net never trained on.
- Labels: 48 yes/no Silas queries over the viewer's frame, each evaluated on a belief that
  resolves no hidden tile (a query reading a hidden hand is refused), plus the 343 raw inputs as
  the trivial baseline. A pole's label is the concept of highest precision on its top 1% of
  rows; clean means precision ≥ 0.9.
- **Result.** 222 of 372 live poles clean, **0.597 [0.581, 0.608]** (row Wilson envelope, about 1.17× too narrow in SE by the split-half A/A of #191, which replicates the share at 0.594 / 0.599 on disjoint deal halves); shuffled-label chance
  **0.000**. 122 by a Silas concept (where the seat sits in the trick 60, partner winning 10, no
  trump held 10, trump led 6, the boss trump 3, three-plus trumps 5, voids 10, count 7), 100 by
  a raw input, of which 43 are **declaration detectors**: the net spends an eighth of its live
  poles knowing which pip is trump, a concept the sealed list had no word for. 12 of the 64
  second-layer neurons never fire.
- **A cheap reader agrees.** A language model shown 12 of a pole's top cards (viewer's
  information only) names the pole and then marks 20 held-out cards: 282 of 372 poles detected
  (0.758; mean balanced accuracy 0.843; chance 0.5) against 46 (0.124) when handed another
  pole's description.
- **Against the teacher's ladder.** The teacher's vocabulary has a ladder: the trump double, the
  highest unplayed trump, the master of any suit. In the student every "master" pole is a trump
  master: on the three boss-trump poles the precision of `holds-master` and `holds-trump` is
  1.00; no live pole has `holds-master` ≥ 0.9 with `no-trump-held` ≥ 0.5 (nearest 0.70 / 0.28).
  The trick-position poles are conjunctions: three "lead" poles carry `no-trump-held` at 1.00,
  "last" poles carry `safe-trick` or `winner-cannot-be-beaten`. Read-outs:
  `audit/reads.md` in the evidence.
- Predictions on record before the run: the clean share (0.3–0.6) landed inside at the top; the
  reader's detection (0.5–0.8; null about 0.1) landed inside; the layer prediction did not hold;
  the knee prediction (40–120) was wrong.

## 4b. Where the student strays from the teacher

([`lem-fidelity-by-concept-2026-10-16`](../../wiki/experiments/lem-fidelity-by-concept-2026-10-16.md),
pre-registered.) The student's regret under the teacher's plan is 0.0084 [0.0083, 0.0086] make
probability over 434,535 held-out positions, and the top decile of positions carries 0.825 of
it. It sits where the decision is open: on lead (0.0102), as the bidder with the contract within
ten points (0.0101), holding five-count tiles (0.0098); it is lowest where the trick is already
decided (safe trick 0.0054, winner cannot be beaten 0.0066, count on the trick 0.0066) and zero
once the bid is made or set. It grows with how much the teacher cares: 0.0008 where the
teacher's best two moves tie, 0.0214 where they differ by more than 0.1. The lead is both the
student's best-labelled region (32 clean `lead` poles) and its least faithful: a legible unit is
not an accurate one.

## 5. Belief probes

The teacher's belief at a position is exact and computable: for every hidden tile and seat,
P(seat holds tile | the viewer's information), under the uniform fiber and under the field's
posterior. The question: does the student encode it? Linear probes from each hidden layer to
the teacher's posterior, scored against the prior that counting alone gives (the uniform fiber
marginal), on the audit sample; then the same for the book's reads (a seat shown void, a seat
that must hold the master). This is the measurement an answer key makes possible.

**First result** ([`lem-belief-probe-2026-10-16`](../../wiki/experiments/lem-belief-probe-2026-10-16.md),
pre-registered): on 20,000 held-out rows from deals disjoint from the fit, the raw input bits
recover **0.79** (R²) of what the voids tell the teacher's posterior, the first hidden layer
**0.24**, the second **0.04** (deal-clustered 95% bands of about ±0.004, ±0.007 and ±0.003; the reverse split gives 0.78, 0.20, 0.02); shuffled targets give 0 (−0.02 to 0.00). The field net's inputs
carry the teacher's belief and its hidden layers discard most of it on the way to a move: the
student plays without a readable belief over who holds what. Without the input baseline the
first layer's 0.24 would have read as "the net represents the posterior"; with it, it reads as
compression. Next arms: the field's posterior from reading plays as the target, Lem's nets, and
a nonlinear probe as an upper bound.

## 6. Concept formation across data and checkpoints

Exploratory, not pre-registered ([`§ Exploratory`](../../wiki/experiments/lem-field-net-audit-2026-10-15.md),
evidence `series/`): the same sample and labels read against the three published nets of this
architecture, one net per size.

| deals | live | clean share | by Silas | by raw feature (declaration) | not named in v1 | dead | knee |
|---:|---:|---|---:|---:|---:|---:|---:|
| 5,000 | 384 | 0.297 [0.273, 0.333] | 30 | 84 (68) | 35 | 0 | 176 |
| 50,000 | 378 | 0.519 [0.505, 0.553] | 104 | 92 (71) | 20 | 6 | 170 |
| 250,000 | 372 | 0.597 [0.581, 0.608] | 122 | 100 (69) | 14 | 12 | 165 |

With more data the nameable concepts rise and the units list v1 cannot name fall; the declaration is learned
first and stays; the play stays dense. This is a snapshot per size, not checkpoint evidence, and
shows no branching. Checkpointed retraining and a width family are the registered next step
**[gap]** (Cammarata's recursive-generalization hypothesis predicts concepts that extend along
their facets; [`wiki/topics/recursive-generalization.md`](../../wiki/topics/recursive-generalization.md)).

## 7. The feature-count curve

Greedy forward selection of neurons, fidelity = share of rows on which the reduced net picks the
whole net's tile (20,000 fit rows, 20,000 eval rows): bias-only play 0.389; k = 20 for half;
k = 155 for 0.9; **k = 165 of 192 for 0.95**; random 128 neurons give 0.582. So far the answer to
"is there a small set of features that covers 42?" is no, at the level of this net's play, while
its units are individually legible. The curve on nets that grow is the falsifiable form of the
question (stage 4).

**The squeeze** (G2's pilot, registered at [`wiki/experiments/lem-squeeze-2026-10-16.md`](../../wiki/experiments/lem-squeeze-2026-10-16.md)).
A pruned curve overstates what is needed; a net trained at width k is the honest point. Nets
retrained from scratch at 128×64, 32×16, 16×8, 8×4 and 128×8 test the essay's squeeze [1]: narrow
neurons should cover coarser families of the teacher's concepts, not unrelated mixtures. Under a
fine label a coarse neuron scores partial, so the test is whether a pole's best two-concept union
lies within one concept family more often than random families give. **Result:** narrow nets are cleaner, not
mixed. The clean share is 0.610 at 192 neurons and 0.833 at 24 (three seeds 0.65–0.83). Where
units broaden, they broaden within a family, chiefly one coarse unit, "late in the trick". 24
neurons close 0.684 of the gap from random play to the teacher (192 neurons: 0.804; 12: 0.615).
A small clean vocabulary, the trick in front of the mover, carries most of the play. **Formation**
([`wiki/experiments/lem-formation-2026-10-17.md`](../../wiki/experiments/lem-formation-2026-10-17.md),
#194): across 15 in-run checkpoints, 0.82 of the wide net's clean units end in the concept family
they first became clean in (null 0.28). Position and trick units form first, and count and mastery
later. An untrained net already scores 0.15–0.28 clean, the honest floor for every clean share. **The width tree**
([`wiki/experiments/lem-width-tree-2026-10-17.md`](../../wiki/experiments/lem-width-tree-2026-10-17.md),
#195): a narrow unit's closest counterparts in a wide net trained from scratch come from its own
concept family at 2.4–2.9 times the random-family rate. The coarse unit spreads along one
concept's facets (the wide net has 47 `last` units), not into its sibling seats. **The drop-out order**
([`wiki/experiments/lem-dropout-order-2026-10-17.md`](../../wiki/experiments/lem-dropout-order-2026-10-17.md),
#196): three seeds at seven widths, each with its untrained floor. Narrower nets are cleaner down
to 12 neurons (means 0.633 at 192, 0.759 at 24, 0.735 at 12; floors 0.17–0.23; wide seeds
within 0.06). The concepts give out in order. The last words are the mover's place in the trick
(`lead`, the coarse `late`). Before them go the trick in front of the mover and the mover's trump,
then count and trump on the table, then whether the winner can be beaten. Mastery outside trump
and the setters' voids need 136 neurons. Two neurons close only 0.28 of the gap. Two of three
six-neuron nets failed to learn the play, and their units fell to the floor with it. **Naming** ([`wiki/experiments/lem-naming-2026-10-17.md`](../../wiki/experiments/lem-naming-2026-10-17.md),
#197): with concept list v2 and Haiku readers, 201 of the 210 live units of the 12- and 24-neuron
nets have a name, and readers detect 0.80–0.87 of the units the list cannot name (null 0.13). Of the
nine named by neither, seven of the eight that got a reader description describe who is winning the
trick together with where the mover sits: a gap in the list, not in the inputs. The list's size moves the floor: an untrained net scores 0.11–0.23
clean under the 48 concepts of v1 and 0.28–0.39 under the 70 of v2, while trained nets barely move.
Every clean share is quoted against the floor for its own list.

## 8. Models of others **[gap]**

The book's lessons assume opponents who model you, partners who read a system, fallible
opponents, a styled field ([`wiki/topics/book-assumptions.md`](../../wiki/topics/book-assumptions.md)).
The first measured case where the right strategy depends on who you face: against a Demon that
contests point bids, Zeb 0.4.1's refit wins 57.0% [53.1, 60.9] of games where 0.4.0 won 32.6%
[28.9, 36.1] on the same seeds, and gives back about 3 points against a table that passes
([`rematch-zeb-041-vs-demon-2026-10-15`](../../wiki/experiments/rematch-zeb-041-vs-demon-2026-10-15.md)).
A hand-tuned constant per opponent type does not scale; a net that infers the table in context
is the scalable form. The league (#182): mixed-partner self-play with hand-written Silas and
language-model seats as the bootstrap, partners drawn from the population too, a payoff matrix
with Nash-averaged ratings so cycles stay visible. The interpretability question: do
table-type units form, and do they read what the teacher's posterior over policies would read?

## 9. Honest limits

- One contract in every pour so far (seat 0 bids 30); pip trumps only in the shipped net (pour A,
  all nine playable declarations, is running: [`lem-pour-a-2026-10-16`](../../wiki/experiments/lem-pour-a-2026-10-16.md)).
- Precision on top activations is correlation, not use; no per-concept ablation or patching yet.
- "Clean" is relative to a list; the readers kept finding conjunctions the list lacked
  (concept list v2, #185). An unnamed unit is a gap in the list before it is a mixture.
- The inputs are one position against uniform beliefs: no bidding history, no record of how
  a seat has played, no model of the table. A unit computing something those inputs cannot show
  would read as unnamed here (#202), as the essay's comma neuron read as mixed until a longer
  context showed it tracked Colorado.
- One net per data size; no seeds; the series is exploratory.
- The reader is a language model; its agreement with the exact label is read, not scored.
- The teacher is a finite-budget planner, not an optimum; fidelity to the teacher is not
  strength, and strength is measured separately on the ladder.

## Gaps: the work order

| # | measurement the paper needs | status |
|---|---|---|
| G1 | **Belief probes**: linear probes from hidden activations to the teacher's exact posterior P(seat holds tile), against the counting prior; then the book's reads | first arm done (play-only, linear): inputs 0.79, a1 0.24, a2 0.04; next arms: the reading posterior, Lem's nets, a nonlinear bound |
| G2 | **Checkpointed retraining + width family** on the 250k pour, 3 seeds; concept formation and the knee per checkpoint and width | width pilot (the squeeze) done 2026-10-17: narrow nets cleaner (0.61 → 0.83); checkpoints next; $0 on the M5 Max |
| G3 | **Causal tests**: per-concept ablation and activation patching between matched legal positions, against matched random units | not started |
| G4 | **The first field-v3 net** on pour A (all nine declarations) and its audit with concept list v2 | done 2026-10-17: clean share 0.628 (halves 0.633/0.625), pip regret 0.0082, doubles/no trump 0.0120; the declaration detected (32 doubles units) but doubles played worse |
| G5 | **Lem's transformer** with a next-play head over league data (#178); its audit | not started |
| G6 | **Table-type probes**: does a net trained on a mixed population encode who it faces | needs G5 and the league (#177, #182) |
| G7 | **Fidelity to the teacher by concept region**: where the student's regret under the teacher's plan concentrates | done ([`lem-fidelity-by-concept-2026-10-16`](../../wiki/experiments/lem-fidelity-by-concept-2026-10-16.md)): mean 0.0084, top decile carries 0.825; highest on lead and with the contract close, lowest where the trick is decided; the lead is the best-labelled and least faithful region |
| G8 | **Meet the book**: the clean units against the book's rules (Silas gates); which rules have a unit and which do not | not started |
| G9 | **Replication across seeds** for every share and interval in §4 | needs G2 |
| G10 | **THE ALREADY-CLEAN TEST** (#193): formation #194 (done: family kept 0.82 vs 0.28), width tree #195 (done: own family 2.4–2.9× random), drop-out order #196 (done: cleaner to 12 neurons; the last words are the trick position), naming #197 (done: 201 of 210 narrow units named; the floor rises with the list), causal #198, pour A #199, family statistic #200 | the core thesis; queued ahead of G3–G8 |

## References

1. Nick Cammarata, "Neural Networks Are Already Clean", Goodfire, <https://already-clean.goodfire.com/>.
   The program's foundation: a network's own neurons are mostly clean when tools, architecture,
   data and training are sound; remaining mixing is recursive generalization ("generalization is
   fractal"); squeezed layers grow coarser concepts, not mixed ones; formation is watched across
   checkpoints. Informal essay, companion paper announced. Summary in our words:
   [`wiki/topics/clean-neurons-42.md`](../../wiki/topics/clean-neurons-42.md).

## Links

Program: [`wiki/topics/lem-the-teacher.md`](../../wiki/topics/lem-the-teacher.md). Desk notebook:
[Lem Desk](https://claude.ai/artifact/6nYMW1xcA4wxpqdJsytTgr). Epics #176–#180, league #182.
