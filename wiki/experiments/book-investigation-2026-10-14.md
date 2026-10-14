---
title: The book against Zeb, investigated — the method, the book's own game, its opponents, its scope (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Question

[[book-head-to-head-2026-10-13]] seated Winning 42's 18 rules ([[w42]], as [[silas]]
policies) against Zeb and found them losing 1.4–3 marks a game, with no rule gaining
anywhere. Jason, 2026-10-13: "some of these would be earth shattering revelations and need
extra extra careful review… dig in to the meat of these disagreements, always trying to
credit the book or make the fair but generous split or the benefit of the doubt. these
strategies lose here. where might they win? what if we constrain play to totally
book-explained situations, or ones we derive. or what about our whole game and policy
blindness (throw em off with a risky play they won't expect you to make) and uniform
beliefs. scientifically speaking the reputation of the book is very high and the
reputation of Zeb, however good its h2h scores, is not. yet."

The default posture is that the book is right and the measurement wrong until shown
otherwise. Is the measurement sound? Who wins the book's own game, points to 250? Does the
book fare better against opponents who read plays? Does it lose inside the situations it
explains? Where, if anywhere, does it win?

## The answer, in short

- **The arena and its accounting hold.** The method audit found no code bug and no leak
  toward Zeb. The published run reproduces exactly, 24 departures re-derive exactly, and the
  84 fix was in every pooled run.
- **The encoding does not fully hold: 44% of block b's loss is not the book's rule against
  Zeb.** Of that loss:
  - 29% is the encoding firing outside a rule's stated scope or against the headers' order;
  - 11% is the least tile picked inside a set that also held Zeb's tile;
  - 4% is unclear.

  The rules' own disagreements with Zeb, with nothing flagged, cost −0.042 [−0.051, −0.033]
  marks a departure: about 1.1 marks a game to first order, against the published 1.9. The
  same departures gain +0.27 [+0.10, +0.45] points.
- **The book loses its own game,** points to 250, in every form that plays its rules where they
  disagree with Zeb. It wins 20.5–37.9% of games; its best fair form wins 37.9% [34.3, 41.4],
  and with its own bidding 20.7%.
- **The book wins its own game as Zeb's tie-break.** Zeb breaking its near-ties (within
  0.02) the book's way wins 53.0% of games to 250, and again on new deals in a registered
  replication (pooled 53.2% [51.1, 55.2], +1.2 settlement points a hand, no marks moved). This
  is measured against Zeb's lowest-id tie rule, not against a points-aware one.
- **Opponents who read plays by the field net's set change nothing.** The reader's gain
  against the book less its gain against Zeb is null under both likelihoods, though the
  book's plays leave it without an explaining world more often (21% of its decisions, against
  16%). A reader who reads the book player by the book changes nothing either: −0.04
  [−0.27, +0.19] marks a game, and it loses +0.50 to plain Zeb ([[book-reads-by-the-book-2026-10-14]]).
- **Inside the book's scope it still loses** (S-core −0.057 marks a departure). It ties only
  inside the three rules Zeb's own plan sides with. On its own eight narrated deals its side
  outplays Zeb's, which carries no interval.
- **After the repairs (#146) the book loses 1.41 marks a game,** 0.47 [0.33, 0.62] less than
  published, as the census forecast, and wins 38.4% of games to 250 (§ After the repairs).
- **The benefit of the doubt does not reverse the sign.**
  - The most generous order gains +0.22 marks a game, and the book still loses 1.2.
  - The winner's curse is 12% of Zeb's forecast price.
  - Another continuation moves a departure's cost by a hundredth of a mark at most.
  - The game state, the contract level and the seat change nothing in marks.

What this does not show: that the book is wrong at the table it was written for, where
partners both know it and the opponents are people who read plays and hold expectations; or
that its value as a tie-break exceeds a principled points tie-break. The first was measured
next, with Zebs as the readers ([[book-reads-by-the-book-2026-10-14]]): the book fares the
same there.

## Pre-registration (written and committed before any pooled batch of this page)

Fixed before the first pooled batch; not moved afterwards.

**Common design.** As [[book-head-to-head-2026-10-13]]: the arena's paired design (each
deal played in both orientations, pooled halves), 36 games a batch, base seeds 1–11 (396
games, 198 deal clusters a block), every batch under `timeout 300` (a block whose batches
hit the cap runs at 24 games a batch and says so); timing batches at seed 900 are in no
number. Zeb is deployed Zeb, `walt:k0,n512,F=margin_d250k_h128x64,f=16`; bidding is the
arena's `heuristic` (Silas's `bid-only-enough`) for both teams unless a block says
`risk-budget`. Intervals are 95% percentile bootstraps over deal clusters (10,000
resamples); a contrast between blocks is paired by cluster (the blocks share seeds and
deals). The book player's variants are named as in the head-to-head: (a) the lowest legal
tile where no rule fires, (b) Zeb's tile there, (c) as (b) with Zeb's best tile inside a
rule's set.

**I. The method audit.** Two independent reviews of the head-to-head, neither by the
session that ran it: a method audit (the code paths, leaks and biases, the statistics, at
least twelve departures re-derived by hand: the verdict, Zeb's tile and q, the rollout) and
a rule-fidelity census (six departures of each departing policy judged against the
policy's own stated rule: in scope, outside it, decided by the wrong policy of the order,
or unclear). Findings are reported as bugs (a number moves), biases (with a size or how to
size it) or none. The published block b, seed 1, is re-run on this branch first: it must
reproduce game for game.

**II. The book's game: points to 250.** The arena plays the book's game
(`ArenaConfig.points_to_win`; Silas's `Scoring::Points{250}`, settled by S-0017-c: made,
each side scores the points it took, a mark bid its value; set, the bidders nothing and the
setters the bid plus their points; both sides reaching 250 on one hand, the bidder's side
wins). Play is score-blind on both teams, so each hand is played as in the marks game;
only the scoring and the end of the game change. Blocks: P-zeb (control, one batch: exactly
50%), P-a, P-b, P-c, P-d, P-t0, P-t1 (variants of III), P-b-rb and P-rb (the book's
bidding, `risk-budget`, with the book's and with Zeb's play).
- *Headline:* the book team's game share (the share of games it wins), mean over clusters.
  The book **wins its game** where the interval lies above 50%, **loses** where below, else
  **ties**.
- *Beside it:* the settlement margin per hand (what the book team scores less what the
  other side scores, by the book's rule), the final score margin per game, made rates.
- *The departures in the book's units:* the head-to-head's decompositions of blocks b and c
  re-scored (no new rollouts; every rollout outcome carries its points): each departure's
  change in the hand's settlement margin, Zeb continuing; pooled, per policy, and by the
  contract at the departure (21+, 11–20, 1–10 points still needed, decided). A gain is an
  interval above 0.

**III. Benefit of the doubt.**
- (d) *any rule:* where rules fire, Zeb's best tile inside the union of the sets of every
  policy of the phase that fires (whichever book rule allows the tile Zeb prefers decides).
  No fixed priority order with Zeb's tie-break inside the set can do better at a decision,
  so (d) bounds the best fair order from above.
- (t0) *Zeb, with the book at Zeb's exact ties:* Zeb decides, except where the deciding rule
  allows a tile whose q equals Zeb's best (the mover's side), where the book's tile among
  those plays (#120: 40% of the pour's decisions are exact ties under `make`).
- (t1) as (t0), within 0.02 of Zeb's best (the plan's own re-plan noise is 0.005–0.016,
  [[book-study-2026-10-12]] § The face-off).
- Blocks d, t0, t1 (marks) and P-d, P-t0, P-t1 (points). Bands as the head-to-head (marks:
  an interval above or below 0) and as II (points). (t0) and (t1) depart from Zeb only
  inside its own indifference, so they price the book's rules as a tie-break.

**IV. Opponents who read plays.** The reading Zeb of 3262eea0 (another session's, merged
unchanged): Zeb plus `d=posterior,r=<likelihood>:2^22`, its worlds drawn from the posterior
that reads every other seat's plays by the field net's set (δ = 2 points), `consistent` or
`uniform` (restricted choice). Blocks (marks): R-zc and R-zu, plain Zeb against two readers;
R-bc and R-bu, the book (b) against two readers.
- *Contrast, per likelihood:* C = [book against readers − book against plain Zeb
  (the head-to-head's block b)] − [Zeb against readers − 0], marks per game, paired by
  cluster. C above 0 (interval clear): a reading opponent is **thrown off** by the book's
  play more than by Zeb's; below 0: the reader exploits the book's play more than Zeb's;
  else null.
- *Mechanism:* the reader's fallback rate (decisions where no world explains the record
  under the net's set), against the book and against Zeb.
- *Not measured, stated as a gap:* an opponent who reads the book player by the book's own
  policies (a book likelihood inside the posterior draw).

**V. The book's scope.**
- *Its own deals:* the eight full narrated deals of `silas/examples/book/hands.json` (Hands
  28–35) under the narrated contract: the book (c) declaring against Zeb setters, Zeb against
  Zeb, Zeb declaring against book (c) setters, and the book's narrated line where complete;
  each at the four rotations of the table (Zeb's samples differ by seat). Made or set and the
  points, per deal. Descriptive: eight deals carry no interval.
- *Strata of the head-to-head's departures* (blocks b and c): S-single (the deciding policy
  is the only one firing), S-one-tile (the rule names one tile), S-bid30 (contract 30),
  S-open (contract undecided), S-sides (helper-stretch, partner-leads, protect-the-count:
  the rules Zeb's plan sides with on the pour), S-core (S-single, S-one-tile and S-bid30
  together). Marks and points per departure with intervals, beside the rest. The book
  **wins in scope** if S-core's marks interval is above 0.

**VI. Selection effects.**
- *The winner's curse:* every departure of block b, seeds 1–2, re-planned with Zeb at seed 1
  (an independent sample of 512 worlds) and at 2,048 worlds; the curse is the logged price
  (q of the book's tile less q of Zeb's pick, the mover's side) less the fresh one.
- *The continuation:* the same departures, each tile rolled out to the end under Zeb at seed
  1 at all four seats, and under the book continuing (the book team as block b's player, Zeb
  opponents); marks and points per departure beside the original (Zeb at seed 0
  continuing). The original decomposition is **biased against the book** where the
  book-continuing mean exceeds it with the paired interval above 0.

**VII. Where might the book win?** Every cell examined (block, game, rule, stratum) is listed
with its interval in the data; every interval above 0 is reported here with the count of
cells examined, since about one null cell in forty lies above 0 by chance.

**Replication (registered after block P-t1 read 53.0% [50.3, 56.1], before any replication
batch).** The one block whose band reads "wins" is run again, unchanged, at base seeds 12–22
(new deals; 396 games, 198 clusters). The claim replicates if the replication's game-share
interval lies above 50%. The pooled seeds 1–22 are reported beside it, and the first run is
not discounted or dropped.

## I. The method

**Reproduction.** Block b, seed 1, re-run on this branch, reproduces the published batch game
for game: 36 of 36 games and all 5,726 book decisions are identical. Every points game passes
its own bookkeeping check: each hand is settled as S-0017-c, the tallies add up, the game ends
on reaching 250, and the winner is right.

**The method audit found no code bug and no leak toward Zeb.**
- *Routing:* across 233,198 logged decisions, the first firing policy of the phase decides
  and none decides outside its phase.
- *Fallbacks and tie-breaks:* as specified. The (c) setter sign is right in the code and in
  the data.
- *The frame:* the mover's hand, the contract and the public plays, nothing else.
- *Scoring, pairing and determinism:* every cluster has both orientations, the first deal is
  identical across a, b and c, and Zeb's tile and q reproduce at 224 positions re-asked. No
  batch came near the cap (the longest took 92 s).
- *Twenty-four departures re-derived by hand:* the verdict, Zeb's tile, every q to 1e-9, and
  the rollout's d and d_random. All are exact.
- *The 84 fix:* only rb and b-rb have 84s. Every Zeb play at an 84 hand matches the fixed
  planner (4,956 of 4,956 in rb, 2,422 of 2,422 in b-rb); the raw-84 planner would match 3,647
  and 1,968 of them. Every pooled run used the fix.
- *Two suspected handicaps, both null:* Zeb's model of its partner (the field net predicts the
  book player's tile as well as Zeb's own, 53.5% against 52.3%), and the net's training at bid
  30 only (the departures cost the same at 30, −0.056, as above it, −0.047).

**The rule-fidelity census** judged six departures of each departing policy against the
policy's own header. 78 of 102 were in scope, 13 outside it, 3 decided by the wrong policy of
the order, and 8 unclear. Over all 14,370 departures of block b, the census labels each kind:
- *Outside a rule's stated scope* (522), in six patterns:
  - low-from-a-pair declines a trick its higher tile takes: 210;
  - "one off may go first" is played as an instruction rather than a permission: 100;
  - the top-trump handoff fires on a tile that cannot beat the trick: 98;
  - follow-with-the-off judges the off by the tile's own suit: 43;
  - the safe-trick prelude counts setters who have already played: 39;
  - protect-the-count fires when the mover's own tile beats the lead: 32.
- *Against the headers' own order:*
  - the no-count rules choose a throw outside every co-firing throw set: 292;
  - the dead double is cashed while the bidder holds the boss trump and his draw also fired: 241.
- *A least-id pick where the rule means one tile* (557): setter-leads' reach 218, a count trump
  led 183, the top trumps' rank 156.
- *Unresolved by the headers:* the dead double thrown over shed's lone tile, 344.
- *The least-id pick where the rule's set holds Zeb's tile:* 3,019 (21%).

The census's ten concrete repairs are #146.

**What the artifacts cost.** The census's labels joined to the published decomposition
(block b; marks a hand is the contribution to the book team's −0.173):

| departures | n | marks/departure | points | marks/hand | share of the loss |
|---|---|---|---|---|---|
| encoding artifacts (outside scope, order, tile picks) | 1,612 | −0.132 [−0.166, −0.097] | −2.00 | −0.050 | 29% |
| least-id pick inside a set holding Zeb's tile | 2,312 | −0.035 [−0.062, −0.008] | −0.53 | −0.019 | 11% |
| unclear (crisis evidence, the dead double over the lone tile, a one-trump trump set) | 559 | −0.054 [−0.097, −0.011] | −0.90 | −0.007 | 4% |
| **the rules against Zeb, nothing flagged** | 9,887 | **−0.042 [−0.051, −0.033]** | **+0.27 [+0.10, +0.45]** | −0.098 [−0.119, −0.076] | 56% |
| all | 14,370 | −0.051 [−0.058, −0.045] | −0.16 | −0.173 | |

To first order (taking each hand's later departures as unchanged), the rules' own
disagreements with Zeb cost about 1.06 marks a game, against the published 1.88. They gain
points while costing marks. In the book's units (II) they cost −1.14 [−1.50, −0.78]
settlement points a departure. The audit's correction of block c is of the same shape and
size: about a sixth of its −1.41 is encoding and order. The head-to-head's page is corrected
in place: where it ranked rules, read block c's policies as its block b's, and called the
chance baseline and the forecast.

## II. The book's game

Points to 250, each hand settled by the book's rule. Play is score-blind on both teams, so
these are the head-to-head's hands under another scoring and another end.

| book team | game share [95%] | verdict | settlement / hand | final score margin | made, book / Zeb |
|---|---|---|---|---|---|
| (a) rules, else the lowest legal tile | 20.5% [17.2, 23.7] | loses | −10.9 [−11.9, −9.8] | −94.6 | 41.3% / 70.2% |
| (b) rules, else Zeb's tile | 33.3% [29.8, 36.9] | loses | −5.5 [−6.4, −4.6] | −49.5 | 50.4% / 66.9% |
| (c) as (b), Zeb's best inside a rule's set | 35.9% [32.6, 39.1] | loses | −4.3 [−5.1, −3.5] | −39.3 | 52.5% / 65.7% |
| (d) as (c), any firing rule's set | 37.9% [34.3, 41.4] | loses | −3.7 [−4.6, −2.9] | −34.6 | 54.0% / 65.6% |
| (b) with the book's bidding (`risk-budget`) | 20.7% [17.2, 24.5] | loses | −10.6 [−12.1, −9.2] | −89.8 | 41.1% / 67.6% |
| Zeb's play with the book's bidding | 39.4% [35.1, 43.7] | loses | −3.6 [−5.0, −2.2] | −31.4 | 53.0% / 62.4% |
| (t0) Zeb, the book at Zeb's exact ties | 51.0% [49.7, 52.5] | ties | **+0.88 [+0.62, +1.14]** | +8.2 [+5.8, +10.6] | 60.9% / 61.4% |
| (t1) Zeb, the book within 0.02 of Zeb's best | **53.0% [50.3, 56.1]** | **wins** | **+1.04 [+0.37, +1.74]** | +9.6 [+3.4, +16.0] | 61.5% / 61.5% |
| (t1), the registered replication (seeds 12–22) | **53.3% [50.3, 56.3]** | **wins** | **+1.40 [+0.73, +2.05]** | | |
| (t1), seeds 1–22 pooled (792 games) | **53.2% [51.1, 55.2]** | | **+1.22 [+0.76, +1.69]** | | |
| Zeb against Zeb (control, one batch) | 50% | | 0 | 0 | 62.0% / 62.0% |

- *The book loses its own game in every form that plays its rules where they disagree with
  Zeb.* Its best fair form, (d), wins 38% of games to 250.
- *The book's bidding loses in the book's game too,* at 21% with the book's play and 39% with
  Zeb's.
- *Zeb breaking its near-ties the book's way wins the book's game.* It wins 53% of games to
  250 in the first run and again in the registered replication on new deals (pooled 53.2%
  [51.1, 55.2], +1.2 settlement points a hand). The marks stay the same: −0.00 [−0.10, +0.10]
  marks a game pooled, and t1 is null in the marks game (III). At exact ties only, the edge is
  +0.9 points a hand and 51.0% of games, a tie by the band.
- *What that edge is measured against:* deployed Zeb breaks its ties by the lowest tile id,
  which knows nothing of the count. Whether the book's rules beat a tie-break that does, such
  as make-then-margin (#120), is not measured.
- *Deployed Zeb plays for the make.* The make-or-set swing dominates the book's settlement
  (below), so the book loses its game even to a Zeb that ignores every point past the make.

**The departures in the book's units.** These are the head-to-head's decompositions
re-scored by the book's settlement, with Zeb continuing. Each cell is the change in the
hand's settlement margin a departure makes (points the book team scores less the other
side's); beside it, the book's tile against a random other one.

| at the departure | n (b) | settlement / departure (b) | rule − random | (c) |
|---|---|---|---|---|
| the bidding side needs 21+ | 7,107 | −2.70 [−3.18, −2.22] | +1.98 | −2.70 |
| needs 11–20 | 3,272 | −2.24 [−2.95, −1.52] | +2.43 | −2.16 |
| needs 1–10 | 1,452 | −1.00 [−1.83, −0.18] | +1.11 | −1.23 |
| **decided** | 2,539 | **+1.27 [+1.09, +1.45]** | +0.59 | **+1.23 [+1.05, +1.41]** |
| all | 14,370 | −1.72 [−1.97, −1.47] | +1.75 [+1.37, +2.12] | −1.55 |

- *Per hand:* the book team scores −5.80 [−6.63, −4.94] settlement points against pure Zeb in
  its seats (block b), and −4.14 [−4.88, −3.39] in block c.
- *Once the contract is decided, every departing policy gains points in the book's game.*
  Take-the-trick gains +2.55 [+2.00, +3.10] settlement a departure, partner-leads +2.01, the
  donation +1.34, the pounce +1.38 and the bidder's plan +1.35.
- *While the contract is open, the departures lose much more than that.* Under the book's
  rule a set swings the hand by the bid plus the setters' points (61 points at a 30). A
  point taken after the make is worth two at most.

## III. Benefit of the doubt

Games to 7 marks, the book team against Zeb:

| book team | marks/game [95%] | verdict | settlement / hand |
|---|---|---|---|
| (c) Zeb's best inside the deciding rule's set (the head-to-head) | −1.41 [−1.60, −1.22] | loses | |
| (d) Zeb's best inside any firing rule's set | −1.20 [−1.38, −1.00] | loses | −3.5 [−4.2, −2.7] |
| (t0) Zeb, the book at Zeb's exact ties | −0.03 [−0.08, +0.02] | null | **+0.98 [+0.75, +1.19]** |
| (t1) Zeb, the book within 0.02 of Zeb's best | −0.04 [−0.20, +0.11] | null | **+1.07 [+0.48, +1.65]** |

- *The priority order is not why the book loses.* (d) bounds every fixed order with Zeb's
  tie-break inside the set from above. It is worth +0.22 [+0.07, +0.36] marks a game over (c),
  paired by cluster, and the book still loses 1.2.
- *The book as Zeb's tie-break.* Where Zeb is indifferent, the book's rule costs nothing in
  marks and gains about a point a hand by the book's settlement: it plays for the count Zeb's
  `make` utility does not see. Widening the tie to 0.02 changes neither: the marks stay
  null and the points stay.

## IV. Opponents who read plays

The reader is Zeb with its worlds drawn from a posterior that reads every other seat's plays
by the field net's set (δ = 2 points), under the consistent or the uniform likelihood (the
latter is restricted choice). Games to 7 marks:

| table | marks/game [95%] | the reader falls back to uniform worlds |
|---|---|---|
| Zeb against two consistent readers | −0.02 [−0.19, +0.16] | 15.8% of its decisions |
| Zeb against two uniform readers | +0.02 [−0.16, +0.20] | 16.0% |
| the book (b) against two consistent readers | −1.83 [−2.04, −1.62] | 20.6% |
| the book (b) against two uniform readers | −1.90 [−2.11, −1.67] | 20.9% |
| the book (b) against plain Zeb (the head-to-head) | −1.88 [−2.09, −1.66] | |

- *The pre-registered contrast is null under both likelihoods.* The reader's gain against the
  book less its gain against Zeb is C = +0.071 [−0.194, +0.323] (consistent) and −0.030
  [−0.308, +0.250] (uniform). A reading opponent is neither thrown off by the book's play more than by Zeb's,
  nor exploits it more.
- *The reader gains nothing against Zeb in this table,* so a reader of this kind had little to
  lose to the book in the first place.
- *The book's plays are harder for it to explain.* Against the book, no world explains the
  record under the net's set at 20.6% of the reader's decisions, against 15.8% against Zeb,
  mostly late (the sixth trick: 56% against 48%). The reader then plans over the uniform
  sample, which is plain Zeb.
- *An opponent who reads the book player by the book* is measured in
  [[book-reads-by-the-book-2026-10-14]] (#147): null against the book, thrown off by Zeb.

## V. The book's scope

**The book's own deals.** These are the eight narrated deals under the narrated contract, at
four rotations of the table each. Each cell gives contracts made of four, with the bidding
side's mean points.

| deal | contract | the book's line | book declares, Zeb sets | Zeb at all four | Zeb declares, book sets | book at all four |
|---|---|---|---|---|---|---|
| Hand 28 | 30, blanks | set (25) | 4/4 (30.0) | 4/4 (32.8) | 1/4 (27.8) | 1/4 (27.8) |
| Hand 29 | 30, fours | set (20) | 4/4 (40.5) | 0/4 (24.5) | 0/4 (21.0) | 0/4 (19.0) |
| Hand 30 | 30, sixes | made (42) | 4/4 (41.8) | 4/4 (36.8) | 4/4 (33.0) | 4/4 (42.0) |
| Hand 31 | 30, deuces | set (25) | 1/4 (26.5) | 0/4 (14.0) | 0/4 (21.0) | 0/4 (19.8) |
| Hand 32 | 36, blanks | set (31) | 2/4 (33.5) | 2/4 (32.5) | 0/4 (19.0) | 0/4 (31.0) |
| Hand 33 | 35, fours | made (41) | 4/4 (41.0) | 4/4 (40.2) | 4/4 (39.8) | 4/4 (41.0) |
| Hand 34 | 84, fives | made (42) | 0/4 (41.0) | 2/4 (41.5) | 3/4 (39.2) | 4/4 (42.0) |
| Hand 35 | 84, fours | not stated | 0/4 (41.0) | 0/4 (41.0) | 0/4 (41.0) | 0/4 (41.0) |
| made | | | **19/32** | 16/32 | **12/32** | 13/32 |

- *On its own deals, the book's side does at least as well as Zeb's.* Its declarers make 19
  contracts against Zeb's 16, and its setters allow 12 against Zeb's 16.
- *The largest swing is Hand 29.* The book's declarer makes the contract at all four
  rotations and Zeb's at none. Zeb's setter ducks the low trump lead in both lines; the
  difference comes later in the hand.
- *Hand 34 goes against the book both ways.* Its declarer makes the 84 at no rotation (Zeb's
  at two), and its setters allow three makes against Zeb's two.
- *Hand 28 is the book's defence.* The book's setters take the 17 the book narrates at three
  rotations of four; Zeb's setters never do.
- *A check of the encoding on its own ground.* With the book at all four seats, the player
  reproduces the narrated result at all seven deals that state one, each in three or four of
  four rotations, to the point on Hands 30, 33 and 34.
- *How much this can carry.* Hands 28–31 are one layout of tiles re-seated and re-trumped, and
  32–33 another, so the table is eight contracts on four layouts, chosen by the author to show
  his rules at work. It carries no interval and no claim beyond these deals.

**Strata of the head-to-head's departures** (marks per departure; beside it the book's tile
against a random other one, in block b):

| stratum | n (b) | block b | block c | points (b) | rule − random (b) |
|---|---|---|---|---|---|
| S-single: the deciding policy alone fires | 10,921 | −0.054 [−0.062, −0.046] | −0.048 [−0.057, −0.040] | −0.15 | +0.038 |
| S-one-tile: the rule names one tile | 9,540 | −0.051 [−0.060, −0.042] | −0.047 [−0.055, −0.039] | +0.17 | +0.033 |
| S-bid30: contract 30 | 7,353 | −0.056 [−0.065, −0.047] | −0.051 [−0.060, −0.042] | −0.33 | +0.027 |
| S-open: contract undecided | 11,831 | −0.063 [−0.070, −0.055] | −0.062 [−0.071, −0.054] | −0.68 | +0.043 |
| S-sides: the three rules Zeb's plan sides with | 1,161 | −0.047 [−0.083, −0.011] | **−0.006 [−0.055, +0.041]** | −0.24 | +0.048 |
| **S-core**: single, one tile, bid 30 | 3,861 | **−0.057 [−0.073, −0.042]** | −0.044 [−0.059, −0.030] | +0.14 | +0.033 |

- *The book does not win in scope.* S-core costs in both blocks.
- *It ties in one place.* Inside the three rules Zeb's own plan sides with on the pour, with
  Zeb's best tile inside the rule's set (block c), the cost is null.
- *The game state changes nothing.* The departures cost the same with the book team trailing
  (−0.051), level (−0.055) or leading (−0.048), late in the game (−0.053) or not (−0.051), at
  a contract of 30 (−0.056) or 31–34 (−0.046).

## VI. Selection effects

**The winner's curse.** These are the 2,653 departures of block b, seeds 1–2, each
re-planned on an independent sample.
- *Size:* the price runs −0.039 [−0.043, −0.036] logged, −0.035 [−0.038, −0.032] at seed 1,
  and −0.035 [−0.038, −0.031] at 1,024 worlds and seed 2. The curse is +0.0043 [+0.0034,
  +0.0051] and +0.0046 [+0.0039, +0.0054] of the book team's success probability: about 12%
  of the price, and a third of the forecast's excess over the realized cost.
- *The picks:* a fresh plan keeps the logged pick at 77–78% of departures and picks the
  book's tile at 10%.
- *Per policy:* every policy's fresh price stays below zero, from the crisis (−0.096) to
  follow-low (−0.009).

**The continuation.** The same departures, each tile rolled out to the end under three
continuations:

| continuation | marks/departure | points |
|---|---|---|
| Zeb at seed 0 at all four seats (the published d) | −0.056 [−0.070, −0.042] | −0.13 |
| Zeb at seed 1 at all four seats | −0.050 [−0.069, −0.031] | −0.08 |
| the book continuing (block b's player), Zeb opponents | −0.057 [−0.088, −0.028] | −0.04 |

- *The paired differences against the original are null:* +0.006 [−0.011, +0.024] and
  −0.001 [−0.027, +0.023]. The audit's own book continuation over 6,257 departures reads
  +0.015 [−0.000, +0.030].
- *So the decomposition leans toward Zeb's tile by at most a hundredth of a mark a
  departure.* The code is not the cause: under seed 0 the same rollouts reproduce the
  published d at 222 of 222 departures.
- *One policy moves:* setter-leads (−0.076 to −0.038 under either other continuation; the
  audit's +0.062 [+0.016, +0.111]). Its departures are valued better when the side that led
  the count keeps playing.

## VII. Where the book wins

The cells examined:
- the departures' per-policy, per-rule and per-contract cells of blocks b and c, each in
  marks, points and the book's settlement (807 intervals);
- the scope and game-state strata (34 cells in each unit);
- the census's labels (17);
- the head-to-head's per-rule marks of block b-rb (about 100);
- the 16 arena blocks.

Every interval above zero among them:

- **In marks, almost nowhere.** None among the departures of blocks b and c, in any stratum of
  scope or game state, or in any arena block where the book's rules decide. The single marks
  interval above zero is first-lead's *lead the top trumps* in block b-rb (+0.084 [+0.006,
  +0.166], 308 departures). It is one cell among about three hundred, which is what chance
  alone gives.
- **In the book's game, as Zeb's tie-break.** Zeb breaking its near-ties (within 0.02) the
  book's way wins 53.0% [50.3, 56.1] of games to 250, which the band reads as a win.
  The registered replication on new deals reads the same: 53.3% [50.3, 56.3], pooled 53.2%
  [51.1, 55.2] over 792 games. This is the one win the investigation found. It is measured
  against Zeb's lowest-id tie rule, not against a points-aware one (#120). At Zeb's exact ties it plays even (51.0% [49.7, 52.5]). Both score about a
  point a hand more by the book's settlement (+0.88 and +1.04). In the marks game they gain
  as much (+0.98, +1.07) while the marks stay null.
- **In points, once the contract is decided.** Every departing policy gains there. Pooled,
  a departure gains +2.28 [+1.95, +2.61] points and +1.27 [+1.09, +1.45] of the book's
  settlement (block c +1.23). Take-the-trick gains +4.84 points, partner-leads +3.00, the
  donation +2.55, the pounce +2.28.
- **In points, the rules that take or give count,** whether the contract is open or not:
  - take-the-trick +1.88 [+1.19, +2.58] points a departure, while it costs −0.092 marks;
  - the safe trick's donation +1.06 [+0.22, +1.88], null in marks;
  - setter-leads' double that calls for count +2.95, and +1.83 of the settlement;
  - the bidder's cash of his other winners +2.14, and the off's double +5.95.
- **The rules' clean disagreements with Zeb gain points** (+0.27 [+0.10, +0.45] a departure)
  while they cost marks (I).
- **On its own deals** (V), descriptively, its side outplays Zeb's declaring (19 makes to 16)
  and defending (12 allowed to 16).
- **Against chance,** the book's tile beats a random other tile by +0.064 [+0.047, +0.080]
  marks a departure (block b).
- **Ties, not wins:**
  - the three rules Zeb's plan sides with, with Zeb's best tile inside the set (block c,
    −0.006 [−0.055, +0.041]);
  - partner-leads and follow-low in block c;
  - both readers.

## What this means

- **Credit the book where it plays for count.** At a decided contract, and among Zeb's
  near-ties, the book's rules collect points that deployed Zeb throws away: its `make` utility
  does not see them, and its lowest-id tie rule ignores them. In the book's own game that is
  worth 53% of games against Zeb, replicated. The book is right about something Zeb does not
  see.
- **Where the contract is open, the book's rules lose to Zeb's plan** in both games and in
  every scope tested: about a mark a game cleanly in marks to 7, and 2.4 settlement points a
  departure in points to 250. Under the book's own scoring the make dominates, since a set
  swings 61 points at a 30 and a point after the make is worth two.
- **About half of what the head-to-head charged to the book was the encoding:** rules firing
  outside their stated scope, the stack's order, least-id picks (#146). The ranking of rules
  changes with it.
  - Encoding: low-from-a-pair, first-lead's *lead a trump* and the dead double's cash.
  - The rules: *the count that sets*, take-the-trick, *double-up, lead low* and *draw the
    trumps*.
- **The opponent the book assumes changes nothing.** A reader by the net's set neither
  punishes nor rewards the book's departures, and neither does a reader who knows the book
  ([[book-reads-by-the-book-2026-10-14]], #147).
- **Nothing is adopted.** The candidate for Zeb is a tie rule, not a book rule: break `make`'s
  near-ties for the count, as #120 proposes, and grade it against the book's tie-break in the
  book's game.

## Deviations from the pre-registration

- The re-plan at 2,048 worlds ran at 1,024 (seed 2): the compiled core caps a rung at 1,024,
  and 2,048 needs the Python referee.
- The replication of P-t1 was added after its first run and registered before any
  replication batch.
- Batches ran beside other sessions' work. 177 batches: median 123 s, the slowest 253 s, none
  at the cap, none failed. Queues were paused and resumed only at batch boundaries; two
  in-flight batches were detached from a stopped queue and completed.

## After the repairs (#146, 2026-10-14)

The census's repairs were made in the Silas book policies and the arena's stack, each rule
brought back inside its own header's scope and pinned by a `repair_` test
(`silas/tests/book_rules.rs`, `arena/test_book_play.py`). Where the census asked for a choice
the book does not make, nothing was guessed; it is recorded in `silas/GAPS.md` § The book's
silences.

| # | census label | repair |
|---|---|---|
| 1 | low-from-a-pair declines a trick (210) | silent where the higher tile takes the bidder's side's trick |
| 2 | "one off may go first" as an instruction (100) | the permission stays in the header; no rule allows the off, and the book's hands of the shape lead trump |
| 3 | top-trump handoff that cannot beat the trick (98) | the tile must beat the tile taking the trick |
| 4 | follow-with-the-off judged by its own suit (43) | the off must also be able to lose in the suit followed; a count off is a recorded silence |
| 5 | the safe trick counts setters who have played (39) | the prelude's `winner-cannot-be-beaten` asks only setters still to play |
| 6 | protect-the-count on a lead the mover beats (32) | silent where a tile of his beats the lead |
| 7 | the no-count rules choose the throw (292) | they only forbid count; the later firing policies choose within them (`arena/book_play.py`, `decide`) |
| 8 | the dead double cashed before the draw (241) | the cash waits while the bidder holds the highest trump out and is not in crisis |
| 9 | the crisis without evidence (196) | the prelude's `crisis` needs the partner shown out of trump (never the first lead) |
| 10 | least-id tile picks (557) | a trump count tile leads only when every trump is count (183); the rank of the sure trumps (156) and the order of setting leads (218) are silences, not encoded |

The dead double over shed's lone tile (344) is unresolved by the headers and kept in the
stated order. Core pins moved: twelve, listed in `silas/tests/core_pins.rs`.

### Pre-registration (written and committed before any pooled batch of this section)

- *Blocks.* b′: the head-to-head's block b (`book:zeb,least`, Zeb deployed, `heuristic`
  bidding for both teams, games to 7 marks) with the repaired encoding; P-b′: the same, to 250
  points settled by the book's rule (II). Base seeds 1–11, 36 games a batch (396 games, 198
  clusters each), every batch under `timeout 300`; a one-batch pilot at seed 900 is in no
  number. Intervals as above (cluster bootstrap, 10,000 resamples).
- *Headline.* b′'s marks/game and P-b′'s game share, each with its interval, beside the
  chance baseline: Zeb against Zeb reads exactly 0 marks and 50% of games (deterministic per
  coordinate). The book **loses** where the interval lies below it.
- *The comparison.* b′ − b (published, −1.88) and P-b′ − P-b (33.3%), paired by cluster. The
  repairs **moved** the loss where the paired interval lies above 0.
- *The forecast, from the census's costs (first order):* the repaired labels carry −0.046
  marks a hand, about 0.50 marks a game, so b′ ≈ −1.4. The same number reads off the other
  side: the clean disagreements' −1.06 plus the labels left in place (least-id picks inside a
  set holding Zeb's tile, the two silences, the dead double over the lone tile; about −0.3).
  The arena alone cannot split b′ into clean and residual without the decomposition's
  rollouts, which are not re-run; b′ is read against the forecast.
- *Descriptive:* decisions a hand each policy decides, and its departures (tile ≠ Zeb's),
  before and after.

### Results

Both blocks ran as registered: 22 batches, none near the cap (slowest 127 s), the pilot
excluded. The published block b reproduces from its files (−1.884).

| book team, rules else Zeb's tile (b) | marks/game [95%] | game share [95%] | made, book / Zeb | settlement / hand |
|---|---|---|---|---|
| published (the head-to-head) | −1.88 [−2.09, −1.66] | 25.0% | 50.0% / 67.5% | |
| **repaired** | **−1.41 [−1.63, −1.18]** | 32.1% [28.3, 35.9] | 53.6% / 66.5% | |
| repaired − published, paired | **+0.47 [+0.33, +0.62]** | | | |
| points to 250 (P-b), published | | 33.3% [29.8, 36.9] | 50.4% / 66.9% | −5.5 [−6.4, −4.6] |
| **points to 250, repaired** | | **38.4% [34.6, 42.2]** | 53.7% / 66.0% | −3.5 [−4.4, −2.5] |
| repaired − published, paired | | **+5.1 points [+2.3, +8.1]** | | |
| Zeb against Zeb (the chance baseline) | 0 | 50% | | 0 |

- *The repairs moved the loss, by what the census said they would.* +0.47 [+0.33, +0.62]
  marks a game, against the +0.50 forecast from the repaired labels' first-order cost.
- *The book still loses, in both games.* −1.41 marks a game, and 38% of games to 250, both
  intervals clear of the baseline.
- *The clean loss is where the census put it.* What remains is −1.41. The census's
  accounting has −1.06 for the rules' clean disagreements with Zeb and about −0.3 for the
  labels left in place (least-id picks inside a set holding Zeb's tile, the two silences, the
  dead double over the lone tile). The two agree. The clean part itself was not re-decomposed
  (no rollouts were run), so the −1.06 stands as the census measured it, not re-measured.
- *What changed at the table.* Per hand: departures 3.37 → 3.20, decisions a rule decides
  6.89 → 6.76.
  - the off on the first lead: 166 → 0;
  - the dead double's cash: 0.109 → 0.039 decisions a hand;
  - the crisis: 0.121 → 0.063;
  - low from a pair: 0.487 → 0.404;
  - the top-trump handoff: 0.082 → 0.056.
  The draw, the top-trump leads and the other bidder's-plan leads take up what these leave.
- *In the book's game the repairs are worth two settlement points a hand,* and five points of
  game share. The book's best fair form before the repairs, (d), won 37.9%; the repaired
  rules with no tie-break help win 38.4%.

Data: `book/encoding-repairs/` (below).

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `book/investigation/` (CC0, `SHA256SUMS`,
`CARD.md`):
- every arena block's games, book decisions and summaries (`out/`, the replication in
  `rep/`);
- the timing smokes and the reproduced head-to-head batch (`smoke/`, `regress/`);
- the departures re-scored by scope and game state (`departures.json`);
- the census's labels and their costs (`fidelity_labels.jsonl`, `fidelity_cost.json`);
- the narrated deals (`narrated.json`);
- the re-plans and continuations (`sel/`);
- `results.json` and `analysis.log`, and the scripts.

After the repairs (#146): `book/encoding-repairs/` (CC0, `SHA256SUMS`, `CARD.md`): both
blocks' games, book decisions and summaries (`out/`), the pilot, the logs, `results.json`,
`analysis.log` and the scripts.

Code: `arena/engine.py` (`points_to_win`), `arena/auction.py` (`settle_points`) and
`arena/book_play.py` (`ties=union`, `w<eps>`); the reader is 3262eea0's.

## Links

[[book-head-to-head-2026-10-13]] · [[book-study-2026-10-12]] · [[metasplat-2026-10-13]] ·
[[lumps-voi-2026-10-13]] · [[silas]] · [[scenario-player]] · [[w42]] ·
[[strategy-fusion-fence]]
