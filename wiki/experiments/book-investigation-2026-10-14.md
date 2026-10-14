---
title: The book against Zeb, investigated — the method, the book's own game, its opponents, its scope (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
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

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `book/investigation/` (CC0, `SHA256SUMS`).

## Links

[[book-head-to-head-2026-10-13]] · [[book-study-2026-10-12]] · [[metasplat-2026-10-13]] ·
[[lumps-voi-2026-10-13]] · [[silas]] · [[scenario-player]] · [[w42]] ·
[[strategy-fusion-fence]]
