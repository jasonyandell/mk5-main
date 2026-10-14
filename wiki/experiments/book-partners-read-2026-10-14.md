---
title: Book partners read each other by the book — the receiving half of the conventions, seated (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

> **Read under [[book-authority]].** Assumption A2 of [[book-assumptions]]: the book's partners
> play one system and read each other by it. Every earlier book block seated the sending half
> only: the book team sent its conventions to a partner whose worlds stayed uniform. This page
> seats the receiving half. Pre-registered below; results follow when the batches finish.

## Question

Issue #150. In the head-to-head and every block after it, the book player (b) holds both seats
of its team, plays the book's rules, and where no rule fires plays Zeb's tile. That Zeb draws
its partner's hand uniformly. So the team pays for every convention it sends (the donation, the
top-trump lead, the partner's leads, the priorities of `partner-leads`, the calls of
`bid-only-enough`) and reads none of them. Does the book team score more when each partner
reads the other's plays and calls by the book's own conventions?

## Prior work: were the conventions ever measured with the receiver reading?

Yes, once, narrowly. The convention overlay of 2026-10-12 ([[scenario-player]] § A convention
overlay; [[book-study-2026-10-12]]) paired a sender and a reader in S/R/SR arms over 22,000
deals: two Zebs against two nets, at bid 30. The handoff's sender cost −0.013 marks/hand and its
reading recovered 0.000 ± 0.002; the donation was null in every arm. Three limits separate it
from this question:

- *The encoding.* The overlay read four hand-written gates (`handoff`, `handoff-singleton`,
  `biggest`, `donation`), not the eighteen book-faithful policies of `silas/examples/book/`. The
  gates were later found to part from the book at ten or more positions (#123).
- *The table.* Its senders were self-play Zebs with one gate switched on, not the book player.
  The receiver read one convention at a time.
- *The calls.* No arm read the auction.

The book's own team, sending and reading the whole stack, has not been seated. The within-hand
reader of [[book-reads-by-the-book-2026-10-14]] read the book player from the opponents' chairs,
with the partner unread; its gaps section names this measurement.

## The reader

`BookPlay(read=...)` (`arena/book_play.py`; spec `book:<fallback>,<ties>,read=<likelihood>[+calls]/<walt>`):
the book player whose Zeb weights its worlds by the likelihood that its **partner** played and
called by the book. The machinery is #147's (`arena/book_read.py`), pointed at the partner.

- *Plays.* At each of the partner's past discretionary plays, the book stack's verdict at the
  partner's frame of the time (its hand in the world plus the tiles it played since, and the
  record before the play), narrowed as `book_play.decide` narrows it. Consistent: weight 1 when
  the set holds the play, else 0. A play no rule decides is silence, as Silas reads a fallback.
  The least-id pick inside a set is a tie-break and is not read, as in Silas.
- *Calls.* The partner's call at its turn, and its declaration if it declared, under
  `bid-only-enough` (the arena's `heuristic`, which the book team bids with, equal call for call:
  `arena/test_book_play.py`): weight 1 when the world's dealt hand makes them, else 0, as Silas
  reads the bid ([[silas]] § The bid as evidence). The shaker's 30 into three passes is silence,
  since it may be the forced opening and the record does not show which.
- *The opponents* stay uniform, as plain Zeb holds them. The two arms differ only in what each
  book seat believes about its partner.
- *The draw* is #147's: the core samples plain Zeb's 512 uniform worlds, and the weights enter
  through the belief seam. Where the fiber holds at most 512 worlds the weights are the exact
  posterior; where it is sampled, a self-normalized importance estimate, with its ESS logged.
  An empty posterior falls back to the uniform and is counted.
- *Where the belief acts.* In (b), only where no rule fires: Zeb's tile there is planned over the
  read worlds. A rule's set is decided by its least tile, which no belief moves. In (c)
  (`book:zeb,zeb`), also inside a rule's set: Zeb's best tile there by the read q. The rules'
  guards read the frame, never a belief (`silas/GAPS.md` 1), so the book's own receiving rules
  (a guard that says "partner showed…") cannot consult the reading. That is a scope limit of this
  reader, stated before measuring.
- *The soft form.* #145 softened the set because the field net's sets starve against players
  who leave them. The book partner never leaves its own set (the stack decides its play and the
  reader asks the same stack; `test_the_partner_read_never_excludes_a_book_partner`), so the
  maximum-likelihood ε of the soft model on the partner's plays is 0, and the soft reader at its
  fitted ε is the consistent reader. It is not run separately.
- *The fence.* The likelihood is the book's verdict at the partner's frame and the bidder's
  rule on the partner's dealt hand in each world; every value is Zeb's plan over the reader's own
  cell ([[strategy-fusion-fence]]). Nothing reads the deal.

**Pilot** (seed 900, 12 games an arm, excluded from every number; seen before this
pre-registration). U took 61 s and R 120 s at a load average near 30 on 18 cores, at 8 core
threads and 8 belief processes. In R the posterior never emptied (0 of 1,113 decisions); its
ESS fraction ran 0.45, 0.36, 0.37, 0.37, 0.41, 0.58 at tricks 1–6 (the calls are evidence from
trick one). The read moved the book player's tile at 29 of 1,113 decisions with a choice (2.6%),
all of them at the 249 where no rule fired (12%). Against the 0.3.0 opponents at 512 worlds a
batch of 8 read games did not finish inside 300 s at a load near 60, which fixed the opponents'
world count below.

## Pre-registration (written and committed before any measured batch)

**Players.** Both teams bid `heuristic` (`bid-only-enough`), games to 7 marks unless stated.

- *The opponents:* deployed Zeb 0.3.0, as the package plays it: its default 160 worlds and its
  two defaults, `walt:k0,n160,F=margin_d250k_h128x64,f=16,d=posterior,r=uniform:4194304:0.28,t=margin`
  (0.3.0 is merged on main and built; [[zeb-030-confirmation-2026-10-14]]). Note that the
  opponents read the book seats softly by the field net's sets, which is not the book's model.
- *The book player's Zeb:* the head-to-head's (b), `k0,n512,F=margin_d250k_h128x64,f=16`,
  uniform draw (the belief seam needs it).

| arm | team A | role |
|---|---|---|
| U | `book:zeb,least/Z2` | (b), partner unread |
| R | `book:zeb,least,read=consistent+calls/Z2` | (b), partner read by the book |
| A | `book:zeb,least/Z2,s1` | the A/A: U with the book's Zeb at planner seed 1 |
| Uc | `book:zeb,zeb/Z2` | (c), partner unread |
| Rc | `book:zeb,zeb,read=consistent+calls/Z2` | (c), partner read |
| P-U, P-R | U and R | games to 250 points, settled by the book's rule |

- *Seeds and size.* Base seeds 1–22, 12 games a batch (6 clusters), 132 clusters an arm. One
  batch at a time, 8 core threads, 8 belief processes. Every batch under `timeout 300`; a batch
  that exits nonzero is dropped, not rerun, and listed; its clusters leave every contrast that
  uses its arm. No early stop, no top-up. Order: per seed, U, R, A; then P-U, P-R; then Uc, Rc.
- *Sizing.* The pilot's paired R − U SD was 0.67 marks/game over 6 clusters (too few to trust);
  the #147 paired contrast between two book arms implies about 1.35. Taking 1.2, 132 clusters
  give a half-width near 0.20 marks/game.
- *The scripts* (`scratch/partners-read/`, published with the data), sha256:
  `run.py` `3b2cddd1d8a08de0d56a2eb30282178322d18d989888244b7dcca917ee68c043`,
  `analyze.py` `e583c320d5d2c739a1be36a1bebae9e0721fa3152ca72e52b46ba2b4db425f48`,
  `queue.sh` `31b84cff7ba436d6329bdde7d9b49d2a6c067da9b7a7c2b1ae5e5e32ad6bccc5`.

**Measures.** Team A's mark margin per game (a cluster's two orientations averaged); game share
to 250 and settlement per game in the points arms. Intervals: 95% percentile bootstrap over
clusters, 10,000 resamples, seed 150; contrasts paired by cluster. Bars: **pays** where the
lower bound is above 0, **costs** where the upper is below 0, else **null**.

**Primary.** Δ = R − U in marks/game.

**Registered secondaries.**
- The A/A, A − U: should cover 0. It is the size of a change that moves nothing but the book
  Zeb's sample.
- Δ_c = Rc − Uc: the read where it also chooses inside a rule's set.
- Δ_P = P-R − P-U: game share to 250 and settlement per game.
- Absolute: U's and R's marks/game against the 0.3.0 field (context; Zeb against itself reads 0).

**Mechanism (descriptive, no bar).** ESS fraction, effective worlds and fallback rate by trick
(R, Rc, P-R); the share of book decisions with a choice at which the read moved the tile (the
shadow: the same decision planned unread on the same sample, in R and Rc), split by fallback and
rule-decided, and by trick.

**Exploratory (marked so when reported).** The interaction (Rc − Uc) − (R − U); Uc − U, the
price of Zeb's choice inside the rule's set against this field.

**How the result will be read** ([[book-authority]]):
- *Pays:* the book's partnership edge, invisible until both halves of its conventions were seated.
  A2 restored, the book team scores more.
- *Null:* our reader of this form cannot see it. Its omissions are then named: it reads only at
  the fallback in (b) and inside the set in (c), never in a rule's guard (gap 1); its calls are
  read under one bidding policy; the 84 defence's discards (Hand 35) wait on #128; the sampled
  fibers are importance-weighted, not drawn exactly; and the opponents are 0.3.0, who read the
  book seats by a different model.
- *Costs:* the reading is mis-specified somewhere (the encoding, presumed wrong first), and the
  reason report at the moved decisions is the next measurement.

## Results

Pending: the batches run after this commit.

## Links

[[book-authority]] · [[book-assumptions]] · [[w42]] · [[book-reads-by-the-book-2026-10-14]] ·
[[book-head-to-head-2026-10-13]] · [[book-investigation-2026-10-14]] ·
[[reads-plays-soft-2026-10-14]] · [[book-study-2026-10-12]] · [[zeb-030-confirmation-2026-10-14]] ·
[[silas]] · [[scenario-player]] · [[strategy-fusion-fence]]
