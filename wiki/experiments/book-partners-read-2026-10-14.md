---
title: Book partners read each other by the book — the receiving half of the conventions, seated; +0.10 [−0.06, +0.26] marks/game, a null at this power (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-22
status: complete
---

> **Read under [[book-authority]].** Assumption A2 of [[book-assumptions]]: the book's partners
> play one system and read each other by it. Every earlier book block seated the sending half
> only: the book team sent its conventions to a partner whose worlds stayed uniform. This page
> seats the receiving half. Pre-registered at `cf89472b`, before any measured batch.

## The answer, in short

- **Seated against deployed Zeb 0.3.0 in an arena where no seat models another across hands
  (A1), opponents never lose track (A3), the field has no bidding styles (A6) and only pip trumps
  are declared (A10), with A2 restored by this reader, the book team whose partners read each
  other by the book scores +0.10 [−0.06, +0.26] marks/game more than the same team unread.**
  That is null at this power. The A/A (a change of the book Zeb's sample alone) reads
  −0.05 [−0.19, +0.09].
- **Every registered contrast points the same way and none clears its bar.** Inside the rule's
  set, form (c): +0.15 [−0.02, +0.33]. To 250 points: +1.9 [−1.1, +4.9] points of game share and
  +1.7 [−4.1, +7.6] settled points a game.
- **The reader is well fed and rarely acts.** Its posterior never starves (0.02% fallbacks;
  about 190 effective worlds of 512 through trick 4, the exact posterior from trick 5). It moves
  the book player's tile at 3.3% of decisions with a choice in (b) and 7.7% in (c).
- **What this rules out is narrow.** A reader of this form, with Zeb choosing at the fallback or
  inside a rule's set over a partner posterior, does not lift the book team by more than about a
  quarter of a mark. The book's receiving rules, which would need a belief inside a rule's guard,
  are not seated (`silas/GAPS.md` 1). That is the measurement that would seat them.

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

All 154 batches ran as registered and exited 0 (slowest 185 s, at load averages of 18–77 on 18
cores beside other sessions), so no batch was dropped. The scripts' hashes at analysis match the
registered ones. Run 13:16–17:04, 2026-10-14, at `cf89472b`.

| arm (team A against deployed Zeb 0.3.0) | marks/game [95%] | games won to 250 | made, A / B | A declares |
|---|---|---|---|---|
| U: (b), partner unread | −1.44 [−1.68, −1.19] | | 52.9% / 65.8% | 49.7% |
| R: (b), partner read by the book | −1.34 [−1.60, −1.08] | | 53.2% / 65.4% | 49.9% |
| A: U at planner seed 1 (the A/A) | −1.49 [−1.73, −1.25] | | 52.4% / 65.8% | 50.0% |
| Uc: (c), partner unread | −1.12 [−1.36, −0.87] | | 54.0% / 64.2% | 50.2% |
| Rc: (c), partner read | −0.97 [−1.24, −0.69] | | 54.9% / 63.9% | 50.4% |
| P-U: U, to 250 | | 31.8% [27.3, 36.4] | 53.5% / 65.9% | 49.8% |
| P-R: R, to 250 | | 33.7% [29.2, 38.3] | 54.2% / 65.8% | 50.1% |
| baseline: Zeb against itself | 0 | 50% | | |

Each arm is 264 games, 132 clusters.

| contrast (paired by cluster, 132 clusters) | estimate [95%] | bar |
|---|---|---|
| **primary: R − U, marks/game** | **+0.10 [−0.06, +0.26]** | null |
| A/A: A − U, marks/game | −0.05 [−0.19, +0.09] | null (as it should be) |
| Δ_c = Rc − Uc, marks/game | +0.15 [−0.02, +0.33] | null |
| Δ_P = P-R − P-U, game share to 250 | +1.9 [−1.1, +4.9] points | null |
| Δ_P, settled points/game | +1.7 [−4.1, +7.6] | null |
| *exploratory:* (Rc − Uc) − (R − U) | +0.05 [−0.13, +0.25] | null |
| *exploratory:* R − A | +0.15 [+0.00, +0.31] | at the edge; not registered as a test |
| *exploratory:* Uc − U, Zeb's choice inside the rule's set | +0.32 [+0.10, +0.54] | pays |

- *The primary is null.* Its half-width came out at 0.16, tighter than the 0.20 planned. It
  would have shown an effect the size of the soft reading's (+0.37). It cannot separate +0.10
  from 0.
- *The signs agree.* R − U, Rc − Uc and both points contrasts are positive, and R − A sits at
  the edge of its interval. Read together, they suggest a small effect below this study's power.
  The registered verdict stays null.
- *Uc − U* repeats the head-to-head's (c) − (b) (+0.47 there, against 0.2.0) against the 0.3.0
  field: Zeb's choice inside a rule's set beats the least tile by +0.32 [+0.10, +0.54].

**Mechanism** (every book decision with a choice; R, Rc and P-R agree to a point):

| trick | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ESS fraction of the partner read | 0.48 | 0.38 | 0.37 | 0.38 | 0.44 | 0.60 |
| effective worlds | 246 | 196 | 189 | 181 | 93 | 10 |
| fallback | 0% | 0% | 0.1% | 0% | 0% | 0% |
| exact fiber | 0% | 0% | 0% | 13% | 76% | 100% |
| read moved the tile, (b) | 2.8% | 4.2% | 4.6% | 4.2% | 3.3% | 1.1% |
| read moved the tile, (c) | 11.4% | 11.4% | 10.0% | 7.9% | 4.8% | 1.3% |

- *The calls carry the early evidence.* At trick 1 the partner has made at most one play, yet the
  ESS fraction is already 0.48: the partner's call and declaration under `bid-only-enough`
  remove about half the sampled worlds.
- *In (b) the read acts only where no rule fires.* That is 22% of the book player's decisions with
  a choice. At 15% of those it moves Zeb's tile, which is 3.3% of all its decisions, about
  three tiles a game. Inside a rule's set the least tile never moves, by construction. In (c) the
  read also moves 5.6% of rule-decided tiles, 7.7% of all decisions.
- *Per moved tile* the primary's estimate is about +0.03 marks. That is the size of a single
  departure's effect in the head-to-head's decomposition, so the read is not inert where it acts.
  It acts too rarely to show in marks at 132 clusters.

## What this means

- **A2, as this reader restores it, does not visibly lift the book team.** Its partners now read
  each other's plays and calls by the book, the posterior is right (the real deal never
  excluded) and well fed, and the team scores +0.10 [−0.06, +0.26] marks/game more. That is the
  scope of our model: a reader that uses the partner posterior only to choose where the book is
  silent (b), or inside the set the book allows (c).
- **The book's receiving half is mostly inside its rules, and our rules cannot read.** The
  conventions this page sets out to credit are receiving rules: the partner who knows the trick
  is safe because the top trump was led (S-0041-a/b), the leads that answer a partner's signal
  (S-0057-a..d), the 84 defender's watch over a partner's discards (S-0150-d..i). In the Silas stack
  a rule's guard reads the frame, never a belief (gap 1). The stack's guards already compute
  "provably safe" from the record alone, so the reading here adds information only where the
  record leaves a placement open and Zeb, not a rule, decides. The measurement that would seat
  the book's receivers is a guard that may read a partner posterior (#128's policy likelihood,
  `silas/GAPS.md` 1), and the 84 defence, which needs it most (#128).
- **Nothing is adopted.** The partner read is an arena option (`read=`); Zeb's defaults are
  unchanged.

## Deviations from the pre-registration

None in the arms, seeds, batch size, bars or analysis. The queue ran per seed in the registered
arm order (U, R, A, P-U, P-R, Uc, Rc).

## Caveats

- **Opponents at 160 worlds.** The 0.3.0 opponents play as the package does, at 160 worlds; the
  book's own Zeb plans at 512. Both arms of every contrast face the same opponents.
- **One bidding policy read.** The calls are read under `bid-only-enough`, which is how the book
  team bids here. Under another book bidding policy (`risk-budget`) the read would be another
  likelihood. The plays and the calls were not separated.
- **The soft form was not run.** Its fitted ε on a book partner is 0 (above), so it is this reader.
- **Importance weights on a uniform sample.** Exact draws on sampled fibers were not used, and
  the planner's cell floor counts worlds, not weight (as in [[book-reads-by-the-book-2026-10-14]]).
- **The exploratory signs** (R − A at its edge; all four contrasts positive) are not a finding.
  A replication at about four times the clusters would be needed to resolve +0.10.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `book/partners-read/` (CC0, `SHA256SUMS`,
`CARD.md`): every game of every arm (`out/<arm>/s<seed>.games.jsonl.gz`), every partner-read
decision's ESS row (`*.reads.jsonl.gz`) and shadow row (`*.moved.jsonl.gz`), the pilot, the logs,
`results.json`, `analysis.log` and the scripts. Code: `arena/book_play.py` (`read=`, the
shadow), `arena/book_read.py` (`seats`, `calls_weight`), `arena/test_book_read.py`.

## Links

[[book-authority]] · [[book-assumptions]] · [[w42]] · [[book-reads-by-the-book-2026-10-14]] ·
[[book-head-to-head-2026-10-13]] · [[book-investigation-2026-10-14]] ·
[[reads-plays-soft-2026-10-14]] · [[book-study-2026-10-12]] · [[zeb-030-confirmation-2026-10-14]] ·
[[silas]] · [[scenario-player]] · [[strategy-fusion-fence]]

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
