---
title: Make, then margin — Zeb's exact ties broken by the plan's mean margin (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Question

Issue #120. Under the `make` utility over 512 worlds, exact ties are common, and Zeb's tie
rule (≺ in [[scenario-player]]'s Θ) takes the lowest tile id. Two studies point at those
ties. In [[metasplat-2026-10-13]], at a decided contract the book's tile has the higher mean
margin in 44% of departures against Zeb's 21%. In [[book-head-to-head-2026-10-13]], book
departures at a decided hand cost no marks and gain +2.28 points each. Three questions
follow. Is the book's tile the margin-best tile of Zeb's tied set? Does a lexicographic rule
(make, then the plan's margin) change Zeb's strength? And how much does it raise Zeb's
book agreement?

## Pre-registration (committed before any measurement)

**What a tie is.** At a Walt decision, the tied set T is the legal tiles whose make value q
equals the best q exactly, as a double, in the decision's own sample. That sample is 512
worlds at the deployed size, or the whole fiber when the fiber is smaller. There is no noise
bar. Under uniform weights q is (worlds made)/n, so a tie means equal counts of worlds made.
A noise bar would let the rule give up a measured make edge for margin, which changes the
objective rather than breaking a tie. Every tie in this study is that exact tie.

**The rule, `make-then-margin`** (`Config.tie = "margin"`; arena `t=margin`; default
`"lowest"`). When |T| ≥ 2 the decision picks the tile of T with the best mean margin for its
own side under the plan: M(t) = Σ_r w_r·m_r(t). Here m_r(t) is the bidder-team final margin
that scenario r reaches when the root plays t and the plan continues. That is
`Analysis.outcomes`: the plan's line, then the tape where the plan stops (contract decided,
or horizon). The bidding side takes the largest M, a setter the smallest, and ties in M go
to the lowest tile id. The rule applies at every decision a Walt makes: the live seat's
root, every modeled rung's decision, and a convention sender's restricted choice. The plan
itself is unchanged: its interior choices still take the lowest id among equal values, so
q is unchanged wherever no modeled rung decides. It is `make` only. Both engines,
bit-identical; default off; Zeb's fixture untouched.

**Step 1, the tied sets.** Positions: the face-off's 20,977 gated rows (#119;
`scratch/faceoff/out/positions.jsonl`, 21 gates of about 1,000 rows each). Each row is
re-planned by deployed Zeb at current code (`walt:k0,n512,F=margin_d250k_h128x64,f=16`,
compiled, seed 0). The population is the rows where |T| ≥ 2 and T holds a book tile.
- **Headline:** the share where the margin-best tile of T is a book tile.
- **Chance:** the expected share for a uniformly random tile of T, |T∩B|/|T|.
- **Beside it:** the lowest-id rule's share.
- **Subsets:** the rows where the lowest-id tile is outside the book (the ties the book
  lost); dead rows (every legal tile ties) against live rows; and the share where the
  margin does not separate the tiles.

Rows are pooled with equal weight, as in the face-off. The 95% intervals are percentile
bootstraps over deals (chunk, deal), 10,000 resamples.

**Step 3, the book-agreement column.** On all rows, book agreement for re-planned Zeb under
each rule, with the exact random-tile baseline, plus the paired difference with its deal
interval. The face-off's pour figure (51.1%) is quoted beside them; it came from an older
player.

**Step 2, the grade.** Arena: `bid30` both sides (the hub's protocol), games to 7 marks,
each deal played in both orientations. Batches are capped at 300 s and pooled over base
seeds 1, 2, …. A cluster is one game index of one batch, both orientations. The headline is
marks/game for the challenger, with a 95% percentile bootstrap over clusters (10,000).
Points per hand stand beside it, and a game to 250 points if the arena supports one. The
challenger is deployed Zeb with `t=margin`, graded on two blocks.
- **G1: against deployed Zeb**, at least 768 games.
- **G2: against the champion `k1,n512,m8`**, at least 384 games. The standing block for
  plain Zeb is +0.04 [−0.27, +0.33].

**Decision.** The rule becomes a default, or a new zeb version, only if G1's marks
interval clears zero. Even then this page proposes the change and does not make it.
Points alone do not move a default. The champion stays `k1,n512,m8` unless G2 clears zero.

## Links

[[scenario-player]] · [[metasplat-2026-10-13]] · [[book-head-to-head-2026-10-13]] ·
[[book-study-2026-10-12]]
