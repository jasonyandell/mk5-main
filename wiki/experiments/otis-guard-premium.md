---
title: Otis guard-premium probe — is the junk two worth more when it guards the 3-2?
kind: experiment
first_seen: 2026-07-15
last_updated: 2026-07-15
status: complete
---

## Question

The [[count-fate-ledger]]'s founding example, asked as a measurement: does a
junk two (2-0, 2-1, or non-trump 2-2) carry a measurably higher retention
price **because** the hand also holds the 3-2 it guards? Jason's framing,
verbatim: *"protection/threats, refining strategy to take advantage of
them."* This probes the protection half — the guard's price conditional on
the thing it guards.

Night 2 context ([[otis-phase-1]]): the retention override tied (V1) /
lost (V2) at power, so instrument prices are NOT table-value claims. This
probe is a **legibility** measurement: do the instruments see the guard
relationship at all? [[otis-v0]] W6 established the category (retention
prices exist; fusion gap +2.99 vs clairvoyant — best cell was a *blank*
guard, keep 0-0, and 2 of 3 qualifying cells sat on now-purged decl 8).
The specific conditional contrast — junk-2 premium **given** 3-2 in hand —
has never been measured. If it doesn't register, we have the wrong test,
and that is a graded outcome, not a failure.

## Design (registered before any pricing run)

**Cells**: on-policy incumbent self-play (`margin:wp(r8)+lens:ev` both
sides, n_samples=10, MPS, fresh seed block 11000000), collect triggered
slough decisions (the [[otis-phase-1]] W6 predicate) where additionally:
the actor still holds the 3-2 (unplayed), **exactly one** junk two is
among the slough candidates, at least one non-two junk candidate exists,
and decl ∉ {twos, threes} (the 3-2 must be a countable non-trump tile for
the guard story to be well-posed).

**Twin (matched pair)**: same state with the 3-2 swapped out of the
actor's hand for a *neutral* tile from an opponent's remaining hand —
non-count, non-double, pips disjoint from {2, 3, trump pip, current led
suit pip}, unplayed. The full play history must replay legally
(follow-suit) for both swapped seats under the twin deal; try all
opponents × qualifying tiles, drop the cell if no legal twin exists.
The twin differs from the real state by exactly one tile in the actor's
information state: the threat (3-2) is present in arm A, absent in arm B.

**Retention margin** of tile g at state s with candidate set C:
`margin(g; s) = max_{c ∈ C\{g}} price(c) − price(g)` — how much better
discarding the best alternative is than discarding g. Positive ⇒ keep g.
Computed over arm A's candidate set in BOTH arms (the neutral tile is a
legal extra discard in arm B; it is excluded from the primary metric and
reported descriptively).

**Guard premium** per cell = `margin(junk2; A) − margin(junk2; B)`.

**Pricers** (both run on every cell): V1's tied-rollout pricer
(`arena/slough_override.py` internals verbatim: repaired MRV sampling
M=50, gus v3 belief weights, common random worlds per arm) and V2's fate
head (`otis/models/otis_play_v0.pt`). G3 additionally reads the mechanism
channel directly: P(my team captures the 3-2) from the fate head on each
candidate's successor in arm A.

## Predictions — registered before any run

| # | claim | band (pass) | wrong-test / falsifier | graded |
|---|---|---|---|---|
| M-A | The context exists on-policy | ≥ 100 legal matched cells from ≤ 512 collection games | < 30 cells in 512 games — the conditional context is too rare on-policy for this harness; probe redesign needed (constructed fixtures) | **PASS — 620 cells / 512 games** (4.5% of 13,806 triggers; 487 priced after 21% twin-drop) |
| G1 | Tied rollouts price the guard conditionally | mean guard premium > 0, 95% bootstrap CI excluding 0, N ≥ 100 cells. HOPE: mean ≥ +0.3 pts (order of W6's median junk spread 0.68, discounted for an interaction) | N ≥ 150, CI includes 0, \|mean\| < 0.15 — a powered ≈0: the tied instrument does not see the guard premium at this surface; **we have the wrong test** | **NEITHER — SIGN SURPRISE: mean −0.156, CI [−0.296, −0.006], N=487** — a CI-excluding *negative*: the tied price of the junk two falls when the 3-2 is in hand, concentrated early (tricks 0–1: −0.460 [−0.695, −0.227]); late (2+): +0.033 [−0.147, +0.220], a tie |
| G2 | The fate head prices the guard conditionally | same contrast on fate-head margins: mean > 0, CI excluding 0 | same powered-≈0 shape — the learned instrument does not encode the guard | **POWERED ≈0, positive lean — mean +0.0072, CI [−0.0017, +0.0164], N=487** (fate-ledger pts); the net premium does not register |
| G3 | The mechanism channel is visible | in arm A, P(my team captures 3-2) is higher under keep-junk-2 than discard-junk-2 in ≥ 60% of cells, mean Δ > 0 | ≤ 50% of cells (coin flip) — the fate head never learned the guard channel; G2 pass would then be for the wrong reason | **PASS — 61.0% of cells, mean Δ +0.0069** capture-prob; strongest early (tricks 0–1: 68.3%, +0.0115) — the protection channel is real and learned, worth ≈0.03–0.06 pts of 3-2 value |

Caveat registered up front: a PASS is an **instrument-legibility** result.
Night 2 measured these same pricers claiming +0.75 pts per override that
the table did not pay — any premium found here inherits that calibration
question until realized outcomes referee it ([[otis-phase-1]]'s exported
constraint).

## Receipts

### The run (2026-07-15, same night as registration)

Collection: 512 incumbent self-play games (seed 11000000), 105 s, 5,663
hands, 13,806 triggers, 620 qualifying cells
(`scratch/otis-night2/collect_guard_cells.py` in the night-2 worktree).
Pricing: 487 matched pairs priced in 412 s (133 dropped — no legal neutral
twin, mostly late-hand); both pricers verbatim from [[otis-phase-1]]
(`price_guard_cells.py`); receipts `guard_premiums.jsonl`, grades
`guard_grades.txt`, curated to `champion/evidence/otis_guard_premium_2026-07-15/`.

### Reading

The probe registered — decisively — and the sign is the surprise. Three
facts, one picture:

1. **The protection channel exists and is learned** (G3): keeping the junk
   two raises the fate head's P(my team captures the 3-2) in 61% of cells
   (68% early), worth ≈0.03–0.06 pts. The guard is real.
2. **The net conditional price runs the other way, early** (G1): with the
   3-2 in hand, the tied-rollout retention price of the junk two *drops*
   by ≈0.46 pts in tricks 0–1 (CI excludes zero). Retention margins are
   positive in BOTH arms (+0.32 with 3-2, +0.48 without) — keeping junk
   is generally priced good; it is the *conditional* premium that inverts.
3. **Late-hand, the contrast is a tie** — the direction the guard story
   wants, but unresolved at this N.

So the protection channel is an order of magnitude smaller than an
opposing early-hand effect the tied policy prices. Candidate mechanisms
(unidentified, filed as follow-up): coverage redundancy (the 3-2 already
covers twos-follows, making the junk two's cheap-follow role duplicative
while both block twos-voidness), comparator interactions (holding 5 extra
count shifts the *relative* value of the other discards), or
count-delivery planning (with 3-2 in hand the tied policy may prefer
engineering the void to slough the 3-2 onto partner's winners — for which
the junk two is an obstacle, not a guard). Registered caveat applies with
force: [[otis-phase-1]] measured these instruments claiming +0.75 pts the
table did not pay — every sign here is an instrument reading, refereed by
nothing realized yet.

## Links

[[otis]] · [[otis-phase-1]] · [[otis-v0]] · [[count-fate-ledger]] ·
[[strategy-fusion]] · [[gus]]
