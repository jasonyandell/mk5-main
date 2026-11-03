---
title: "A field net written by hand: every weight a rule, no gradient (Lem)"
kind: experiment
first_seen: 2026-11-02
last_updated: 2026-11-02
status: complete
---

Jason's commission (2026-11-02): write Zeb's field net from scratch, choosing every weight by hand.
Each neuron is a rule in the book's words, compiled into integer weights. Nothing in the net is
trained or fine-tuned. The generator is `lem/handnet.py` (CC0); `python -u -m lem.handnet OUT`
writes `hand.npz`, `hand.fnet` and `neurons.md`, the table of every unit.

## The net

field-v2 (the shipped layout: pip trumps only, the canonical frame where trump is pip 6), shifts 0,
**150 layer-1 units × 124 layer-2 units** (layer-1 units no rule reads are pruned). The shipped net is 128 × 64.

- **Layer 1 reads the table.** Each unit is "all of these inputs, none of those": weights ±1 or −2
  and a bias, so its value is exactly 0 or 1. A count ("a trump is still out", "I hold three trumps")
  is a step pair: two units whose difference is the 0/1 step. Families: position (lead, follow, third,
  last), who holds the trick, the bidding side, the declaration (six gates, one per canonical count
  layout; twos and threes put the count in the same slots, so the frame cannot tell them apart), the
  led suit by who led it, trump out, trump length, count on the trick (one step pair per layout), each
  tile's "no live tile outside my hand beats it in its suit" (walker / boss trump), each tile's
  "nothing on the trick beats it in suit s", and "partner led a tile nothing live beats".
- **Layer 2 states the rules.** Each unit is a conjunction of layer-1 literals, exactly 0 or 1:
  take the trick from an opponent (per tile and suit), take with count, secure partner's beatable lead
  with trump, feed partner's safe trick (last seat; third seat when partner's lead cannot lose its
  suit, at a third of the dose), keep count off an opponent's trick, don't lead count, draw trump
  (bidders with two trumps, anyone with three; the boss trump first), lead a walker (doubles first).
- **The output ranks tiles.** A rule votes for its tiles with its priority (take 4000 > secure 3200 >
  pull 2800 > walk-double 2600 > walk 2400; count 60 a point). The static bias plays the weakest tile:
  offs before trumps, singles before doubles, low pips first.

Every unit is 0 or 1 on all 86,495 dev rows and on 400 random states (`lem/tests/test_handnet.py`);
the compiled core (`scenario/core/src/field.rs`) gives the referee's scores to the integer.

**How the priorities were chosen.** By hand, on a dev split the graded rows never touch: pour 0's
deals with `deal % 10 == 1` (training deals for the shipped net; 86,495 rows), scored by mean regret
against Walt's `q`. Seven rounds, every variant listed in `variants.md` on the Hub; frozen at
dev regret 0.0150 (shipped 0.0092 on rows it trained on, random legal 0.0418). Rules that lost on
dev were removed, not kept at zero.

## Two arms (the commission's addendum, 2026-11-02)

The protocol mirrors Rikiya T.'s weight-space programming ([post](https://rikiyat.github.io/blog/weight-space-programming/)):
- **Zero forward passes** (`build(zero=True)`, 100 × 85): the rules as first written, from the input
  layout and the book alone, before the net was ever run: take, feed last, keep count, don't lead count,
  draw with the boss trump, draw, walk, the static ranking. It is exactly v1 (dev regret 0.0175, the same
  number before and after the code was refactored to hold both arms).
- **Budgeted forward passes** (the frozen net, 150 × 124): edits kept only where they lowered dev regret.
  Budget spent: **26 forward passes of candidate nets over the 86,495 dev rows (18 distinct candidates)**,
  plus 5 reads of Walt's choices on dev rows that ran no net (`variants.md`). One deviation from
  "keep an edit only if it wins": round 2 added five rules as one bundle, and later rounds tuned or
  removed them one at a time.

## Grading plan, registered before any graded number

All on the audit's **434,535 held-out rows** (the five training pours' `deal % 10 == 0`,
`scratch/lem/audit-2026-10-15/sample.npz`), never seen while writing the rules.

**G1. Fidelity to Walt.** For both arms, the shipped net, a trained net of the same width
(165 × 124, started before pruning settled the width at 150; the squeeze recipe: `lem.probe squeeze-train`, the same five pours, 60 epochs, seed 0),
its untrained twin (epoch 0) and uniform random legal play (analytic):
- mean regret in make probability, with a deal-clustered bootstrap 95% interval (`lem.probe.fidelity`);
- exact match with Walt's chosen tile, and zero-regret agreement (q ties count);
- both by trick position (lead, second, third, last) and by list-v2 concept (`labels_shipped_v2.npz`);
- the share of the random-to-shipped regret gap the hand net closes.

**G2. The arena.** Zeb with the hand field against Zeb with the shipped field: both
`walt:k0,n512,F=<net>,f=16`, heuristic bidding on both sides, mirrored game pairs (`arena.cli`'s
rotated halves), games to 7 marks, base seeds 1–24 at 16 games each (384 games), one `cpu-slot` call
per seed. FAIR CLOCKS: both decide by worlds, never wall time. Report marks per game with the arena's
bootstrap interval, games won, and the minimum detectable effect at this N. A tie is not agreement.

**G3. The instrument check.** Audit the hand net with the probe kit as the unit papers read nets
(`lem.probe audit … --baseline n_plays`, list v2, read raw with the play-count baseline, #233). Beside
it, two nulls read the same way: the untrained 165 × 124 twin (epoch 0 of the trained twin's recipe),
and a structure-matched random net (`handnet.random_twin`, seed 0: every unit a conjunction of the same
arity and signs over random inputs, so it is as boolean as the hand net and means nothing). The audited
copy carries three **decoys**, layer-1 units nothing reads (play is unchanged, checked on all 434,535
rows): `decoy-offshoot` = 2·partner-winning + 1·third (an idea and a weaker related facet; key:
`partner-winning`), `decoy-unrelated` = partner-winning + "fives are trump" (key: the audit should not
call it clean), `decoy-near-miss` = lead ∧ no double held (key: `lead` is a facet; calling it clean
`lead` is over-broad naming).

**Ties.** A hand unit's pre-activation is a small integer, so its top 1% is mostly a tie broken by row
order, and the sample runs in pour order: a unit on in more than 1% of rows takes its top set from
pour 0's earliest deals, where a per-deal concept like `trump-is-X` can win the label by accident. Many units are on in under 1% of rows; for them the top set is padded with near misses, and the true
rule's precision cannot exceed on-rate ÷ 0.01 (`onrates.json` on the Hub, computed before the audit).
So the audit runs twice: as the instrument stands, and on the sample with its rows permuted by a fixed
seed. The difference between the two is itself a finding about the instrument. Score the audit's labels against the
answer key below, written before the audit runs:
- **M1 exact recovery:** of the units whose rule is exactly a v2 concept, the share whose `+` pole
  label is that concept; also with the play-count-excluded column counted (`label or excluded`).
- **M2 facet recovery:** of all live units, the share whose `+` label is one of the rule's conjuncts
  that v2 names (or the raw input the unit copies).
- **M3 false naming (the headline):** of the per-tile units (walker, boss trump, tops, partner-led-top,
  take, secure, walk, boss), which no v2 word names, the share the audit calls *clean*; and over all live
  units, the share called clean with a label outside the key's set (denominator: live units; an `x:`
  label counts as correct only where the key lists it).
- **M4 the on/off column (#236):** M1 again, with the unit's best on/off match in place of the label.
- The clean share against both nulls, beside the shipped net's. Position units (`lead`, `third`, `last`,
  `led-…`) are reported apart, since the play-count baseline can never grant them (#236).
- `−` poles are reported descriptively, not scored: a hand unit's shut side is "one or more required
  inputs missing, or a forbidden one present", which list v2 mostly has no word for.

**What G3 can and cannot show.** It is lopsided. Good recovery does not license the audit's 0.597 on a
trained net, whose units are graded dials with offshoots, not saturated detectors built from list words.
Bad recovery does count against the audit.

**The answer key** (unit family → exact v2 concept, if any; acceptable facets):

| family | exact | facets |
|---|---|---|
| `lead`, `third`, `last` | `lead`, `third`, `last` | same |
| `follow` | none (second ∪ third ∪ last) | `second`, `third`, `last` |
| `partner-winning` | `partner-winning` | same |
| `opponent-winning` | none | — |
| `bidding-side` | `bidding-side` | same |
| `decl-X` | `trump-is-X` (twos/threes: none) | `trump-is-twos`, `trump-is-threes` |
| `led-trump@1`, `@3` | `second-trump-led`, `last-trump-led` | `trump-led` |
| `led-trump@2`, `led-s@n` (pips) | none | `trump-led` for trump; position by n |
| `trump-out±` | `trump-out` | same |
| `trumps-3+±` | `trumps-3plus` | `holds-trump` |
| `trumps-2+±` | none | `holds-trump` |
| `count-on-trick:X±` | none (∧ the declaration) | `count-on-trick`, `trump-is-X` |
| `ten:t`, `five:t` | the raw count input | `x:count…` of t |
| `walker:t`, `boss trump:t`, `tops:t/s`, `partner-led-top:u` | none | — (no v2 word: these are per-tile) |
| `take…`, `take-ten/five…` (layer 2) | none | `can-take-it` only where t is held; position, `trump-led` |
| `secure:t` | none | `third-partner-winning`, `third`, `partner-winning` |
| `feed:X` | none (`last-partner-winning` ∧ the declaration) | `last-partner-winning`, `last`, `partner-winning`, `trump-is-X` |
| `feed-third:X` | none | `third-partner-winning`, `third`, `partner-winning`, `trump-is-X` |
| `keep-count:X` | none | `trump-is-X` |
| `lead-no-count:X` | none | `lead`, `trump-is-X` |
| `pull`, `pull-long` | none | `lead`, `bidding-side`, `trump-out`, `trumps-3plus` |
| `boss:t` | none | `lead`, `trump-out`, `lead-boss-trump` |
| `walk:t` | none | `lead` |

## Predictions, sealed before G1–G3 run

0. **The zero-pass arm**: held-out regret in [0.015, 0.020], above the budgeted arm's.
1. **Replication, not prediction.** Dev regret was 0.0150, closing 0.82 of the random-to-shipped gap
   on dev. On the held-out rows: mean regret in [0.013, 0.018]; gap closed 0.75–0.88.
2. **Match.** The wiki's 0.754 for the shipped net is zero-regret agreement (`fidelity.run`'s `agree`);
   its exact match with Walt's tile was 0.529 on dev. Hand exact match within 0.03 of the shipped net's
   exact match; hand zero-regret agreement 0.66–0.71.
3. **Where.** Per row, the hand net's excess regret over the shipped net (hand − shipped, mean within
   the region) is largest on the lead.
4. **The trained twin** (one seed; the squeeze recipe, 60 epochs, which is not exactly the shipped
   net's): regret within 0.002 of the shipped net, its interval clear of the hand net's.
5. **The arena.** Game margins have SD ≈ 5–6 marks; with 384 games in mirrored pairs the 95% half-width is
   about 0.55 and the minimum detectable effect (80% power) about 0.8 marks per game. Prediction: the
   interval includes 0, and its lower end is above −0.8 (the hand field costs Zeb less than 0.8
   marks a game). A field effect smaller than the MDE is invisible here; a tie is not agreement.
6. **Recovery.** Over the non-position exact units, M1 by label ≥ 0.7. Over position units, label
   alone ≤ 0.2 and label-or-excluded ≥ 0.8.
7. **False naming.** Of the per-tile units, ≥ 0.3 are called clean by some v2 word (most often a
   facet: `lead`, a position, `trump-is-X`); over all live units M3 ≥ 0.15. The shuffle changes at least
   10 labels.
8. **The boundary column.** M4 exceeds M1 by label by at least 0.1 (a bootstrap over units for the
   interval).
9. **Decoys.** The offshoot is labelled `partner-winning`; the near miss is called clean `lead`
   (over-broad, as expected of a top-set reader); the unrelated OR is not called clean.
10. **Nulls.** The hand net's clean share exceeds the random conjunction net's by at least 0.2; the
    random net's own share is above the untrained twin's (booleanness alone buys some cleanliness).

## Results

Run 2026-11-02 as sealed (70988a02), on the 434,535 held-out rows. Evidence on the Hub (link below).

**G1. Fidelity to Walt** (mean regret in make probability, deal-bootstrap 95%; chance is uniform legal play):

| net | weights | regret | lead | 2nd | 3rd | last | exact match | zero-regret agree |
|---|---|---|---|---|---|---|---:|---:|
| random legal | — | 0.0413 [0.0410, 0.0417] | 0.0455 | 0.0346 | 0.0379 | 0.0450 | 0.335 | — |
| untrained 165 × 124 | random init | 0.0434 [0.0430, 0.0438] | 0.0539 | 0.0329 | 0.0361 | 0.0445 | 0.329 | 0.550 |
| **hand, zero passes** (100 × 85) | by hand | **0.0170** [0.0168, 0.0172] | 0.0184 | 0.0146 | 0.0214 | 0.0126 | 0.524 | 0.685 |
| **hand, budgeted** (150 × 124) | by hand | **0.0143** [0.0141, 0.0145] | 0.0175 | 0.0133 | 0.0138 | 0.0108 | 0.535 | 0.700 |
| shipped (128 × 64) | trained | 0.0084 [0.0083, 0.0086] | 0.0102 | 0.0083 | 0.0076 | 0.0066 | 0.533 | 0.754 |
| trained twin (165 × 124) | trained | 0.0076 [0.0075, 0.0077] | 0.0094 | 0.0078 | 0.0066 | 0.0058 | 0.539 | 0.764 |

- Written from the rules alone, with no forward pass, the net closes **0.74** of the random-to-shipped gap.
  26 budgeted passes take it to **0.82**. The shipped net stays 1.7× better.
- The hand net picks Walt's exact tile as often as the shipped net (0.535 against 0.533) and loses on
  zero-regret agreement (0.700 against 0.754). It agrees on the easy rows and misses the costly ones.
- The lead carries the largest excess per row (hand − shipped: lead 0.0073, second 0.0050, third 0.0062,
  last 0.0042), as on dev. It is the shipped net's weakest region too.

**G2. The arena.** Zeb with the hand field (budgeted arm) against Zeb with the shipped field, 384 games
(24 seeds × 16, 192 mirrored pairs), heuristic bidding on both sides, games to 7 marks, both deciding by
worlds:
- **−0.55 marks per game [−0.76, −0.34]** for the hand field (bootstrap over pairs, 10,000 resamples);
  it won 167 of 384 games (0.435). Chance is 0.
- The game margin's SD was 3.3 marks, not the 5–6 assumed, so the MDE was 0.31, not 0.8.
- **The field's fidelity moves Zeb's marks.** A field 1.7× less faithful to Walt costs about half a mark
  a game, measured. Zeb plans against the field, so a cruder model of the other seats is a worse plan.
- Cost: 9.25e13 instructions retired over the 24 calls.

**G3. The instrument check.** The audit read the hand net (with its three decoys) as it reads any field net.

| net (read R, list v2) | clean share of live poles |
|---|---:|
| hand net, every unit one known rule | **0.360** (195 of 542) |
| the same, rows shuffled | 0.356 |
| random conjunctions, same shape | 0.378 |
| untrained 165 × 124 | 0.320 |
| for scale: trained 128 × 64, three seeds (#233) | 0.542 (floor 0.276) |

- **The clean share cannot tell rules from random conjunctions.** A net built entirely of named rules,
  with no superposition by construction, reads 0.360. Two thirds of its live poles come out partial or
  unnamed. The trained net's 0.542 sits further above its floor than the hand net does above its own.
  So an unnamed pole really is, first, a word list v2 lacks (#202). Here we know the missing words: the
  per-tile rules.
- **Exact recovery**, the `+` label against the key: 0.35 (12 of 34). Most misses are not the instrument
  failing.
  - 19 of the 34 are copies of one count input. The audit named a declaration, or a coarser count slot,
    that holds on exactly those rows or a superset of them: a synonym the key did not list.
  - `bidding-side` (base rate 0.52) and `trump-out` (0.863) can never be labels, because the audit caps
    a label's base rate at 0.5.
  - Of the rest, 9 of 12 are named exactly. These are the five declaration gates, `partner-winning`, both
    halves of `trumps-3plus`, and the offshoot decoy.
- **Position**: the play-count baseline never grants `lead`, `third` or `last` (#236), and the excluded
  column names all three (3 of 3).
- **The on/off column (#236) recovers more** than the top-set label: 0.59 against 0.35.
- **Per-tile rules**: 0.20 of the 188 live per-tile units are called clean. The labels are mostly a facet
  (`partner-winning` ×8, `trump-is-fours/fives`) or a raw `gone` bit. Over all live units, 0.21 are called
  clean with a label outside the key's set. That is 0.61 of everything called clean.
- **The decoys.**
  - The offshoot (partner holds the trick, or more weakly partner led) is labelled `partner-winning`,
    clean, as keyed.
  - **The unrelated OR (partner holds the trick, or fives are trump) is also called clean
    `partner-winning`** at precision 1.000. The top 1% sees only the stronger half. A top-set reader
    cannot see packing hidden in a pole's weaker branch. The on/off column names the other half
    (`trump-is-fives`).
  - The near miss (`lead` and no double held) reads unnamed: `lead` is position, which the baseline
    never grants.
- **Ties** moved 26 labels between the row orders but not the shares (0.360 against 0.356).

**Predictions.**

| # | prediction | result |
|---|---|---|
| 0 | zero-pass regret in [0.015, 0.020], above budgeted | **held** (0.0170) |
| 1 | replication: [0.013, 0.018]; gap closed 0.75–0.88 | **held** (0.0143; 0.82) |
| 2 | exact match within 0.03 of shipped; agreement 0.66–0.71 | **held** (0.535 vs 0.533; 0.700) |
| 3 | the lead carries the largest per-row excess | **held** (0.0073) |
| 4 | twin within 0.002 of shipped, clear of hand | **held** (0.0076) |
| 5 | arena interval includes 0, lower end above −0.8 | **missed** (−0.55 [−0.76, −0.34] excludes 0; the bound held) |
| 6 | non-position exact M1 ≥ 0.7; position label ≤ 0.2 and with excluded ≥ 0.8 | **missed** (0.35; position 0.33 by label, 1.0 with excluded). Diagnosed above: synonyms and the base cap. |
| 7 | per-tile clean ≥ 0.3; M3 ≥ 0.15; shuffle moves ≥ 10 labels | **2 of 3** (0.20 missed; 0.21 held; 26 held) |
| 8 | on/off beats the label by ≥ 0.1 | **held** (0.59 against 0.35) |
| 9 | offshoot `partner-winning`; near miss clean `lead`; unrelated OR not clean | **1 of 3**: the unrelated OR passed as clean, the finding |
| 10 | hand exceeds random conjunctions by ≥ 0.2; random above untrained | **missed / held**: hand 0.360 < random 0.378; random > untrained 0.320 |

**Exploratory, not registered: does the shipped net read voids?** A blind read of its weights found
the void block's mean |w1| about equal to the constant bid block's (commission addendum 2). On the
held-out rows, zeroing the void block changes 12.8% of the shipped net's moves where some seat has
shown a void on a live tile (62% of rows). Regret there rises from 0.0090 to 0.0100, and overall from
0.0084 to 0.0091. Weight size undercounted a rarely-on input: the net reads voids, modestly. The hand
net reads none. The one void rule tried, a walker an opponent can ruff, did not move Walt on dev
(739 rows). Reading voids on purpose is the next rule to write.

## What it taught us

- **The field can be written by hand.** About 270 named boolean rules play within 1.7× of the trained
  net's regret, and the compiled core runs them bit for bit. A model of the other seats can now be read,
  and varied by editing one named weight.
- **The book's rules carry most of it, and the lead is where they stop.** The trained twin's edge is
  largest on the lead,, where Walt is softest: holding a walker, it leads one only 0.60 of the time on dev. The book's lead
  rules are a priority list; the teacher's leads weigh more than one at a time.
- **On a net of known rules, the audit's clean share reads as an untrained net does.** Its top-set
  label misses rules list v2 has no word for, picks synonyms, and passes an unrelated OR as clean
  whenever the OR's stronger half fills the top 1%.
  - For the already-clean test, this cuts both ways. A low clean share is weak evidence of mixing, since
    a perfectly clean net scores 0.36. A clean label is weak evidence against packing in a pole's weak
    branch.
  - The essay's own tools, read together, do better. The on/off column found the hidden half; growth
    and use would test it.
  - Scope: one hand-built net of saturated detectors, list v2, these rows. Trained units are graded
    dials, so this calibrates the instrument's blind spots, not the 0.597.

## The review before sealing

One fresh Opus reader read the essay whole, the commission and this plan (2026-11-02). It found G3
mostly tested whether the audit finds units built from list words ("Brooklyn made of Brooklyn
pieces"). Its edits, all taken: the tie problem and its on-rate ceilings; decoys for growth, packing
and a near miss; per-tile false naming as the headline; a structure-matched null beside the 165 × 124
twin; the asymmetry stated; P1 relabelled a replication; P2's metric settled; P3 given a metric; the
arena MDE written before running. Not taken: scoring `−` poles (reported descriptively; v2 has few
words for a conjunction's shut side).

## Scope

This net, these inputs (one position, uniform beliefs, no history), list v2. A rule the net lacks is a
rule we did not write, not one the game lacks.

## Links

Evidence: [Hub, `lem/hand-written-field-net-2026-11-02/`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-11/tree/7b3ddc3ae095f0bf09abad72bb91f360dfd0ff03/lem/hand-written-field-net-2026-11-02).


[[clean-neurons-42]] · [[lem-the-teacher]] · [[lem-fidelity-by-concept-2026-10-16]] ·
[[lem-field-net-audit-2026-10-15]] · [[lem-fixed-conditioning-2026-10-20]] · [[scenario-player]]
