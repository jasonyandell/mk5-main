---
title: The metasplat — how Zeb's splat differs from the book's, finer than a mean (2026-10-13)
kind: experiment
first_seen: 2026-10-13
last_updated: 2026-10-13
status: complete
---

## Summary

Jason, 2026-10-13: "I am very interested to study the metasplat formed by the Zeb model
both where it agrees and disagrees and see what we can learn. it's not refutation or
validation. it's just learning how it disagrees. we've not yet had a contrast much finer
than brute winrate."

The book study ([[book-study-2026-10-12]]) priced each of the book's rules by one scalar
per position: the book's best pmake less Zeb's. This page compares the whole **splats**
([[splat-visualizer]], [[candlewax]]) instead: at every live position of the 250,000-deal
pour where a book-faithful rule fires, Zeb's pick against one contrast tile, world by
world, named by one of nine shapes fixed before any position was classified.

**What it shows, in one line.** Half of Zeb's departures from the book are not about the
contract: 44% carry a make edge beyond the pour's noise bar, 45% sit at an even make and
11% are ties; where the make does separate them, Zeb's tile is usually better at every
margin (dominant), and the "safer tail at a lower mean" trade is 3%. At a settled contract
the book plays for points and Zeb, under `make`, does not. This is description: it explains
how the two differ, not which is right.

## The shapes, fixed before counting

Fixed in `scenario/metasplat.py` and committed (`d2ed2134`) before the first position was
classified; the sample rule too. Nothing below moved after the counts.

**The pair.** Z is Zeb's pick, the tile the pour played. Where Z is outside the rule's
allowed set (Zeb *leaves* the book), the contrast X is the book's best tile; where Z is
inside it (Zeb *keeps to* the book), X is the best tile the book rules out. "Best" is the
mover's pmake, then the mover's mean margin, then the lowest tile: the book's strongest
case. A rule that allows every legal tile has no contrast (125,678 rows).

**Units and bar.** The mover's: success is the contract made for the bidders, set for the
setters; the margin is signed toward the mover. Δp is Z's success probability less X's, Δμ
its mean margin less X's, in points. Both moves are played out in the same worlds, so the
noise bar is **twice the paired standard error over the position's worlds**; a cell under
512 worlds is the whole fiber, enumerated, with a bar of zero. The distribution test asks,
at every margin threshold, whether Z's mass above it exceeds X's beyond that threshold's own
bar (*ahead*) or falls short of it (*behind*); Δp is that test at the contract.

| shape | definition (first match) |
|---|---|
| identical | the same margin in every world |
| noise-tie | ahead and behind nowhere, Δμ inside its bar: **a tie on noise, not a disagreement** |
| behind-on-make | Δp below minus its bar |
| dominant | Δp above its bar, behind nowhere (first-order dominance within the bar) |
| make-for-margin | Δp above its bar, behind somewhere, Δμ below minus its bar: the safer tail at a lower mean |
| make-for-tail | Δp above its bar, behind somewhere, Δμ not below its bar |
| even-make-mean-up / -down | Δp inside its bar, Δμ above / below its bar |
| even-make-spread | Δp and Δμ inside their bars, ahead or behind at some threshold |

**Population and sample.** The book study's: the 18 book-faithful rules of `book/book-reprice-3`
at all 4,343,426 planned positions, less the rows decided by the banked score: 3,536,649
(rule, position) rows over 2,782,903 positions (a position counts once per rule that fires
there). The sample for the page and the reasons: from chunk s100001, per shape, three
departures from different rules and one agreement, at random (seed 20261013).

**Producing player.** The pour's: `k0,n512`, utility `make`, bid 30, seat 0 bids, field nets
`field_make_d5000_v2` and `v2/margin_d50k_h128x64`, before the #110 and #112 fixes; not the
deployed Zeb. Every number is that player's 512-world forecast.

## The metasplat

Shares of rows, %, with 95% intervals clustered by deal (under ±0.1 point here).

| | n | identical | noise-tie | dominant | make-for-margin | make-for-tail | even, μ up | even, μ down | even, spread |
|---|---|---|---|---|---|---|---|---|---|
| Zeb leaves the book | 1,592,675 | 4.1 | 6.9 | 27.0 | 2.9 | 13.9 | 12.4 | 11.8 | 21.0 |
| Zeb keeps to the book | 1,818,296 | 5.1 | 6.3 | 38.6 | 1.5 | 11.0 | 15.4 | 5.8 | 16.4 |

behind-on-make never occurs: the pour's pick always has its own sample's most make.

- **Leaving:** make edge 44%, even make 45%, ties 11%. **Keeping:** 51%, 38%, 11%. An
  agreement is barely more decided than a departure.
- **Live rows** (some legal tile makes differently in Zeb's worlds; 1,354,944 departures):
  make edge 51%, even 41%, ties 8%; keeping, 60% make edge. **Dead rows** (237,731
  departures, every legal tile makes alike): identical 26%, μ down 44%, μ up 21%.
- **Sizes** (medians of departures): dominant Δp +0.074 (bar 0.034) with Δμ +3.8 points;
  make-for-tail Δp +0.062, Δμ +1.7; make-for-margin Δp +0.036, Δμ −1.75 (61% of these in
  enumerated late cells); even μ down Δμ −2.7; spread Δp +0.012 under a bar of 0.030, and
  Δp > 0 in 91% of spread rows.

**Where it lives** (departures; % of each bucket's rows):

| axis | bucket | make edge | identical | μ down | spread |
|---|---|---|---|---|---|
| trick | 1 / 2 / 3 / 4 / 5 / 6 | 42 / 42 / 44 / 50 / 52 / 23 | 0 / 0 / 0 / 2 / 11 / 43 | 6 / 8 / 12 / 16 / 22 / 23 | 31 / 26 / 22 / 15 / 3 / 0 |
| role | bidder / left setter / partner / right setter | 42 / 45 / 46 / 43 | 3 / 4 / 4 / 6 | 9 / 13 / 10 / 15 | 29 / 18 / 19 / 16 |
| count at risk (table + the larger tile's) | 0 / 5 / 10 / 15 / 20+ | 39 / 49 / 48 / 45 / 45 | 3 / 5 / 5 / 7 / 7 | 9 / 14 / 14 / 20 / 27 | 28 / 16 / 14 / 11 / 5 |
| log2 worlds | 0 / (0,6] / (6,9] / (9,12] / (12,16] / (16,20] / >20 | 4 / 30 / 67 / 47 / 46 / 43 / 41 | 74 / 31 / 5 / 3 / 1 / 0 / 0 | 17 / 26 / 19 / 17 / 14 / 11 / 6 | 0 / 0 / 0 / 15 / 19 / 23 / 30 |
| setters need | 1–4 / 5–8 / 9–13 | 42 / 44 / 44 | 10 / 9 / 2 | 20 / 18 / 10 | 11 / 12 / 24 |

make-for-margin is 17% of departures in cells of 2^6–2^9 worlds and 11% in trick 5, under 3%
elsewhere. Every rule, rule inside it and axis, for both sides: `tables.md` (Data).

## How Zeb leaves each rule

Per book rule, the share of its departures (n) with a make edge / at an even make / tied,
and the share of its agreements with a make edge (how decided Zeb's agreement is).
Intervals are under ±1 point except where shown.

| book rule | leaves (n) | make edge | even make | ties | dominant | μ down | keeps: make edge |
|---|---|---|---|---|---|---|---|
| partner-plays-the-top-trump | 25,367 | 53 | 35 | 12 | 49 | 3 | 62 |
| setter-leads | 134,101 | 51 | 43 | 6 | 35 | 12 | 60 |
| partner-leads | 67,206 | 50 | 45 | 5 | 27 | 10 | 70 |
| first-lead | 139,701 | 48 | 50 | 2 | 34 | 4 | 35 |
| pounce | 107,107 | 47 | 46 | 8 | 27 | 14 | 73 |
| dead-double | 240,901 | 45 | 40 | 15 | 29 | 4 | 23 |
| shed-for-a-void | 255,374 | 45 | 39 | 17 | 30 | 14 | 36 |
| donate-when-safe | 86,329 | 44 | 47 | 9 | 25 | 18 | 66 |
| low-from-a-pair | 88,764 | 44 | 39 | 18 | 36 | 3 | 27 |
| protect-the-count | 23,914 | 44 | 51 | 5 | 30 | 23 | 72 |
| follow-with-the-off | 9,220 | 43 | 45 | 12 | 20 | 13 | 55 |
| take-the-trick | 174,424 | 40 | 53 | 7 | 12 | 27 | 62 |
| helper-stretch | 1,485 | 38 ± 2.5 | 53 ± 2.5 | 9 ± 1.5 | 21 ± 2.1 | 31 ± 2.3 | 81 ± 1.3 |
| crisis | 119,476 | 37 | 58 | 5 | 20 | 11 | 30 |
| bidders-plan | 60,892 | 34 | 44 | 22 | 15 | 17 | 65 |
| trump-set | 4,173 | 29 ± 1.5 | 61 ± 1.7 | 10 ± 1.0 | 12 ± 1.0 | 36 ± 1.6 | 50 ± 2.6 |
| follow-low | 54,241 | 26 | 46 | 27 | 21 | 3 | 27 |

In words (n = departures; shares of them):

- **partner-plays-the-top-trump.** Zeb leaves the handoff where keeping the top trump is
  better at every margin: 49% dominant, median Δp +0.096. The edge is 55% of departures in
  tricks 1–4 and 21% after.
- **first-lead.** Never dead, 2% ties. Zeb leaves the book's permissions with a lead better
  everywhere: *one off may go first* (4,096) 77% dominant, *double-up, lead low* (17,626)
  61%. Among trumps it leaves by shape, not by make: *lead the top trumps* (4,206) 41%
  spread and 2% dominant; *lead a trump* (52,494) 36% spread, 29% μ up.
- **setter-leads.** 51% make edge (35 dominant, median Δp +0.082). Where one point sets
  (6,710), 37% μ down: the book's double banks more at a settled set.
- **partner-leads.** 50% make edge, 18% make-for-tail: Zeb's lead gains the make and gives
  up a tail. Its agreements are decided (70% make edge).
- **pounce.** *A ten now* (41,495): 38% dominant. *No count on a lost trick* (18,750): 43%
  μ down, 21% identical, 12% make-for-margin; half of these rows are dead. Zeb's setter
  puts count on the trick the book calls lost mostly where the set does not turn on it, and
  in one row in eight trades points for set chance.
- **take-the-trick.** 53% even make. Zeb declines the book's trick either for the make at a
  cost in a tail (make-for-tail 23%; *take it last* 27%) or at a settled contract where the
  trick's points are all that differ (μ down 27%, 73% of them dead).
- **dead-double.** 45% make edge, 15% ties, μ down only 4%. *Throw it* (145,880): 21% ties;
  *cash it* (95,021): 31% spread.
- **shed-for-a-void.** 17% ties; *throw a lone tile* (78,553) 25% ties (17 noise, 8
  identical); *throw what he need not keep* (149,567) 33% dominant.
- **crisis.** The even make is the rule: 58%, 39% spread (*a count double away from his
  suit*, 78,120: 45% spread). Zeb's choice among doubles reshapes the splat by less than
  the bar can price on the make. *The six-five bait* (673): 53% dominant.
- **low-from-a-pair.** 36% dominant, 17% noise ties, 26% spread. The make edge is 41% of
  departures before trick 5 and 78% after: the pair's order decides late.
- **donate-when-safe.** *Give a five* (14,395): 50% μ down, 12% make-for-margin. Zeb
  withholds the five on the safe trick; at an even make the book's five has the higher
  mean, and one row in eight trades the points for make. *No count on a trick that can be
  lost* (36,053): 31% dominant, 24% make-for-tail.
- **bidders-plan.** 22% ties, nearly all *the donation lead* (below). *Draw the trumps*
  (28,864): make-for-tail 25%, μ down 21%, dominant 10%.
- **protect-the-count, helper-stretch, trump-set.** Departures at an even make with the
  book's tile ahead in the mean: μ down 23%, 31 ± 2.3% and 36 ± 1.6%, 49%, 84% and 86% of
  them dead. *Feed the trump holder* (992) is the exception: 52% make-for-tail, 30% dominant.
  helper-stretch's agreements are the most decided of any rule (81 ± 1.3% make edge).
- **follow-low.** The least decided: 27% ties, 46% even. Where it decides, it decides late
  (make edge 24% of departures before trick 5, 90% after).

## What it teaches

1. **Half the departures are not about the contract.** 44% of departures have a make edge
   beyond the bar, 45% an even make, 11% a tie. Agreements are 51% make edge. The rule's
   boundary does not separate where Zeb cares from where it does not; a price, which is 0 at
   every exact tie and a small negative inside the bar, cannot show it.
2. **When the make separates them, Zeb's tile is usually better everywhere.** Dominant is 27
   of the 44 points (61%), at a median Δp +0.074 and Δμ +3.8 points: the make edge carries
   margin with it. The classic trade, the safer tail at a lower mean, is 2.9% of departures,
   61% of them in enumerated late cells. Zeb does not leave the book to play safe; where the
   contract turns, its tile wins more often and by more.
3. **At a settled contract the book plays for points; Zeb does not.** On dead rows the book's
   tile has the higher mean in 44% of departures against Zeb's in 21%. μ down rises with the
   count at risk (9% at none, 27% at 20 or more) and through the hand (6% at trick 1, 23% at
   trick 6), and gathers in take-the-trick, trump-set, helper-stretch, protect-the-count and
   the five donated. The explanation is the utility: under `make` Zeb is indifferent there,
   its tie rule takes the lowest tile (#120), and the margin past a decided contract is the
   tape's (#107). A make-then-margin tiebreak would remove most of this class; unmeasured.
4. **Early, the departure is a reshaping the sample cannot price.** even-make-spread is 31%
   of trick-1 departures and 30% where the belief holds over 2^20 worlds; in 91% of spread
   rows Zeb is ahead on the make inside the bar (median +0.012 against 0.030). The opening
   trump choice and the crisis's choice of double live here; 512 worlds cannot decide them.
5. **Ties on noise are concentrated, not general.** 11% of departures overall, but
   identical outcomes are 43% of trick-6 departures (the plan has decided the hand).

**Flagged: departures mostly ties.** *The donation lead* (bidders-plan, 14,978): 60% ties,
56% identical in every world, 67% dead; 82% ties or even make. Its disagreement is
illusory. *Take the rest* (228): 70% ties, 100% ties or even make. *The lowest trump*
(follow-low, 16,627): 39% noise ties, 88% ties or even make. Mostly make-indifferent (ties
or even make above 70%): *lead the top trumps* 82%, *lead a trump* 70%, *give a five* 75%,
*no count on a lost trick* 76%, *win with the double* 74%, *one point sets, lead a double*
75%, *lead a high trump* (trump-set) 87%, *cash the other winners* 74%, *the double of the
off's suit* 76%. No book rule as a whole is mostly ties
(the most, follow-low, 27%).

## The sample, the reasons and the page

The 32 sampled rows (every shape the pour has; 3 departures and 1 agreement each) are
`scenario/viz/data/metasplat_sample.json`: open
`/scenario/viz/?data=data/metasplat_sample.json` ([[splat-visualizer]] § Book against
Zeb). For each, the reason report's separating-fact search ([[scenario-player]] § The
reason report) ran on the pour's own worlds without re-planning, splitting the worlds where
Z succeeds and X fails from the reverse (or, at an even make in every world, by margin).
Of the 28 not identical, a reason clears the report's threshold and chance level at 12 (10
as two facts, 2 as one): none of the 4 noise ties, none of the 4 μ-up rows, 3 of 4 dominant,
3 of 4 spread. The facts found are holdings (at sample deal 9, *take the trick: trump it*: *the
partner holds the 6-4 or is void in fours*, 32% of Zeb's side against 88% of the book's);
nothing recurs across a shape at this size. Twenty-eight positions are an illustration, not a census.

## Caveats

- The bar is the pour's sampling only. The plan's own error (another seed, a stronger
  rung) adds 0.005–0.016 pmake per decision ([[book-study-2026-10-12]] § The face-off): a
  make edge just outside this bar can be inside that one.
- Zeb's pick is its own sample's argmax, so Δp is biased toward it by selection, and the bar
  does not correct that. The book study's four-times-stronger re-plan kept 91–97% of the
  large negative prices.
- Rows are pooled across rules; a position where three rules fire counts three times.
- Bid 30, seat 0 bids, self-play partners who never follow the book: as in the book study.

## Data

Rows, summaries, the sample and the scripts are public in
`jasonyandell/texas-42-zeb-research-2026-10` under `book/metasplat/` (CC0, with
`SHA256SUMS`; `NOTES.md` lists the columns). Recompute
([publish-data-to-huggingface](../playbooks/publish-data-to-huggingface.md)):

```bash
# per chunk, 40–80 s (five in parallel, about 85 s); --reprice is book/book-reprice-3/out/
python -u -m scenario.metasplat run --pour scratch/atlas/pour_make_netfield_s100001_d50000 \
    --reprice <book-reprice-3>/out --label s100001 --out scratch/metasplat/out/s100001.npz
python -u -m scenario.metasplat summarize scratch/metasplat/out/s*.npz --out scratch/metasplat/out   # 6 s
python -u -m scenario.metasplat sample scratch/metasplat/out/s*.npz \
    --pour scratch/atlas/pour_make_netfield_s100001_d50000 --out scenario/viz/data/metasplat_sample.json   # 30 s
```

Tests: `scenario/tests/test_metasplat.py` (each shape on a built pair; the batch is the
reference; the fiber count is `scenario.fiber`'s) and `test_metasplat_pour.py` (a small pour
read against a stand-in rule: every row is the reference reading of its pair; the summary
counts each row once; the sample is the page's own document; the checked-in sample reads off
its own moves).

## Links

[[book-study-2026-10-12]] · [[splat-visualizer]] · [[candlewax]] · [[scenario-player]] ·
[[w42]] · [[research-night-2026-10-12]] · [[splat-search-bounds]] ·
[[lumps-voi-2026-10-13]] (the same margins split by who holds one hidden tile: hedges and the value of knowing it)
