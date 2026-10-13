# The book's odds, exactly

Every probability *Winning 42* (Roberson) prints, as a Silas query, checked in
`tests/book_odds.rs`. Most are in Chapter 16; the rest are scattered through
Chapters 2 to 12. Anchors name the private, reviewed capture
(`shot-NNNN`); the claims are restated in our words.

Three numbers per claim:

- **printed**: the book's figure, as a number;
- **book model**: each missing tile equally likely in each of the three other
  hands, so all 3^m placements of m tiles count the same;
- **exact**: the uniform deal of the 21 unseen tiles, seven to a hand, given
  the stated hand (authority `exact`, from `Belief::uniform_resolving`), and
  derived a second time in plain Rust by weighting each placement by its
  completions.

A shape claim ("four trumps, missing the second-highest") is checked on two
hands with different trumps and other tiles; they must agree, and they do.
Whole-deal claims, with no hand given, are a gap (`GAPS.md`, deal view): they
are enumerated in the test over all 1,184,040 hands, and each file's query is
checked hand by hand against that enumeration.

Verdicts: **matches**; **arithmetic** (the book's own model does not give the
printed figure); **model** (the printed figure is the model's, and the model is
wrong because hand sizes matter); **ambiguous** (both readings given).

| claim | anchor | printed | book model | exact | verdict |
|---|---|---|---|---|---|
| c16-4t-ways (count, in the test) | shot-0232 | 27 | 27 | 27 placements | matches; not equally likely |
| [c16-4t-guarded-second](c16-4t-guarded-second.silas) | shot-0232 | 0.37 | 10/27 = 0.370 | 33/95 = 0.347 | model |
| [c16-4t-works](c16-4t-works.silas) | shot-0232 | 0.63 | 17/27 = 0.630 | 62/95 = 0.653 | model |
| [c16-4t-two-of-three](c16-4t-two-of-three.silas) | shot-0232 | 0.52 | 14/27 = 0.519 | 47/95 = 0.495 | model; under one half, so the lean the book draws reverses |
| [c16-4t-both-high](c16-4t-both-high.silas) | shot-0233 | 6 of 14 | 6/27 = 0.222, or 3/7 = 0.429 inside the 14 | 1/5, or 19/47 = 0.404 given the bad case | ambiguous; model either way |
| [c16-four-of-a-suit](c16-four-of-a-suit.silas) (GAP) | shot-0233 | 0.28 | none | 0.295 (four or more in a suit), 0.268 (longest suit exactly four) | ambiguous |
| c16-3t-ways (count, in the test) | shot-0233 | 81 | 81 | 81 placements | matches |
| [c16-3t-guarded-second](c16-3t-guarded-second.silas) | shot-0233 | 0.47 | 38/81 = 0.469 | 388/855 = 0.454 | model |
| [c16-3t-partner-second](c16-3t-partner-second.silas) | shot-0233 | 0.26 (21 of 81) | 27/81 = 0.333 | 1/3 | arithmetic; the 81 split 38 + 27 + 16, and no event we tried counts 21 |
| [c16-3t-opponent-three](c16-3t-opponent-three.silas) | shot-0233 | 18 of 81 = 0.222 | 18/81 | 10/57 = 0.175 | model |
| c16-2t-ways (count, in the test) | shot-0233 | 225 | 243 | 243 placements | arithmetic |
| [c16-2t-spread](c16-2t-spread.silas) | shot-0233 | 0.40 (90 of 225) | 90/243 = 0.370 | 147/323 = 0.455 | arithmetic (90 is right, 225 is not); model too |
| [c16-2t-partner-three](c16-2t-partner-three.silas) | shot-0233 | 0.20 (45 of 225) | 51/243 = 0.210 | 176/969 = 0.182 | arithmetic; model too |
| [c16-2t-favorable](c16-2t-favorable.silas) | shot-0233 | 0.60 | 47/81 = 0.580 | 617/969 = 0.637 | arithmetic; model too; the truth is better than printed |
| same file, Hand 26 | shot-0121 | 0.60 (of 225 ways) | 47/81 = 0.580 | 617/969 = 0.637 | arithmetic; model too |
| [c12-two-trump-outtrumped](c12-two-trump-outtrumped.silas) | shot-0123 | 0.40 | 34/81 = 0.420 | 352/969 = 0.363 | arithmetic; model too |
| c16-hands (count, in the test) | shot-0234 | 1,184,040 | none | C(28,7) = 1,184,040 | matches |
| [c16-suits-represented](c16-suits-represented.silas) (GAP) | shot-0234 | 0.41, 0.48, 0.10, 0.01 (seven to four suits) | none | 0.423, 0.470, 0.103, 0.0035 | arithmetic: the first two are a point off, the last is a third of one percent |
| [c16-doubles](c16-doubles.silas) (GAP) | shot-0234 | 0.10, 0.32, 0.36, 0.18, 0.04, under 0.005, 0.0002 (zero to six) | none | 0.098, 0.321, 0.361, 0.177, 0.039, 0.0037, 0.000124 | matches to the printed rounding, except six doubles: arithmetic (147 hands, about 1/80 of a percent, not 1/50) |
| [c16-two-to-four-doubles](c16-two-to-four-doubles.silas) (GAP) | shot-0234 | at least 0.36 | none | 341677/592020 = 0.577 | ambiguous: true as a bound, but 0.36 is the two-double row alone |
| [c16-modal-hand](c16-modal-hand.silas) (GAP) | shot-0234 | two doubles, one void | none | that cell, 0.180, is the largest | matches |
| [c16-partner-two-doubles](c16-partner-two-doubles.silas) | shot-0234 | a decent chance | 131/243 = 0.539 (bidder holds two doubles), 11/27 = 0.407 (three) | 180/323 = 0.557, 116/285 = 0.407; 0.581 with no hand known | matches (no number printed) |
| [c02-partner-double-five](c02-partner-double-five.silas) | shot-0026, shot-0047 | 1/3 | 1/3 | 1/3 | matches |
| [c02-opponent-double-five](c02-opponent-double-five.silas) | shot-0027 | 2/3 | 2/3 | 2/3 | matches |
| [c03-double-ahead-forces-count](c03-double-ahead-forces-count.silas) | shot-0042 | 0.53 | 43/81 = 0.531 | 467/855 = 0.546 | model, on our reading (partner gives it; an opponent only when forced; four deuces out) |
| [c03-double-ahead-later](c03-double-ahead-later.silas) (GAP) | shot-0042 | 0.05 | not determined; 6/81 on one play reading | not computed | ambiguous: depends on how the hidden seats play two tricks |
| [c03-off-first-partner](c03-off-first-partner.silas) | shot-0042 | 0.33 | 1/3 | 1/3 | matches |
| [c05-setter-partner-holds-winner](c05-setter-partner-holds-winner.silas) | shot-0066 | 2/3 | 2/3 before the deal; 1/3 from the setter | 2/3 before the deal; 7/20 from the setter who cannot follow, 1/2 if he also knows the bidder lacks it | ambiguous |
| [c07-straight-off-double](c07-straight-off-double.silas) | shot-0089, shot-0101 | 2/3 | 2/3 | 2/3 | matches |
| [c09-partner-five-blank](c09-partner-five-blank.silas) | shot-0104 | 1/3 | 1/3 | 1/3 | matches |
| [c09-doubles-trump-void](c09-doubles-trump-void.silas) | shot-0109 | 0.80 | 7/9 = 0.778 | 23/30 = 0.767 | model; 0.80 is a coarse rounding of either |
| [c12-only-trey](c12-only-trey.silas) | shot-0124 | 2/3 | 16/81 = 0.198 | 7/33 = 0.212 | arithmetic: two to one is one tile's odds; the event needs all four live treys elsewhere (record ours) |

Not encoded, because they print no number or depend on how the seats play:
the rare three-trump opponent (shot-0073), "odds very low" (shot-0044), "half
of the hands" not worth a bid (shot-0054), and the straight-off 84's set rate
of about two in three against good players (shot-0089), which is the 2/3 above
only if the double's holder always keeps it to the last trick.

## Against the earlier figures

The study ([book-study-2026-10-12](../../../../wiki/experiments/book-study-2026-10-12.md))
and the audit catalogs computed several of these first. Every one is
confirmed: 33/95, 47/95, 23/30, 0.557, 0.407 (116/285; the book model's 11/27
also rounds to 0.407), 38/81 and 18/81 as model counts (exactly 0.454 and
0.175), 27/81 against the printed 21/81, 243 against 225, six doubles at
about 1/80 of a percent against the printed 1/50, and two to four doubles at
0.577 against "at least 36 percent". The study's per-placement weights (0.043,
0.037, 0.026 for 1-1-1, 2-1-0 and 3-0-0 splits) check too. One wording: the
study quotes "partner holds two or more doubles 58% of the time" as the
book's; the book prints no 58 (it says "a decent chance"); 0.581 is the
unconditional exact figure.
