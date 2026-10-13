# The study's exact-check leads, checked

The book study's coverage matrix (private; § 5) lists 35 claims that its
catalogs computed with throwaway tools: 14 double-dummy rows on the eight full
deals (Hands 28-35) and 21 probability or arithmetic rows. Each is checked
here with Silas's own exact tools, one test per row in
[`tests/book_leads.rs`](../../tests/book_leads.rs), named by the row id. The
claims are in our words.

- **Double-dummy** rows: `Solver` root values at the position the narrated
  line reaches ([`hands.json`](hands.json), the bidder in S0), under both
  objectives. A value is the bidder's side's final points under best play by
  all four seats (from `Objective::Margin`); *made* / *set* is
  `Objective::Make` against the deal's bid (a marks bid needs 42). The solver
  sees every hand; these are facts about open positions.
- **Probability** rows: exact over the uniform deal of the unseen tiles,
  given the record (`Belief::uniform_resolving`, authority `exact`).
- **CONFIRMED**: ours equals the claim. **REFUTED**: it does not; both values
  are given. **BLOCKED**: the claim needs a capability Silas lacks
  ([`GAPS.md`](../../GAPS.md) number); nothing is computed in its place.

**Totals.** 22 rows confirmed whole; 4 refuted in part (DD-09, DD-11, EX-12,
EX-13); 3 blocked whole (EX-10, EX-14, EX-19); 6 confirmed where computable
and blocked for the rest (EX-04, EX-11, EX-15, EX-16, EX-17, EX-20). Every
refutation is listed with both values.

## Double-dummy leads (Hands 28-35)

| row | claim | our exact value | verdict |
|---|---|---|---|
| DD-01 | Hand 28 (blanks, 30) is a lay-down from the opening lead; at trick 1 the 6-0, 0-0 or 5-0 and either partner trump keep it; at trick 2 the 0-0 and both opponents' narrated follows keep it | 42, made, at every one | CONFIRMED |
| DD-02 | Hand 28, trick 3 lead: 3-1 or 5-3 sets, 5-0, 6-1, 6-6 make; after the 6-1 the partner overtakes with the 6-4 and runs 2-2, 6-2, 6-5 while the bidder sheds his offs | 19 set / 42 made; the line is margin-best throughout (the bidder throws the 6-6 on the 2-2, the offs on the sixes) | CONFIRMED |
| DD-03 | Hand 28, trick 3 left opponent: 3-2 holds the bidder to 19, 3-3 gives exactly 30 | 19 set; 30 made | CONFIRMED |
| DD-04 | Hand 28, trick 4 right opponent: 1-1 → 19, 5-4 or 5-2 → 25, 4-2 → 36; at a 31 bid the 1-1 is still best | 19 / 25 / 25 / 36 (only the 4-2 makes); at 31 the 1-1 is the unique margin-best | CONFIRMED |
| DD-05 | Hand 28, tricks 5-6: the 4-1 and 4-4 leads both give 25 (the bidder's 5-0 ruffs either); the bidder's ruff and order and both opponents' throws are indifferent | 25 everywhere; the 5-0 is legal on and beats either lead | CONFIRMED |
| DD-06 | Hand 29 (fours, 30) is worth 26; trick 1: 4-0 → 20, 4-4, 3-2, 5-5 → 26; left opponent's 6-4 → 20, 4-3 → 42 | 26 set; 20 / 26 / 26 / 26; 20 / 42 made | CONFIRMED |
| DD-07 | Hand 29: trick 2 left opponent's 2-2 → 20, any other lead 26; trick 4 partner 6-3, 5-2, 5-4 → 20, 3-0 → 19; bidder 4-1 or 4-4 → 20, thrown 3-3 or 5-5 → 8; trick 5 4-4 → 20 | exactly so, all set | CONFIRMED |
| DD-08 | Hand 30 (sixes) worth 42, its deuces reseating (Hand 31) 29; only a 1-0 opening drops (35); trick 2 partner 0-0, 5-0, 5-3 → 42, a six or 3-1 → 41; trick 3 partner 3-1 → 41, 5-0 or 5-3 → 42; right opponent 3-2 → 42, 3-3 → 41; every other narrated play indifferent | exactly so; the only margin-short narrated plays are those two at trick 3 | CONFIRMED |
| DD-09 | Hand 31 (deuces, 30) worth 29, with values at tricks 1-5 and the book's two post-mortem alternatives not making | deal 29 set; trick 1, trick 2 (6-5, 6-4, partner, 4-0, 5-5), trick 3, trick 4 (3-1, sixes, the two throws), trick 5 (4-4, 5-5, thrown 1-1 → 30) as claimed; both alternatives 24, set | REFUTED in part: trick 2 bidder 6-2 claim 24, ours 25; trick 2 right opponent 4-1 claim 29, ours **30, made**; trick 4 partner 5-3 claim 24, ours 25; trick 5 right opponent 4-1 claim 24, ours 25; trick 5 left opponent's trumps claim 24, ours 25 (the narrated trick-3 throw 5-1 is worth 25, not the best 24) |
| DD-10 | Hand 32 (blanks, 36) worth 31; trick 1 partner 4-1 → 31, 6-3 → 26; right opponent 6-2 fine, 3-3 → 42; trick 2 right opponent 5-3 fine, 3-3 → 42; trick 3 left opponent 6-4 or 5-5 → 31, else 41; right opponent's 3-3 the only setting play; trick 4 bidder 1-0 → 31, 2-2 → 20 | exactly so | CONFIRMED |
| DD-11 | Hand 33 (fours, 35) worth 42; trick 2 bidder 6-4 → 40, 1-1, 4-0, 5-5 → 42; right opponent 3-0 → 40, 2-2 → 42; trick 3 partner 5-2 or 6-2 → 40, 3-2 → 35; trick 4 right opponent 5-0, 1-0, 3-1 → 41, 0-0 → 40; the rest indifferent | 42 made; all made; 1-1, 4-0, 5-5, 2-2, 5-0, 1-0, 3-1 as claimed; the only margin-short narrated play is the bidder's 6-4 | REFUTED in part: bidder 6-4 claim 40, ours 41; right opponent 3-0 claim 40, ours 41; partner 5-2 / 6-2 claim 40, ours 41; partner 3-2 claim 35, ours 36; right opponent 0-0 claim 40, ours 41 |
| DD-12 | Hand 34 (fives, 84) made from the lead; 5-5 sets, 5-4, 5-2, 5-0, 2-2 make; after 5-4 the partner overtakes with the 6-5, leads the 6-2, and the left opponent must follow with the 6-4; values at tricks 3-6; the left opponent holds 6-4 and 4-3 | exactly so (the 6-4 is the left opponent's only legal tile) | CONFIRMED |
| DD-13 | Hand 35 (fours, 84) made from the lead; 4-4 sets, 4-3, 4-0, 5-5, 6-6 make; after 4-3 the partner's 5-4 wins, he leads the 6-2, the right opponent plays the 6-1, the bidder's 6-3 wins, the 6-6 later fells the 6-5; values at tricks 1-6 | exactly so (the 6-5 is forced under the 6-6) | CONFIRMED |
| DD-14 | Every narrated line reproduces its printed score, with Hand 33's trick-6 revoke | 25-17, 22 after two tricks, 42-0, 25-17, 31-11, 41-1 (the right opponent's 0-0 is illegal; his only tile is the 3-1), 42-0, set on trick 7 at 41-1 | CONFIRMED |

## Probability and arithmetic leads

| row | claim | our exact value | verdict |
|---|---|---|---|
| EX-01 | Hand 33 after trick 1, from the bidder: the left opponent holds one of the two trumps out 17/22; the partner both 5/22; in Hand 32 the same is certain | 17/22, 5/22; 1 | CONFIRMED |
| EX-02 | Four trumps out: an opponent holds three or more 10/57 (5/57 each); refinements in Hands 13 and 19 | 10/57, 5/57; Hand 13 with the 3-5 among them 23/171 (no number claimed); Hand 19 (read as sixes) never without a trump above the 6-2 (0) | CONFIRMED |
| EX-03 | Hand 2: all four treys out in one hand 1/171, an opponent 2/171, any hand 1/57 | 1/171, 2/171, 1/57 | CONFIRMED |
| EX-04 | Three trumps out: one hand 1/38, any 3/38, an opponent 1/19; in Hands 17 and 18 that opponent keeps a trump above the bidder's low ones and always sets | 1/38, 3/38, 1/19 on both hands; the contrary holding has probability 0; "always sets" not computed | CONFIRMED / BLOCKED (gap 7) |
| EX-05 | One missing tile: partner 1/3, opponent 2/3; from the setter's seat 7/20, or 1/2 knowing the bidder lacks it | 1/3, 2/3 (and 1/3, 2/3 in Hands 23 and the straight-off 84); 7/20, 1/2 | CONFIRMED |
| EX-06 | Hand 25, two doubles out: some opponent holds neither 23/30 | 23/30 | CONFIRMED |
| EX-07 | Five trumps out (Hand 26): 617/969, 352/969, 147/323, 176/969 | the same | CONFIRMED |
| EX-08 | 33/95, 47/95, 1/5 (19/47 given the bad case), 388/855, 1/3, 10/57 | the same | CONFIRMED |
| EX-09 | Whole-hand frequencies: exactly four of a suit 26.8%, four or more 29.5%; suits present 42.31/46.98/10.35/0.35%; doubles 9.82/32.08/36.09/17.69/3.93/0.37/0.0124% (147 six-double hands); two to four 57.7%, two or more 58.1%; mode (2 doubles, 1 void) 18.0%, next (1, 0) 16.2% | the same (longest suit exactly four, the reading that gives 26.8%); exact enumeration over 1,184,040 hands, since a deal view is gap 6 | CONFIRMED |
| EX-10 | Hand 3 with fours: the bidder takes six of seven tricks in every deal | not computed; tile facts only (the two trumps out, 4-0 and 5-4, are below his 4-4 and 6-4) | BLOCKED (gap 7) |
| EX-11 | With the double ahead of the deuce-blank: 53% forced, 5% later, 33% partner holds the top deuce | 1/3; 467/855 on `odds/`'s reading, so 53% needs a stated policy as the lead says; 5% not computed | CONFIRMED / BLOCKED (gap 7) |
| EX-12 | Hand 16: set only on one layout after tricks 1-4; at trick 3 the 3-2 ruff is overtrumped by the left opponent's last trump (3-4, 3-5 or 3-0) with no blank, the 3-3 is safe | layout claim not computed (play-out, and the record has unnamed plays); the 3-4 and 3-5 overtrump, the 3-3 is unbeatable | REFUTED in part: the 3-0 is a trump *below* the 3-2 and cannot overtrump it; BLOCKED (gap 7) for the layout claim |
| EX-13 | Hand 12, trick 2: only the 6-5 makes the partner's donation safe; a 6-4 or 6-3 could lose to the 6-2 | the 6-5 is beaten only by the fallen 6-6; the 6-4 by 6-6, 6-5; the 6-3 by 6-6, 6-5, 6-4 | CONFIRMED for the 6-5; REFUTED for the threat: the 6-2 beats neither, the threat is the 6-5 (and the 6-4 over the 6-3) |
| EX-14 | Hand 23: the book's set path and a fourth path | not computed | BLOCKED (gap 7) |
| EX-15 | Hand 26: setters take at most one trick and 6 once trumps clear; the override; at 28 banked + 8 left, 35 needs the 5-0 and two tricks, one trick reaches only 34 | 35 - 28 = 7 = 5 + 2; 28 + 5 + 1 = 34; the play claims not computed | CONFIRMED (arithmetic) / BLOCKED (gap 7; gap 1 for the override) |
| EX-16 | Hand 25: the only set path; at 30 the bidder may lose 12, an off with the 6-4 costs 11 plus the off's count | 12; 11; the set path not computed | CONFIRMED (arithmetic) / BLOCKED (gap 7) |
| EX-17 | Hand 1: a finite set probability under a setter policy; a ten sets alone | 42 - 36 = 6 < 11; the probability not computed | CONFIRMED (arithmetic) / BLOCKED (gap 7) |
| EX-18 | Hand 9: the 4-0 off can lose 6-4 and 4-1 together (16); 11 + 1 = 12 fails 31; four trumps out with only the 5-5 on top, so a second trump loser is possible | 16; 12 > 11; out: 5-0, 5-3, 5-4, 6-5; the 6-5, 5-4, 5-3 beat the bidder's 5-2 | CONFIRMED |
| EX-19 | Nel-O sample hand: a third set route by the 1-0 | not computed: no Nel-O physics | BLOCKED (no physics) |
| EX-20 | Sevens: 3 sevens, 4 sixes, 3 eights; 12 tiles sum 0-5, 12 sum 7-12; holding one seven, an opponent holds both others 1/5; the sample hand survives three eights and a seven | the counts; 1/5 on two hands; the sample hand not computed | CONFIRMED / BLOCKED (no Sevens physics) |
| EX-21 | 32 means five tricks and one five; 33 four tricks and a five; 39, 40, 41 three, two, one tricks; 42 - 15 = 27 | the unique decompositions as claimed; 27 | CONFIRMED |
