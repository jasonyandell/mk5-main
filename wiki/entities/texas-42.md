---
title: Texas 42
kind: entity
first_seen: 2026-04-09
last_updated: 2026-07-11
status: active
---

## What it is

Texas 42 is a trick-taking partnership game played with a standard double-six domino set —
the game this whole project is about. This page is the terse quick-reference for the
straight tournament form, the form every model and experiment in this repo plays.

**The game-of-42 cluster** — where to go deeper:

- [[rules-of-42]] — the complete ruleset: full bidding rules, special contracts
  (nello/plunge/splash/sevens), conduct, tournament standards, terminology.
- [[suit-algebra]] — how the algebraic model of suits came to be, and what it replaced.
- [[suit-algebra-spec]] — the formal specification: called sets, power, the three-tier
  trick order, the unique-winner theorem, S₇ symmetry, machine encoding.
- [[play-phase-algebra]] — the play-phase state model, signed rewards, and the graded DAG
  that [[the-oracle]] solves.
- [[the-book-enters]] / [[w42]] — Roberson's *Winning 42* and the empirical validation of
  its strategy claims.

## Equipment and players

- **Set**: 28 unique dominoes, each half showing 0–6 pips.
- **Players**: 4, in 2 fixed partnerships. Partners sit across the table; play alternates
  between teams.
- **Deal**: All 28 dominoes are dealt (7 per player). No boneyard.

## Suits

A non-double domino belongs to the pip suit of each of its two halves. A double belongs
only to the suit of its pips. The double of a suit is its highest member; remaining members
rank by off-pip descending. Example (fives, no trump): 5-5 > 6-5 > 5-4 > 5-3 > 5-2 > 5-1 > 5-0.

## Trump declarations

The winning bidder names one of ten declarations, which reduce to three kinds:

1. **Pip suit trump** (blanks, ones, twos, threes, fours, fives, or sixes): every domino
   containing that pip is trump. The double of the trump suit is the highest trump. Trumps
   leave their other pip suit entirely.
2. **Doubles trump**: only the 7 doubles are trump, ranked 6-6 (high) through 0-0 (low).
   Doubles are no longer members of their pip suits.
3. **No-trump**: no trump suit exists. The highest domino of the led suit always wins.

(lem/rules/primer.md @ a8bccfa)

## The led suit rule

The led domino commits the trick's led suit:

1. If the led domino is trump, the led suit is trump.
2. Otherwise, the led suit is the **higher** of its two pips.

Examples: with twos trump, leading 5-3 leads fives (not threes); with fours trump, leading
4-2 leads trump (not twos); with doubles trump, leading 5-5 leads trump. (lem/rules/primer.md @ a8bccfa)

## Following suit

A player holding any domino of the led suit must play one. A player void in the led suit
may play any domino, including a trump. Under doubles-as-trump, doubles cannot follow a
non-double pip-suit lead, and non-doubles cannot follow a double lead. (lem/rules/primer.md @ a8bccfa)

## Winning a trick

If any trumps were played, the highest trump wins. Otherwise the highest domino of the led
suit wins. Off-suit discards cannot win. The trick winner leads the next trick. Seven tricks
are played per hand. (lem/rules/primer.md @ a8bccfa)

## Count and scoring

**Five count dominoes** (35 count points total):

| Domino | Count points |
|---|---|
| 5-5 | 10 |
| 6-4 | 10 |
| 5-0 | 5 |
| 4-1 | 5 |
| 3-2 | 5 |

All other 23 dominoes are worth 0 count points. Each of the 7 tricks is worth 1 trick
point. Total hand value: 35 + 7 = **42 points**.

The bidding team must take at least as many of the 42 hand points as the bid (for point
bids 30–41) or all 42 points (for mark bids) to make the contract. Falling short by any
amount scores the contract for the defending team. (lem/rules/primer.md @ a8bccfa)

## Communication constraint

Partners may not communicate hand information during play. All decisions are based on public
information: the bid, trump declaration, tricks played, the current lead, and the player's
own hand. (lem/rules/primer.md @ a8bccfa)
