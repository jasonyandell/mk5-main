---
title: Rules of Texas 42 — complete ruleset
kind: topic
first_seen: f989134f
last_updated: pending-this-ingest
status: active
---

The complete ruleset: tournament standard ("Straight 42") plus the traditional variations.
[[texas-42]] is the terse quick-reference for the straight game; this page is the full
specification, including special contracts, conduct rules, and tournament standards. The
formal mathematical treatment of suits and tricks is [[suit-algebra-spec]]; the play-phase
state model is [[play-phase-algebra]].

## Provenance and authority

Compiled from the National 42 Players Association (N42PA, the tournament sanctioning body),
Dennis Roberson's *Winning 42* (the canonical published text — see [[the-book-enters]] and
[[w42]]), the Texas State Historical Association, and major tournament authorities
(Hallettsville Texas State Championship, Austin 42 Club). Texas 42 was invented in 1887 by
William Thomas (12) and Walter Earl (14) in Trapp Spring (now Garner), Parker County, Texas,
to circumvent religious prohibitions on card games; it was transmitted orally for a century
before systematic documentation in the 1990s.

Migrated from `docs/rules.md` (first committed f989134f). A sibling doc, `rules-tournament.md`,
was retired unmigrated at this ingest: it contained outright rules errors ("all seven doubles
are ALWAYS trump regardless of declared suit"; "Follow-Me: trump suit changes each trick")
that contradict this ruleset and the engine. Nothing cites it; nothing should.

## Equipment and setup

- One standard double-six domino set (28 dominoes), 4 players in 2 fixed partnerships,
  partners opposite. Composition by highest pip: blanks 0-0…0-6 (7), ones 1-1…1-6 (6),
  twos (5), threes (4), fours (3), fives (2), sixes (1) — 7+6+5+4+3+2+1 = 28; at 4 dominoes
  per trick that is exactly 7 tricks.
- **First shaker**: all four draw one domino face-down; highest pip total shakes first;
  ties redraw.
- **Draw (tournament)**: shaker shuffles face-down; non-shaking team draws first (7 each),
  then shaker's partner, then shaker. No boneyard. Regional play may deal instead.
- **Arrangement (tournament)**: dominoes in 4-3 or 3-4 formation; once bidding begins they
  cannot be rearranged.

## Bidding

- Player left of the shaker bids first; clockwise; each player bids or passes exactly once.
- **Point bids**: 30–41. **Mark bids**: 1 mark (42 points), 2 marks (84), 3 marks (126), …
- Each bid must exceed the previous. After 42 (1 mark), bids move in whole-mark increments.
- **Opening ceiling**: 2 marks. Any player may bid up to 2 marks when 2 marks has not been
  bid; subsequent bids may add only one mark. 3 marks can only follow an existing 2-mark bid.
- **Plunge exception**: the only jump bid. Requires 4+ doubles in hand; worth 4 marks minimum
  (5 if bidding already reached 4); may open or jump.
- **All pass (tournament)**: reshake, next player shakes. Common variation: shaker forced to
  bid 30.
- **Bid conduct (tournament)**: single words only ("thirty," "pass," "two marks"); no
  inflection, gestures, or commentary. Out-of-turn bids stand but cannot be modified.

## Trump declaration and play

The winning bidder declares trump before the first play: any pip suit (blanks through sixes),
doubles-as-trump, or no-trump ("follow-me"). Then:

- **Leading**: bid winner leads trick 1; each trick's winner leads the next. Any domino may
  be led.
- **Led suit**: if the led domino contains a trump pip, it leads trump; otherwise its higher
  pip determines the suit led. (With 4s trump, leading 4-2 leads trump, not twos.)
- **Following**: must play the led suit if able; if void, may play anything, including trump.
- **Winning**: highest trump played wins; otherwise highest domino of the led suit wins.
  Off-suit discards never win.

### Doubles

- Standard: a double belongs to its natural suit and is that suit's highest member
  (6-6 is the highest six, etc.).
- **Doubles-as-trump**: only the seven doubles are trump, ranked 6-6 high through 0-0 low.
  Doubles then cannot follow pip-suit leads and non-doubles cannot follow double leads.
- Variations exist (doubles as own suit without power — used by nello; doubles-low is rare).

### Renege

Failure to follow suit when able. Tournament: immediate loss of hand plus penalty marks.
Casual: usually loss of hand. May be called when noticed and verified from played dominoes.

## Scoring

- **Count dominoes** (35 points): 5-5 and 6-4 worth 10 each; 5-0, 4-1, 3-2 worth 5 each.
- **Tricks**: 1 point each, 7 total. Hand total 35 + 7 = 42.
- **Marks (tournament)**: game to 7 marks. A made point bid (30–41) or 1-mark bid earns
  1 mark; higher mark bids earn their bid. A set contract awards the marks bid to the
  defenders.
- **Points (traditional)**: game to 250 (variants 150/500); actual points taken accumulate.

## Special contracts

Prohibited in N42PA tournament play ("Straight 42 only"); standard in traditional play.

- **Nel-O (nello, low)**: 1-mark minimum. Bidder must lose every trick; partner sits out,
  dominoes face-down; no trump. **Doubles treatment in this engine**: doubles form their own
  suit (suit 7), ranked 6-6 high to 0-0 low, with no power — the standard tournament
  treatment. (Alternative high/low-in-suit treatments exist and are not implemented.)
- **Plunge**: 4+ doubles required, 4 marks minimum, may open or jump. Partner names trump
  without consultation and leads; must win all 7 tricks.
- **Splash (crash)**: 3+ doubles, 2–3 marks (regional). Partner names trump and leads; must
  win all 7 tricks.
- **Sevens**: 1+ mark. Domino closest to 7 total pips wins the trick; 3-4 is unbeatable;
  equidistant ties go to the first played. Must win all tricks. Play is forced — sevens has
  no suit structure and sits outside [[suit-algebra-spec]] by design. Rarely accepted in
  serious play.

Engine note: nello, plunge, and splash are implemented as layers
(`src/game/layers/nello.ts`, `plunge.ts`, `splash.ts`), which cite this section.

## Conduct

No table talk, physical signals, count announcements, or timing tells — all partner
communication about the hand is prohibited during bidding and play. Tournament specifics:
first domino touched must be played; exposed dominoes play at first legal opportunity; play
out of turn stands if legal. Penalties escalate from warning to mark-to-opponents to
ejection; renege costs the hand plus penalty.

## Tournament standards (N42PA)

Straight 42 only; 25-minute round-robin games (current hand completes at time); seeding by
win record, then marks earned/against; elimination brackets typically untimed. Tournament
director is final authority.

## Terminology

**Bidder** (won the auction) · **count/counters** (the five point dominoes) · **marks**
(tournament scoring units) · **offs** (dominoes likely to lose tricks — see
[[at-risk-points]]) · **renege** (failing to follow when able) · **set** (defeating the
contract) · **shaker** (dealer) · **lay down** (claiming remaining tricks) · **follow-me**
(no-trump) · **low-boy** (nel-o).

## Links

[[texas-42]] · [[suit-algebra-spec]] · [[play-phase-algebra]] · [[the-book-enters]] ·
[[at-risk-points]] · [[w42]] · [[engine]]
