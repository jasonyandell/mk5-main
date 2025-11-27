# Texas 42 Game Rules

## Overview

Texas 42 is a trick-taking game played with dominoes. Four players in two partnerships compete to reach 7 marks by bidding and winning tricks.

## Equipment
- 28 double-six dominoes (0-0 through 6-6)
- 4 players in 2 partnerships (partners sit opposite)
- Each player draws 7 dominoes (no boneyard)

## Scoring Summary

### Counting Dominoes (35 points total)
| Domino | Points |
|--------|--------|
| 5-5, 6-4 | 10 each |
| 5-0, 4-1, 3-2 | 5 each |
| All others | 0 |

### Trick Points
- Each trick won: 1 point
- 7 tricks total: 7 points
- **Hand total: 35 (count) + 7 (tricks) = 42 points**

### Mark System
- Game to 7 marks
- Successful 30-41 bid: 1 mark
- Successful 42+ bid: marks equal to bid level
- Failed bid: opponents get marks bid

## Bidding

### Order
1. Player left of dealer bids first
2. Clockwise, each player bids once or passes
3. Minimum bid: 30, maximum opening: 2 marks (84)
4. Each bid must exceed previous

### Bid Values
- Point bids: 30-41
- Mark bids: 1 mark (42), 2 marks (84), 3+ marks

### Critical Rules
- Can open at 2 marks maximum
- 3 marks only after 2 marks already bid
- Plunge (4+ marks) only with 4+ doubles

## Trump Declaration

Winner declares trump before playing:
- **Suit trump**: blanks, ones, twos, threes, fours, fives, or sixes
- **Doubles trump**: all 7 doubles form trump suit
- **No-trump (follow-me)**: no trump suit

## Play Mechanics

### Leading
- Bid winner leads first trick
- Trick winner leads next
- Any domino may be led

### Following Suit
1. **Suit determination**: Higher end of non-trump domino = suit
2. **Must follow**: Play suit if able
3. **Can't follow**: May trump or play anything

### Winning Tricks
1. Highest trump wins
2. If no trump, highest of led suit wins
3. Ties: first played wins (in Sevens)

### Doubles
- Doubles belong to their natural suit
- 6-6 is highest six, 5-5 is highest five, etc.
- When doubles are trump, only doubles are trump

## Special Contracts

### Nello (Nel-O)
**Bid**: 1+ marks
**Objective**: Bidder loses ALL tricks
**Rules**:
- Partner sits out (dominoes face-down)
- No trump declared
- 3 players play (trick complete at 3 dominoes)
- Doubles form own suit (blanks through sixes, then doubles)

### Splash
**Bid**: 2-3 marks
**Requires**: 3+ doubles in hand
**Rules**:
- Partner declares trump and leads
- Must win ALL 7 tricks
- Fails if any trick lost

### Plunge
**Bid**: 4+ marks
**Requires**: 4+ doubles in hand
**Rules**:
- Partner declares trump and leads
- Must win ALL 7 tricks
- Only case where jump bid or 4+ mark opening allowed

### Sevens
**Bid**: 1+ marks
**Rules**:
- Domino closest to 7 total pips wins trick
- Ties: first played wins
- Must win ALL 7 tricks (rarely accepted in serious play)

## Tournament Rules (Straight 42)

N42PA tournament standard:
- **No special contracts** (no Nello, Splash, Plunge, Sevens)
- Strict communication prohibition
- First to 7 marks
- If all pass, reshuffle (or dealer forced to bid 30)

## Implementation Notes

### Layer Mapping
| Contract | Layer | Key Rule Override |
|----------|-------|-------------------|
| Standard | `base.ts` | Default 4-player tricks |
| Nello | `nello.ts` | `isTrickComplete` → 3 plays |
| Splash | `splash.ts` | Partner leads, must sweep |
| Plunge | `plunge.ts` | 4+ doubles, partner leads |
| Sevens | `sevens.ts` | Distance from 7 wins |
| Tournament | `tournament.ts` | Filters out special bids |

### Trump Type Discrimination
```typescript
type Trump =
  | { type: 'suit'; suit: Suit }
  | { type: 'doubles' }
  | { type: 'follow-me' }
  | { type: 'nello' }
  | { type: 'splash'; suit: Suit }
  | { type: 'plunge'; suit: Suit }
  | { type: 'sevens' }
```

Layers check `state.trump.type` to activate their rules.

### Hand Outcome
```typescript
type HandOutcome =
  | { isDetermined: false }
  | { isDetermined: true; reason: string; decidedAtTrick?: number }
```

Special contracts can determine outcome early (nello/splash lose on first trick lost).

## Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **Bidder** | Player who wins bidding |
| **Count/Counters** | Dominoes worth 5 or 10 points |
| **Marks** | Scoring units (game to 7) |
| **Offs** | Dominoes likely to lose tricks |
| **Renege** | Failure to follow suit when able |
| **Set** | Defeating bidder's contract |
| **Shaker** | Dealer who shuffles |
| **Trump** | Suit that beats all others |
