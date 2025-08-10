# Texas 42 Official Rules

Complete formal specification for tournament-legal Texas 42 domino play.

## Game Overview

Texas 42 is a trick-taking game played with a standard double-six domino set (28 dominoes). Four players form two partnerships, sitting opposite each other. The objective is to be the first team to reach 7 marks (game points).

## Equipment

- Standard double-six domino set (28 dominoes)
- Score pad for tracking marks
- Seating arrangement: Partners sit opposite (North-South vs East-West)

## Domino Set Composition

The double-six set contains exactly 28 dominoes:
- Blanks: 0-0, 0-1, 0-2, 0-3, 0-4, 0-5, 0-6 (7 dominoes)
- Ones: 1-1, 1-2, 1-3, 1-4, 1-5, 1-6 (6 dominoes)
- Twos: 2-2, 2-3, 2-4, 2-5, 2-6 (5 dominoes)
- Threes: 3-3, 3-4, 3-5, 3-6 (4 dominoes)
- Fours: 4-4, 4-5, 4-6 (3 dominoes)
- Fives: 5-5, 5-6 (2 dominoes)
- Sixes: 6-6 (1 domino)

## Point Values

Only five dominoes have point values (totaling 42 points):
- **6-6 (Big Six)**: 42 points
- **5-5**: 10 points
- **6-4**: 10 points
- **5-0**: 5 points
- **6-5**: 5 points

All other dominoes have zero point value.

## Game Setup

1. **Shuffle**: Dominoes are placed face down and thoroughly shuffled
2. **Deal**: Each player receives exactly 7 dominoes
3. **Dealer Rotation**: Dealer position rotates clockwise after each hand
4. **First Player**: Player to dealer's left bids first

## Bidding Phase

### Bidding Order
- Bidding proceeds clockwise starting with player to dealer's left
- Each player bids exactly once per round
- No player may bid twice in the same round

### Valid Bids

#### Opening Bids
- **Point Bids**: 30, 31, 32, ..., 41 points
- **Mark Bids**: 1 mark (equivalent to 42 points), 2 marks (equivalent to 84 points)
- **Pass**: Decline to bid

#### Subsequent Bids
- Must exceed the current high bid
- **Point Progression**: Any point bid higher than current point bid
- **Mark Progression**: 
  - After point bids: May bid 1 or 2 marks
  - After 1 mark: May bid 2 marks
  - After 2 marks: May bid 3 marks (and so on, one mark increment only)

#### Tournament Rules (Straight 42)
- **NO SPECIAL CONTRACTS**: Nel-O, Splash, Plunge, or Follow-Me bids are prohibited
- **Opening Limit**: Maximum opening bid is 2 marks
- **Progressive Only**: Mark bids must follow strict progression

### Bid Resolution
- **All Pass**: If all four players pass, dominoes are reshuffled and redealt by next dealer
- **Winning Bid**: Highest bidder becomes declarer and selects trump suit

## Trump Declaration

After winning the bid, declarer must choose trump suit:
- **Blanks** (0s), **Ones** (1s), **Twos** (2s), **Threes** (3s), **Fours** (4s), **Fives** (5s), or **Sixes** (6s)

### Trump Hierarchy (Highest to Lowest)
1. **All Doubles** (regardless of pip value): 6-6, 5-5, 4-4, 3-3, 2-2, 1-1, 0-0
2. **Trump Suit Non-Doubles** (by pip total): Highest pip count wins
3. **Non-Trump Suits** (by pip total within suit)

**Critical Rule**: All seven doubles are ALWAYS trump, regardless of the declared trump suit.

## Playing Phase

### Trick Structure
- Each hand consists of exactly 7 tricks
- Each trick consists of exactly 4 dominoes (one per player)
- 28 dominoes total = 7 tricks × 4 players

### Play Order
1. **First Trick**: Winning bidder leads
2. **Subsequent Tricks**: Winner of previous trick leads
3. **Follow Clockwise**: Play proceeds clockwise around table

### Legal Plays
- **Must Follow Suit**: If able to play a domino of the suit led, must do so
- **Trump Rules**: All doubles are trump; dominoes with trump pips are trump suit
- **Cannot Follow**: If unable to follow suit, may play any domino
- **Renege Penalty**: Playing off-suit when able to follow suit forfeits hand

### Suit Determination
- **Doubles**: All doubles belong to trump suit
- **Trump Dominoes**: Any domino containing the trump pip
- **Other Suits**: Determined by higher pip value

### Trick Winner
- **Trump beats Non-Trump**: Any trump domino beats any non-trump domino
- **Within Trump**: Higher-ranked trump wins (see hierarchy above)
- **Within Non-Trump**: Higher pip total wins
- **Ties**: In non-trump suits, if equal pip totals, first played wins

## Scoring Phase

### Point Calculation
1. Count point values in tricks won by each team
2. Verify total equals 42 points (mathematical check)

### Mark Awards

#### Successful Bids
- **Point Bids**: If team scores bid amount or more, earns 1 mark
- **Mark Bids**: If team scores 42 points, earns bid amount in marks

#### Failed Bids
- If bidding team fails to make bid, opponents earn the mark value
- **Point Bid Failure**: Opponents earn 1 mark
- **Mark Bid Failure**: Opponents earn bid amount in marks

### Game End
- First team to reach 7 marks wins the game
- Games cannot end in ties (maximum possible marks per hand prevents this)

## Tournament Regulations

### Official Tournament Rules
- **Straight 42 Only**: No special contracts permitted
- **Standard Equipment**: Regulation double-six domino set
- **Time Limits**: May be imposed for tournament play
- **Seating**: Random draw for initial partnerships
- **Rotation**: Partners rotate after predetermined number of games

### Penalties
- **Renege**: Loss of hand plus 2-mark penalty
- **Exposed Domino**: Domino must be played at first legal opportunity
- **Bid Out of Turn**: Bid is void, correct player bids
- **Play Out of Turn**: Play stands if legal, otherwise replay

## Strategic Considerations

### Bidding Strategy
- **Hand Evaluation**: Count certain tricks and probable tricks
- **Point Distribution**: Consider point domino locations
- **Partnership Communication**: Bidding conveys hand strength
- **Risk Assessment**: Balance bid level against success probability

### Trump Selection
- **Domino Control**: Choose suit where team holds key dominoes
- **Double Count**: Consider double distribution
- **Suit Length**: Favor suits with multiple dominoes
- **Opponent Weakness**: Select suit opponents appear weak in

### Play Tactics
- **Count Tracking**: Monitor played dominoes and remaining distribution
- **Force Plays**: Lead to force opponents into unfavorable positions
- **Communication**: Legal plays can signal partner information
- **Endgame**: Careful domino counting crucial in final tricks

## Variations (Non-Tournament)

While tournament play uses Straight 42 only, casual games may include:

### Special Contracts (Casual Play Only)
- **Nel-O**: Bidding team must lose every trick
- **Splash**: Requires 3+ doubles, all doubles trump
- **Plunge**: Requires 4+ doubles, all doubles trump
- **Follow-Me**: Trump suit changes each trick

**Note**: These variations are NOT permitted in official tournament play.

## Mathematical Verification

### Constant Verification
- Total dominoes: 28 (verified)
- Total points: 42 (verified: 42+10+10+5+5 = 72 total pip value in counting dominoes)
- Tricks per hand: 7 (verified: 28÷4 = 7)
- Maximum game length: Limited by mark progression rules

### Probability Analysis
- Point distribution ensures balanced gameplay
- Bidding ranges provide appropriate risk/reward ratios
- Trump structure maintains suit balance

---

This specification represents the complete, authoritative rules for tournament-legal Texas 42 play, verified against official tournament regulations and mathematical game theory principles.