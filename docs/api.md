# Texas 42 Game Engine API

Complete API reference for the mk5 Texas 42 game engine.

## Core Concepts

The game engine follows a pure functional architecture with immutable state and explicit state transitions.

### State Machine Pattern
- All game state is immutable
- State transitions are explicit via `getNextStates()`
- No side effects in game logic
- Predictable and testable behavior

## Types

### GameState
```typescript
interface GameState {
  phase: GamePhase;
  players: Player[];
  currentPlayer: number;
  dealer: number;
  bids: Bid[];
  currentBid: Bid | null;
  winningBidder: number | null;
  trump: Trump | null;
  tricks: Trick[];
  currentTrick: Play[];
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;
  tournamentMode: boolean;
}
```

### Player
```typescript
interface Player {
  id: number;
  name: string;
  hand: Domino[];
  teamId: 0 | 1;
  marks: number;
}
```

### Domino
```typescript
interface Domino {
  high: number;
  low: number;
  id: string;
}
```

### Bid
```typescript
interface Bid {
  type: BidType;
  value?: number;
  player: number;
}

type BidType = 'pass' | 'points' | 'marks' | 'nello' | 'splash' | 'plunge';
```

### StateTransition
```typescript
interface StateTransition {
  id: string;
  label: string;
  newState: GameState;
}
```

## Core Functions

### State Management

#### createInitialState()
```typescript
function createInitialState(): GameState
```
Creates a fresh game state with dealt hands and proper initialization.

**Returns**: New `GameState` with all players dealt 7 dominoes each.

**Example**:
```typescript
import { createInitialState } from './game';

const state = createInitialState();
console.log(state.phase); // 'bidding'
console.log(state.players.length); // 4
```

#### getNextStates()
```typescript
function getNextStates(state: GameState): StateTransition[]
```
Core state machine function returning all valid transitions from current state.

**Parameters**:
- `state` - Current game state

**Returns**: Array of valid state transitions with descriptive labels.

**Example**:
```typescript
import { createInitialState, getNextStates } from './game';

const state = createInitialState();
const actions = getNextStates(state);

// Bidding phase actions
actions.forEach(action => {
  console.log(action.label); // "Pass", "Bid 30 points", etc.
});

// Execute an action
const newState = actions[0].newState;
```

#### cloneGameState()
```typescript
function cloneGameState(state: GameState): GameState
```
Creates a deep copy of game state for immutable operations.

**Parameters**:
- `state` - State to clone

**Returns**: Deep copy of the game state.

#### validateGameState()
```typescript
function validateGameState(state: GameState): string[]
```
Validates game state integrity and rule compliance.

**Parameters**:
- `state` - State to validate

**Returns**: Array of error messages (empty if valid).

### Rule Validation

#### isValidBid()
```typescript
function isValidBid(
  state: GameState, 
  bid: Bid, 
  playerHand?: Domino[]
): boolean
```
Validates whether a bid is legal in the current game context.

**Parameters**:
- `state` - Current game state
- `bid` - Bid to validate
- `playerHand` - Optional player hand for special contract validation

**Returns**: `true` if bid is valid, `false` otherwise.

**Example**:
```typescript
import { isValidBid, BID_TYPES } from './game';

const bid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
const isValid = isValidBid(state, bid);
```

#### isValidPlay()
```typescript
function isValidPlay(
  state: GameState, 
  domino: Domino, 
  playerId: number
): boolean
```
Validates whether a domino play is legal.

**Parameters**:
- `state` - Current game state
- `domino` - Domino to play
- `playerId` - Player making the play

**Returns**: `true` if play is valid, `false` otherwise.

#### getValidPlays()
```typescript
function getValidPlays(
  hand: Domino[], 
  currentTrick: Play[], 
  trump: Trump
): Domino[]
```
Gets all valid domino plays for current situation.

**Parameters**:
- `hand` - Player's current hand
- `currentTrick` - Current trick plays
- `trump` - Current trump suit

**Returns**: Array of playable dominoes.

### Domino Utilities

#### getDominoSuit()
```typescript
function getDominoSuit(domino: Domino, trump: number | null): number
```
Determines the suit of a domino considering trump rules.

**Important**: All doubles are trump regardless of declared trump suit.

#### getDominoValue()
```typescript
function getDominoValue(domino: Domino, trump: number | null): number
```
Gets the trick-taking value of a domino.

**Hierarchy**:
1. Trump doubles (100+): 6-6, 5-5, 4-4, 3-3, 2-2, 1-1, 0-0
2. Trump non-doubles (50+): By pip total
3. Non-trump (0-20): By pip total

#### getDominoPoints()
```typescript
function getDominoPoints(domino: Domino): number
```
Gets the scoring points for a domino.

**Point Values**:
- 6-6: 42 points
- 5-5: 10 points
- 6-4: 10 points
- 5-0: 5 points
- 6-5: 5 points
- All others: 0 points

### Scoring

#### calculateTrickWinner()
```typescript
function calculateTrickWinner(trick: Play[], trump: number): number
```
Determines the winner of a completed trick.

#### calculateRoundScore()
```typescript
function calculateRoundScore(state: GameState): [number, number]
```
Calculates mark awards at the end of a hand.

**Returns**: Updated team marks `[team0Marks, team1Marks]`.

## Constants

### Game Constants
```typescript
const GAME_CONSTANTS = {
  TOTAL_DOMINOES: 28,
  HAND_SIZE: 7,
  TOTAL_POINTS: 42,
  TRICKS_PER_HAND: 7,
  PLAYERS: 4,
  TEAMS: 2,
  MIN_BID: 30,
  MAX_BID: 41,
  DEFAULT_GAME_TARGET: 7,
};
```

### Bid Types
```typescript
const BID_TYPES = {
  PASS: 'pass',
  POINTS: 'points',
  MARKS: 'marks',
  NELLO: 'nello',    // Casual mode only
  SPLASH: 'splash',  // Casual mode only
  PLUNGE: 'plunge',  // Casual mode only
};
```

### Trump Suits
```typescript
const TRUMP_SUITS = {
  BLANKS: 0,
  ONES: 1,
  TWOS: 2,
  THREES: 3,
  FOURS: 4,
  FIVES: 5,
  SIXES: 6,
};
```

## Usage Patterns

### Basic Game Loop
```typescript
import { createInitialState, getNextStates } from './game';

let state = createInitialState();

while (state.phase !== 'game_end') {
  const actions = getNextStates(state);
  
  if (actions.length === 0) {
    console.log('No valid actions available');
    break;
  }
  
  // Select action (e.g., user input, AI decision)
  const selectedAction = actions[0];
  
  // Execute action
  state = selectedAction.newState;
  
  console.log(`Phase: ${state.phase}, Current Player: ${state.currentPlayer}`);
}
```

### State Validation
```typescript
import { validateGameState } from './game';

const errors = validateGameState(state);
if (errors.length > 0) {
  console.error('Game state errors:', errors);
}
```

### Custom Game Scenarios
```typescript
import { GameTestHelper } from './tests/helpers/gameTestHelper';

// Create specific scenarios for testing
const biddingState = GameTestHelper.createBiddingScenario(0, []);
const playingState = GameTestHelper.createPlayingScenario(3, 0);

// Validate tournament rules
const tournamentErrors = GameTestHelper.validateTournamentRules(state);
```

## Error Handling

The API uses explicit error reporting rather than exceptions:

```typescript
// Functions return error arrays
const errors = validateGameState(state);
if (errors.length > 0) {
  // Handle validation errors
}

// Functions return boolean for validity
if (!isValidBid(state, bid)) {
  // Handle invalid bid
}

// State transitions may return empty arrays
const actions = getNextStates(state);
if (actions.length === 0) {
  // No valid actions available
}
```

## Performance Considerations

- **Immutability**: All state operations create new objects
- **Caching**: Consider memoizing `getNextStates()` for repeated calls
- **Memory**: Clone operations are deep but efficient for game-sized data
- **Validation**: Rule validation is computationally lightweight

## Thread Safety

The pure functional design makes the engine inherently thread-safe:
- No shared mutable state
- No side effects in game logic
- All operations are deterministic
- Safe for concurrent access patterns

---

This API enables building robust Texas 42 applications with confidence in rule compliance and state management.