# Texas 42 - Pure Functional RuleSet Composition with Composition Points

**⚠️ HISTORICAL DOCUMENT**: This document uses legacy terminology. "Layer" is now "RuleSet", "Variant" is now "ActionTransformer". Content preserved for reference.

**Created**: 2025-10-25
**Branch**: mk8
**Status**: Architecture validated, ready for implementation (terminology updated 2025)
**Supersedes**: remix-layers.md (adds composition points pattern)

---

## Executive Summary

This document specifies the complete architecture for implementing Texas 42 special contracts (nello, plunge, splash, sevens) using **pure functional layer composition** with **minimal composition points**.

**Key Innovation**: Instead of refactoring all executors to use composed rules (heavy), we add **two intermediate phases** (`bidding_complete`, `trump_selection_complete`) that act as composition hooks where layers can intervene. This is minimal, pure, and handles all special contracts.

**Validation**: All special contracts from `docs/rules.md` are fully covered by this architecture.

---

## Context and Motivation

### The Problem

Texas 42 is a **family of related games** that share mechanics but have different rules:

1. **Standard Texas 42** - 4 players, regular bidding, normal trump, standard trick-taking
2. **Nello** - 3 players (partner sits out), no trump, doubles own suit, lose all tricks
3. **Plunge** - Partner selects trump and leads (not bidder), must win all tricks
4. **Splash** - Same as plunge but with 3+ doubles instead of 4+
5. **Sevens** - Completely different win condition (closest to 7 pips)

**Current architecture cannot support these** because:
- Executors hardcode `currentPlayer = bidder` for trump selection
- Executors hardcode `currentPlayer = bidder` for first leader
- Trick completion assumes 4 players
- No composition point for variants to override behavior

### Why Not Refactor All Executors?

The layers document (remix-layers.md) proposes refactoring executors to use composed rules:

```typescript
// Refactored executor
function executeTrumpSelection(rules, state, player, selection) {
  const firstLeader = rules.getFirstLeader(state, player);
  return { ...state, currentPlayer: firstLeader };
}
```

**This is correct but heavy** - requires refactoring ~10 executor functions and threading rules through everything.

### The Lightweight Solution: Composition Points

Add **two intermediate phases** that act as composition hooks:

```typescript
type GamePhase =
  | 'setup'
  | 'bidding'
  | 'bidding_complete'           // ← NEW: composition point
  | 'trump_selection'
  | 'trump_selection_complete'   // ← NEW: composition point
  | 'playing'
  | 'scoring'
  | 'game_end';
```

**Benefits:**
- ✅ Minimal changes to existing executors
- ✅ Clean composition points for layers
- ✅ Pure functional
- ✅ No executor refactoring needed
- ✅ Handles all special contracts

---

## Key Architectural Insights

### 1. Nello is NOT a Bid Type

**Critical realization from this session:**

```typescript
// WRONG: Nello as bid type
{ type: 'bid', bid: 'nello', value: 1 }  // ❌ This doesn't exist!

// RIGHT: Nello as trump type
// During bidding:
{ type: 'bid', bid: 'marks', value: 1 }  // Player bids marks

// During trump selection:
{ type: 'select-trump', trump: { type: 'nello' } }  // Player declares nello
```

**Why this matters:**
- Nello is revealed during **trump selection**, not bidding
- During bidding, a nello bid looks identical to a marks bid
- Only when bidder selects "nello" instead of a suit does it become nello
- This is why we check `state.trump?.type === 'nello'` not `state.winningBid?.bid === 'nello'`

### 2. Plunge/Splash ARE Bid Types

```typescript
// During bidding:
{ type: 'bid', bid: 'plunge', value: 4 }  // ✅ Plunge is a bid type

// Check during play:
if (state.winningBid?.bid === 'plunge') {
  // Apply plunge rules
}
```

**Why this matters:**
- Plunge/splash are declared during bidding (player says "I plunge")
- They modify who selects trump (partner, not bidder)
- They modify who leads (partner, not bidder)
- But trump is still selected normally (suit, doubles, or no-trump)

### 3. Contract State Lives in Two Places

```typescript
interface GameState {
  winningBid: Bid;         // { type: 'plunge', value: 4, player: 0 }
  trump: TrumpSelection;   // { type: 'suit', suit: 6 } or { type: 'nello' }
}

// Check for plunge:
state.winningBid?.bid === 'plunge'

// Check for nello:
state.trump?.type === 'nello'

// Standard game:
state.winningBid?.bid === 'marks' && state.trump?.type === 'suit'
```

### 4. Variants vs Layers Terminology

**"Variants" in config** = user-selected game modes:
```typescript
config.variants = [
  { type: 'nello' },      // Enable nello
  { type: 'plunge' },     // Enable plunge
  { type: 'speed' },      // Enable speed mode
  { type: 'tournament' }  // Enable tournament restrictions
]
```

**"Layers" in code** = implementation of those variants:
```typescript
const layers = [
  baseLayer,
  nelloLayer,
  plungeLayer,
  speedLayer,
  tournamentLayer
];
```

We use "layer" for code, "variant" for user-facing configuration.

---

## Core Architecture

### Type Definitions

```typescript
/**
 * Bid types - what you bid during bidding phase
 */
export type BidType =
  | 'pass'
  | 'points'    // 30-41 point bids
  | 'marks'     // 1+ mark bids (42+ points)
  | 'plunge'    // 4+ marks, requires 4+ doubles
  | 'splash'    // 2-3 marks, requires 3+ doubles
  | 'sevens';   // 1+ marks, distance from 7 wins

/**
 * Trump types - what you select during trump selection phase
 */
export interface TrumpSelection {
  type: 'not-selected' | 'suit' | 'doubles' | 'no-trump' | 'nello';
  suit?: RegularSuit;  // Only when type === 'suit'
}

/**
 * Game phases - including composition points
 */
export type GamePhase =
  | 'setup'
  | 'bidding'
  | 'bidding_complete'           // NEW: composition point after bidding
  | 'trump_selection'
  | 'trump_selection_complete'   // NEW: composition point after trump selection
  | 'playing'
  | 'scoring'
  | 'game_end';

/**
 * Game actions - including composition point actions
 */
export type GameAction =
  | { type: 'bid'; player: number; bid: BidType; value?: number }
  | { type: 'pass'; player: number }
  | { type: 'select-trump'; player: number; trump: TrumpSelection }
  | { type: 'play'; player: number; dominoId: string }
  | { type: 'redeal' }
  | { type: 'advance-to-trump-selection' }   // NEW: composition point action
  | { type: 'advance-to-playing' };          // NEW: composition point action
```

### GameRules Interface

```typescript
/**
 * Composable rules that layers can override
 * Each method follows (state, prev) => result pattern
 */
interface GameRules {
  // Structural rules (enable nello 3-player tricks)
  getTrickSize(state: GameState, prev: number): number;
  shouldSkipPlayer(state: GameState, playerId: number, prev: boolean): boolean;
  isTrickComplete(state: GameState, prev: boolean): boolean;

  // Suit and trump rules
  getSuitOfDomino(state: GameState, domino: Domino, prev: number): number;
  getTrumpForTrick(state: GameState, prev: TrumpSelection): TrumpSelection;

  // Trick winner calculation
  calculateTrickWinner(state: GameState, trick: Play[], prev: number): number;

  // Turn order
  getNextPlayer(state: GameState, current: number, prev: number): number;

  // Hand outcome determination (early termination)
  checkHandOutcome(state: GameState, prev: HandOutcome | null): HandOutcome | null;

  // Domino comparison (for "doubles low" variation)
  compareDominoes(state: GameState, d1: Domino, d2: Domino, suit: number, prev: number): number;
}
```

### GameLayer Interface

```typescript
/**
 * A layer can override any combination of:
 * - Action generation (getValidActions)
 * - Action execution (executeAction)
 * - Game rules (rules)
 */
interface GameLayer {
  name: string;

  /**
   * Transform the list of valid actions
   * Pattern: filter, annotate, or add actions
   */
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[];

  /**
   * Transform the result of action execution
   * Pattern: override specific fields of prev result
   */
  executeAction?: (state: GameState, action: GameAction, prev: GameState) => GameState;

  /**
   * Override specific rule methods
   * Each rule follows (state, ...args, prev) => result pattern
   */
  rules?: Partial<GameRules>;
}
```

---

## Composition Points Pattern

### The Problem They Solve

Base executors hardcode who selects trump and who leads:

```typescript
// In executeBid (when bidding completes)
return {
  ...state,
  phase: 'trump_selection',
  currentPlayer: winningBid.player  // ← Always bidder
};

// In executeTrumpSelection (when trump selected)
return {
  ...state,
  phase: 'playing',
  currentPlayer: player  // ← Always trump selector (bidder)
};
```

**Plunge needs:**
- Partner selects trump (not bidder)
- Partner leads (not bidder)

**Without composition points**, we'd need to refactor these executors to use rules.

### The Solution: Intermediate Phases + Advance Actions

#### 1. Bidding Complete Composition Point

```typescript
// Base executor sets intermediate phase
function executeBid(state, player, bidType, value) {
  // ... validate and record bid ...

  if (newBids.length === 4 && nonPassBids.length > 0) {
    const winningBid = /* find highest */;

    return {
      ...state,
      bids: newBids,
      currentBid: winningBid,
      phase: 'bidding_complete',        // ← Intermediate phase
      winningBidder: winningBid.player,
      currentPlayer: winningBid.player  // ← Default: bidder
    };
  }
}

// getValidActions returns auto-advance action
function getValidActions(state) {
  if (state.phase === 'bidding_complete') {
    return [{ type: 'advance-to-trump-selection' }];
  }
  // ...
}

// Base executor completes the transition
function executeAction(state, action) {
  if (action.type === 'advance-to-trump-selection') {
    return {
      ...state,
      phase: 'trump_selection'
      // currentPlayer stays as-is (layers may have overridden it)
    };
  }
  // ...
}

// Plunge layer intercepts to override currentPlayer
plungeLayer.executeAction = (state, action, prev) => {
  if (action.type === 'advance-to-trump-selection' &&
      state.winningBid?.bid === 'plunge') {
    const partner = (state.winningBidder + 2) % 4;
    return {
      ...prev,
      currentPlayer: partner  // ← Override before advancing
    };
  }
  return prev;
};
```

#### 2. Trump Selection Complete Composition Point

```typescript
// Base executor sets intermediate phase
function executeTrumpSelection(state, player, selection) {
  // ... validate trump selection ...

  return {
    ...state,
    phase: 'trump_selection_complete',  // ← Intermediate phase
    trump: selection,
    currentPlayer: player  // ← Default: trump selector (bidder or partner)
  };
}

// getValidActions returns auto-advance action
function getValidActions(state) {
  if (state.phase === 'trump_selection_complete') {
    return [{ type: 'advance-to-playing' }];
  }
  // ...
}

// Base executor completes the transition
function executeAction(state, action) {
  if (action.type === 'advance-to-playing') {
    return {
      ...state,
      phase: 'playing'
      // currentPlayer stays as-is
    };
  }
  // ...
}

// Plunge layer can intercept if needed (but partner already selected trump, so no override needed)
// Nello layer intercepts to set next player as leader
nelloLayer.executeAction = (state, action, prev) => {
  if (action.type === 'select-trump' && action.trump.type === 'nello') {
    const nextPlayer = (state.winningBidder + 1) % 4;
    return {
      ...prev,
      currentPlayer: nextPlayer  // ← Player after bidder leads
    };
  }
  return prev;
};
```

### Why This Works

1. **Minimal changes**: Add 2 phases, 2 action types
2. **Clean hooks**: Layers intercept `advance-*` actions to override state
3. **Pure functional**: Layers return `{ ...prev, field: newValue }`
4. **Explicit composition**: The intermediate phases clearly signal "layers can modify here"
5. **Action history complete**: Advance actions appear in history for debugging

---

## Layer Implementations

### Base Layer

```typescript
const baseLayer: GameLayer = {
  name: 'base',

  getValidActions: (state, prev) => {
    // Base generates actions from scratch (ignores prev)
    switch (state.phase) {
      case 'bidding':
        return getBiddingActions(state);

      case 'bidding_complete':
        return [{ type: 'advance-to-trump-selection' }];

      case 'trump_selection':
        return getTrumpSelectionActions(state);

      case 'trump_selection_complete':
        return [{ type: 'advance-to-playing' }];

      case 'playing':
        return getPlayingActions(state);

      case 'scoring':
        return getScoringActions(state);

      default:
        return [];
    }
  },

  executeAction: (state, action, prev) => {
    // Base executes all core actions
    // prev is ignored (base is first in pipeline)
    return coreExecuteAction(state, action);
  },

  rules: {
    getTrickSize: (state, prev) => 4,
    shouldSkipPlayer: (state, playerId, prev) => false,
    isTrickComplete: (state, prev) => state.currentTrick.length === 4,
    getSuitOfDomino: (state, domino, prev) => domino.high,  // Higher end
    getTrumpForTrick: (state, prev) => state.trump,
    calculateTrickWinner: (state, trick, prev) => standardTrickWinner(state, trick),
    getNextPlayer: (state, current, prev) => (current + 1) % 4,
    checkHandOutcome: (state, prev) => {
      // Base checks for standard points/marks bids
      if (state.phase !== 'playing' && state.phase !== 'scoring') {
        return null;
      }
      if (state.phase === 'scoring') {
        return { isDetermined: true, reason: 'Hand complete' };
      }

      const bid = state.currentBid;
      if (!bid || bid.player === -1 || bid.type === 'pass') {
        return null;
      }

      // Only handle standard bids here
      if (bid.type === 'points') {
        return checkPointsBidOutcome(state, bid);
      }
      if (bid.type === 'marks') {
        return checkMarksBidOutcome(state, bid);
      }

      return prev;  // Let layers handle special contracts
    },
    compareDominoes: (state, d1, d2, suit, prev) => standardComparison(d1, d2, suit)
  }
};
```

### Nello Layer

```typescript
const nelloLayer: GameLayer = {
  name: 'nello',

  getValidActions: (state, prev) => {
    // Add nello as trump option ONLY if bid was marks 1+
    if (state.phase === 'trump_selection' &&
        state.winningBid?.bid === 'marks' &&
        state.winningBid.value >= 1) {
      return [...prev, {
        type: 'select-trump',
        player: state.winningBidder,
        trump: { type: 'nello' }
      }];
    }

    return prev;
  },

  executeAction: (state, action, prev) => {
    // When nello selected: next player leads (not bidder)
    if (action.type === 'select-trump' && action.trump.type === 'nello') {
      const nextPlayer = (state.winningBidder + 1) % 4;
      return {
        ...prev,
        currentPlayer: nextPlayer
      };
    }

    return prev;
  },

  rules: {
    // 3-player tricks (partner sits out)
    getTrickSize: (state, prev) =>
      state.trump?.type === 'nello' ? 3 : prev,

    // Partner doesn't play
    shouldSkipPlayer: (state, playerId, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      const partner = (state.winningBidder + 2) % 4;
      return playerId === partner;
    },

    // Trick complete with 3 plays
    isTrickComplete: (state, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      return state.currentTrick.length === 3;
    },

    // Doubles form own suit (suit 7)
    getSuitOfDomino: (state, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      return domino.high === domino.low ? 7 : domino.high;
    },

    // No trump in nello
    getTrumpForTrick: (state, prev) =>
      state.trump?.type === 'nello' ? { type: 'no-trump' } : prev,

    // Custom trick winner (no trump, doubles as suit 7)
    calculateTrickWinner: (state, trick, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      return calculateNelloTrickWinner(state, trick);
    },

    // Skip partner in turn order
    getNextPlayer: (state, current, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      const partner = (state.winningBidder + 2) % 4;
      let next = (current + 1) % 4;

      if (next === partner) {
        next = (next + 1) % 4;
      }

      return next;
    },

    // Check for early termination (bidder wins trick)
    checkHandOutcome: (state, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      if (prev?.isDetermined) return prev;  // Already determined

      // Check if bidder won any tricks
      const bidderWonTrick = state.tricks.some(trick =>
        trick.winner === state.winningBidder
      );

      if (bidderWonTrick) {
        return {
          isDetermined: true,
          reason: 'Bidding team won a trick on nello',
          decidedAtTrick: state.tricks.length
        };
      }

      return prev;
    }
  }
};
```

### Plunge Layer

```typescript
const plungeLayer: GameLayer = {
  name: 'plunge',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    // Add plunge bid if player has 4+ doubles
    const player = state.players[state.currentPlayer];
    if (countDoubles(player.hand) >= 4) {
      // Determine value based on current bidding
      const highestBid = getHighestMarkBid(state.bids);
      const plungeValue = highestBid >= 4 ? 5 : 4;

      return [...prev, {
        type: 'bid',
        player: state.currentPlayer,
        bid: 'plunge',
        value: plungeValue
      }];
    }

    return prev;
  },

  executeAction: (state, action, prev) => {
    // When plunge wins: partner selects trump
    if (action.type === 'advance-to-trump-selection' &&
        state.winningBid?.bid === 'plunge') {
      const partner = (state.winningBidder + 2) % 4;
      return {
        ...prev,
        currentPlayer: partner  // Override: partner selects trump
      };
    }

    // When trump selected in plunge: partner leads
    // (Actually partner already selected trump, so currentPlayer is already partner)
    // No override needed for advance-to-playing

    return prev;
  },

  rules: {
    // Check for early termination (opponents win trick)
    checkHandOutcome: (state, prev) => {
      if (state.winningBid?.bid !== 'plunge') return prev;
      if (prev?.isDetermined) return prev;

      const biddingTeam = state.winningBidder % 2;
      const defendingTeamWonTrick = state.tricks.some(trick =>
        trick.winner !== undefined && trick.winner % 2 !== biddingTeam
      );

      if (defendingTeamWonTrick) {
        return {
          isDetermined: true,
          reason: 'Defending team won a trick on plunge',
          decidedAtTrick: state.tricks.length
        };
      }

      return prev;
    }
  }
};
```

### Splash Layer

```typescript
const splashLayer: GameLayer = {
  name: 'splash',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    // Add splash bid if player has 3+ doubles
    const player = state.players[state.currentPlayer];
    if (countDoubles(player.hand) >= 3) {
      return [...prev,
        { type: 'bid', player: state.currentPlayer, bid: 'splash', value: 2 },
        { type: 'bid', player: state.currentPlayer, bid: 'splash', value: 3 }
      ];
    }

    return prev;
  },

  executeAction: (state, action, prev) => {
    // Same pattern as plunge: partner selects trump and leads
    if (action.type === 'advance-to-trump-selection' &&
        state.winningBid?.bid === 'splash') {
      const partner = (state.winningBidder + 2) % 4;
      return {
        ...prev,
        currentPlayer: partner
      };
    }

    return prev;
  },

  rules: {
    // Check for early termination (same as plunge)
    checkHandOutcome: (state, prev) => {
      if (state.winningBid?.bid !== 'splash') return prev;
      if (prev?.isDetermined) return prev;

      const biddingTeam = state.winningBidder % 2;
      const defendingTeamWonTrick = state.tricks.some(trick =>
        trick.winner !== undefined && trick.winner % 2 !== biddingTeam
      );

      if (defendingTeamWonTrick) {
        return {
          isDetermined: true,
          reason: 'Defending team won a trick on splash',
          decidedAtTrick: state.tricks.length
        };
      }

      return prev;
    }
  }
};
```

### Sevens Layer

```typescript
const sevensLayer: GameLayer = {
  name: 'sevens',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    // Add sevens bid options (1-2 marks)
    return [...prev,
      { type: 'bid', player: state.currentPlayer, bid: 'sevens', value: 1 },
      { type: 'bid', player: state.currentPlayer, bid: 'sevens', value: 2 }
    ];
  },

  rules: {
    // Custom trick winner: closest to 7 total pips
    calculateTrickWinner: (state, trick, prev) => {
      if (state.winningBid?.bid !== 'sevens') return prev;

      const distances = trick.map(play =>
        Math.abs(7 - (play.domino.high + play.domino.low))
      );
      const minDistance = Math.min(...distances);

      return trick.findIndex(play =>
        Math.abs(7 - (play.domino.high + play.domino.low)) === minDistance
      );
    },

    // Check for early termination (opponents win trick)
    checkHandOutcome: (state, prev) => {
      if (state.winningBid?.bid !== 'sevens') return prev;
      if (prev?.isDetermined) return prev;

      const biddingTeam = state.winningBidder % 2;
      const defendingTeamWonTrick = state.tricks.some(trick =>
        trick.winner !== undefined && trick.winner % 2 !== biddingTeam
      );

      if (defendingTeamWonTrick) {
        return {
          isDetermined: true,
          reason: 'Defending team won a trick on sevens',
          decidedAtTrick: state.tricks.length
        };
      }

      return prev;
    }
  }
};
```

### Tournament Layer

```typescript
const tournamentLayer: GameLayer = {
  name: 'tournament',

  getValidActions: (state, prev) => {
    // Filter out special contracts during bidding
    if (state.phase === 'bidding') {
      return prev.filter(action =>
        action.type !== 'bid' ||
        !['nello', 'plunge', 'splash', 'sevens'].includes(action.bid)
      );
    }

    // Filter out nello trump option
    if (state.phase === 'trump_selection') {
      return prev.filter(action =>
        action.type !== 'select-trump' ||
        action.trump.type !== 'nello'
      );
    }

    return prev;
  }
};
```

### Speed Layer

```typescript
const speedLayer: GameLayer = {
  name: 'speed',

  getValidActions: (state, prev) => {
    if (state.phase !== 'playing') return prev;

    // Group actions by player
    const actionsByPlayer = new Map<number, GameAction[]>();
    const neutralActions: GameAction[] = [];

    for (const action of prev) {
      if ('player' in action) {
        const player = action.player;
        if (!actionsByPlayer.has(player)) {
          actionsByPlayer.set(player, []);
        }
        actionsByPlayer.get(player)!.push(action);
      } else {
        neutralActions.push(action);
      }
    }

    // Auto-execute if only one action for a player
    const result: GameAction[] = [];

    for (const [player, actions] of actionsByPlayer.entries()) {
      if (actions.length === 1) {
        result.push({
          ...actions[0],
          autoExecute: true,
          meta: {
            ...('meta' in actions[0] ? actions[0].meta : {}),
            speedMode: true,
            reason: 'only-legal-action'
          }
        });
      } else {
        result.push(...actions);
      }
    }

    result.push(...neutralActions);
    return result;
  }
};
```

### Hints Layer

```typescript
const hintsLayer: GameLayer = {
  name: 'hints',

  getValidActions: (state, prev) => {
    // Add hint metadata to all actions
    return prev.map(action => ({
      ...action,
      meta: {
        ...('meta' in action ? action.meta : {}),
        hint: generateHint(state, action),
        requiredCapabilities: [{ type: 'see-hints' }]
      }
    }));
  }
};
```

---

## Engine Composition

### Composition Function

```typescript
function composeEngine(layers: GameLayer[]): ComposedEngine {
  // Compose getValidActions (reduce pipeline)
  const getValidActions = (state: GameState): GameAction[] => {
    return layers.reduce(
      (actions, layer) =>
        layer.getValidActions?.(state, actions) ?? actions,
      [] as GameAction[]
    );
  };

  // Compose executeAction (reduce pipeline)
  const executeAction = (state: GameState, action: GameAction): GameState => {
    return layers.reduce(
      (prevState, layer) =>
        layer.executeAction?.(state, action, prevState) ?? prevState,
      state  // Base layer will execute, others will transform
    );
  };

  // Compose each rule method
  const rules: GameRules = {} as GameRules;

  const ruleNames: (keyof GameRules)[] = [
    'getTrickSize',
    'shouldSkipPlayer',
    'isTrickComplete',
    'getSuitOfDomino',
    'getTrumpForTrick',
    'calculateTrickWinner',
    'getNextPlayer',
    'checkHandOutcome',
    'compareDominoes'
  ];

  for (const ruleName of ruleNames) {
    rules[ruleName] = (state: GameState, ...args: any[]) => {
      const initialValue = getInitialValue(ruleName);

      return layers.reduce(
        (prev, layer) =>
          layer.rules?.[ruleName]?.(state, ...args, prev) ?? prev,
        initialValue
      );
    } as any;
  }

  return { getValidActions, executeAction, rules };
}

function getInitialValue(ruleName: keyof GameRules): any {
  const defaults: Record<string, any> = {
    getTrickSize: null,
    shouldSkipPlayer: false,
    isTrickComplete: false,
    getSuitOfDomino: null,
    getTrumpForTrick: null,
    calculateTrickWinner: null,
    getNextPlayer: null,
    checkHandOutcome: null,
    compareDominoes: 0
  };

  return defaults[ruleName];
}
```

### GameHost Integration

```typescript
class GameHost {
  private engine: ComposedEngine;       // Composed once, never changes
  private mpState: MultiplayerGameState; // Pure data, always serializable

  constructor(gameId: string, config: GameConfig, sessions: PlayerSession[]) {
    // Map config variants to layer implementations
    const layers = [
      baseLayer,  // Always first
      ...config.variants.map(v => getLayerByType(v.type))
    ];

    // Compose engine once
    this.engine = composeEngine(layers);

    // Create initial state
    const initialState = createInitialState(config);

    // Store config in state for replay
    this.mpState = createMultiplayerGame({
      gameId,
      coreState: initialState,
      players: sessions,
      enabledVariants: config.variants
    });
  }

  getValidActions(playerId: string): GameAction[] {
    // Get all valid actions from composed engine
    const allActions = this.engine.getValidActions(this.mpState.coreState);

    // Filter by player's capabilities (multiplayer layer)
    const session = this.mpState.players.find(p => p.playerId === playerId);
    if (!session) return [];

    return filterActionsForSession(session, allActions);
  }

  executeAction(request: ActionRequest): Result<MultiplayerGameState> {
    // Use composed engine to execute
    const newCoreState = this.engine.executeAction(
      this.mpState.coreState,
      request.action
    );

    // Update multiplayer state
    this.mpState = {
      ...this.mpState,
      coreState: newCoreState,
      lastActionAt: Date.now()
    };

    return ok(this.mpState);
  }
}
```

### Layer Registry

```typescript
const LAYER_REGISTRY: Record<string, GameLayer> = {
  'nello': nelloLayer,
  'plunge': plungeLayer,
  'splash': splashLayer,
  'sevens': sevensLayer,
  'tournament': tournamentLayer,
  'speed': speedLayer,
  'hints': hintsLayer
};

function getLayerByType(type: string): GameLayer {
  const layer = LAYER_REGISTRY[type];
  if (!layer) {
    throw new Error(`Unknown layer type: ${type}`);
  }
  return layer;
}
```

---

## Variant Selection

### User Configuration

```typescript
// User creating a game chooses which contracts to enable
const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],

  variants: [
    { type: 'nello' },    // Enable nello
    { type: 'plunge' },   // Enable plunge
    { type: 'speed' },    // Enable speed mode
    { type: 'hints' }     // Enable hints
  ],

  shuffleSeed: 12345
};

// Tournament game: no special contracts
const tournamentConfig: GameConfig = {
  playerTypes: ['human', 'human', 'human', 'human'],
  variants: [{ type: 'tournament' }],  // Filters out special contracts
  shuffleSeed: 67890
};
```

### Variant Types

**Special Contracts** (modify game rules):
- `nello` - 3-player tricks, partner sits out, no trump
- `plunge` - Partner selects trump and leads, must win all
- `splash` - Same as plunge but 3+ doubles
- `sevens` - Distance from 7 wins tricks

**Game Modifiers** (modify actions/UI):
- `tournament` - Filter out special contracts
- `speed` - Auto-execute single options
- `hints` - Add hint metadata to actions

**Note**: Tournament mode is mutually exclusive with special contracts. If user selects tournament, don't allow nello/plunge/splash/sevens.

---

## Implementation Plan

### Phase 1: Update Types

**Files**: `src/game/types.ts`, `src/game/constants.ts`

1. Add new `BidType` values: `'plunge' | 'splash' | 'sevens'`
2. Add new `TrumpSelection.type` value: `'nello'`
3. Add new `GamePhase` values: `'bidding_complete' | 'trump_selection_complete'`
4. Add new `GameAction` types: `'advance-to-trump-selection' | 'advance-to-playing'`

### Phase 2: Update Base Executors (Minimal Changes)

**Files**: `src/game/core/actions.ts`

1. **executeBid**: Change final phase from `'trump_selection'` to `'bidding_complete'`
2. **executeTrumpSelection**: Change final phase from `'playing'` to `'trump_selection_complete'`
3. **executeAction**: Add handlers for `'advance-to-trump-selection'` and `'advance-to-playing'`

```typescript
// In executeBid
if (newBids.length === 4 && nonPassBids.length > 0) {
  return {
    ...state,
    phase: 'bidding_complete',  // ← Changed from 'trump_selection'
    // ...
  };
}

// In executeTrumpSelection
return {
  ...state,
  phase: 'trump_selection_complete',  // ← Changed from 'playing'
  // ...
};

// In executeAction
if (action.type === 'advance-to-trump-selection') {
  return { ...state, phase: 'trump_selection' };
}

if (action.type === 'advance-to-playing') {
  return { ...state, phase: 'playing' };
}
```

### Phase 3: Update getValidActions (Add Advance Actions)

**Files**: `src/game/core/gameEngine.ts`

```typescript
export function getValidActions(state: GameState): GameAction[] {
  switch (state.phase) {
    case 'bidding':
      return getBiddingActions(state);

    case 'bidding_complete':
      return [{ type: 'advance-to-trump-selection' }];  // NEW

    case 'trump_selection':
      return getTrumpSelectionActions(state);

    case 'trump_selection_complete':
      return [{ type: 'advance-to-playing' }];  // NEW

    case 'playing':
      return getPlayingActions(state);

    case 'scoring':
      return getScoringActions(state);

    default:
      return [];
  }
}
```

### Phase 4: Define Layer Interfaces

**New files**: `src/game/layers/types.ts`

```typescript
export interface GameRules {
  getTrickSize(state: GameState, prev: number): number;
  shouldSkipPlayer(state: GameState, playerId: number, prev: boolean): boolean;
  isTrickComplete(state: GameState, prev: boolean): boolean;
  getSuitOfDomino(state: GameState, domino: Domino, prev: number): number;
  getTrumpForTrick(state: GameState, prev: TrumpSelection): TrumpSelection;
  calculateTrickWinner(state: GameState, trick: Play[], prev: number): number;
  getNextPlayer(state: GameState, current: number, prev: number): number;
  shouldEndHandEarly(state: GameState, prev: boolean): boolean;
  compareDominoes(state: GameState, d1: Domino, d2: Domino, suit: number, prev: number): number;
}

export interface GameLayer {
  name: string;
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[];
  executeAction?: (state: GameState, action: GameAction, prev: GameState) => GameState;
  rules?: Partial<GameRules>;
}

export interface ComposedEngine {
  getValidActions(state: GameState): GameAction[];
  executeAction(state: GameState, action: GameAction): GameState;
  rules: GameRules;
}
```

### Phase 5: Refactor checkHandOutcome

**Files**: `src/game/core/handOutcome.ts`

1. Split monolithic `checkHandOutcome()` into:
   - `checkPointsBidOutcome()` - Points bid logic (30-41)
   - `checkMarksBidOutcome()` - Marks bid logic (42 points)
2. Move nello/plunge/splash logic to their respective layers
3. Base `checkHandOutcome` rule only handles standard bids
4. Special contract layers compose their early termination logic

### Phase 6: Implement Layers

**New files**: `src/game/layers/`

1. `base.ts` - Base layer (wraps existing executors)
2. `nello.ts` - Nello layer (includes `checkHandOutcome` override)
3. `plunge.ts` - Plunge layer (includes `checkHandOutcome` override)
4. `splash.ts` - Splash layer (includes `checkHandOutcome` override)
5. `sevens.ts` - Sevens layer (includes `checkHandOutcome` override)
6. `tournament.ts` - Tournament layer
7. `speed.ts` - Speed layer (move from variants/)
8. `hints.ts` - Hints layer (move from variants/)
9. `registry.ts` - Layer registry
10. `compose.ts` - Engine composition function

### Phase 7: Update GameHost

**Files**: `src/server/game/GameHost.ts`

```typescript
class GameHost {
  private engine: ComposedEngine;

  constructor(gameId: string, config: GameConfig, sessions: PlayerSession[]) {
    const layers = [
      baseLayer,
      ...config.variants.map(v => getLayerByType(v.type))
    ];

    this.engine = composeEngine(layers);

    // ... initialize mpState ...
  }

  // Use this.engine for all operations
}
```

### Phase 8: Update Tests

1. Add tests for composition points (bidding_complete, trump_selection_complete)
2. Add tests for each special contract layer
3. Add integration tests for layer composition
4. Update existing tests for new phases

---

## Validation: All Special Contracts Covered

### ✅ Nello (§8.1.1)
- Add nello trump option: `getValidActions`
- 3-player tricks: `rules.getTrickSize`
- Partner sits out: `rules.shouldSkipPlayer`
- Doubles own suit: `rules.getSuitOfDomino`
- Early termination: `rules.checkHandOutcome`
- Next player leads: `executeAction` override

### ✅ Plunge (§8.1.2)
- Add plunge bid: `getValidActions`
- Partner selects trump: `executeAction` on `advance-to-trump-selection`
- Partner leads: automatic (partner selected trump, so already currentPlayer)
- Early termination: `rules.checkHandOutcome`

### ✅ Splash (§8.1.3)
- Same as plunge, just different doubles requirement

### ✅ Sevens (§8.1.4)
- Add sevens bid: `getValidActions`
- Custom winner: `rules.calculateTrickWinner`
- Early termination: `rules.checkHandOutcome`

### ✅ Tournament Mode (§10)
- Filter special bids: `getValidActions`

### ✅ Regional Variations
- Forced bidding: `getValidActions` filter
- Doubles variations: `rules.getSuitOfDomino`, `rules.compareDominoes`

**All special contracts fully covered with zero refactoring of core executors.**

---

## Key Architectural Invariants

1. **Engine composed once** - At GameHost init, never changes
2. **State is pure data** - No function references, perfect serialization
3. **Layers check state at runtime** - `state.trump?.type === 'nello'` conditionals
4. **Composition points are explicit** - Intermediate phases clearly signal composition hooks
5. **Pure functional pipeline** - Reduce over layers, each returns new result
6. **Event sourcing works** - `replayActions(config, actions)` is trivial
7. **Time travel works** - `replayActions(config, actions.slice(0, N))` works perfectly

---

## Summary

This architecture achieves:

✅ **All special contracts** - nello, plunge, splash, sevens fully supported
✅ **Minimal changes** - Just 2 new phases, 2 new actions
✅ **No executor refactoring** - Base executors stay mostly unchanged
✅ **Pure functional** - Everything composes via reduce pipeline
✅ **Clean composition points** - Explicit hooks for layers to intervene
✅ **Perfect event sourcing** - State serializable, replay trivial
✅ **Extensible** - Easy to add new layers without touching core

**The path forward is validated. Ready to implement.** 🎯
