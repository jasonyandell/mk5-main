# Texas 42: Pure Functional RuleSet Composition via Threaded Rules

**⚠️ HISTORICAL DOCUMENT**: This document uses legacy terminology. "Layer" is now "RuleSet", "Variant" is now "ActionTransformer". Content preserved for reference.

**Created**: 2025-10-25
**Branch**: mk8
**Status**: Implementation plan validated and ready (terminology updated 2025)
**Philosophy**: Pure functions, parametric polymorphism, correct by construction

---

## Executive Summary

This document specifies the architecture for implementing Texas 42 special contracts (nello, plunge, splash, sevens) using **pure functional composition with threaded rules**.

**Core Innovation**: Instead of checking state conditionally throughout executors, we **parameterize executors with composable rules**. Executors become pure, polymorphic functions that delegate all variant-specific behavior to injected rules.

**Key Benefits**:
- ✅ **Pure functional** - no conditional logic in executors, all behavior parameterized
- ✅ **Correct by construction** - type system guarantees all rules provided
- ✅ **Parametric polymorphism** - executors know nothing about special contracts
- ✅ **Extensible** - add new contracts without touching core
- ✅ **Testable** - inject mock rules for testing

---

## Context and Motivation

### The Multiplayer Vision

The multiplayer architecture specification (`docs/remixed-855ccfd5.md`) established pure functional composition for **action generation**:

```typescript
// Variants transform action generation
type Variant = (StateMachine) → StateMachine
type StateMachine = (state: GameState) => GameAction[]

// Example: Tournament mode filters special contracts
tournamentMode: (base) => (state) =>
  base(state).filter(action => !isSpecialContract(action))
```

This works perfectly for:
- Filtering available actions (tournament mode)
- Annotating actions with metadata (hints, autoExecute)
- Adding new actions (special bids)

**But it has a fundamental limitation**: Variants can only transform what's **possible**, not what **happens when executed**.

### The Problem: Execution Semantics

Special contracts need to change execution behavior:

**Nello:**
- WHO leads: Next player (not bidder)
- WHEN trick completes: After 3 plays (not 4)
- HOW suits work: Doubles form own suit

**Plunge:**
- WHO selects trump: Partner (not bidder)
- WHO leads: Partner (not bidder)
- WHEN hand ends: First lost trick (early termination)

**Sevens:**
- HOW trick winner determined: Distance from 7 total pips (not trump/suit)
- HOW plays validated: No follow-suit requirement

**These cannot be expressed as action transformations.** They require changing the execution logic itself.

### Why Conditional Logic Becomes Intractable

The naive approach is to check state in executors:

```typescript
function executeTrumpSelection(state, player, selection) {
  let firstLeader = player;  // Default: bidder leads

  // Nello override
  if (selection.type === 'nello') {
    firstLeader = (state.winningBidder + 1) % 4;
  }

  // Plunge override
  if (state.winningBid?.bid === 'plunge') {
    // Wait, partner already selected trump, so...
    firstLeader = player;  // Or is it...?
  }

  // What if plunge + nello? 🤯
  // What if we add more contracts?

  return { ...state, phase: 'playing', currentPlayer: firstLeader };
}
```

**Problems:**
1. Couples executors to all special contracts
2. Conditional logic explodes with each new contract
3. Interactions between contracts unclear
4. Not extensible - new contract requires base changes
5. Can't disable contracts (no composition)

**This violates our core principles**: purity, separation of concerns, extensibility.

### The Solution: Threaded Rules

**Insight**: Executors should be **parametrically polymorphic** - they execute the same way regardless of variant, but delegate variant-specific decisions to **injected rules**.

```typescript
// Executor is parameterized by rules
function executeTrumpSelection(
  state: GameState,
  player: number,
  selection: TrumpSelection,
  rules: GameRules  // ← Injected dependency
): GameState {
  // Delegate "who leads" decision to rules
  const firstLeader = rules.getFirstLeader(state, player, selection);

  // Executor has NO conditional logic
  return {
    ...state,
    phase: 'playing',
    currentPlayer: firstLeader
  };
}
```

**Rules compose via layers:**

```typescript
// Base rule: bidder leads
const baseRules = {
  getFirstLeader: (state, player, selection) => player
};

// Nello layer: next player leads when nello selected
const nelloLayer = {
  rules: {
    getFirstLeader: (state, player, selection, prev) =>
      selection.type === 'nello'
        ? (state.winningBidder + 1) % 4
        : prev
  }
};

// Compose: base → nello → final
const composedRules = composeRules([baseRules, nelloLayer]);

// Execute with composed rules
executeAction(state, action, composedRules);
```

**This is pure, composable, and correct by construction.**

---

## Architectural Principles

### 1. Pure Functions Everywhere

**Definition**: Functions with no side effects that always return the same output for the same input.

**Application**:
- Executors: `(state, action, rules) => newState` (pure)
- Rules: `(state, ...params) => result` (pure)
- Composition: `reduce` over layers (pure)

**No mutation**, no global state, no hidden dependencies.

### 2. Parametric Polymorphism

**Definition**: Functions that work uniformly across all types/variants via abstraction.

**Application**:
- Executors don't inspect state to determine behavior
- They call `rules.method()` and trust the result
- Same executor code works for standard 42, nello, plunge, sevens
- **Variant behavior is injected, not hardcoded**

### 3. Separation of Concerns

**Three independent layers**:

| Layer | Responsibility | Example |
|-------|----------------|---------|
| **Core Executors** | Apply actions to state | `executeTrumpSelection` transitions to playing phase |
| **Rules** | Define variant-specific algorithms | `nelloRules.getFirstLeader` returns next player |
| **Variants** | Transform action generation | `tournamentMode` filters special contract bids |

**No cross-cutting concerns.** Each layer has a single, well-defined job.

### 4. Correct by Construction

**Type system guarantees**:
- Executors **must** receive `rules` parameter (compiler enforces)
- Rules **must** implement all 7 methods (interface contract)
- Composition produces valid `GameRules` (type-safe reduce)

**Impossible to**:
- Call executor without rules
- Forget to implement a rule
- Compose incompatible layers

**Runtime correctness via compile-time guarantees.**

### 5. Composition Over Configuration

**Not configuration**:
```typescript
// ❌ Feature flags
if (config.enableNello) { ... }
```

**Composition**:
```typescript
// ✅ Function composition
const rules = compose([baseRules, nelloRules, plungeRules])
```

**Benefits**: Layers are first-class values, can be tested independently, compose in any order.

---

## Core Architecture

### GameRules Interface (7 Methods)

Rules are grouped by concern into three categories:

#### WHO Rules: Determine which player acts

```typescript
interface GameRules {
  /**
   * Who selects trump after bidding completes?
   *
   * Base: Winning bidder
   * Plunge: Partner of winning bidder
   */
  getTrumpSelector(state: GameState, winningBid: Bid): number;

  /**
   * Who leads the first trick after trump selected?
   *
   * Base: Trump selector (bidder)
   * Nello: Next player after bidder
   * Plunge: Partner (who selected trump)
   */
  getFirstLeader(
    state: GameState,
    trumpSelector: number,
    trump: TrumpSelection
  ): number;

  /**
   * Who plays next after current player?
   *
   * Base: (current + 1) % 4
   * Nello: Skip partner, so (current + 1 or 2) % 4
   */
  getNextPlayer(state: GameState, currentPlayer: number): number;
}
```

#### WHEN Rules: Determine timing and completion

```typescript
interface GameRules {
  /**
   * Is the current trick complete?
   *
   * Base: 4 plays
   * Nello: 3 plays (partner sits out)
   */
  isTrickComplete(state: GameState): boolean;

  /**
   * Should the hand end early (before all 7 tricks)?
   *
   * Base: null (play all tricks)
   * Nello: Bidder wins any trick = hand over
   * Plunge/Splash: Opponents win any trick = hand over
   * Sevens: Opponents win any trick = hand over
   *
   * Returns null to continue, or HandOutcome if determined
   */
  checkHandOutcome(state: GameState): HandOutcome | null;
}
```

#### HOW Rules: Determine game mechanics

```typescript
interface GameRules {
  /**
   * What suit does a domino lead when played?
   *
   * Base: Higher pip (or 7 if doubles-trump)
   * Nello (doubles-as-own-suit variant): Doubles = 7, else higher pip
   */
  getLedSuit(state: GameState, domino: Domino): LedSuit;

  /**
   * Who won this trick?
   *
   * Base: Trump > suit, higher value wins
   * Sevens: Closest to 7 total pips wins (no trump/suit)
   */
  calculateTrickWinner(state: GameState, trick: Play[]): number;
}
```

**Total: 7 rules** - all algorithmically necessary for special contracts.

### GameLayer Interface (2 Surfaces)

Layers compose two orthogonal concerns:

```typescript
interface GameLayer {
  name: string;

  /**
   * Transform action generation
   *
   * Pattern: Filter, annotate, or add actions
   *
   * Example: Tournament layer filters special contract bids
   */
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[];

  /**
   * Override specific rules
   *
   * Pattern: Check state, return override or prev
   *
   * Example: Nello layer returns 3 for isTrickComplete
   */
  rules?: Partial<GameRules>;
}
```

**Two composition surfaces**:
1. **Action generation** (existing) - what's possible
2. **Rule algorithms** (new) - how things execute

**No third surface** - execution composition via rules, not separate `executeAction` hook.

### Rule Composition Pattern

**Monadic composition**: Each rule is a reducer that takes previous result and returns new result.

```typescript
function composeRules(layers: GameLayer[]): GameRules {
  // Use reduce with identity function as base
  const identity = <T>(x: T) => x;

  return {
    getTrumpSelector: (state, bid) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getTrumpSelector?.(state, bid, prev) ?? prev,
        bid.player  // Base identity: bidder selects trump
      ),

    isTrickComplete: (state) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.isTrickComplete?.(state, prev) ?? prev,
        state.currentTrick.length === 4  // Base identity: 4 plays
      ),

    getNextPlayer: (state, current) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getNextPlayer?.(state, current, prev) ?? prev,
        getNextPlayer(current)  // Base identity: use pure helper
      ),

    getFirstLeader: (state, selector, trump) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getFirstLeader?.(state, selector, trump, prev) ?? prev,
        selector  // Base identity: selector leads
      ),

    checkHandOutcome: (state) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.checkHandOutcome?.(state, prev) ?? prev,
        null  // Base identity: no early termination
      ),

    getLedSuit: (state, domino) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getLedSuit?.(state, domino, prev) ?? prev,
        getLedSuitBase(state, domino)  // Base identity: use helper
      ),

    calculateTrickWinner: (state, trick) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.calculateTrickWinner?.(state, trick, prev) ?? prev,
        calculateTrickWinnerBase(state, trick)  // Base identity: use helper
      )
  };
}
```

**Each layer rule gets**:
- Current state
- Method parameters
- **Previous layer's result** (can override or pass through via identity)

**Composition semantics**: `reduce` left-to-right, last layer wins.

**Monoid property**: Identity element (base rule) + associative operation (override or pass through).

### Executor Threading Pattern

```typescript
// Before: Hardcoded behavior
function executeCompleteTrick(state: GameState): GameState {
  if (state.currentTrick.length !== 4) {  // ❌ Hardcoded to 4
    throw new Error('Trick not complete');
  }

  const winner = calculateTrickWinner(  // ❌ Global function, not composable
    state.currentTrick,
    state.trump,
    state.currentSuit
  );

  // ...
}

// After: Parameterized by rules
function executeCompleteTrick(
  state: GameState,
  rules: GameRules  // ✅ Injected
): GameState {
  if (!rules.isTrickComplete(state)) {  // ✅ Call composed rule
    throw new Error('Trick not complete');
  }

  const winner = rules.calculateTrickWinner(state, state.currentTrick);  // ✅ Composed

  // Check for early termination
  const outcome = rules.checkHandOutcome(updatedState);  // ✅ Composed
  if (outcome?.isDetermined) {
    newPhase = 'scoring';
  }

  // ...
}
```

**Pattern**: Executors become pure functions that delegate all decisions to rules.

---

## Shared Helper Functions

Pure utility functions used by layers to avoid duplication and modulus arithmetic:

```typescript
/**
 * Get partner of a player (opposite seat)
 * Uses existing player relationships, not modulus
 */
function getPartner(playerIndex: number): number {
  // Assuming state.players structure maintains partnerships
  // Partner is always +2 seats, but use helper to avoid modulus
  return getPlayerAtOffset(playerIndex, 2);
}

/**
 * Get player at offset using existing getNextPlayer logic
 */
function getPlayerAtOffset(from: number, offset: number): number {
  let current = from;
  for (let i = 0; i < offset; i++) {
    current = getNextPlayer(current);
  }
  return current;
}

/**
 * Get player's team (0 or 1)
 */
function getPlayerTeam(state: GameState, playerIndex: number): number {
  return state.players[playerIndex].teamId;
}

/**
 * Check if bidding team must win all tricks (or must NOT win any)
 * Used by: plunge, splash, sevens, nello
 *
 * @param mustWin - true for "must win all", false for "must win none"
 * @returns HandOutcome if determined, null otherwise
 */
function checkMustWinAllTricks(
  state: GameState,
  biddingTeam: number,
  mustWin: boolean
): HandOutcome | null {
  // Check each completed trick
  for (let i = 0; i < state.tricks.length; i++) {
    const trick = state.tricks[i];
    if (trick.winner === undefined) continue;

    const winnerTeam = getPlayerTeam(state, trick.winner);
    const biddingTeamWon = winnerTeam === biddingTeam;

    // For "must win all": fail if opponents won
    // For "must win none": fail if bidding team won
    if (mustWin && !biddingTeamWon) {
      return {
        isDetermined: true,
        reason: `Defending team won trick ${i + 1}`,
        decidedAtTrick: i + 1
      };
    }

    if (!mustWin && biddingTeamWon) {
      return {
        isDetermined: true,
        reason: `Bidding team won trick ${i + 1}`,
        decidedAtTrick: i + 1
      };
    }
  }

  return null; // Not determined yet
}

/**
 * Get highest marks bid value from bids so far
 * Used by: plunge, splash to determine automatic bid value
 */
function getHighestMarksBid(bids: Bid[]): number {
  return bids
    .filter(b => b.type === 'marks' || b.type === 'plunge' || b.type === 'splash')
    .reduce((max, bid) => Math.max(max, bid.value || 0), 0);
}
```

---

## Layer Implementations

### Base Layer (Standard Texas 42)

```typescript
const baseLayer: GameLayer = {
  name: 'base',

  rules: {
    // WHO rules
    getTrumpSelector: (state, bid) => bid.player,

    getFirstLeader: (state, selector, trump) => selector,

    getNextPlayer: (state, current) => (current + 1) % 4,

    // WHEN rules
    isTrickComplete: (state) => state.currentTrick.length === 4,

    checkHandOutcome: (state) => {
      // Only end after all 7 tricks in standard play
      if (state.tricks.length < 7) return null;

      return {
        isDetermined: true,
        reason: 'All tricks played'
      };
    },

    // HOW rules
    getLedSuit: (state, domino) => {
      // Standard: higher pip leads (or 7 if doubles-trump)
      const trumpSuit = getTrumpSuit(state.trump);

      if (trumpSuit === DOUBLES_AS_TRUMP) {
        return domino.high === domino.low ? 7 : domino.high;
      }

      // Trump dominoes lead trump suit
      if (trumpSuit >= 0 && trumpSuit <= 6) {
        if (domino.high === trumpSuit || domino.low === trumpSuit) {
          return trumpSuit;
        }
      }

      return domino.high;
    },

    calculateTrickWinner: (state, trick) => {
      // Standard trick-taking logic:
      // 1. Trump beats non-trump
      // 2. Higher trump wins
      // 3. Following suit beats non-following
      // 4. Higher value wins

      const leadPlay = trick[0];
      const trumpSuit = getTrumpSuit(state.trump);
      const leadSuit = state.currentSuit;

      let winningPlay = leadPlay;
      let winningValue = getDominoValue(leadPlay.domino, state.trump);
      let winningIsTrump = isDominoTrump(leadPlay.domino, trumpSuit);

      for (let i = 1; i < trick.length; i++) {
        const play = trick[i];
        const playValue = getDominoValue(play.domino, state.trump);
        const playIsTrump = isDominoTrump(play.domino, trumpSuit);

        if (playIsTrump && !winningIsTrump) {
          winningPlay = play;
          winningValue = playValue;
          winningIsTrump = true;
        } else if (playIsTrump && winningIsTrump && playValue > winningValue) {
          winningPlay = play;
          winningValue = playValue;
        } else if (!playIsTrump && !winningIsTrump &&
                   dominoFollowsSuit(play.domino, leadSuit) &&
                   playValue > winningValue) {
          winningPlay = play;
          winningValue = playValue;
        }
      }

      return winningPlay.player;
    }
  }
};
```

**Base layer knows nothing about special contracts.** Pure standard 42 logic.

### Nello Layer

From `docs/rules.md` §8.A:
- Must bid at least 1 mark
- Partner sits out with dominoes face-down (3-player tricks)
- No trump suit declared
- Doubles form own suit (standard variant)
- Objective: Bidder must lose every trick
- **Bidder leads normally** (not next player)

```typescript
const nelloLayer: GameLayer = {
  name: 'nello',

  getValidActions: (state, prev) => {
    // Add nello as trump option when marks bid won
    if (state.phase === 'trump_selection' &&
        state.winningBid?.bid === 'marks') {
      return [...prev, {
        type: 'select-trump',
        player: state.winningBidder,
        trump: { type: 'nello' }
      }];
    }

    return prev;
  },

  rules: {
    // Bidder leads normally in nello (no override needed)
    // getFirstLeader passes through prev

    // Skip partner in turn order
    getNextPlayer: (state, current, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      const partner = getPartner(state.winningBidder);
      let next = getNextPlayer(current);

      if (next === partner) {
        next = getNextPlayer(next);
      }

      return next;
    },

    // 3-player tricks (partner sits out)
    isTrickComplete: (state, prev) =>
      state.trump?.type === 'nello'
        ? state.currentTrick.length === 3
        : prev,

    // Hand ends if bidder wins any trick
    checkHandOutcome: (state, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      if (prev?.isDetermined) return prev; // Already ended

      // Use shared helper: bidding team must not win any tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, false);

      return outcome || prev;
    },

    // Doubles form own suit (suit 7)
    getLedSuit: (state, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      return domino.high === domino.low ? 7 : domino.high;
    },

    // Standard trick-taking but with no trump (already handled by getLedSuit)
    // calculateTrickWinner uses prev (base implementation works)
  }
};
```

**Key insight**: Nello only overrides 5 of 7 rules. Others pass through to base.

### Plunge Layer

From `docs/rules.md` §8.A:
- Requires 4+ doubles in hand
- Bid value: Automatic based on current high bid (4+ marks, jumps over existing bids)
- Partner declares trump and leads
- Must win all 7 tricks

```typescript
const plungeLayer: GameLayer = {
  name: 'plunge',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    // Add plunge bid if player has 4+ doubles
    const player = state.players[state.currentPlayer];
    const doubles = countDoubles(player.hand);

    if (doubles >= 4) {
      // Plunge value = highest marks bid + 1, minimum 4
      const highestMarksBid = getHighestMarksBid(state.bids);
      const plungeValue = Math.max(4, highestMarksBid + 1);

      return [...prev, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'plunge' as const,
        value: plungeValue  // Automatic, not user choice
      }];
    }

    return prev;
  },

  rules: {
    // Partner selects trump (not bidder)
    getTrumpSelector: (state, bid, prev) =>
      bid.bid === 'plunge'
        ? getPartner(bid.player)
        : prev,

    // Partner leads (they selected trump, so already currentPlayer)
    // getFirstLeader passes through (prev already correct)

    // Hand ends if opponents win any trick
    checkHandOutcome: (state, prev) => {
      if (state.winningBid?.bid !== 'plunge') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
```

**Minimal override**: Only 2 rules change. Everything else standard.

### Splash Layer

From `docs/rules.md` §8.A:
- Requires 3+ doubles in hand
- Bid value: Automatic based on current high bid (2-3 marks, jumps over existing bids)
- Partner declares trump and leads
- Must win all 7 tricks

```typescript
const splashLayer: GameLayer = {
  name: 'splash',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    const player = state.players[state.currentPlayer];
    const doubles = countDoubles(player.hand);

    if (doubles >= 3) {
      // Splash value = highest marks bid + 1, minimum 2, maximum 3
      const highestMarksBid = getHighestMarksBid(state.bids);
      const splashValue = Math.min(3, Math.max(2, highestMarksBid + 1));

      return [...prev, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'splash' as const,
        value: splashValue  // Automatic, not user choice
      }];
    }

    return prev;
  },

  rules: {
    // Same as plunge: partner selects trump
    getTrumpSelector: (state, bid, prev) =>
      bid.bid === 'splash'
        ? getPartner(bid.player)
        : prev,

    // Same as plunge: early termination on lost trick
    checkHandOutcome: (state, prev) => {
      if (state.winningBid?.bid !== 'splash') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
```

**Nearly identical to plunge** - just different doubles requirement.

### Sevens Layer

From `docs/rules.md` §8.A:
- Must bid marks (not sevens bid type)
- Select "sevens" as trump type (like nello)
- Domino closest to 7 total pips wins trick
- Ties won by first played
- No follow-suit requirement
- Must win all tricks
- **Bidder leads normally**

**Critical insight**: Sevens is a trump selection option (like nello), not a bid type.

```typescript
const sevensLayer: GameLayer = {
  name: 'sevens',

  getValidActions: (state, prev) => {
    // Add sevens as trump option when marks bid won
    if (state.phase === 'trump_selection' &&
        state.winningBid?.bid === 'marks') {
      return [...prev, {
        type: 'select-trump',
        player: state.winningBidder,
        trump: { type: 'sevens' }
      }];
    }

    return prev;
  },

  rules: {
    // Bidder leads normally in sevens (no override needed)
    // getFirstLeader passes through prev

    // Completely different trick winner algorithm
    calculateTrickWinner: (state, trick, prev) => {
      if (state.trump?.type !== 'sevens') return prev;

      // Distance from 7 total pips
      const distances = trick.map(play =>
        Math.abs(7 - (play.domino.high + play.domino.low))
      );

      const minDistance = Math.min(...distances);

      // First domino with minimum distance wins (tie = first)
      return trick.findIndex(play =>
        Math.abs(7 - (play.domino.high + play.domino.low)) === minDistance
      );
    },

    // Early termination on lost trick
    checkHandOutcome: (state, prev) => {
      if (state.trump?.type !== 'sevens') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
```

**Note**: Sevens has no follow-suit requirement. This is already handled in `getValidPlays` (existing code) - it doesn't check suit when determining legal plays. The rule composition only needs to handle trick winner determination.

### Tournament Layer

From `docs/rules.md` §10:
- No special contracts (nello, plunge, splash, sevens)
- Strict communication prohibition
- Standardized equipment
- Time limits

```typescript
const tournamentLayer: GameLayer = {
  name: 'tournament',

  getValidActions: (state, prev) => {
    // Filter out all special contract bids and trump selections

    if (state.phase === 'bidding') {
      return prev.filter(action =>
        action.type !== 'bid' ||
        !['nello', 'plunge', 'splash', 'sevens'].includes(action.bid)
      );
    }

    if (state.phase === 'trump_selection') {
      return prev.filter(action =>
        action.type !== 'select-trump' ||
        action.trump.type !== 'nello'
      );
    }

    return prev;
  }

  // No rule overrides - tournament uses standard rules
};
```

**Pure action filtering** - no execution changes needed.

---

## Integration with Existing Systems

### Compatibility with Multiplayer Architecture

The threaded rules architecture **extends** (not replaces) the multiplayer variant system:

**Current multiplayer system** (`docs/remixed-855ccfd5.md`):
- Variants transform action generation: `Variant = (StateMachine) → StateMachine`
- Used for: tournament mode, speed mode, hints, oneHand

**Threaded rules adds**:
- Layers provide rule implementations: `GameLayer.rules: Partial<GameRules>`
- Used for: special contracts that change execution

**They compose independently**:

```typescript
// Variant transforms actions (existing)
const tournamentVariant: Variant = (base) => (state) =>
  base(state).filter(a => !isSpecialContract(a));

// Layer provides rules (new)
const nelloLayer: GameLayer = {
  getValidActions: (state, prev) => [...prev, nelloTrumpOption],
  rules: { isTrickComplete: (state) => state.currentTrick.length === 3 }
};

// Compose both
const gameHost = new GameHost({
  variants: [tournamentVariant, speedVariant],  // Action transforms
  layers: [baseLayer, nelloLayer, plungeLayer]  // Rule implementations
});
```

**Single composition point** in GameHost:
1. Compose variants → modified `getValidActions` function
2. Compose layers → `GameRules` implementation
3. Thread rules through executors

### AutoExecute Stays Unchanged

AutoExecute is **orthogonal** to layers architecture:

**What it is**: Metadata annotation on actions that hints to client to auto-execute.

**Used by**: oneHand variant to script bidding/trump selection.

**Example**:
```typescript
const oneHandVariant: Variant = (base) => (state) => {
  if (state.phase === 'bidding') {
    // Force player 0 to bid 30
    return [{
      type: 'bid',
      player: 0,
      bid: 'points',
      value: 30,
      autoExecute: true,  // ✅ Client auto-executes
      meta: { delay: 500, reason: 'one-hand-mode' }
    }];
  }

  return base(state);
};
```

**No changes needed** - autoExecute is pure metadata, layers don't care about it.

### GameHost Integration

```typescript
class GameHost {
  private rules: GameRules;
  private getValidActionsFn: (state: GameState) => GameAction[];

  constructor(config: GameConfig, sessions: PlayerSession[]) {
    // 1. Compose layers into rules
    const layers = [
      baseLayer,
      ...config.enabledLayers.map(name => LAYER_REGISTRY[name])
    ];

    this.rules = composeRules(layers);

    // 2. Compose variants into action generator
    this.getValidActionsFn = applyVariants(
      getValidActions,  // Base action generator
      config.enabledVariants
    );

    // 3. Initialize state
    this.mpState = createMultiplayerGame({
      gameId: config.gameId,
      coreState: createInitialState(config),
      players: sessions,
      enabledVariants: config.enabledVariants,
      enabledLayers: config.enabledLayers  // Store for replay
    });
  }

  getValidActions(playerId: string): GameAction[] {
    // Get all valid actions (variants applied)
    const allActions = this.getValidActionsFn(this.mpState.coreState);

    // Filter by player capabilities (multiplayer layer)
    const session = this.mpState.players.find(p => p.playerId === playerId);
    if (!session) return [];

    return filterActionsForSession(session, allActions);
  }

  executeAction(request: ActionRequest): Result<MultiplayerGameState> {
    // Use composed rules
    const newCoreState = executeAction(
      this.mpState.coreState,
      request.action,
      this.rules  // ✅ Thread rules
    );

    this.mpState = {
      ...this.mpState,
      coreState: newCoreState,
      lastActionAt: Date.now()
    };

    return ok(this.mpState);
  }
}
```

**Key integration points**:
1. Compose rules from layers
2. Compose action generator from variants
3. Thread rules through `executeAction`

---

## Implementation Plan

### Phase 1: Define Type Interfaces

**Files**: `src/game/layers/types.ts`

```typescript
// GameRules interface (7 methods)
export interface GameRules {
  getTrumpSelector(state: GameState, winningBid: Bid): number;
  getFirstLeader(state: GameState, trumpSelector: number, trump: TrumpSelection): number;
  getNextPlayer(state: GameState, currentPlayer: number): number;
  isTrickComplete(state: GameState): boolean;
  checkHandOutcome(state: GameState): HandOutcome | null;
  getLedSuit(state: GameState, domino: Domino): LedSuit;
  calculateTrickWinner(state: GameState, trick: Play[]): number;
}

// GameLayer interface (2 surfaces)
export interface GameLayer {
  name: string;
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[];
  rules?: Partial<GameRules>;
}

// HandOutcome type
export interface HandOutcome {
  isDetermined: boolean;
  reason: string;
  decidedAtTrick?: number;
}
```

**Validation**: Type-check compiles, exports work.

### Phase 2: Implement Base Layer

**Files**: `src/game/layers/base.ts`

1. Extract current logic from executors into base rules
2. Implement all 7 rules with standard 42 behavior
3. Add JSDoc comments for each rule explaining standard behavior
4. Unit test each rule independently

**Example**: Extract `calculateTrickWinner` from `scoring.ts` into `baseLayer.rules.calculateTrickWinner`.

**Validation**: All existing tests still pass (base layer = current behavior).

### Phase 3: Refactor Executors to Thread Rules

**Files**: `src/game/core/actions.ts`

**Executors to refactor**:

1. **`executeAction`** - Add `rules` parameter, thread to all sub-executors
2. **`executeBid`** - Call `rules.getTrumpSelector()` when bidding completes
3. **`executeTrumpSelection`** - Call `rules.getFirstLeader()` when trump selected
4. **`executePlay`** - Call `rules.getLedSuit()` and `rules.getNextPlayer()`
5. **`executeCompleteTrick`** - Call `rules.isTrickComplete()`, `rules.calculateTrickWinner()`, `rules.checkHandOutcome()`

**Pattern for each**:
```typescript
// Before
function executeFoo(state: GameState, ...): GameState {
  const x = hardcodedLogic();
  // ...
}

// After
function executeFoo(state: GameState, ..., rules: GameRules): GameState {
  const x = rules.method(state, ...);
  // ...
}
```

**Validation**:
- Inject base rules, all tests pass (same behavior)
- Type errors if rules not provided (correct by construction)

### Phase 4: Implement Rule Composition

**Files**: `src/game/layers/compose.ts`

```typescript
export function composeRules(layers: GameLayer[]): GameRules {
  // For each of 7 rules, create composed version
  return {
    getTrumpSelector: composeRule(
      layers,
      'getTrumpSelector',
      (bid) => bid.player  // Default
    ),

    isTrickComplete: composeRule(
      layers,
      'isTrickComplete',
      (state) => state.currentTrick.length === 4  // Default
    ),

    // ... 5 more
  };
}

function composeRule<T>(
  layers: GameLayer[],
  ruleName: keyof GameRules,
  baseImpl: (...args: any[]) => T
): (...args: any[]) => T {
  return (...args) => {
    let result = baseImpl(...args);

    for (const layer of layers) {
      const rule = layer.rules?.[ruleName];
      if (rule) {
        result = rule(...args, result);
      }
    }

    return result;
  };
}
```

**Validation**:
- Compose `[baseLayer]`, all tests pass
- Compose `[baseLayer, mockLayer]`, mock overrides work

### Phase 5: Implement Special Contract Layers

**Files**:
- `src/game/layers/nello.ts`
- `src/game/layers/plunge.ts`
- `src/game/layers/splash.ts`
- `src/game/layers/sevens.ts`

For each layer:
1. Implement `getValidActions` to add special bid/trump options
2. Implement rule overrides (see layer implementations above)
3. Unit test each rule independently
4. Integration test full hand with layer enabled

**Validation criteria**:
- Nello: 3-player tricks, bidder loses all tricks, partner skipped
- Plunge: Partner selects trump/leads, opponents win = hand ends
- Splash: Same as plunge with 3+ doubles
- Sevens: Distance from 7 wins, no follow-suit

### Phase 6: Implement Layer Registry

**Files**: `src/game/layers/registry.ts`, `src/game/layers/index.ts`

```typescript
import { baseLayer } from './base';
import { nelloLayer } from './nello';
import { plungeLayer } from './plunge';
import { splashLayer } from './splash';
import { sevensLayer } from './sevens';

export const LAYER_REGISTRY: Record<string, GameLayer> = {
  'nello': nelloLayer,
  'plunge': plungeLayer,
  'splash': splashLayer,
  'sevens': sevensLayer
};

export function getLayerByName(name: string): GameLayer {
  const layer = LAYER_REGISTRY[name];
  if (!layer) {
    throw new Error(`Unknown layer: ${name}`);
  }
  return layer;
}

// Export everything
export * from './types';
export * from './base';
export * from './compose';
export { baseLayer, nelloLayer, plungeLayer, splashLayer, sevensLayer };
```

**Validation**: Can import and use all layers.

### Phase 7: GameHost Integration

**Files**: `src/server/game/GameHost.ts`

1. Add `rules: GameRules` property
2. Compose rules in constructor from config
3. Thread rules through `executeAction` calls
4. Store enabled layers in `MultiplayerGameState` for replay

```typescript
constructor(config: GameConfig, sessions: PlayerSession[]) {
  // Compose layers
  const layerImpls = config.enabledLayers.map(name => getLayerByName(name));
  this.rules = composeRules([baseLayer, ...layerImpls]);

  // ... existing variant composition ...
}

executeAction(request: ActionRequest) {
  const newState = executeAction(
    this.mpState.coreState,
    request.action,
    this.rules  // ✅ Thread rules
  );

  // ...
}
```

**Validation**: Create game with layers, execute actions, rules apply correctly.

### Phase 8: Configuration Updates

**Files**: `src/game/types/config.ts`

Add layer configuration:

```typescript
export interface GameConfig {
  playerTypes: PlayerType[];
  shuffleSeed: number;
  enabledVariants: VariantConfig[];
  enabledLayers: string[];  // ✅ NEW: e.g., ['nello', 'plunge']
}
```

**Validation**: Config validates, serializes correctly.

### Phase 9: Testing

**Unit tests** (per layer):
- Base layer: All 7 rules with standard behavior
- Nello layer: Each rule override independently
- Plunge layer: Each rule override independently
- Splash layer: Each rule override independently
- Sevens layer: Each rule override independently
- Composition: Multiple layers compose correctly

**Integration tests** (full hands):
- Standard game (base layer only)
- Nello hand (bidder loses all tricks)
- Nello hand (bidder wins trick early, hand ends)
- Plunge hand (opponents win trick early, hand ends)
- Sevens hand (distance from 7 determines winner)
- Mixed layers (nello + plunge enabled but plunge bid wins)

**Consensus tests** (nello with 3 active players):
- Trick completion: 3 players agree (partner excluded from consensus)
- Hand scoring: 3 players agree (partner excluded from consensus)
- Validation: Consensus size matches active player count

**Replay tests**:
- Save action history with layer config
- Replay from scratch, same result
- Validates determinism

**Edge cases**:
- Nello partner turn (should skip to next player)
- Nello consensus (only 3 players, not 4)
- Sevens tie (first domino wins)
- Multiple layers enabled (correct composition order)

---

## Consensus Handling for Special Contracts

**Current implementation**: Consensus actions (`agree-complete-trick`, `agree-score-hand`) require all 4 players to agree before executing system actions (`complete-trick`, `score-hand`).

**Nello challenge**: Only 3 players are active (partner sits out).

**Solution**: Keep consensus at 4 players for now, but partner auto-agrees.

### Partner Auto-Agrees (Simplest)

```typescript
// In nello layer's getValidActions
getValidActions: (state, prev) => {
  if (state.trump?.type === 'nello' && state.phase === 'playing') {
    const partner = getPartner(state.winningBidder);

    // If partner hasn't agreed yet and trick/hand is waiting, auto-add agreement
    if (state.currentTrick.length === 3 &&
        !state.consensus.completeTrick.has(partner)) {
      return [...prev, {
        type: 'agree-complete-trick',
        player: partner,
        autoExecute: true  // Client auto-executes
      }];
    }

    if (state.phase === 'scoring' &&
        !state.consensus.scoreHand.has(partner)) {
      return [...prev, {
        type: 'agree-score-hand',
        player: partner,
        autoExecute: true
      }];
    }
  }

  return prev;
}
```

**Benefits**:
- No changes to consensus counting (still waits for 4)
- Partner's agreement is explicit in action history (good for replay)
- Uses existing autoExecute mechanism
- Simple to implement

**This is the recommended approach** - keeps consensus logic unchanged, uses composition to add partner auto-agreement.

---

## Validation Against Requirements

### All Special Contracts Covered

| Contract | Requirements | Rules Used |
|----------|-------------|------------|
| **Nello** | Partner sits out (3 tricks), next player leads, doubles as suit 7, lose all tricks | `isTrickComplete`, `getNextPlayer`, `getFirstLeader`, `getLedSuit`, `checkHandOutcome` |
| **Plunge** | Partner selects trump/leads, early termination | `getTrumpSelector`, `checkHandOutcome` |
| **Splash** | Same as plunge with 3+ doubles | `getTrumpSelector`, `checkHandOutcome` |
| **Sevens** | Distance from 7 wins, no follow-suit | `calculateTrickWinner`, `checkHandOutcome` |
| **Tournament** | Disable special contracts | `getValidActions` filter only |

✅ **All covered with 7 composable rules.**

### No Conditional Logic in Executors

```typescript
// ❌ BEFORE: Conditional logic
if (state.trump?.type === 'nello') {
  // special case
} else if (state.winningBid?.bid === 'plunge') {
  // another special case
}

// ✅ AFTER: Parameterized
const firstLeader = rules.getFirstLeader(state, selector, trump);
```

✅ **Executors are pure, polymorphic functions.**

### Type-Safe Composition

```typescript
interface GameRules {
  // All 7 methods required
  getTrumpSelector(...): number;
  // ...
}

// Compiler enforces
executeAction(state, action, rules);  // ✅ Must provide rules
executeAction(state, action);         // ❌ Compile error
```

✅ **Impossible to forget rules, type system guarantees.**

### Extensibility

Adding a new special contract:

1. Create new layer file
2. Implement `getValidActions` (add bid option)
3. Implement rule overrides (just the ones that differ)
4. Register in `LAYER_REGISTRY`
5. **No changes to core executors**

✅ **Open for extension, closed for modification.**

---

## References and Onboarding

### Essential Reading

**1. Game Rules** (`docs/rules.md`)
- Section 8.A: Special Contracts (nello, plunge, splash, sevens)
- Section 10: Tournament Standards (no special contracts)
- Complete formal specification of Texas 42

**2. Multiplayer Architecture** (`docs/remixed-855ccfd5.md`)
- Pure functional composition for action generation
- Variant system: `Variant = (StateMachine) → StateMachine`
- Capability system for authorization
- Server-client architecture

**3. Current Codebase**

Key files:
- `src/game/core/actions.ts` - Executors to be refactored
- `src/game/core/scoring.ts` - Current `calculateTrickWinner` logic
- `src/game/core/gameEngine.ts` - `getValidActions` structure
- `src/game/variants/` - Existing variant system (action transforms only)
- `src/server/game/GameHost.ts` - Integration point

### Key Concepts

**Pure Functions**: No side effects, deterministic, same input → same output.

**Parametric Polymorphism**: Functions work uniformly across types via abstraction (executors don't know about variants).

**Composition**: Building complex behavior from simple, composable pieces via function composition.

**Separation of Concerns**: Each layer has single responsibility (executors = state transitions, rules = algorithms, variants = action filtering).

**Correct by Construction**: Type system guarantees certain classes of bugs impossible (can't call executor without rules).

### Why This Matters

**The problem we're solving**: Special contracts require changing execution semantics, which action-only variants cannot express.

**The insight**: Thread composable rules through executors to make them parametrically polymorphic.

**The result**: Pure, extensible architecture where special contracts are independent layers that compose via reduce.

**The guarantee**: Type system prevents forgetting rules, executors have no conditional logic, new contracts don't touch core.

---

## Philosophical Foundation

### Purity as a Design Constraint

**Pure functions are predictable**: Given same inputs, always produce same outputs. No hidden dependencies, no surprises.

**Benefits**:
- Easier to test (no mocking, just call with inputs)
- Easier to reason about (no spooky action at a distance)
- Easier to parallelize (no shared state)
- Enables time-travel debugging (replay any state)

**In this architecture**:
- Executors: `(state, action, rules) => newState` (pure)
- Rules: `(state, ...params) => result` (pure)
- Composition: `reduce` over layers (pure)
- **No mutation anywhere**

### Parametric Polymorphism as Extensibility

**Polymorphic functions don't inspect their parameters** - they work uniformly across all types.

**Example**: Array.map doesn't care what's in the array, it just applies the function.

**In this architecture**: Executors don't inspect state to determine behavior - they call rules and trust the result.

**Benefit**: Add new contracts without changing executors. Executors are **closed for modification, open for extension**.

### Composition as Simplicity

**Complex systems built from simple, composable pieces.**

**Not**:
```typescript
// ❌ Monolithic
function executeTrick(state) {
  if (nello) { ... }
  else if (plunge) { ... }
  else if (sevens) { ... }
  // Grows with each contract
}
```

**Instead**:
```typescript
// ✅ Composable
const rules = compose([baseRules, nelloRules, plungeRules]);
executeTrick(state, rules);  // Same executor, composed behavior
```

**Benefit**: Each layer is simple and testable. Composition creates complexity, but composition itself is simple (reduce).

### Correct by Construction

**Use the type system to make bugs impossible.**

**Not**:
```typescript
// ❌ Runtime check
if (!rules) throw new Error('Rules required');
```

**Instead**:
```typescript
// ✅ Compile-time guarantee
function execute(state: GameState, rules: GameRules): GameState
```

**Benefit**: Class of bugs (forgot to provide rules) eliminated at compile time. No need to test for it.

---

## Success Criteria

This architecture succeeds if:

1. ✅ **All special contracts implemented** without conditional logic in executors
2. ✅ **Type-safe composition** - compiler enforces correctness
3. ✅ **Extensible** - new contracts don't touch core
4. ✅ **Testable** - each layer tested independently
5. ✅ **Pure functional** - no mutation, no side effects
6. ✅ **Correct by construction** - type system prevents common bugs
7. ✅ **Compatible** with existing multiplayer and variant systems
8. ✅ **Documented** clearly for future maintainers

**We will have succeeded when**: A developer can add a new special contract by creating a single layer file, without reading or modifying any executor code.

---

## Conclusion

This architecture achieves **pure functional composition** for game execution via **threaded rules**:

- **7 composable rules** replace conditional logic
- **Parametric polymorphism** makes executors variant-agnostic
- **Type safety** guarantees correctness
- **Composition** enables extensibility
- **Purity** enables testing and reasoning

**The result**: A clean, correct, extensible foundation for Texas 42 that handles all special contracts while maintaining the purity and composability of the multiplayer vision.

**Next steps**: Implement phase by phase, validate at each step, ship when all tests pass.

**The foundation is solid. Let's build.** 🎯
