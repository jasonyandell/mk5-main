/**
 * TestLayer - Isolation utility for layer unit testing.
 *
 * Enables testing layers WITHOUT the full base layer by providing
 * controllable prev values for all rule methods.
 *
 * Key insight: Tests should verify what a layer CHANGES, not what it passes through.
 * Using TestLayer with sentinel values proves passthrough definitively.
 *
 * @example Testing plunge's passthrough behavior
 * ```typescript
 * const testLayer = createTestLayer({
 *   getTrumpSelector: () => PASSTHROUGH_SENTINEL
 * });
 * const rules = composeRules([testLayer, plungeLayer]);
 *
 * // Non-plunge bid should pass through
 * const bid = { type: 'marks', value: 2, player: 0 };
 * expect(rules.getTrumpSelector(mockState, bid)).toBe(PASSTHROUGH_SENTINEL);
 * ```
 *
 * @example Testing nello's isTrickComplete override
 * ```typescript
 * const testLayer = createTestLayer({
 *   isTrickComplete: () => false // Sentinel: base would say true
 * });
 * const rules = composeRules([testLayer, nelloLayer]);
 *
 * // With nello trump, should return true at 3 plays (overrides sentinel)
 * const state = StateBuilder.inPlayingPhase({ type: 'nello' })
 *   .withCurrentTrick([...3 plays...])
 *   .build();
 * expect(rules.isTrickComplete(state)).toBe(true);
 * ```
 */

import type { Layer, HandOutcome } from '../../game/layers/types';
import type { GameState, Bid, TrumpSelection, Domino, Play, LedSuit, GamePhase } from '../../game/types';

/**
 * Sentinel value for verifying passthrough behavior.
 * When a rule returns this value, we know the layer under test passed through.
 */
export const PASSTHROUGH_SENTINEL = 999;

/**
 * Type for layer rule overrides that TestLayer accepts.
 * Each rule receives the same parameters as the Layer.rules type but must return
 * the appropriate type (not receive prev, since TestLayer IS the prev provider).
 */
export interface TestLayerOverrides {
  getTrumpSelector?: (state: GameState, bid: Bid) => number;
  getFirstLeader?: (state: GameState, selector: number, trump: TrumpSelection) => number;
  getNextPlayer?: (state: GameState, current: number) => number;
  isTrickComplete?: (state: GameState) => boolean;
  checkHandOutcome?: (state: GameState) => HandOutcome;
  getLedSuit?: (state: GameState, domino: Domino) => LedSuit;
  calculateTrickWinner?: (state: GameState, trick: Play[]) => number;
  isValidPlay?: (state: GameState, domino: Domino, playerId: number) => boolean;
  getValidPlays?: (state: GameState, playerId: number) => Domino[];
  isValidBid?: (state: GameState, bid: Bid, playerHand?: Domino[]) => boolean;
  getBidComparisonValue?: (bid: Bid) => number;
  isValidTrump?: (trump: TrumpSelection) => boolean;
  calculateScore?: (state: GameState) => [number, number];
  getPhaseAfterHandComplete?: (state: GameState) => GamePhase;
}

/**
 * Creates a minimal test layer for isolated layer testing.
 *
 * The test layer acts as a "mock base layer" that provides controllable
 * prev values. When composed with a layer under test:
 *   composeRules([testLayer, layerUnderTest])
 *
 * The layerUnderTest receives testLayer's values as prev, allowing
 * isolated testing of just that layer's logic.
 *
 * @param overrides Functions that return the prev value for each rule
 * @returns A Layer suitable for composition in tests
 */
export function createTestLayer(overrides: TestLayerOverrides = {}): Layer {
  return {
    name: 'test',
    rules: {
      // WHO rules
      getTrumpSelector: (state, bid, prev) =>
        overrides.getTrumpSelector?.(state, bid) ?? prev,

      getFirstLeader: (state, selector, trump, prev) =>
        overrides.getFirstLeader?.(state, selector, trump) ?? prev,

      getNextPlayer: (state, current, prev) =>
        overrides.getNextPlayer?.(state, current) ?? prev,

      // WHEN rules
      isTrickComplete: (state, prev) =>
        overrides.isTrickComplete?.(state) ?? prev,

      checkHandOutcome: (state, prev) =>
        overrides.checkHandOutcome?.(state) ?? prev,

      // HOW rules
      getLedSuit: (state, domino, prev) =>
        overrides.getLedSuit?.(state, domino) ?? prev,

      calculateTrickWinner: (state, trick, prev) =>
        overrides.calculateTrickWinner?.(state, trick) ?? prev,

      // VALIDATION rules
      isValidPlay: (state, domino, playerId, prev) =>
        overrides.isValidPlay?.(state, domino, playerId) ?? prev,

      getValidPlays: (state, playerId, prev) =>
        overrides.getValidPlays?.(state, playerId) ?? prev,

      isValidBid: (state, bid, playerHand, prev) =>
        overrides.isValidBid?.(state, bid, playerHand) ?? prev,

      // SCORING rules
      getBidComparisonValue: (bid, prev) =>
        overrides.getBidComparisonValue?.(bid) ?? prev,

      isValidTrump: (trump, prev) =>
        overrides.isValidTrump?.(trump) ?? prev,

      calculateScore: (state, prev) =>
        overrides.calculateScore?.(state) ?? prev,

      // LIFECYCLE rules
      getPhaseAfterHandComplete: (state, prev) =>
        overrides.getPhaseAfterHandComplete?.(state) ?? prev,
    }
  };
}

/**
 * Sentinel value for numeric rules where we CAN use a sentinel.
 * Player indices 0-3, so 99 is clearly a sentinel.
 */
export const PLAYER_SENTINEL = 99;

/**
 * Create a test layer that returns sentinel/default values for all rules.
 * Useful for verifying that a layer passes through ALL rules it doesn't override.
 *
 * For numeric player indices, uses PLAYER_SENTINEL (99).
 * For constrained types (LedSuit, boolean, etc.), uses type-valid defaults.
 */
export function createSentinelLayer(): Layer {
  return createTestLayer({
    getTrumpSelector: () => PLAYER_SENTINEL,
    getFirstLeader: () => PLAYER_SENTINEL,
    getNextPlayer: () => PLAYER_SENTINEL,
    isTrickComplete: () => false, // Type-constrained: boolean
    checkHandOutcome: () => ({ isDetermined: false }), // Type-constrained
    getLedSuit: () => 0, // Type-constrained: 0-7
    calculateTrickWinner: () => PLAYER_SENTINEL,
    isValidPlay: () => true, // Type-constrained: boolean
    getValidPlays: () => [], // Type-constrained: Domino[]
    isValidBid: () => true, // Type-constrained: boolean
    getBidComparisonValue: () => PASSTHROUGH_SENTINEL,
    isValidTrump: () => true, // Type-constrained: boolean
    calculateScore: () => [PASSTHROUGH_SENTINEL, PASSTHROUGH_SENTINEL],
    getPhaseAfterHandComplete: () => 'bidding', // Type-constrained: GamePhase
  });
}
