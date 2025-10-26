/**
 * Core types for the threaded rules architecture.
 *
 * This enables pure functional composition for special contracts via parametric polymorphism.
 * Executors become variant-agnostic by delegating decisions to injected rules.
 */

import type { GameState, TrumpSelection, Bid, Play, Domino, LedSuit } from '../types';

/**
 * Result of checking if hand outcome is determined early.
 * Used by special contracts that end before all 7 tricks (nello, plunge, splash, sevens).
 */
export interface HandOutcome {
  isDetermined: boolean;
  reason?: string;
  decidedAtTrick?: number;
}

/**
 * GameRules interface - 7 composable rules that define game execution semantics.
 *
 * Rules are grouped into three categories:
 * - WHO: Determine which player acts
 * - WHEN: Determine timing and completion
 * - HOW: Determine game mechanics
 *
 * Executors call these rules instead of hardcoding behavior, enabling
 * special contracts to override specific rules without touching executor code.
 */
export interface GameRules {
  // ============================================
  // WHO RULES: Determine which player acts
  // ============================================

  /**
   * Who selects trump after bidding completes?
   *
   * Base: Winning bidder
   * Plunge/Splash: Partner of winning bidder
   */
  getTrumpSelector(state: GameState, winningBid: Bid): number;

  /**
   * Who leads the first trick after trump is selected?
   *
   * Base: Trump selector (bidder)
   * Nello/Sevens: Bidder leads
   * Plunge/Splash: Partner (who selected trump) leads
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

  // ============================================
  // WHEN RULES: Determine timing and completion
  // ============================================

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
   * Plunge/Splash/Sevens: Opponents win any trick = hand over
   *
   * Returns null to continue, or HandOutcome if determined.
   */
  checkHandOutcome(state: GameState): HandOutcome | null;

  // ============================================
  // HOW RULES: Determine game mechanics
  // ============================================

  /**
   * What suit does a domino lead when played?
   *
   * Base: Higher pip (or 7 if doubles-trump)
   * Nello: Doubles = 7 (own suit), else higher pip
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

/**
 * GameLayer interface - Composable layer that can override rules and/or transform actions.
 *
 * Layers have two orthogonal composition surfaces:
 * 1. Action generation (getValidActions) - what's possible
 * 2. Rule algorithms (rules) - how things execute
 *
 * Layers compose via reduce, with later layers overriding earlier ones.
 */
export interface GameLayer {
  name: string;

  /**
   * Transform action generation (existing variant system).
   *
   * Pattern: Filter, annotate, or add actions.
   *
   * Example: Tournament layer filters special contract bids
   * Example: Nello layer adds nello trump selection option
   *
   * @param state Current game state
   * @param prev Actions from previous layers (base or other layers)
   * @returns Transformed action list
   */
  getValidActions?: (state: GameState, prev: import('../types').GameAction[]) => import('../types').GameAction[];

  /**
   * Override specific rules (new threaded rules system).
   *
   * Pattern: Check state, return override or pass through prev.
   *
   * Each rule method receives the previous layer's result as its last parameter.
   * Return prev to pass through, or return new value to override.
   *
   * Example: Nello layer returns 3 for isTrickComplete
   * Example: Plunge layer returns partner index for getTrumpSelector
   *
   * Partial<GameRules> means you only implement rules you want to override.
   */
  rules?: {
    getTrumpSelector?: (state: GameState, winningBid: Bid, prev: number) => number;
    getFirstLeader?: (state: GameState, trumpSelector: number, trump: TrumpSelection, prev: number) => number;
    getNextPlayer?: (state: GameState, currentPlayer: number, prev: number) => number;
    isTrickComplete?: (state: GameState, prev: boolean) => boolean;
    checkHandOutcome?: (state: GameState, prev: HandOutcome | null) => HandOutcome | null;
    getLedSuit?: (state: GameState, domino: Domino, prev: LedSuit) => LedSuit;
    calculateTrickWinner?: (state: GameState, trick: Play[], prev: number) => number;
  };
}
