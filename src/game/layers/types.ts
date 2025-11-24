/**
 * Core types for the threaded rules architecture.
 *
 * This enables pure functional composition for special contracts via parametric polymorphism.
 * Executors become rule-set-agnostic by delegating decisions to injected rules.
 */

import type { GameState, TrumpSelection, Bid, Play, Domino, LedSuit, GamePhase } from '../types';

/**
 * Result of checking if hand outcome is determined early.
 * Uses discriminated union to make invalid states unrepresentable.
 * Aligns with Result<T> pattern in multiplayer/types.ts.
 */
export type HandOutcome =
  | { isDetermined: false }
  | { isDetermined: true; reason: string; decidedAtTrick?: number };

/**
 * GameRules interface - 13 composable rules that define game execution semantics.
 *
 * Rules are grouped into four categories:
 * - WHO: Determine which player acts (3 rules)
 * - WHEN: Determine timing and completion (2 rules)
 * - HOW: Determine game mechanics (2 rules)
 * - VALIDATION: Determine what's legal (3 rules)
 * - SCORING: Determine bid ordering and final marks (3 rules)
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
   * Base: { isDetermined: false } (play all tricks)
   * Nello: Bidder wins any trick = hand over
   * Plunge/Splash/Sevens: Opponents win any trick = hand over
   *
   * Returns { isDetermined: false } to continue, or { isDetermined: true, reason } if outcome determined.
   */
  checkHandOutcome(state: GameState): HandOutcome;

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

  // ============================================
  // VALIDATION RULES: Determine what's legal
  // ============================================

  /**
   * Is this domino play valid for this player?
   *
   * Base: Standard follow-suit logic
   * Sevens: No follow-suit requirement (always valid if domino in hand)
   */
  isValidPlay(state: GameState, domino: Domino, playerId: number): boolean;

  /**
   * Get all valid plays for this player.
   *
   * Base: Follow-suit constrained
   * Sevens: All dominoes in hand
   */
  getValidPlays(state: GameState, playerId: number): Domino[];

  /**
   * Is this bid valid?
   *
   * Base: Standard bid validation
   * Layers can override for special bid rules
   */
  isValidBid(state: GameState, bid: Bid, playerHand?: Domino[]): boolean;

  // ============================================
  // SCORING RULES: Determine bid ordering and final marks
  // ============================================

  /**
   * Get bid comparison value for ordering (e.g., 30 points vs 2 marks).
   *
   * Base: Points = value, Marks = value * 42
   * Special contracts will override to add their types
   */
  getBidComparisonValue(bid: Bid): number;

  /**
   * Is this trump selection valid?
   *
   * Base: suit/doubles/no-trump
   * Nello: Also allows 'nello' trump type
   * Sevens: Also allows 'sevens' trump type
   */
  isValidTrump(trump: TrumpSelection): boolean;

  /**
   * Calculate final marks awarded for the hand.
   *
   * Base: Standard points/marks scoring
   * Nello: 0 tricks required
   * Splash/Plunge: All tricks required
   * Sevens: All tricks required with sevens trump
   *
   * Returns [team0Marks, team1Marks]
   */
  calculateScore(state: GameState): [number, number];

  // ============================================
  // LIFECYCLE RULES: Determine game flow transitions
  // ============================================

  /**
   * What phase should we transition to after hand is scored?
   *
   * Base: 'bidding' (continue to next hand)
   * OneHand: 'one-hand-complete' (terminal state, don't deal new hand)
   *
   * Used by executeScoreHand to determine next phase when game is not complete.
   */
  getPhaseAfterHandComplete(state: GameState): GamePhase;
}

/**
 * Layer interface - Composable layer that can override rules and/or transform actions.
 *
 * Layers have two orthogonal composition surfaces:
 * 1. Action generation (getValidActions) - what's possible
 * 2. Rule algorithms (rules) - how things execute
 *
 * Layers compose via reduce, with later layers overriding earlier ones.
 */
export interface Layer {
  name: string;

  /**
   * Transform action generation.
   *
   * Pattern: Filter, annotate, or add actions.
   *
   * Example: Tournament layer filters special contract bids
   * Example: Nello layer adds nello trump selection option
   * Example: Speed layer auto-executes forced moves
   * Example: Hints layer annotates actions with educational hints
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
    checkHandOutcome?: (state: GameState, prev: HandOutcome) => HandOutcome;
    getLedSuit?: (state: GameState, domino: Domino, prev: LedSuit) => LedSuit;
    calculateTrickWinner?: (state: GameState, trick: Play[], prev: number) => number;
    isValidPlay?: (state: GameState, domino: Domino, playerId: number, prev: boolean) => boolean;
    getValidPlays?: (state: GameState, playerId: number, prev: Domino[]) => Domino[];
    isValidBid?: (state: GameState, bid: Bid, playerHand: Domino[] | undefined, prev: boolean) => boolean;
    getBidComparisonValue?: (bid: Bid, prev: number) => number;
    isValidTrump?: (trump: TrumpSelection, prev: boolean) => boolean;
    calculateScore?: (state: GameState, prev: [number, number]) => [number, number];
    getPhaseAfterHandComplete?: (state: GameState, prev: GamePhase) => GamePhase;
  };
}
