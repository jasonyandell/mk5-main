/**
 * StateBuilder - Fluent test state factory for Texas 42
 *
 * Provides a clean, type-safe API for building game states in tests.
 * Eliminates 164+ duplicated state creation patterns across the codebase.
 *
 * @example Basic usage
 * ```typescript
 * const state = StateBuilder
 *   .inBiddingPhase()
 *   .withDealer(2)
 *   .withPlayerHand(0, ['6-6', '6-5', '5-5'])
 *   .build();
 * ```
 *
 * @example Complex scenario
 * ```typescript
 * const state = StateBuilder
 *   .inPlayingPhase({ type: 'suit', suit: ACES })
 *   .withTricksPlayed(3)
 *   .withTeamScores(15, 8)
 *   .withCurrentTrick([
 *     { player: 0, domino: '6-5' },
 *     { player: 1, domino: '5-4' }
 *   ])
 *   .build();
 * ```
 */

import type { GameState, Domino, Bid, Trick, Play, TrumpSelection } from '../../game/types';
import { createInitialState, cloneGameState } from '../../game/core/state';
import { BID_TYPES } from '../../game/constants';
import { ACES } from '../../game/types';
import { getLedSuitBase } from '../../game/layers/rules-base';
import {
  generateDealFromConstraints,
  type DealConstraints,
  type PlayerConstraint,
  type ConstraintDomino
} from './dealConstraints';

/**
 * Helper class for building Domino objects from string IDs or pairs
 */
export class DominoBuilder {
  /**
   * Parse a domino from string format "6-5" or "5-6"
   * @example DominoBuilder.from("6-5") // { high: 6, low: 5, id: "6-5" }
   */
  static from(id: string): Domino {
    const parts = id.split('-');
    if (parts.length !== 2) {
      throw new Error(`Invalid domino ID: ${id}. Expected format "high-low" (e.g., "6-5")`);
    }

    const [a, b] = parts.map(Number);
    if (a === undefined || b === undefined || isNaN(a) || isNaN(b)) {
      throw new Error(`Invalid domino ID: ${id}. Both parts must be numbers`);
    }

    if (a < 0 || a > 6 || b < 0 || b > 6) {
      throw new Error(`Invalid domino ID: ${id}. Pips must be 0-6`);
    }

    // Normalize to high-low order
    const high = Math.max(a, b);
    const low = Math.min(a, b);

    return {
      high,
      low,
      id: `${high}-${low}`,
      points: this.calculatePoints(high, low)
    };
  }

  /**
   * Create a domino from high and low pip values
   * @example DominoBuilder.fromPair(6, 5) // { high: 6, low: 5, id: "6-5" }
   */
  static fromPair(high: number, low: number): Domino {
    if (high < 0 || high > 6 || low < 0 || low > 6) {
      throw new Error(`Invalid domino pips: [${high}, ${low}]. Must be 0-6`);
    }

    return {
      high,
      low,
      id: `${high}-${low}`,
      points: this.calculatePoints(high, low)
    };
  }

  /**
   * Create a double domino
   * @example DominoBuilder.doubles(6) // { high: 6, low: 6, id: "6-6" }
   */
  static doubles(value: 0 | 1 | 2 | 3 | 4 | 5 | 6): Domino {
    return this.fromPair(value, value);
  }

  /**
   * Calculate points for a domino (mk4 rules: 35 points total)
   */
  private static calculatePoints(high: number, low: number): number {
    const sum = high + low;

    // Count-all (5 pips = 1 point)
    if (sum === 10) return 10; // 5-5 or 6-4
    if (sum === 5) return 5;   // 5-0, 4-1, or 3-2
    if (high === 6 && low === 5) return 5; // 6-5
    if (high === 6 && low === 6) return 2; // 6-6

    return 0;
  }
}

/**
 * Helper class for building hands (arrays of Domino objects)
 */
export class HandBuilder {
  /**
   * Create a hand with specified number of doubles
   * @example HandBuilder.withDoubles(3) // [0-0, 1-1, 2-2, 3-0, 4-0, 5-0, 6-0]
   */
  static withDoubles(count: number): Domino[] {
    if (count < 0 || count > 7) {
      throw new Error(`Invalid double count: ${count}. Must be 0-7`);
    }

    const hand: Domino[] = [];

    // Add doubles
    for (let i = 0; i < count; i++) {
      hand.push(DominoBuilder.doubles(i as 0 | 1 | 2 | 3 | 4 | 5 | 6));
    }

    // Fill remaining with non-doubles
    let nextHigh = count;
    let nextLow = 0;
    while (hand.length < 7) {
      if (nextHigh > 6) break;
      if (nextLow >= nextHigh) {
        nextHigh++;
        nextLow = 0;
        continue;
      }

      hand.push(DominoBuilder.fromPair(nextHigh, nextLow));
      nextLow++;
    }

    return hand;
  }

  /**
   * Parse a hand from array of string IDs
   * @example HandBuilder.fromStrings(["6-6", "6-5", "5-5"])
   */
  static fromStrings(ids: string[]): Domino[] {
    return ids.map(id => DominoBuilder.from(id));
  }

  /**
   * Generate a random hand (for when exact dominoes don't matter)
   * @param seed Optional seed for deterministic randomness
   */
  static random(seed?: number): Domino[] {
    const allDominoes: Domino[] = [];
    for (let high = 0; high <= 6; high++) {
      for (let low = 0; low <= high; low++) {
        allDominoes.push(DominoBuilder.fromPair(high, low));
      }
    }

    // Simple shuffle (not cryptographically secure, just for tests)
    const rng = seed !== undefined ? this.seededRandom(seed) : Math.random;
    for (let i = allDominoes.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [allDominoes[i], allDominoes[j]] = [allDominoes[j]!, allDominoes[i]!];
    }

    return allDominoes.slice(0, 7);
  }

  /**
   * Simple seeded random number generator for deterministic tests
   */
  private static seededRandom(seed: number): () => number {
    let state = seed;
    return () => {
      state = (state * 1664525 + 1013904223) % 2147483648;
      return state / 2147483648;
    };
  }
}

/**
 * Fluent builder for creating GameState objects in tests.
 *
 * Key features:
 * - Immutable: each modifier returns a new builder
 * - Type-safe: leverages TypeScript for correctness
 * - Chainable: fluent API for readability
 * - Comprehensive: covers all common test scenarios
 */
export class StateBuilder {
  private state: GameState;
  private dealConstraints: DealConstraints | null = null;
  private constraintFillSeed: number = 0;

  /**
   * Private constructor - use static factory methods instead
   */
  private constructor(state: GameState, constraints?: DealConstraints | null, fillSeed?: number) {
    this.state = cloneGameState(state);
    this.dealConstraints = constraints ?? null;
    this.constraintFillSeed = fillSeed ?? 0;
  }

  // ============================================================================
  // Factory Methods (Entry Points)
  // ============================================================================

  /**
   * Create a state in bidding phase with dealt hands
   * @param dealer Dealer position (0-3)
   * @example StateBuilder.inBiddingPhase(0)
   */
  static inBiddingPhase(dealer: number = 0): StateBuilder {
    const state = createInitialState({ dealer });
    return new StateBuilder(state);
  }

  /**
   * Create a state in trump selection phase (after winning bid)
   * @param winningBidder Player who won the bid
   * @param bidValue Bid value (default 30)
   * @example StateBuilder.inTrumpSelection(0, 32)
   */
  static inTrumpSelection(winningBidder: number = 0, bidValue: number = 30): StateBuilder {
    const state = createInitialState({ dealer: (winningBidder + 3) % 4 });
    state.phase = 'trump_selection';
    state.currentPlayer = winningBidder;
    state.winningBidder = winningBidder;
    state.currentBid = { type: BID_TYPES.POINTS, value: bidValue, player: winningBidder };

    // Create bid history showing winning bid
    state.bids = [
      { type: BID_TYPES.POINTS, value: bidValue, player: winningBidder }
    ];

    return new StateBuilder(state);
  }

  /**
   * Create a state in playing phase (trump selected, no tricks played)
   * @param trump Trump selection (optional, defaults to Aces)
   * @example StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
   */
  static inPlayingPhase(trump?: TrumpSelection): StateBuilder {
    const trumpSelection = trump || { type: 'suit', suit: ACES };
    const state = createInitialState({ dealer: 0 });
    state.phase = 'playing';
    state.trump = trumpSelection;
    state.winningBidder = 0;
    state.currentBid = { type: BID_TYPES.POINTS, value: 30, player: 0 };
    state.currentPlayer = 1; // Player left of dealer leads
    state.bids = [
      { type: BID_TYPES.POINTS, value: 30, player: 0 },
      { type: BID_TYPES.PASS, player: 1 },
      { type: BID_TYPES.PASS, player: 2 },
      { type: BID_TYPES.PASS, player: 3 }
    ];

    return new StateBuilder(state);
  }

  /**
   * Create a state with N tricks already played
   * @param count Number of tricks (1-7)
   * @param trump Trump selection
   * @example StateBuilder.withTricksPlayed(3)
   */
  static withTricksPlayed(count: number, trump?: TrumpSelection): StateBuilder {
    if (count < 0 || count > 7) {
      throw new Error(`Invalid trick count: ${count}. Must be 0-7`);
    }

    let builder = StateBuilder.inPlayingPhase(trump);

    // Add dummy tricks
    for (let i = 0; i < count; i++) {
      const trickPlays: Play[] = [
        { player: 0, domino: DominoBuilder.fromPair(i, 0) },
        { player: 1, domino: DominoBuilder.fromPair(i, 1) },
        { player: 2, domino: DominoBuilder.fromPair(i, 2) },
        { player: 3, domino: DominoBuilder.fromPair(Math.min(i, 6), Math.min(3, 6)) }
      ];

      builder = builder.addTrick(trickPlays, i % 4, 0) as StateBuilder; // Winner rotates, no points
    }

    // Remove those dominoes from hands
    const newState = cloneGameState(builder.state);
    newState.players = newState.players.map((player, idx) => {
      const hand = player.hand.filter(d => {
        // Remove dominoes that were played in tricks
        for (let i = 0; i < count; i++) {
          const playedId = `${i}-${idx}`;
          if (d.id === playedId) return false;
        }
        return true;
      });

      return { ...player, hand: hand.slice(0, 7 - count) };
    });

    return new StateBuilder(newState);
  }

  /**
   * Create a state in scoring phase (all tricks played)
   * @param teamScores Final scores [team0, team1]
   * @example StateBuilder.inScoringPhase([30, 12])
   */
  static inScoringPhase(teamScores: [number, number]): StateBuilder {
    const builder = StateBuilder.withTricksPlayed(7);
    builder.state.phase = 'scoring';
    builder.state.teamScores = teamScores;
    builder.state.players = builder.state.players.map(p => ({ ...p, hand: [] }));

    return builder;
  }

  /**
   * Create a terminal game_end state
   * @param winningTeam Which team won (0 or 1)
   * @example StateBuilder.gameEnded(0)
   */
  static gameEnded(winningTeam: 0 | 1): StateBuilder {
    const marks: [number, number] = winningTeam === 0 ? [7, 0] : [0, 7];
    const builder = StateBuilder.inScoringPhase([0, 0]);
    builder.state.phase = 'game_end';
    builder.state.teamMarks = marks;

    return builder;
  }

  // ============================================================================
  // Special Contract Presets
  // ============================================================================

  /**
   * Create a Nello contract state (marks bid, 3-player tricks)
   * @param bidder Player who bid Nello
   * @example StateBuilder.nelloContract(0)
   */
  static nelloContract(bidder: number = 0): StateBuilder {
    const state = createInitialState({ dealer: (bidder + 3) % 4 });
    state.phase = 'trump_selection';
    state.currentPlayer = bidder;
    state.winningBidder = bidder;
    state.currentBid = { type: BID_TYPES.MARKS, value: 2, player: bidder };
    state.bids = [
      { type: BID_TYPES.MARKS, value: 2, player: bidder }
    ];

    return new StateBuilder(state);
  }

  /**
   * Create a Splash contract state (2-3 marks, partner selects trump)
   * @param bidder Player who bid Splash
   * @param value Splash value (2 or 3 marks)
   * @example StateBuilder.splashContract(0, 2)
   */
  static splashContract(bidder: number = 0, value: number = 2): StateBuilder {
    if (value < 2 || value > 3) {
      throw new Error(`Invalid splash value: ${value}. Must be 2-3`);
    }

    const state = createInitialState({ dealer: (bidder + 3) % 4 });
    state.phase = 'trump_selection';

    // Partner selects trump in splash
    const partner = (bidder + 2) % 4;
    state.currentPlayer = partner;
    state.winningBidder = bidder;
    state.currentBid = { type: 'splash', value, player: bidder };
    state.bids = [
      { type: 'splash', value, player: bidder }
    ];

    return new StateBuilder(state);
  }

  /**
   * Create a Plunge contract state (4+ marks, partner selects trump)
   * @param bidder Player who bid Plunge
   * @param value Plunge value (4-7 marks)
   * @example StateBuilder.plungeContract(0, 4)
   */
  static plungeContract(bidder: number = 0, value: number = 4): StateBuilder {
    if (value < 4 || value > 7) {
      throw new Error(`Invalid plunge value: ${value}. Must be 4-7`);
    }

    const state = createInitialState({ dealer: (bidder + 3) % 4 });
    state.phase = 'trump_selection';

    // Partner selects trump in plunge
    const partner = (bidder + 2) % 4;
    state.currentPlayer = partner;
    state.winningBidder = bidder;
    state.currentBid = { type: 'plunge', value, player: bidder };
    state.bids = [
      { type: 'plunge', value, player: bidder }
    ];

    return new StateBuilder(state);
  }

  /**
   * Create a Sevens contract state (high card must lead)
   * @param bidder Player who bid Sevens
   * @example StateBuilder.sevensContract(0)
   */
  static sevensContract(bidder: number = 0): StateBuilder {
    const state = createInitialState({ dealer: (bidder + 3) % 4 });
    state.phase = 'trump_selection';
    state.currentPlayer = bidder;
    state.winningBidder = bidder;
    state.currentBid = { type: BID_TYPES.MARKS, value: 1, player: bidder };
    state.bids = [
      { type: BID_TYPES.MARKS, value: 1, player: bidder }
    ];

    return new StateBuilder(state);
  }

  // ============================================================================
  // Chainable Modifiers
  // ============================================================================

  /**
   * Set the dealer position
   * @example builder.withDealer(2)
   */
  withDealer(dealer: number): this {
    if (dealer < 0 || dealer > 3) {
      throw new Error(`Invalid dealer: ${dealer}. Must be 0-3`);
    }

    const newState = cloneGameState(this.state);
    newState.dealer = dealer;
    return new StateBuilder(newState) as this;
  }

  /**
   * Set the current player
   * @example builder.withCurrentPlayer(1)
   */
  withCurrentPlayer(player: number): this {
    if (player < 0 || player > 3) {
      throw new Error(`Invalid player: ${player}. Must be 0-3`);
    }

    const newState = cloneGameState(this.state);
    newState.currentPlayer = player;
    return new StateBuilder(newState) as this;
  }

  /**
   * Set all bids
   * @example builder.withBids([{type: 'points', value: 30, player: 0}])
   */
  withBids(bids: Bid[]): this {
    const newState = cloneGameState(this.state);
    newState.bids = [...bids];

    // Update currentBid to highest bid
    if (bids.length > 0) {
      const highestBid = bids.reduce((highest, bid) => {
        if (bid.type === BID_TYPES.PASS) return highest;
        if (highest.type === BID_TYPES.PASS) return bid;

        const highestValue = highest.value || 0;
        const bidValue = bid.value || 0;

        return bidValue > highestValue ? bid : highest;
      }, bids[0]!);

      newState.currentBid = highestBid;
    }

    return new StateBuilder(newState) as this;
  }

  /**
   * Set trump selection
   * @example builder.withTrump({ type: 'suit', suit: ACES })
   */
  withTrump(trump: TrumpSelection): this {
    const newState = cloneGameState(this.state);
    newState.trump = trump;
    return new StateBuilder(newState) as this;
  }

  /**
   * Set winning bid and bidder
   * @example builder.withWinningBid(0, {type: 'points', value: 32, player: 0})
   */
  withWinningBid(player: number, bid: Bid): this {
    const newState = cloneGameState(this.state);
    newState.winningBidder = player;
    newState.currentBid = bid;

    // Add to bids array if not already there
    if (!newState.bids.some(b => b.player === player && b.type === bid.type && b.value === bid.value)) {
      newState.bids.push(bid);
    }

    return new StateBuilder(newState) as this;
  }

  /**
   * Set a specific player's hand
   * @param playerIndex Player index (0-3)
   * @param dominoes Array of Domino objects or string IDs
   * @example builder.withPlayerHand(0, ['6-6', '6-5', '5-5'])
   */
  withPlayerHand(playerIndex: number, dominoes: Domino[] | string[]): this {
    if (playerIndex < 0 || playerIndex > 3) {
      throw new Error(`Invalid player index: ${playerIndex}. Must be 0-3`);
    }

    const newState = cloneGameState(this.state);
    const hand = Array.isArray(dominoes) && typeof dominoes[0] === 'string'
      ? HandBuilder.fromStrings(dominoes as string[])
      : (dominoes as Domino[]);

    newState.players[playerIndex] = {
      ...newState.players[playerIndex]!,
      hand
    };

    return new StateBuilder(newState) as this;
  }

  /**
   * Set all player hands at once
   * @param hands Array of 4 hands (Domino arrays or string arrays)
   * @example builder.withHands([['6-6'], ['5-5'], ['4-4'], ['3-3']])
   */
  withHands(hands: Array<Domino[] | string[]>): this {
    if (hands.length !== 4) {
      throw new Error(`Must provide exactly 4 hands, got ${hands.length}`);
    }

    return hands.reduce(
      (acc, hand, idx) => acc.withPlayerHand(idx, hand),
      this as this
    );
  }

  /**
   * Set current trick in progress
   * @param plays Array of {player, domino} plays
   * @example builder.withCurrentTrick([{player: 0, domino: '6-5'}])
   */
  withCurrentTrick(plays: Array<{ player: number; domino: Domino | string }>): this {
    const newState = cloneGameState(this.state);

    newState.currentTrick = plays.map(play => ({
      player: play.player,
      domino: typeof play.domino === 'string'
        ? DominoBuilder.from(play.domino)
        : play.domino
    }));

    // Update currentSuit if there's a lead domino
    if (newState.currentTrick.length > 0 && newState.trump.type !== 'not-selected') {
      const leadDomino = newState.currentTrick[0]!.domino;
      newState.currentSuit = getLedSuitBase(newState, leadDomino);
    }

    return new StateBuilder(newState) as this;
  }

  /**
   * Set completed tricks
   * @example builder.withTricks([{plays: [...], winner: 0, points: 10}])
   */
  withTricks(tricks: Trick[]): this {
    const newState = cloneGameState(this.state);
    newState.tricks = tricks.map(trick => ({
      ...trick,
      plays: [...trick.plays]
    }));

    return new StateBuilder(newState) as this;
  }

  /**
   * Add a completed trick
   * @param plays Array of Play objects
   * @param winner Winner player index
   * @param points Points scored
   * @example builder.addTrick([{player: 0, domino: dom}], 0, 10)
   */
  addTrick(plays: Play[], winner: number, points: number): this {
    const newState = cloneGameState(this.state);

    const ledSuit = newState.trump.type !== 'not-selected' && plays[0]
      ? getLedSuitBase(newState, plays[0].domino)
      : undefined;

    const trick: Trick = {
      plays: [...plays],
      winner,
      points,
      ...(ledSuit !== undefined && { ledSuit })
    };

    newState.tricks.push(trick);

    // Update team scores
    const winnerTeam = newState.players[winner]!.teamId;
    newState.teamScores[winnerTeam] += points;

    return new StateBuilder(newState) as this;
  }

  /**
   * Set team scores
   * @example builder.withTeamScores(25, 17)
   */
  withTeamScores(team0: number, team1: number): this {
    const newState = cloneGameState(this.state);
    newState.teamScores = [team0, team1];
    return new StateBuilder(newState) as this;
  }

  /**
   * Set team marks
   * @example builder.withTeamMarks(3, 2)
   */
  withTeamMarks(team0: number, team1: number): this {
    const newState = cloneGameState(this.state);
    newState.teamMarks = [team0, team1];
    return new StateBuilder(newState) as this;
  }

  /**
   * Set shuffle seed for deterministic randomness
   * @example builder.withSeed(12345)
   */
  withSeed(seed: number): this {
    const newState = cloneGameState(this.state);
    newState.shuffleSeed = seed;
    newState.initialConfig.shuffleSeed = seed;
    return new StateBuilder(newState) as this;
  }

  /**
   * Merge partial config into initialConfig
   * @example builder.withConfig({ playerTypes: ['human', 'human', 'ai', 'ai'] })
   */
  withConfig(config: Partial<GameState['initialConfig']>): this {
    const newState = cloneGameState(this.state);
    newState.initialConfig = {
      ...newState.initialConfig,
      ...config
    };

    // Update top-level fields that mirror config
    if (config.playerTypes) newState.playerTypes = [...config.playerTypes];
    if (config.theme) newState.theme = config.theme;
    if (config.colorOverrides) newState.colorOverrides = { ...config.colorOverrides };

    return new StateBuilder(newState) as this;
  }

  /**
   * Escape hatch: merge arbitrary partial state
   * Use sparingly - prefer specific modifiers
   * @example builder.with({ phase: 'scoring' })
   */
  with(overrides: Partial<GameState>): this {
    const newState = cloneGameState(this.state);
    Object.assign(newState, overrides);
    return new StateBuilder(newState, this.dealConstraints, this.constraintFillSeed) as this;
  }

  // ============================================================================
  // Deal Constraint Methods
  // ============================================================================

  /**
   * Set constraints for a specific player's hand.
   * Constraints are applied during build().
   *
   * @example
   * StateBuilder.inBiddingPhase()
   *   .withPlayerConstraint(0, { minDoubles: 4 })
   *   .build();
   */
  withPlayerConstraint(player: 0 | 1 | 2 | 3, constraint: PlayerConstraint): this {
    const newConstraints: DealConstraints = {
      ...this.dealConstraints,
      players: {
        ...this.dealConstraints?.players,
        [player]: constraint
      },
      fillSeed: this.constraintFillSeed
    };
    return new StateBuilder(this.state, newConstraints, this.constraintFillSeed) as this;
  }

  /**
   * Shorthand: ensure player has minimum number of doubles.
   * Useful for plunge (4+) or splash (3+) scenarios.
   *
   * @example
   * StateBuilder.inBiddingPhase()
   *   .withPlayerDoubles(0, 4)  // Plunge-eligible
   *   .build();
   */
  withPlayerDoubles(player: 0 | 1 | 2 | 3, minDoubles: number): this {
    return this.withPlayerConstraint(player, { minDoubles });
  }

  /**
   * Set full deal constraints for all players at once.
   *
   * @example
   * StateBuilder.inBiddingPhase()
   *   .withDealConstraints({
   *     players: {
   *       0: { exactDominoes: ['6-6'], minDoubles: 3 },
   *       1: { voidInSuit: [6], maxDoubles: 1 }
   *     },
   *     fillSeed: 99999
   *   })
   *   .build();
   */
  withDealConstraints(constraints: DealConstraints): this {
    return new StateBuilder(this.state, constraints, constraints.fillSeed ?? this.constraintFillSeed) as this;
  }

  /**
   * Set the seed for deterministic filling of remaining hand slots.
   * Same seed + same constraints = identical hands.
   *
   * @example
   * StateBuilder.inBiddingPhase()
   *   .withPlayerDoubles(0, 4)
   *   .withFillSeed(42)
   *   .build();
   */
  withFillSeed(seed: number): this {
    const newConstraints: DealConstraints = {
      ...this.dealConstraints,
      fillSeed: seed
    };
    return new StateBuilder(this.state, newConstraints, seed) as this;
  }

  // ============================================================================
  // Build Methods
  // ============================================================================

  /**
   * Build and return the immutable GameState
   * @returns Frozen GameState object
   */
  build(): GameState {
    // Apply deal constraints if specified
    if (this.dealConstraints) {
      return this.buildWithConstraints();
    }

    // Return a deep clone to ensure immutability
    return cloneGameState(this.state);
  }

  /**
   * Internal: Apply deal constraints and return state with constrained hands.
   */
  private buildWithConstraints(): GameState {
    const newState = cloneGameState(this.state);

    // Generate constrained deal
    const constrainedHands = generateDealFromConstraints(this.dealConstraints!);

    // Convert ConstraintDomino[] to Domino[] and update player hands
    newState.players = newState.players.map((player, idx) => {
      const constraintHand = constrainedHands[idx as 0 | 1 | 2 | 3]!;
      const hand: Domino[] = constraintHand.map(this.constraintToDomino);

      return {
        ...player,
        hand
      };
    });

    return newState;
  }

  /**
   * Convert ConstraintDomino to game Domino
   */
  private constraintToDomino(d: ConstraintDomino): Domino {
    return DominoBuilder.from(d.id);
  }

  /**
   * Clone this builder for further modification
   * @returns New StateBuilder with cloned state
   */
  clone(): StateBuilder {
    return new StateBuilder(this.state, this.dealConstraints, this.constraintFillSeed);
  }
}
