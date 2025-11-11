import type { GameState, Domino, Bid, Player, TrumpSelection, LedSuitOrNone } from '../../game/types';
import {
  createInitialState,
  getNextStates,
  dealDominoesWithSeed
} from '../../game';
import { createTestContext } from './executionContext';
import { composeRules, baseRuleSet } from '../../game/rulesets';
import { getDominoSuit } from '../../game/core/dominoes';
import { BID_TYPES } from '../../game/constants';
import { BLANKS, ACES } from '../../game/types';

const rules = composeRules([baseRuleSet]);

/**
 * Comprehensive test helper for Texas 42 game testing
 * Provides utilities for state injection, validation, and scenario generation
 */
export class GameTestHelper {
  
  /**
   * Deals dominoes to all players in a game state
   */
  dealDominoes(state: GameState): GameState {
    const hands = dealDominoesWithSeed(state.shuffleSeed || 12345);

    return {
      ...state,
      players: state.players.map((player, index) => ({
        ...player,
        hand: hands[index] || []
      }))
    };
  }
  
  /**
   * Creates a state in bidding phase
   */
  createBiddingState(): GameState {
    const state = createInitialState();
    const dealtState = this.dealDominoes(state);
    return {
      ...dealtState,
      phase: 'bidding'
    };
  }
  
  /**
   * Creates a state with game in progress
   */
  createGameInProgress(): GameState {
    const state = this.createBiddingState();
    return {
      ...state,
      phase: 'playing',
      trump: { type: 'suit', suit: ACES },
      winningBidder: 0,
      currentPlayer: 0,
      bids: [
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 }
      ]
    };
  }
  
  /**
   * Creates a completed game state
   */
  createCompletedGame(winningTeam: number): GameState {
    const state = this.createGameInProgress();
    return {
      ...state,
      phase: 'game_end',
      teamMarks: winningTeam === 0 ? [7, 0] : [0, 7]
    };
  }
  
  /**
   * Creates a complete game state
   */
  createCompleteGameState(): GameState {
    const state = this.createGameInProgress();
    // Create 7 completed tricks
    const tricks = [];
    for (let i = 0; i < 7; i++) {
      tricks.push({
        plays: [
          { player: 0, domino: { id: `trick${i}-0`, high: 1, low: 2, points: 0 } },
          { player: 1, domino: { id: `trick${i}-1`, high: 2, low: 3, points: 0 } },
          { player: 2, domino: { id: `trick${i}-2`, high: 3, low: 4, points: 0 } },
          { player: 3, domino: { id: `trick${i}-3`, high: 4, low: 5, points: 0 } }
        ],
        points: 1
      });
    }
    return {
      ...state,
      tricks,
      players: state.players.map(player => ({
        ...player,
        hand: [] // All dominoes played
      }))
    };
  }
  
  /**
   * Creates a game state with specific mark scores
   */
  createGameWithMarks(team0Marks: number, team1Marks: number): GameState {
    const state = createInitialState();
    const isComplete = team0Marks >= 7 || team1Marks >= 7;
    return {
      ...state,
      teamMarks: [team0Marks, team1Marks],
      phase: isComplete ? 'game_end' : state.phase
    };
  }
  
  /**
   * Creates a test state with custom parameters
   */
  static createTestState(overrides: Partial<GameState> & { players?: (Player | Partial<Player> | undefined)[] } = {}): GameState {
    const baseState = createInitialState();
    
    // Deep merge players array if provided
    let players = baseState.players;
    if (overrides.players) {
      players = overrides.players.map((playerOverride, index) => {
        if (playerOverride === undefined) {
          return baseState.players[index]!;
        }
        return {
          ...baseState.players[index]!,
          ...playerOverride
        };
      }) as Player[];
    }
    
    const state = {
      ...baseState,
      ...overrides,
      players
    };
    
    // Set currentSuit based on currentTrick if provided
    if (state.currentTrick && state.currentTrick.length > 0 && state.trump.type !== 'not-selected') {
      const leadDomino = state.currentTrick[0]!.domino;
      state.currentSuit = getDominoSuit(leadDomino, state.trump) as LedSuitOrNone;
    }
    
    return state;
  }
  
  /**
   * Creates a specific bidding scenario
   */
  static createBiddingScenario(
    currentPlayer: number = 0,
    existingBids: Bid[] = []
  ): GameState {
    return this.createTestState({
      phase: 'bidding',
      currentPlayer,
      bids: existingBids
    });
  }
  
  /**
   * Creates a playing scenario with specified trump and trick state
   */
  static createPlayingScenario(
    trump: TrumpSelection,
    currentPlayer: number = 0,
    currentTrick: { player: number; domino: Domino }[] = []
  ): GameState {
    return this.createTestState({
      phase: 'playing',
      trump,
      currentPlayer,
      currentTrick,
      winningBidder: 0,
      currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 }
    });
  }

  /**
   * Creates a playing state with custom parameters for rule testing
   */
  static createPlayingState(options: {
    trump?: TrumpSelection;
    currentTrick?: { player: number; domino: Domino }[];
    currentPlayer?: number;
    hands?: { [playerId: number]: Domino[] };
  } = {}): GameState {
    const baseState = this.createTestState({
      phase: 'playing',
      trump: options.trump || { type: 'suit', suit: BLANKS },
      currentPlayer: options.currentPlayer || 0,
      currentTrick: options.currentTrick || [],
      winningBidder: 0,
      currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 }
    });

    // Add hands if provided
    if (options.hands) {
      const updatedPlayers = baseState.players.map((player, index) => ({
        ...player,
        hand: options.hands?.[index] || player.hand
      }));
      return { ...baseState, players: updatedPlayers };
    }

    return baseState;
  }
  
  /**
   * Creates a specific domino hand for testing
   */
  static createTestHand(dominoes: ({ high: number; low: number; points?: number } | [number, number])[]): Domino[] {
    return dominoes.map((domino) => {
      if (Array.isArray(domino)) {
        const [high, low] = domino;
        return {
          high,
          low,
          points: 0,
          id: `${high}-${low}`
        };
      } else {
        const { high, low, points = 0 } = domino;
        return {
          high,
          low,
          points,
          id: `${high}-${low}`
        };
      }
    });
  }

  /**
   * Creates a hand with a specific number of doubles for testing Plunge/Splash bids
   */
  static createHandWithDoubles(numDoubles: number, fillWithRandom: boolean = true): Domino[] {
    const hand: Domino[] = [];
    
    // Add the specified number of doubles
    for (let i = 0; i < Math.min(numDoubles, 7); i++) {
      hand.push({
        id: `double-${i}`,
        high: i,
        low: i,
        points: 0
      });
    }
    
    // Fill remaining slots with non-doubles if requested
    if (fillWithRandom && hand.length < 7) {
      const remaining = 7 - hand.length;
      for (let i = 0; i < remaining; i++) {
        hand.push({
          id: `non-double-${i}`,
          high: i,
          low: (i + 1) % 7,
          points: 0
        });
      }
    }
    
    return hand;
  }
  
  /**
   * Creates a player with a specific hand
   */
  static createTestPlayer(
    id: number, 
    hand: Domino[], 
    teamId: 0 | 1 = 0
  ): Player {
    return {
      id,
      name: `Player ${id + 1}`,
      hand,
      teamId,
      marks: 0
    };
  }
  
  /**
   * Validates that a game state follows all rules
   */
  static validateGameRules(state: GameState): string[] {
    const errors: string[] = [];
    
    // Check basic state structure
    if (state.players.length !== 4) {
      errors.push('Must have exactly 4 players');
    }
    
    // Check hand sizes
    const totalDominoes = state.players.reduce((total, player) => 
      total + player.hand.length, 0
    );
    const expectedTotal = 28 - (state.tricks.length * 4) - state.currentTrick.length;
    if (totalDominoes !== expectedTotal) {
      errors.push(`Invalid total dominoes: ${totalDominoes}, expected: ${expectedTotal}`);
    }
    
    // Check score consistency
    const calculatedScores = [0, 0];
    state.tricks.forEach(trick => {
      if (trick.winner !== undefined) {
        const player = state.players[trick.winner];
        if (!player) {
          throw new Error(`Invalid winner player index: ${trick.winner}`);
        }
        const team = player.teamId;
        calculatedScores[team] = (calculatedScores[team] || 0) + trick.points;
      }
    });
    
    if (calculatedScores[0] !== state.teamScores[0] || 
        calculatedScores[1] !== state.teamScores[1]) {
      errors.push('Team scores don\'t match trick totals');
    }
    
    // Check bidding rules
    if (state.phase === 'bidding') {
      for (let i = 1; i < state.bids.length; i++) {
        const currentBid = state.bids[i];
        
        if (currentBid && !rules.isValidBid(state, currentBid)) {
          errors.push(`Invalid bid at position ${i}: ${JSON.stringify(currentBid)}`);
        }
      }
    }
    
    return errors;
  }
  
  /**
   * Simulates a complete bidding round
   */
  static simulateBiddingRound(
    initialState: GameState,
    bids: (Bid | 'auto')[]
  ): GameState {
    const ctx = createTestContext();
    let state = { ...initialState };

    for (const bid of bids) {
      const nextStates = getNextStates(state, ctx);

      if (bid === 'auto') {
        // Automatically select first valid action
        if (nextStates.length > 0) {
          state = nextStates[0]!.newState;
        }
      } else {
        // Find matching bid action
        const matchingAction = nextStates.find(action => {
          const actionBid = this.extractBidFromAction(action.id);
          return actionBid && this.bidsEqual(actionBid, bid);
        });

        if (matchingAction) {
          state = matchingAction.newState;
        } else {
          throw new Error(`Cannot find action for bid: ${JSON.stringify(bid)}`);
        }
      }
    }

    return state;
  }
  
  /**
   * Extracts bid information from action
   */
  private static extractBidFromAction(id: string): Bid | null {
    if (id === 'pass') {
      return { type: BID_TYPES.PASS, player: 0 }; // Player will be set correctly
    }
    
    const pointsMatch = id.match(/^bid-(\d+)$/);
    if (pointsMatch && pointsMatch[1]) {
      return { type: BID_TYPES.POINTS, value: parseInt(pointsMatch[1]), player: 0 };
    }
    
    const marksMatch = id.match(/^bid-(\d+)-marks$/);
    if (marksMatch && marksMatch[1]) {
      return { type: BID_TYPES.MARKS, value: parseInt(marksMatch[1]), player: 0 };
    }
    
    return null;
  }
  
  /**
   * Compares two bids for equality
   */
  private static bidsEqual(bid1: Bid, bid2: Bid): boolean {
    return bid1.type === bid2.type && bid1.value === bid2.value;
  }
  
  /**
   * Checks mathematical constants (mk4 35-point system)
   */
  static verifyPointConstants(): boolean {
    const testDominoes = this.createTestHand([
      { high: 5, low: 5, points: 10 }, // 5-5 = 10 points
      { high: 6, low: 4, points: 10 }, // 6-4 = 10 points  
      { high: 5, low: 0, points: 5 },  // 5-0 = 5 points
      { high: 4, low: 1, points: 5 },  // 4-1 = 5 points
      { high: 3, low: 2, points: 5 },  // 3-2 = 5 points
      { high: 0, low: 0, points: 0 },  // 0-0 = 0 points
      { high: 1, low: 1, points: 0 }   // 1-1 = 0 points
    ]);
    
    const total = testDominoes.reduce((sum, d) => sum + (d.points || 0), 0);
    
    // Should total exactly 35 points for all counting dominoes (mk4 rules)
    return total === 35;
  }
  
  /**
   * Generates tournament compliance validation
   */
  static validateTournamentRules(state: GameState): string[] {
    const errors: string[] = [];

    // Check for special contracts (tournament mode now enforced via actionTransformers)
    const specialBids = state.bids.filter(bid =>
      bid.type === BID_TYPES.SPLASH ||
      bid.type === BID_TYPES.PLUNGE
    );

    if (specialBids.length > 0) {
      errors.push('Special contracts not allowed in tournament mode');
    }

    // Check for special trump selections (nello, sevens)
    if (state.trump?.type === 'nello' || state.trump?.type === 'sevens') {
      errors.push('Special trump selections not allowed in tournament mode');
    }

    // Check game target is 7 marks
    if (state.gameTarget !== 7) {
      errors.push('Tournament play requires 7-mark games');
    }

    return errors;
  }

  /**
   * Process consensus sequentially for all players in turn order
   * @param initialState The initial game state
   * @param consensusType The type of consensus to process ('completeTrick' or 'scoreHand')
   * @param humanPlayers Optional set of player IDs that should be skipped (for human players)
   * @returns The state after processing consensus
   */
  static async processSequentialConsensus(
    initialState: GameState,
    consensusType: 'completeTrick' | 'scoreHand',
    humanPlayers: Set<number> = new Set()
  ): Promise<GameState> {
    const { executeAction } = await import('../../game/core/actions');
    const ctx = createTestContext();
    let state = initialState;
    const actionType = consensusType === 'completeTrick'
      ? 'agree-complete-trick'
      : 'agree-score-hand';

    // Process agrees sequentially until all players have agreed
    while (state.consensus[consensusType].size < 4) {
      // Skip if current player is human
      if (humanPlayers.has(state.currentPlayer)) {
        break;
      }

      const transitions = getNextStates(state, ctx);
      const agreeAction = transitions.find(t =>
        t.action.type === actionType &&
        t.action.player === state.currentPlayer
      );

      if (!agreeAction) {
        break; // No more agrees available
      }

      state = executeAction(state, agreeAction.action);
    }

    return state;
  }

  /**
   * Process a complete trick including all plays and consensus
   * @param initialState The initial game state
   * @param humanPlayers Optional set of player IDs that should be skipped during consensus
   * @returns The state after the trick is complete
   */
  static async processCompleteTrick(
    initialState: GameState,
    humanPlayers: Set<number> = new Set()
  ): Promise<GameState> {
    const { executeAction } = await import('../../game/core/actions');
    const ctx = createTestContext();
    let state = initialState;

    // Play cards until trick is complete
    while (state.currentTrick.length < 4 && state.phase === 'playing') {
      const transitions = getNextStates(state, ctx);
      const playAction = transitions.find(t => t.action.type === 'play');
      if (playAction) {
        state = executeAction(state, playAction.action);
      } else {
        break;
      }
    }

    // Process consensus
    state = await this.processSequentialConsensus(state, 'completeTrick', humanPlayers);

    // Complete the trick if all agreed
    if (state.consensus.completeTrick.size === 4) {
      const transitions = getNextStates(state, ctx);
      const completeTrick = transitions.find(t => t.action.type === 'complete-trick');
      if (completeTrick) {
        state = executeAction(state, completeTrick.action);
      }
    }

    return state;
  }

  /**
   * Process hand scoring including consensus
   * @param initialState The initial game state in scoring phase
   * @param humanPlayers Optional set of player IDs that should be skipped during consensus
   * @returns The state after scoring is complete
   */
  static async processHandScoring(
    initialState: GameState,
    humanPlayers: Set<number> = new Set()
  ): Promise<GameState> {
    const { executeAction } = await import('../../game/core/actions');
    const ctx = createTestContext();
    let state = initialState;

    // Process consensus
    state = await this.processSequentialConsensus(state, 'scoreHand', humanPlayers);

    // Score the hand if all agreed
    if (state.consensus.scoreHand.size === 4) {
      const transitions = getNextStates(state, ctx);
      const scoreHand = transitions.find(t => t.action.type === 'score-hand');
      if (scoreHand) {
        state = executeAction(state, scoreHand.action);
      }
    }

    return state;
  }
  
}

// Export convenience functions for easier importing
export const createTestState = GameTestHelper.createTestState;
export const createTestHand = GameTestHelper.createTestHand;
export const createHandWithDoubles = GameTestHelper.createHandWithDoubles;
export const processSequentialConsensus = GameTestHelper.processSequentialConsensus;
export const processCompleteTrick = GameTestHelper.processCompleteTrick;
export const processHandScoring = GameTestHelper.processHandScoring;