/**
 * Pre-configured game states for testing.
 *
 * These fixtures provide realistic GameView objects for common test scenarios.
 * Use these instead of creating GameHost instances in tests.
 */

import type { GameView, PlayerInfo } from '../../shared/multiplayer/protocol';
import type { GameConfig } from '../../game/types/config';
import type { GameState, Domino, Player, TrumpSelection } from '../../game/types';
import type { Capability } from '../../game/multiplayer/types';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a base GameState with sensible defaults.
 */
function createBaseState(overrides: Partial<GameState> = {}): GameState {
  const config: GameConfig = {
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    shuffleSeed: 12345,
  };

  return {
    initialConfig: config,
    theme: 'business',
    colorOverrides: {},
    phase: 'setup',
    players: createDefaultPlayers(),
    currentPlayer: 0,
    dealer: 0,
    bids: [],
    currentBid: { type: 'pass', player: -1 },
    winningBidder: -1,
    trump: { type: 'not-selected' },
    tricks: [],
    currentTrick: [],
    currentSuit: -1,
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: 7,
    shuffleSeed: 12345,
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    consensus: {
      completeTrick: new Set(),
      scoreHand: new Set(),
    },
    actionHistory: [],
    ...overrides,
  };
}

/**
 * Create default players with sensible hands.
 */
function createDefaultPlayers(): Player[] {
  return [
    {
      id: 0,
      name: 'Player 0',
      hand: createHand([
        '6-6', '6-5', '6-4', '5-5', '5-4', '4-4', '3-3',
      ]),
      teamId: 0,
      marks: 0,
    },
    {
      id: 1,
      name: 'Player 1',
      hand: createHand([
        '6-3', '6-2', '5-3', '5-2', '4-3', '3-2', '2-2',
      ]),
      teamId: 1,
      marks: 0,
    },
    {
      id: 2,
      name: 'Player 2',
      hand: createHand([
        '6-1', '6-0', '5-1', '5-0', '4-2', '3-1', '2-1',
      ]),
      teamId: 0,
      marks: 0,
    },
    {
      id: 3,
      name: 'Player 3',
      hand: createHand([
        '4-1', '4-0', '3-0', '2-0', '1-1', '1-0', '0-0',
      ]),
      teamId: 1,
      marks: 0,
    },
  ];
}

/**
 * Create dominoes from string representations.
 */
function createHand(dominoIds: string[]): Domino[] {
  return dominoIds.map(id => {
    const [high, low] = id.split('-').map(Number);
    return {
      id,
      high: high!,
      low: low!,
    };
  });
}

/**
 * Create default player info.
 */
function createDefaultPlayerInfo(): PlayerInfo[] {
  const baseCapabilities = (index: number): Capability[] => ([
    { type: 'act-as-player', playerIndex: index as 0 | 1 | 2 | 3 },
    { type: 'observe-own-hand' }
  ]);

  return [
    {
      playerId: 0,
      controlType: 'human',
      connected: true,
      name: 'Player 0',
      capabilities: baseCapabilities(0)
    },
    {
      playerId: 1,
      controlType: 'ai',
      connected: true,
      name: 'Player 1',
      capabilities: [...baseCapabilities(1), { type: 'replace-ai' }]
    },
    {
      playerId: 2,
      controlType: 'ai',
      connected: true,
      name: 'Player 2',
      capabilities: [...baseCapabilities(2), { type: 'replace-ai' }]
    },
    {
      playerId: 3,
      controlType: 'ai',
      connected: true,
      name: 'Player 3',
      capabilities: [...baseCapabilities(3), { type: 'replace-ai' }]
    },
  ];
}

/**
 * Create a GameView with sensible defaults.
 */
function createGameView(state: GameState, validActions: any[] = []): GameView {
  return {
    state,
    validActions,
    players: createDefaultPlayerInfo(),
    metadata: {
      gameId: 'test-game-123',
      created: Date.now(),
      lastUpdate: Date.now(),
    },
  };
}

// ============================================================================
// Common Test Fixtures
// ============================================================================

/**
 * Bidding phase - start of game, player 0's turn to bid.
 */
export function createBiddingPhaseView(): GameView {
  const state = createBaseState({
    phase: 'bidding',
    currentPlayer: 0,
    dealer: 0,
  });

  const validActions = [
    {
      action: { type: 'pass' as const, player: 0 },
      description: 'Pass',
      isRecommended: false,
    },
    {
      action: { type: 'bid' as const, player: 0, bid: 'points' as const, value: 30 },
      description: 'Bid 30',
      isRecommended: false,
    },
    {
      action: { type: 'bid' as const, player: 0, bid: 'points' as const, value: 32 },
      description: 'Bid 32',
      isRecommended: true,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Bidding phase - after some bids, player 2's turn.
 */
export function createMidBiddingPhaseView(): GameView {
  const state = createBaseState({
    phase: 'bidding',
    currentPlayer: 2,
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'points', player: 1, value: 32 },
    ],
    currentBid: { type: 'points', player: 1, value: 32 },
  });

  const validActions = [
    {
      action: { type: 'pass' as const, player: 2 },
      description: 'Pass',
      isRecommended: false,
    },
    {
      action: { type: 'bid' as const, player: 2, bid: 'points' as const, value: 34 },
      description: 'Bid 34',
      isRecommended: true,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Trump selection phase - player won bid and must select trump.
 */
export function createTrumpSelectionPhaseView(): GameView {
  const state = createBaseState({
    phase: 'trump_selection',
    currentPlayer: 0,
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    currentBid: { type: 'points', player: 0, value: 30 },
    winningBidder: 0,
  });

  const validActions = [
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'suit', suit: 0 } as TrumpSelection,
      },
      description: 'Trump: Blanks',
      isRecommended: false,
    },
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'suit', suit: 1 } as TrumpSelection,
      },
      description: 'Trump: Ones',
      isRecommended: true,
    },
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'doubles' } as TrumpSelection,
      },
      description: 'Trump: Doubles',
      isRecommended: false,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Playing phase - start of first trick, dealer leads.
 */
export function createPlayingPhaseView(): GameView {
  const state = createBaseState({
    phase: 'playing',
    currentPlayer: 1, // Player after dealer
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    currentBid: { type: 'points', player: 0, value: 30 },
    winningBidder: 0,
    trump: { type: 'suit', suit: 1 }, // Ones are trump
    tricks: [],
    currentTrick: [],
    currentSuit: -1,
  });

  const validActions = state.players[1]!.hand.map(domino => ({
    action: { type: 'play' as const, player: 1, dominoId: domino.id },
    description: `Play ${domino.id}`,
    isRecommended: false,
  }));

  return createGameView(state, validActions);
}

/**
 * Playing phase - middle of trick, must follow suit.
 */
export function createMidTrickPhaseView(): GameView {
  const state = createBaseState({
    phase: 'playing',
    currentPlayer: 2,
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    currentBid: { type: 'points', player: 0, value: 30 },
    winningBidder: 0,
    trump: { type: 'suit', suit: 1 },
    tricks: [],
    currentTrick: [
      { player: 1, domino: { id: '6-3', high: 6, low: 3 } },
    ],
    currentSuit: 3, // Threes are led
  });

  // Only dominoes with 3 are valid (follow suit)
  const validActions = [
    {
      action: { type: 'play' as const, player: 2, dominoId: '6-3' },
      description: 'Play 6-3',
      isRecommended: true,
    },
    {
      action: { type: 'play' as const, player: 2, dominoId: '5-3' },
      description: 'Play 5-3',
      isRecommended: false,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Playing phase - end of trick, needs consensus to complete.
 */
export function createTrickCompletionPhaseView(): GameView {
  const state = createBaseState({
    phase: 'playing',
    currentPlayer: 0, // Can be anyone
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    currentBid: { type: 'points', player: 0, value: 30 },
    winningBidder: 0,
    trump: { type: 'suit', suit: 1 },
    tricks: [],
    currentTrick: [
      { player: 1, domino: { id: '6-3', high: 6, low: 3 } },
      { player: 2, domino: { id: '5-3', high: 5, low: 3 } },
      { player: 3, domino: { id: '4-3', high: 4, low: 3 } },
      { player: 0, domino: { id: '3-2', high: 3, low: 2 } },
    ],
    currentSuit: 3,
    consensus: {
      completeTrick: new Set([0]), // Player 0 already agreed
      scoreHand: new Set(),
    },
  });

  const validActions = [
    {
      action: { type: 'agree-complete-trick' as const, player: 0 },
      description: 'Complete Trick',
      isRecommended: true,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Scoring phase - hand complete, needs consensus to score.
 */
export function createScoringPhaseView(): GameView {
  const state = createBaseState({
    phase: 'scoring',
    currentPlayer: 0,
    dealer: 0,
    bids: [
      { type: 'points', player: 0, value: 30 },
      { type: 'pass', player: 1 },
      { type: 'pass', player: 2 },
      { type: 'pass', player: 3 },
    ],
    currentBid: { type: 'points', player: 0, value: 30 },
    winningBidder: 0,
    trump: { type: 'suit', suit: 1 },
    tricks: [
      // Mock complete tricks
      { plays: [], winner: 0, points: 10, ledSuit: 3 },
      { plays: [], winner: 1, points: 5, ledSuit: 2 },
      { plays: [], winner: 0, points: 15, ledSuit: 1 },
    ],
    currentTrick: [],
    teamScores: [30, 12],
    consensus: {
      completeTrick: new Set(),
      scoreHand: new Set([0]), // Player 0 already agreed
    },
  });

  const validActions = [
    {
      action: { type: 'agree-score-hand' as const, player: 0 },
      description: 'Score Hand',
      isRecommended: true,
    },
  ];

  return createGameView(state, validActions);
}

/**
 * Game end phase - one team reached target.
 */
export function createGameEndPhaseView(): GameView {
  const state = createBaseState({
    phase: 'game_end',
    currentPlayer: 0,
    dealer: 0,
    teamScores: [7, 3],
    teamMarks: [7, 3],
  });

  return createGameView(state, []);
}

// ============================================================================
// Sequence Fixtures (for state progression tests)
// ============================================================================

/**
 * Complete bidding round sequence (4 states).
 */
export function createBiddingSequence(): GameView[] {
  return [
    createBiddingPhaseView(), // Player 0 bids
    createMidBiddingPhaseView(), // Player 2 responds
    // Add more states as needed for complete sequence
  ];
}

/**
 * Complete trick sequence (4 plays + completion).
 */
export function createTrickSequence(): GameView[] {
  return [
    createPlayingPhaseView(), // Lead play
    createMidTrickPhaseView(), // Follow suit
    createTrickCompletionPhaseView(), // Complete
  ];
}

/**
 * Full hand sequence (bidding → trump → playing → scoring).
 */
export function createFullHandSequence(): GameView[] {
  return [
    createBiddingPhaseView(),
    createTrumpSelectionPhaseView(),
    createPlayingPhaseView(),
    createScoringPhaseView(),
  ];
}

// ============================================================================
// Error/Edge Case Fixtures
// ============================================================================

/**
 * Empty valid actions (no legal moves - shouldn't happen but useful for testing).
 */
export function createNoValidActionsView(): GameView {
  const state = createBaseState({
    phase: 'playing',
  });

  return createGameView(state, []);
}

/**
 * State with empty hands (end of game).
 */
export function createEmptyHandsView(): GameView {
  const baseState = createBaseState({
    phase: 'scoring',
  });

  const state = {
    ...baseState,
    players: baseState.players.map(p => ({ ...p, hand: [] })),
  };

  return createGameView(state, []);
}
