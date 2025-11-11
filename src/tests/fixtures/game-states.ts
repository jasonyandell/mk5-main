/**
 * Pre-configured game states for testing.
 *
 * These fixtures provide realistic GameView objects for common test scenarios.
 * Use these instead of creating Room instances in tests.
 */

import type { GameView, PlayerInfo, ValidAction } from '../../shared/multiplayer/protocol';
import type { GameConfig } from '../../game/types/config';
import type { GameState, FilteredGameState, Domino, Player, TrumpSelection, SuitAnalysis } from '../../game/types';
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
    { type: 'observe-hands', playerIndices: [index] }
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
      capabilities: baseCapabilities(1)
    },
    {
      playerId: 2,
      controlType: 'ai',
      connected: true,
      name: 'Player 2',
      capabilities: baseCapabilities(2)
    },
    {
      playerId: 3,
      controlType: 'ai',
      connected: true,
      name: 'Player 3',
      capabilities: baseCapabilities(3)
    },
  ];
}

/**
 * Create a GameView with sensible defaults.
 *
 * In tests, we often have full GameState objects with complete player hands.
 * This function converts them to FilteredGameState format for use in GameView.
 * By default, it shows all hands (for testing convenience). For realistic filtering,
 * use createObserverGameView() instead.
 */
function createGameView(state: GameState, validActions: ValidAction[] = []): GameView {
  // Convert GameState players to FilteredGameState players by adding handCount
  const filteredState: FilteredGameState = {
    ...state,
    players: state.players.map(player => ({
      id: player.id,
      name: player.name,
      teamId: player.teamId,
      marks: player.marks,
      hand: player.hand, // In basic test fixtures, show all hands
      handCount: player.hand.length,
      ...(player.suitAnalysis && { suitAnalysis: player.suitAnalysis }),
    })),
  };

  // Convert validActions to ViewTransitions
  const transitions = validActions.map(valid => ({
    id: JSON.stringify(valid.action),  // Simple ID for tests
    label: valid.label,
    action: valid.action,
    ...(valid.group ? { group: valid.group } : {}),
    ...(valid.recommended ? { recommended: valid.recommended } : {})
  }));

  return {
    state: filteredState,
    validActions,
    transitions,
    players: createDefaultPlayerInfo(),
    metadata: {
      gameId: 'test-game-123'
    },
  };
}

/**
 * Create a GameView from a GameState with custom player info and valid actions.
 *
 * This is useful when you need full control over the GameView structure,
 * including custom player capabilities or specific valid actions.
 *
 * @param state - The game state (will be converted to FilteredGameState)
 * @param players - Player information with capabilities
 * @param validActions - List of valid actions for the current player
 * @returns A complete GameView object for testing
 */
export function createCustomGameView(
  state: GameState,
  players: PlayerInfo[],
  validActions: ValidAction[] = []
): GameView {
  // Convert GameState to FilteredGameState by adding handCount
  const filteredState: FilteredGameState = {
    ...state,
    players: state.players.map(player => ({
      id: player.id,
      name: player.name,
      teamId: player.teamId,
      marks: player.marks,
      hand: player.hand, // Show all hands in custom views by default
      handCount: player.hand.length,
      ...(player.suitAnalysis && { suitAnalysis: player.suitAnalysis }),
    })),
  };

  // Convert validActions to ViewTransitions
  const transitions = validActions.map(valid => ({
    id: JSON.stringify(valid.action),  // Simple ID for tests
    label: valid.label,
    action: valid.action,
    ...(valid.group ? { group: valid.group } : {}),
    ...(valid.recommended ? { recommended: valid.recommended } : {})
  }));

  return {
    state: filteredState,
    validActions,
    transitions,
    players,
    metadata: {
      gameId: 'test-game-123'
    },
  };
}

/**
 * Create a filtered game state from a full GameState, hiding hands based on observer permissions.
 *
 * This simulates what the server would send to a client with limited observation capabilities.
 * Use this when you want to test realistic scenarios where a player can't see other players' hands.
 *
 * @param state - The full game state
 * @param observerPlayerIndex - Which player's perspective to use (can see own hand only)
 * @returns A FilteredGameState with hidden hands
 */
export function createFilteredState(
  state: GameState,
  observerPlayerIndex: number
): FilteredGameState {
  return {
    ...state,
    players: state.players.map((player, index) => {
      const filtered: {
        id: number;
        name: string;
        teamId: 0 | 1;
        marks: number;
        hand: Domino[];
        handCount: number;
        suitAnalysis?: SuitAnalysis;
      } = {
        id: player.id,
        name: player.name,
        teamId: player.teamId,
        marks: player.marks,
        hand: index === observerPlayerIndex ? player.hand : [], // Only show observer's own hand
        handCount: player.hand.length,
      };

      // Only add suitAnalysis if it exists and this is the observer
      if (index === observerPlayerIndex && player.suitAnalysis) {
        filtered.suitAnalysis = player.suitAnalysis;
      }

      return filtered;
    }),
  };
}

/**
 * Create a GameView with properly filtered state based on observer permissions.
 *
 * This creates a realistic GameView as the server would send it to a client.
 * The observer can only see their own hand, not other players' hands.
 *
 * @param state - The full game state
 * @param observerPlayerIndex - Which player is observing (0-3)
 * @param validActions - Valid actions for the observer
 * @returns A GameView with filtered state
 */
export function createObserverGameView(
  state: GameState,
  observerPlayerIndex: number,
  validActions: ValidAction[] = []
): GameView {
  const filteredState = createFilteredState(state, observerPlayerIndex);

  // Convert validActions to ViewTransitions
  const transitions = validActions.map(valid => ({
    id: JSON.stringify(valid.action),  // Simple ID for tests
    label: valid.label,
    action: valid.action,
    ...(valid.group ? { group: valid.group } : {}),
    ...(valid.recommended ? { recommended: valid.recommended } : {})
  }));

  return {
    state: filteredState,
    validActions,
    transitions,
    players: createDefaultPlayerInfo(),
    metadata: {
      gameId: 'test-game-123'
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

  const validActions: ValidAction[] = [
    {
      action: { type: 'pass' as const, player: 0 },
      label: 'Pass',
      recommended: false,
    },
    {
      action: { type: 'bid' as const, player: 0, bid: 'points' as const, value: 30 },
      label: 'Bid 30',
      recommended: false,
    },
    {
      action: { type: 'bid' as const, player: 0, bid: 'points' as const, value: 32 },
      label: 'Bid 32',
      recommended: true,
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

  const validActions: ValidAction[] = [
    {
      action: { type: 'pass' as const, player: 2 },
      label: 'Pass',
      recommended: false,
    },
    {
      action: { type: 'bid' as const, player: 2, bid: 'points' as const, value: 34 },
      label: 'Bid 34',
      recommended: true,
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

  const validActions: ValidAction[] = [
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'suit', suit: 0 } as TrumpSelection,
      },
      label: 'Trump: Blanks',
      recommended: false,
    },
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'suit', suit: 1 } as TrumpSelection,
      },
      label: 'Trump: Ones',
      recommended: true,
    },
    {
      action: {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'doubles' } as TrumpSelection,
      },
      label: 'Trump: Doubles',
      recommended: false,
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

  const validActions: ValidAction[] = state.players[1]!.hand.map(domino => ({
    action: { type: 'play' as const, player: 1, dominoId: String(domino.id) },
    label: `Play ${domino.id}`,
    recommended: false,
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
  const validActions: ValidAction[] = [
    {
      action: { type: 'play' as const, player: 2, dominoId: '6-3' },
      label: 'Play 6-3',
      recommended: true,
    },
    {
      action: { type: 'play' as const, player: 2, dominoId: '5-3' },
      label: 'Play 5-3',
      recommended: false,
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

  const validActions: ValidAction[] = [
    {
      action: { type: 'agree-complete-trick' as const, player: 0 },
      label: 'Complete Trick',
      recommended: true,
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

  const validActions: ValidAction[] = [
    {
      action: { type: 'agree-score-hand' as const, player: 0 },
      label: 'Score Hand',
      recommended: true,
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
