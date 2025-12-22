/**
 * Projection Security Tests
 *
 * These tests enforce security invariants for the "dumb client" pattern:
 * 1. No hidden state leaks - opponent hands are never visible
 * 2. All rule-aware fields come from server-computed DerivedViewFields
 * 3. Capability-based filtering works correctly
 * 4. FilteredGameState respects visibility boundaries
 */

import { describe, it, expect } from 'vitest';
import { getVisibleStateForSession, humanCapabilities, spectatorCapabilities, buildCapabilities } from '../../multiplayer/capabilities';
import { buildKernelView } from '../../kernel/kernel';
import { createInitialState } from '../../game/core/state';
import { createExecutionContext } from '../../game/types/execution';
import type { GameState } from '../../game/types';
import type { PlayerSession, MultiplayerGameState } from '../../multiplayer/types';

/**
 * Create test game state with known hands
 */
function createTestState(): GameState {
  const state = createInitialState();
  state.phase = 'playing';
  state.trump = { type: 'suit', suit: 3 };

  // Set up known hands for each player
  state.players[0]!.hand = [
    { id: '6-5', high: 6, low: 5 },
    { id: '6-4', high: 6, low: 4 },
    { id: '6-3', high: 6, low: 3 },
    { id: '6-2', high: 6, low: 2 },
    { id: '6-1', high: 6, low: 1 },
    { id: '6-0', high: 6, low: 0 },
    { id: '5-5', high: 5, low: 5 },
  ];

  state.players[1]!.hand = [
    { id: '5-4', high: 5, low: 4 },
    { id: '5-3', high: 5, low: 3 },
    { id: '5-2', high: 5, low: 2 },
    { id: '5-1', high: 5, low: 1 },
    { id: '5-0', high: 5, low: 0 },
    { id: '4-4', high: 4, low: 4 },
    { id: '4-3', high: 4, low: 3 },
  ];

  state.players[2]!.hand = [
    { id: '4-2', high: 4, low: 2 },
    { id: '4-1', high: 4, low: 1 },
    { id: '4-0', high: 4, low: 0 },
    { id: '3-3', high: 3, low: 3 },
    { id: '3-2', high: 3, low: 2 },
    { id: '3-1', high: 3, low: 1 },
    { id: '3-0', high: 3, low: 0 },
  ];

  state.players[3]!.hand = [
    { id: '2-2', high: 2, low: 2 },
    { id: '2-1', high: 2, low: 1 },
    { id: '2-0', high: 2, low: 0 },
    { id: '1-1', high: 1, low: 1 },
    { id: '1-0', high: 1, low: 0 },
    { id: '0-0', high: 0, low: 0 },
    { id: '6-6', high: 6, low: 6 },
  ];

  return state;
}

/**
 * Create a player session for testing
 */
function createSession(playerIndex: 0 | 1 | 2 | 3, type: 'human' | 'spectator' = 'human'): PlayerSession {
  if (type === 'spectator') {
    return {
      playerId: 'spectator',
      playerIndex: 0 as 0 | 1 | 2 | 3,
      controlType: 'human',
      capabilities: spectatorCapabilities()
    };
  }

  return {
    playerId: `player-${playerIndex}`,
    playerIndex,
    controlType: 'human',
    capabilities: humanCapabilities(playerIndex)
  };
}

describe('Projection Security: Hidden State', () => {
  describe('getVisibleStateForSession', () => {
    it('player can see their own hand', () => {
      const state = createTestState();
      const session = createSession(0);

      const visible = getVisibleStateForSession(state, session);

      // Player 0 should see their full hand
      expect(visible.players[0]!.hand).toHaveLength(7);
      expect(visible.players[0]!.handCount).toBe(7);
    });

    it('player cannot see opponent hands', () => {
      const state = createTestState();
      const session = createSession(0);

      const visible = getVisibleStateForSession(state, session);

      // Player 0 should NOT see other players' hands
      expect(visible.players[1]!.hand).toHaveLength(0);
      expect(visible.players[2]!.hand).toHaveLength(0);
      expect(visible.players[3]!.hand).toHaveLength(0);

      // But handCount is visible (public information)
      expect(visible.players[1]!.handCount).toBe(7);
      expect(visible.players[2]!.handCount).toBe(7);
      expect(visible.players[3]!.handCount).toBe(7);
    });

    it('spectator can see all hands', () => {
      const state = createTestState();
      const session = createSession(0, 'spectator');

      const visible = getVisibleStateForSession(state, session);

      // Spectator should see all hands
      expect(visible.players[0]!.hand).toHaveLength(7);
      expect(visible.players[1]!.hand).toHaveLength(7);
      expect(visible.players[2]!.hand).toHaveLength(7);
      expect(visible.players[3]!.hand).toHaveLength(7);
    });

    it('does not mutate original state', () => {
      const state = createTestState();
      const originalHand0 = [...state.players[0]!.hand];
      const session = createSession(0);

      const visible = getVisibleStateForSession(state, session);

      // Original state should be unchanged
      expect(state.players[0]!.hand).toEqual(originalHand0);
      expect(state.players[1]!.hand).toHaveLength(7);

      // Visible state changes shouldn't affect original
      visible.players[0]!.hand = [];
      expect(state.players[0]!.hand).toEqual(originalHand0);
    });

    it('player metadata is always visible', () => {
      const state = createTestState();
      state.players[1]!.name = 'Bob';
      state.players[1]!.teamId = 1;
      state.players[1]!.marks = 3;

      const session = createSession(0);
      const visible = getVisibleStateForSession(state, session);

      // Metadata is always visible
      expect(visible.players[1]!.name).toBe('Bob');
      expect(visible.players[1]!.teamId).toBe(1);
      expect(visible.players[1]!.marks).toBe(3);
      expect(visible.players[1]!.id).toBeDefined();
    });
  });

  describe('custom capabilities', () => {
    it('custom observe-hands capability works', () => {
      const state = createTestState();
      const session: PlayerSession = {
        playerId: 'custom',
        playerIndex: 0,
        controlType: 'human',
        capabilities: buildCapabilities()
          .actAsPlayer(0)
          .observeHands([0, 2])  // Can see player 0 and player 2
          .build()
      };

      const visible = getVisibleStateForSession(state, session);

      expect(visible.players[0]!.hand).toHaveLength(7);  // Own hand
      expect(visible.players[1]!.hand).toHaveLength(0);  // Cannot see
      expect(visible.players[2]!.hand).toHaveLength(7);  // Custom capability
      expect(visible.players[3]!.hand).toHaveLength(0);  // Cannot see
    });
  });
});

describe('Projection Security: Derived Fields', () => {
  function createMultiplayerState(): MultiplayerGameState {
    return {
      gameId: 'test-game',
      coreState: createTestState(),
      players: [
        createSession(0),
        createSession(1),
        createSession(2),
        createSession(3),
      ]
    };
  }

  it('buildKernelView includes derived fields', () => {
    const mpState = createMultiplayerState();
    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

    const view = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });

    expect(view.derived).toBeDefined();
    expect(view.derived.currentTrickWinner).toBeDefined();
    expect(view.derived.handDominoMeta).toBeDefined();
    expect(view.derived.currentHandPoints).toBeDefined();
  });

  it('handDominoMeta contains isTrump from rules', () => {
    const mpState = createMultiplayerState();
    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

    const view = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });

    // Player 0's hand has 6-3 which is trump (trump is suit 3)
    const d63Meta = view.derived.handDominoMeta.find(m => m.dominoId === '6-3');
    expect(d63Meta?.isTrump).toBe(true);

    // 6-5 is not trump
    const d65Meta = view.derived.handDominoMeta.find(m => m.dominoId === '6-5');
    expect(d65Meta?.isTrump).toBe(false);
  });

  it('handDominoMeta contains canFollow from rules', () => {
    const mpState = createMultiplayerState();
    // Set up a trick with led suit
    mpState.coreState.currentTrick = [
      { player: 3, domino: { id: '6-6', high: 6, low: 6 } }
    ];
    mpState.coreState.currentSuit = 6;  // Sixes led
    mpState.coreState.currentPlayer = 0;

    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });
    const view = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });

    // Player 0 has many sixes (6-5, 6-4, 6-3 (trump), 6-2, 6-1, 6-0)
    // 6-5 can follow sixes
    const d65Meta = view.derived.handDominoMeta.find(m => m.dominoId === '6-5');
    expect(d65Meta?.canFollow).toBe(true);

    // 5-5 cannot follow sixes (it's a double, no 6 pip)
    const d55Meta = view.derived.handDominoMeta.find(m => m.dominoId === '5-5');
    expect(d55Meta?.canFollow).toBe(false);
  });

  it('derived fields match what rules would compute', () => {
    const mpState = createMultiplayerState();
    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

    const view = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });

    // Verify each domino's isTrump matches rules.isTrump
    const rules = ctx.rules;
    for (const meta of view.derived.handDominoMeta) {
      const [high, low] = meta.dominoId.split('-').map(Number);
      const domino = { id: meta.dominoId, high: high!, low: low! };
      const expectedIsTrump = rules.isTrump(mpState.coreState, domino);
      expect(meta.isTrump).toBe(expectedIsTrump);
    }
  });
});

describe('Projection Security: No State Leaks in Transitions', () => {
  function createMultiplayerState(): MultiplayerGameState {
    const state = createTestState();
    state.currentPlayer = 0;
    state.currentTrick = [];

    return {
      gameId: 'test-game',
      coreState: state,
      players: [
        createSession(0),
        createSession(1),
        createSession(2),
        createSession(3),
      ]
    };
  }

  it('transitions do not leak opponent hands', () => {
    const mpState = createMultiplayerState();
    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

    const view = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });

    // Check that transitions don't contain domino info from other players
    for (const transition of view.transitions) {
      const action = transition.action;

      // Play actions should only have dominoId from player 0's hand
      if (action.type === 'play' && 'dominoId' in action) {
        const player0DominoIds = view.state.players[0]!.hand.map(d => `${d.high}-${d.low}`);
        expect(player0DominoIds).toContain(action.dominoId);
      }

      // Actions shouldn't have hidden opponent information
      if ('meta' in action && action.meta) {
        const meta = action.meta as Record<string, unknown>;
        // Ensure no opponent hand data in meta
        expect(meta).not.toHaveProperty('opponentHand');
        expect(meta).not.toHaveProperty('allHands');
      }
    }
  });

  it('validActions are filtered per session', () => {
    const mpState = createMultiplayerState();
    const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

    const view0 = buildKernelView(mpState, 'player-0', ctx, { gameId: 'test-game' });
    const view1 = buildKernelView(mpState, 'player-1', ctx, { gameId: 'test-game' });

    // Player 0 (current player) should have play actions
    const play0Actions = view0.validActions.filter(a => a.action.type === 'play');
    expect(play0Actions.length).toBeGreaterThan(0);

    // Player 1 (not current) should not have play actions
    const play1Actions = view1.validActions.filter(a => a.action.type === 'play');
    expect(play1Actions.length).toBe(0);
  });
});

describe('Projection Security: Filtered State Invariants', () => {
  it('FilteredGameState always has handCount for all players', () => {
    const state = createTestState();
    const session = createSession(0);

    const visible = getVisibleStateForSession(state, session);

    for (let i = 0; i < 4; i++) {
      expect(visible.players[i]!.handCount).toBeDefined();
      expect(typeof visible.players[i]!.handCount).toBe('number');
    }
  });

  it('public game state is always visible', () => {
    const state = createTestState();
    state.teamMarks = [3, 2];
    state.tricks = [
      {
        plays: [
          { player: 0, domino: { id: '6-5', high: 6, low: 5 } },
          { player: 1, domino: { id: '5-4', high: 5, low: 4 } },
          { player: 2, domino: { id: '4-3', high: 4, low: 3 } },
          { player: 3, domino: { id: '3-2', high: 3, low: 2 } },
        ],
        winner: 0,
        points: 5
      }
    ];
    state.currentBid = { type: 'points', value: 30, player: 0 };

    const session = createSession(1);  // Non-bidder viewing
    const visible = getVisibleStateForSession(state, session);

    // Public state is always visible
    expect(visible.teamMarks).toEqual([3, 2]);
    expect(visible.tricks).toHaveLength(1);
    expect(visible.tricks[0]!.winner).toBe(0);
    expect(visible.currentBid).toEqual(state.currentBid);
    expect(visible.phase).toBe('playing');
  });

  it('trump selection is visible to all', () => {
    const state = createTestState();
    state.trump = { type: 'suit', suit: 4 };

    const session = createSession(2);
    const visible = getVisibleStateForSession(state, session);

    expect(visible.trump).toEqual({ type: 'suit', suit: 4 });
  });
});

describe('Projection Security: Edge Cases', () => {
  it('empty hand is correctly filtered (not undefined)', () => {
    const state = createTestState();
    state.players[1]!.hand = [];  // Empty hand

    const session = createSession(0);
    const visible = getVisibleStateForSession(state, session);

    // Player 1's hand should be empty array, not undefined
    expect(Array.isArray(visible.players[1]!.hand)).toBe(true);
    expect(visible.players[1]!.hand).toHaveLength(0);
    expect(visible.players[1]!.handCount).toBe(0);
  });

  it('session without act-as-player capability cannot see own hand', () => {
    const state = createTestState();
    const session: PlayerSession = {
      playerId: 'observer',
      playerIndex: 0,
      controlType: 'human',
      capabilities: []  // No capabilities at all
    };

    const visible = getVisibleStateForSession(state, session);

    // With no observe-hands capability, should not see any hands
    expect(visible.players[0]!.hand).toHaveLength(0);
    expect(visible.players[1]!.hand).toHaveLength(0);
    expect(visible.players[2]!.hand).toHaveLength(0);
    expect(visible.players[3]!.hand).toHaveLength(0);
  });

  it('malformed session capabilities are handled gracefully', () => {
    const state = createTestState();
    const session: PlayerSession = {
      playerId: 'test',
      playerIndex: 0,
      controlType: 'human',
      capabilities: [
        { type: 'observe-hands', playerIndices: [] }  // Empty array
      ]
    };

    // Should not throw
    expect(() => getVisibleStateForSession(state, session)).not.toThrow();

    const visible = getVisibleStateForSession(state, session);

    // Empty playerIndices means no hands visible
    expect(visible.players[0]!.hand).toHaveLength(0);
  });
});
