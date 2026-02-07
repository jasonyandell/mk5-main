import { describe, it, expect } from 'vitest';
import { canPlayerExecuteAction, authorizeAndExecute } from '../../multiplayer/authorization';
import type { GameState, GameAction } from '../../game/types';
import type { MultiplayerGameState, PlayerSession } from '../../multiplayer/types';
import { createInitialState, executeAction } from '../../game';
import { getNextPlayer } from '../../game/core/players';
import { createTestContext } from '../helpers/executionContext';
import { StateBuilder } from '../helpers/stateBuilder';
import { composeRules, baseLayer } from '../../game/layers';

describe('Authorization', () => {
  // Helper to create test sessions
  function createTestSessions(): PlayerSession[] {
    return [
      {
        playerId: 'player-0',
        playerIndex: 0,
        controlType: 'human',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 0 },
          { type: 'observe-hands' as const, playerIndices: [0] }
        ]
      },
      {
        playerId: 'ai-1',
        playerIndex: 1,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 1 },
          { type: 'observe-hands' as const, playerIndices: [1] }
        ]
      },
      {
        playerId: 'ai-2',
        playerIndex: 2,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 2 },
          { type: 'observe-hands' as const, playerIndices: [2] }
        ]
      },
      {
        playerId: 'ai-3',
        playerIndex: 3,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 3 },
          { type: 'observe-hands' as const, playerIndices: [3] }
        ]
      }
    ];
  }

  // Helper to create test multiplayer state
  function createTestMPState(gameState?: GameState): MultiplayerGameState {
    const state = gameState || createInitialState();
    const sessions = createTestSessions();

    return {
      gameId: 'test-game',
      coreState: state,
      players: sessions
    };
  }

  describe('canPlayerExecuteAction', () => {
    it('allows any player to execute neutral actions (no player field)', () => {
      // Use all AI players so consensus layer passes through (no gating)
      const ctx = createTestContext({ playerTypes: ['ai', 'ai', 'ai', 'ai'] });
      // Create playing phase with complete trick (all AI players)
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: 0 })
        .withConfig({ playerTypes: ['ai', 'ai', 'ai', 'ai'] })
        .withCurrentTrick([
          { player: 0, domino: '0-0' },
          { player: 1, domino: '0-1' },
          { player: 2, domino: '0-2' },
          { player: 3, domino: '0-3' }
        ])
        .build();
      const sessions = createTestSessions();
      const neutralAction: GameAction = { type: 'complete-trick' };

      // All players should be able to execute neutral actions (no player field)
      expect(canPlayerExecuteAction(sessions[0]!, neutralAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[1]!, neutralAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[2]!, neutralAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[3]!, neutralAction, state, ctx)).toBe(true);
    });

    it('allows only session with act-as-player capability to execute player-specific actions', () => {
      const ctx = createTestContext();
      const state = createInitialState();
      const sessions = createTestSessions();
      const player0Action: GameAction = { type: 'pass', player: 0 };

      expect(canPlayerExecuteAction(sessions[0]!, player0Action, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[1]!, player0Action, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, player0Action, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[3]!, player0Action, state, ctx)).toBe(false);
    });

    it('correctly handles bid actions with capabilities', () => {
      const ctx = createTestContext();
      const rules = composeRules([baseLayer]);
      // Advance state so player 2 is current player
      let state = createInitialState();
      state = executeAction(state, { type: 'pass', player: 0 }, rules);
      state = executeAction(state, { type: 'pass', player: 1 }, rules);
      // Now player 2 is current player in bidding phase

      const sessions = createTestSessions();
      const bidAction: GameAction = { type: 'bid', player: 2, bid: 'points', value: 30 };

      expect(canPlayerExecuteAction(sessions[2]!, bidAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, bidAction, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[1]!, bidAction, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[3]!, bidAction, state, ctx)).toBe(false);
    });

    it('correctly handles trump selection actions with capabilities', () => {
      const ctx = createTestContext();
      const rules = composeRules([baseLayer]);
      // Complete bidding with player 1 as winner, advance to trump_selection
      let state = createInitialState();
      state = executeAction(state, { type: 'pass', player: 0 }, rules);
      state = executeAction(state, { type: 'bid', player: 1, bid: 'points', value: 30 }, rules);
      state = executeAction(state, { type: 'pass', player: 2 }, rules);
      state = executeAction(state, { type: 'pass', player: 3 }, rules);
      // Now in trump_selection phase with player 1 selecting

      const sessions = createTestSessions();
      const trumpAction: GameAction = {
        type: 'select-trump',
        player: 1,
        trump: { type: 'suit', suit: 0 }
      };

      expect(canPlayerExecuteAction(sessions[1]!, trumpAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, trumpAction, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, trumpAction, state, ctx)).toBe(false);
    });

    it('correctly handles play actions with capabilities', () => {
      const ctx = createTestContext();
      // Create playing phase with player 3 as current player and domino in hand
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: 0 })
        .withCurrentPlayer(3)
        .withPlayerHand(3, ['0-0', '1-1'])
        .build();
      const sessions = createTestSessions();
      const playAction: GameAction = { type: 'play', player: 3, dominoId: '0-0' };

      expect(canPlayerExecuteAction(sessions[3]!, playAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, playAction, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[1]!, playAction, state, ctx)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, playAction, state, ctx)).toBe(false);
    });

    it('correctly handles consensus actions with capabilities', () => {
      // Player 1 must be human to have an agree action (consensus only gates humans)
      const ctx = createTestContext({ layers: ['consensus'], playerTypes: ['human', 'human', 'ai', 'ai'] });
      // Create playing phase with complete trick so consensus action is valid
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: 0 })
        .withConfig({ playerTypes: ['human', 'human', 'ai', 'ai'] })
        .withCurrentPlayer(1)
        .withCurrentTrick([
          { player: 0, domino: '0-0' },
          { player: 1, domino: '0-1' },
          { player: 2, domino: '0-2' },
          { player: 3, domino: '0-3' }
        ])
        .build();
      const sessions = createTestSessions();
      const consensusAction: GameAction = { type: 'agree-trick', player: 1 };

      // Consensus actions have player field - only player with capability can execute
      expect(canPlayerExecuteAction(sessions[1]!, consensusAction, state, ctx)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, consensusAction, state, ctx)).toBe(false);
    });
  });

  describe('authorizeAndExecute', () => {
    it('successfully executes authorized action', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      // Pass action should be valid for current player
      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer },
      }, ctx);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.value.coreState.bids.length).toBe(1);
        expect(result.value.coreState.bids[0]?.type).toBe('pass');
      }
    });

    it('rejects action from wrong player', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer === 0 ? 'player-0' : `ai-${wrongPlayer}`,
        action: { type: 'pass', player: currentPlayer }, // Try to act as different player
      }, ctx);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('lacks capability');
      }
    });

    it('rejects invalid player ID', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 'invalid-player-99', // Invalid player ID
        action: { type: 'complete-trick' },
      }, ctx);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('No player found');
      }
    });

    it('allows action without sessionId (for local games)', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer },
      }, ctx);

      expect(result.success).toBe(true);
    });

    it('rejects action that is not valid in current state', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      // Try to play a domino during bidding phase
      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'play', player: currentPlayer, dominoId: '0-0' },
      }, ctx);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('not valid in current game state');
      }
    });

    it('preserves sessions across state transitions', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer },
      }, ctx);

      expect(result.success).toBe(true);
      if (result.success) {
        // Players should be unchanged
        expect(result.value.players).toEqual(mpState.players);
      }
    });

    it('correctly handles neutral actions from any player', () => {
      const ctx = createTestContext();
      // Create state where trick is complete and waiting for consensus
      // We need to manually construct a state in playing phase with complete trick
      // For now, just test that neutral actions are allowed

      const mpState = createTestMPState();

      // Complete-trick is a neutral action
      const result = authorizeAndExecute(mpState, {
        playerId: 'player-0', // Any player
        action: { type: 'complete-trick' },
      }, ctx);

      // This will fail because trick isn't actually complete, but authorization should pass
      // The error should be about state validity, not authorization
      expect(result.success).toBe(false);
      if (!result.success) {
        // Should fail on validity check, not authorization
        expect(result.error).not.toContain('not authorized');
      }
    });
  });

  describe('Edge cases', () => {
    it('handles invalid player IDs', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 'player-99',
        action: { type: 'complete-trick' },
      }, ctx);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('No player found');
      }
    });
  });

  describe('System authority', () => {
    it('executes action with system authority without capability checks', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      // Create an action that wrong player would normally not be able to execute
      // But with system authority, it should bypass capability checks
      const actionWithSystemAuthority: GameAction = {
        type: 'pass',
        player: currentPlayer,
        meta: { authority: 'system' }
      };

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer === 0 ? 'player-0' : `ai-${wrongPlayer}`,
        action: actionWithSystemAuthority,
      }, ctx);

      // Should succeed despite wrong player because of system authority
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.value.coreState.bids.length).toBe(1);
        expect(result.value.coreState.bids[0]?.type).toBe('pass');
      }
    });

    it('system authority action still requires structural validity', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      // Try to play a domino during bidding phase (structurally invalid)
      const invalidActionWithSystemAuthority: GameAction = {
        type: 'play',
        player: currentPlayer,
        dominoId: '0-0',
        meta: { authority: 'system' }
      };

      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: invalidActionWithSystemAuthority,
      }, ctx);

      // Should fail because action is not valid in current game state
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('not valid in current game state');
      }
    });

    it('action without system authority still requires capabilities', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      // Regular action without system authority
      const regularAction: GameAction = {
        type: 'pass',
        player: currentPlayer
      };

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer === 0 ? 'player-0' : `ai-${wrongPlayer}`,
        action: regularAction,
      }, ctx);

      // Should fail due to lack of capability
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('lacks capability');
      }
    });

    it('action with player authority requires capabilities', () => {
      const ctx = createTestContext();
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      // Action with explicit player authority
      const actionWithPlayerAuthority: GameAction = {
        type: 'pass',
        player: currentPlayer,
        meta: { authority: 'player' }
      };

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer === 0 ? 'player-0' : `ai-${wrongPlayer}`,
        action: actionWithPlayerAuthority,
      }, ctx);

      // Should fail due to lack of capability
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error).toContain('lacks capability');
      }
    });
  });
});
