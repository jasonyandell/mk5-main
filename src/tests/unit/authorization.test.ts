import { describe, it, expect } from 'vitest';
import { canPlayerExecuteAction, getValidActionsForPlayer, authorizeAndExecute } from '../../game/multiplayer/authorization';
import type { GameState, GameAction } from '../../game/types';
import type { MultiplayerGameState, PlayerSession } from '../../game/multiplayer/types';
import { createInitialState } from '../../game';
import { getNextPlayer } from '../../game/core/players';

describe('Authorization', () => {
  // Helper to create test multiplayer state
  function createTestMPState(gameState?: GameState): MultiplayerGameState {
    const state = gameState || createInitialState();
    const sessions: PlayerSession[] = [
      { playerId: 0, sessionId: 'session-0', type: 'human' },
      { playerId: 1, sessionId: 'session-1', type: 'ai' },
      { playerId: 2, sessionId: 'session-2', type: 'ai' },
      { playerId: 3, sessionId: 'session-3', type: 'ai' }
    ];
    return { state, sessions };
  }

  describe('canPlayerExecuteAction', () => {
    it('allows any player to execute neutral actions (no player field)', () => {
      const state = createInitialState();
      const neutralAction: GameAction = { type: 'complete-trick' };

      // All players should be able to execute neutral actions
      expect(canPlayerExecuteAction(0, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(1, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(2, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(3, neutralAction, state)).toBe(true);
    });

    it('allows only the specified player to execute player-specific actions', () => {
      const state = createInitialState();
      const player0Action: GameAction = { type: 'pass', player: 0 };

      expect(canPlayerExecuteAction(0, player0Action, state)).toBe(true);
      expect(canPlayerExecuteAction(1, player0Action, state)).toBe(false);
      expect(canPlayerExecuteAction(2, player0Action, state)).toBe(false);
      expect(canPlayerExecuteAction(3, player0Action, state)).toBe(false);
    });

    it('correctly handles bid actions', () => {
      const state = createInitialState();
      const bidAction: GameAction = { type: 'bid', player: 2, bid: 'points', value: 30 };

      expect(canPlayerExecuteAction(2, bidAction, state)).toBe(true);
      expect(canPlayerExecuteAction(0, bidAction, state)).toBe(false);
      expect(canPlayerExecuteAction(1, bidAction, state)).toBe(false);
      expect(canPlayerExecuteAction(3, bidAction, state)).toBe(false);
    });

    it('correctly handles trump selection actions', () => {
      const state = createInitialState();
      const trumpAction: GameAction = {
        type: 'select-trump',
        player: 1,
        trump: { type: 'suit', suit: 0 }
      };

      expect(canPlayerExecuteAction(1, trumpAction, state)).toBe(true);
      expect(canPlayerExecuteAction(0, trumpAction, state)).toBe(false);
      expect(canPlayerExecuteAction(2, trumpAction, state)).toBe(false);
    });

    it('correctly handles play actions', () => {
      const state = createInitialState();
      const playAction: GameAction = { type: 'play', player: 3, dominoId: '0-0' };

      expect(canPlayerExecuteAction(3, playAction, state)).toBe(true);
      expect(canPlayerExecuteAction(0, playAction, state)).toBe(false);
      expect(canPlayerExecuteAction(1, playAction, state)).toBe(false);
      expect(canPlayerExecuteAction(2, playAction, state)).toBe(false);
    });

    it('correctly handles consensus actions', () => {
      const state = createInitialState();
      const consensusAction: GameAction = { type: 'agree-complete-trick', player: 1 };

      // Consensus actions have player field but should only be executed by that player
      expect(canPlayerExecuteAction(1, consensusAction, state)).toBe(true);
      expect(canPlayerExecuteAction(0, consensusAction, state)).toBe(false);
    });
  });

  describe('getValidActionsForPlayer', () => {
    it('returns only actions that player can execute', () => {
      const state = createInitialState();
      // Get the actual current player from the state
      const currentPlayer = state.currentPlayer;
      const otherPlayer = getNextPlayer(currentPlayer);

      const currentPlayerActions = getValidActionsForPlayer(state, currentPlayer);
      const otherPlayerActions = getValidActionsForPlayer(state, otherPlayer);

      // Current player should have bidding actions
      expect(currentPlayerActions.length).toBeGreaterThan(0);
      expect(currentPlayerActions.every(a => !('player' in a) || a.player === currentPlayer)).toBe(true);

      // Other player should have no actions (not their turn)
      expect(otherPlayerActions.length).toBe(0);
    });

    it('filters out actions for other players', () => {
      const state = createInitialState();
      const currentPlayer = state.currentPlayer;

      const validActions = getValidActionsForPlayer(state, currentPlayer);

      // All actions should either be neutral or for the current player
      for (const action of validActions) {
        if ('player' in action) {
          expect(action.player).toBe(currentPlayer);
        }
      }
    });
  });

  describe('authorizeAndExecute', () => {
    it('successfully executes authorized action', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;

      // Pass action should be valid for current player
      const result = authorizeAndExecute(mpState, {
        playerId: currentPlayer,
        action: { type: 'pass', player: currentPlayer },
        sessionId: `session-${currentPlayer}`
      });

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.value.state.bids.length).toBe(1);
        expect(result.value.state.bids[0]?.type).toBe('pass');
      }
    });

    it('rejects action from wrong player', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer,
        action: { type: 'pass', player: currentPlayer }, // Try to act as different player
        sessionId: `session-${wrongPlayer}`
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('not authorized');
      }
    });

    it('rejects invalid player ID', () => {
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 99, // Invalid player ID
        action: { type: 'complete-trick' },
        sessionId: 'session-99'
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('Invalid player ID');
      }
    });

    it('rejects invalid session ID', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: currentPlayer,
        action: { type: 'pass', player: currentPlayer },
        sessionId: 'wrong-session-id'
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('Invalid session ID');
      }
    });

    it('allows action without sessionId (for local games)', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: currentPlayer,
        action: { type: 'pass', player: currentPlayer }
        // No sessionId provided
      });

      expect(result.ok).toBe(true);
    });

    it('rejects action that is not valid in current state', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;

      // Try to play a domino during bidding phase
      const result = authorizeAndExecute(mpState, {
        playerId: currentPlayer,
        action: { type: 'play', player: currentPlayer, dominoId: '0-0' },
        sessionId: `session-${currentPlayer}`
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('not valid in current game state');
      }
    });

    it('preserves sessions across state transitions', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.state.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: currentPlayer,
        action: { type: 'pass', player: currentPlayer },
        sessionId: `session-${currentPlayer}`
      });

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Sessions should be unchanged
        expect(result.value.sessions).toEqual(mpState.sessions);
      }
    });

    it('correctly handles neutral actions from any player', () => {
      // Create state where trick is complete and waiting for consensus
      const initialState = createInitialState();
      // We need to manually construct a state in playing phase with complete trick
      // For now, just test that neutral actions are allowed

      const mpState = createTestMPState();

      // Complete-trick is a neutral action
      const result = authorizeAndExecute(mpState, {
        playerId: 0, // Any player
        action: { type: 'complete-trick' }
      });

      // This will fail because trick isn't actually complete, but authorization should pass
      // The error should be about state validity, not authorization
      expect(result.ok).toBe(false);
      if (!result.ok) {
        // Should fail on validity check, not authorization
        expect(result.error).not.toContain('not authorized');
      }
    });
  });

  describe('Edge cases', () => {
    it('handles negative player IDs', () => {
      const state = createInitialState();
      const action: GameAction = { type: 'pass', player: -1 };

      expect(canPlayerExecuteAction(-1, action, state)).toBe(true);
      expect(canPlayerExecuteAction(0, action, state)).toBe(false);
    });

    it('handles player IDs >= 4', () => {
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 4,
        action: { type: 'complete-trick' }
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('Invalid player ID');
      }
    });
  });
});
