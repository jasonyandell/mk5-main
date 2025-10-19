import { describe, it, expect } from 'vitest';
import { canPlayerExecuteAction, authorizeAndExecute } from '../../game/multiplayer/authorization';
import type { GameState, GameAction } from '../../game/types';
import type { MultiplayerGameState, PlayerSession } from '../../game/multiplayer/types';
import { createInitialState } from '../../game';
import { getNextPlayer } from '../../game/core/players';

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
          { type: 'observe-own-hand' as const }
        ]
      },
      {
        playerId: 'ai-1',
        playerIndex: 1,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 1 },
          { type: 'observe-own-hand' as const },
          { type: 'replace-ai' as const }
        ]
      },
      {
        playerId: 'ai-2',
        playerIndex: 2,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 2 },
          { type: 'observe-own-hand' as const },
          { type: 'replace-ai' as const }
        ]
      },
      {
        playerId: 'ai-3',
        playerIndex: 3,
        controlType: 'ai',
        capabilities: [
          { type: 'act-as-player' as const, playerIndex: 3 },
          { type: 'observe-own-hand' as const },
          { type: 'replace-ai' as const }
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
      players: sessions,
      createdAt: Date.now(),
      lastActionAt: Date.now(),
      enabledVariants: []
    };
  }

  describe('canPlayerExecuteAction', () => {
    it('allows any player to execute neutral actions (no player field)', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const neutralAction: GameAction = { type: 'complete-trick' };

      // All players should be able to execute neutral actions
      expect(canPlayerExecuteAction(sessions[0]!, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[1]!, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[2]!, neutralAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[3]!, neutralAction, state)).toBe(true);
    });

    it('allows only session with act-as-player capability to execute player-specific actions', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const player0Action: GameAction = { type: 'pass', player: 0 };

      expect(canPlayerExecuteAction(sessions[0]!, player0Action, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[1]!, player0Action, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, player0Action, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[3]!, player0Action, state)).toBe(false);
    });

    it('correctly handles bid actions with capabilities', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const bidAction: GameAction = { type: 'bid', player: 2, bid: 'points', value: 30 };

      expect(canPlayerExecuteAction(sessions[2]!, bidAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, bidAction, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[1]!, bidAction, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[3]!, bidAction, state)).toBe(false);
    });

    it('correctly handles trump selection actions with capabilities', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const trumpAction: GameAction = {
        type: 'select-trump',
        player: 1,
        trump: { type: 'suit', suit: 0 }
      };

      expect(canPlayerExecuteAction(sessions[1]!, trumpAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, trumpAction, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, trumpAction, state)).toBe(false);
    });

    it('correctly handles play actions with capabilities', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const playAction: GameAction = { type: 'play', player: 3, dominoId: '0-0' };

      expect(canPlayerExecuteAction(sessions[3]!, playAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, playAction, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[1]!, playAction, state)).toBe(false);
      expect(canPlayerExecuteAction(sessions[2]!, playAction, state)).toBe(false);
    });

    it('correctly handles consensus actions with capabilities', () => {
      const state = createInitialState();
      const sessions = createTestSessions();
      const consensusAction: GameAction = { type: 'agree-complete-trick', player: 1 };

      // Consensus actions have player field - only player with capability can execute
      expect(canPlayerExecuteAction(sessions[1]!, consensusAction, state)).toBe(true);
      expect(canPlayerExecuteAction(sessions[0]!, consensusAction, state)).toBe(false);
    });
  });

  describe('authorizeAndExecute', () => {
    it('successfully executes authorized action', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      // Pass action should be valid for current player
      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer }
      });

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.value.coreState.bids.length).toBe(1);
        expect(result.value.coreState.bids[0]?.type).toBe('pass');
      }
    });

    it('rejects action from wrong player', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;
      const wrongPlayer = getNextPlayer(currentPlayer);

      const result = authorizeAndExecute(mpState, {
        playerId: wrongPlayer === 0 ? 'player-0' : `ai-${wrongPlayer}`,
        action: { type: 'pass', player: currentPlayer } // Try to act as different player
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('not authorized');
      }
    });

    it('rejects invalid player ID', () => {
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 'invalid-player-99', // Invalid player ID
        action: { type: 'complete-trick' }
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('No player found');
      }
    });

    it('allows action without sessionId (for local games)', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer }
      });

      expect(result.ok).toBe(true);
    });

    it('rejects action that is not valid in current state', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      // Try to play a domino during bidding phase
      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'play', player: currentPlayer, dominoId: '0-0' }
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('not valid in current game state');
      }
    });

    it('preserves sessions across state transitions', () => {
      const mpState = createTestMPState();
      const currentPlayer = mpState.coreState.currentPlayer;

      const result = authorizeAndExecute(mpState, {
        playerId: `player-${currentPlayer}`,
        action: { type: 'pass', player: currentPlayer }
      });

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Players should be unchanged
        expect(result.value.players).toEqual(mpState.players);
      }
    });

    it('correctly handles neutral actions from any player', () => {
      // Create state where trick is complete and waiting for consensus
      // We need to manually construct a state in playing phase with complete trick
      // For now, just test that neutral actions are allowed

      const mpState = createTestMPState();

      // Complete-trick is a neutral action
      const result = authorizeAndExecute(mpState, {
        playerId: 'player-0', // Any player
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
    it('handles invalid player IDs', () => {
      const mpState = createTestMPState();

      const result = authorizeAndExecute(mpState, {
        playerId: 'player-99',
        action: { type: 'complete-trick' }
      });

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toContain('No player found');
      }
    });
  });
});
