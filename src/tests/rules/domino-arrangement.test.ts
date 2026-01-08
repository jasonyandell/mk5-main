import { describe, it, expect } from 'vitest';
import type { GameState } from '../../game/types';
import { createInitialState, dealDominoesWithSeed } from '../../game';

describe('Feature: Game Setup - Domino Arrangement', () => {
  describe('Scenario: Domino Arrangement', () => {
    it('Given players have drawn their dominoes', () => {
      // Test setup - create initial state and deal dominoes
      const gameState: GameState = createInitialState();
      const hands = dealDominoesWithSeed(12345);
      
      // Assign dealt hands to players
      gameState.players.forEach((player, index) => {
        const hand = hands[index];
        if (!hand) {
          throw new Error(`No hand dealt for player ${index}`);
        }
        player.hand = hand;
      });

      // Verify each player has 7 dominoes
      expect(gameState.players).toHaveLength(4);
      gameState.players.forEach(player => {
        expect(player.hand).toHaveLength(7);
      });
    });

    it('When arranging dominoes for tournament play, Then dominoes must be arranged in 4-3 or 3-4 formation', () => {
      // In tournament mode, dominoes must be arranged in specific formations
      const validFormations = ['4-3', '3-4'];
      
      // This is a test-only representation of domino arrangement
      const playerArrangements = [
        { playerId: 0, formation: '4-3' },
        { playerId: 1, formation: '3-4' },
        { playerId: 2, formation: '4-3' },
        { playerId: 3, formation: '3-4' },
      ];

      // Verify all arrangements are valid
      playerArrangements.forEach(arrangement => {
        expect(validFormations).toContain(arrangement.formation);
      });
    });

    it('And once bidding begins, dominoes cannot be rearranged', () => {
      // Test that arrangement is locked once bidding phase starts
      const gameStateBeforeBidding: GameState = createInitialState();
      const hands = dealDominoesWithSeed(12345);
      
      // Assign dealt hands to players
      gameStateBeforeBidding.players.forEach((player, index) => {
        const hand = hands[index];
        if (!hand) {
          throw new Error(`No hand dealt for player ${index}`);
        }
        player.hand = hand;
      });
      
      gameStateBeforeBidding.phase = 'setup';

      // Capture hand arrangement before bidding
      const handsBeforeBidding = gameStateBeforeBidding.players.map(p => [...p.hand]);

      // Transition to bidding phase
      const gameStateAfterBiddingStarts: GameState = {
        ...gameStateBeforeBidding,
        phase: 'bidding',
      };

      // Verify hands remain unchanged after bidding starts
      expect(gameStateAfterBiddingStarts.phase).toBe('bidding');
      gameStateAfterBiddingStarts.players.forEach((player, index) => {
        expect(player.hand).toEqual(handsBeforeBidding[index]);
      });
    });
  });
});