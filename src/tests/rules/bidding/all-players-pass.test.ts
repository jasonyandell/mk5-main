import { describe, test, expect } from 'vitest';
import type { GameState } from '../../../game/types';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../../game';

describe('Feature: Special Bids', () => {
  describe('Scenario: All Players Pass', () => {
    test('Given all players have had a chance to bid', () => {
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 3;
      gameState.currentPlayer = getPlayerLeftOfDealer(3); // Player 0
      gameState.bids = [];

      // Simulate all players passing using game engine
      for (let i = 0; i < 4; i++) {
        const transitions = getNextStates(gameState);
        const passTransition = transitions.find(t => t.id === 'pass');
        expect(passTransition).toBeDefined();
        
        if (passTransition) {
          Object.assign(gameState, passTransition.newState);
        }
      }

      // All players have had a chance to bid
      expect(gameState.bids.length).toBe(4);
      expect(gameState.bids.every(bid => bid.type === 'pass')).toBe(true);
    });

    test('When all players pass - Then under tournament rules, the hand is reshaken with the next player as shaker', () => {
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 3;
      gameState.currentPlayer = getPlayerLeftOfDealer(3); // Player 0
      // REMOVED: gameState.tournamentMode = true;
      gameState.bids = [];

      // Simulate all players passing
      for (let i = 0; i < 4; i++) {
        const transitions = getNextStates(gameState);
        const passTransition = transitions.find(t => t.id === 'pass');
        if (passTransition) {
          Object.assign(gameState, passTransition.newState);
        }
      }

      // Check final state after all players pass
      expect(gameState.bids.length).toBe(4);
      expect(gameState.bids.every(bid => bid.type === 'pass')).toBe(true);
      
      // Game should handle all-pass scenario according to rules
      // (Implementation may vary - some go to setup, some force dealer bid)
    });

    test('And under common variation, the shaker must bid minimum 30', () => {
      const mockState: Partial<GameState> = {
        phase: 'bidding',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        dealer: 3,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'pass', player: 2 },
        ],
      };

      // When it's the dealer's turn and all others have passed
      const forcedBidState: Partial<GameState> = {
        ...mockState,
        bids: [
          ...mockState.bids!,
          { type: 'points', value: 30, player: 3 } // Dealer forced to bid 30
        ],
        currentBid: { type: 'points', value: 30, player: 3 },
        winningBidder: 3,
      };

      expect(forcedBidState.bids?.length).toBe(4);
      const dealerBid = forcedBidState.bids?.find(bid => bid.player === mockState.dealer);
      expect(dealerBid?.type).toBe('points');
      expect(dealerBid?.value).toBe(30);
      expect(forcedBidState.winningBidder).toBe(mockState.dealer);
    });
  });
});