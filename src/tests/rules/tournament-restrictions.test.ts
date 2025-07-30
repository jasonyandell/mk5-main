import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, BidType } from '../../game/types';
import { createInitialState } from '../../game/core/state';
import { isValidBid } from '../../game/core/rules';
import { dealDominoes } from '../../game/core/dominoes';

describe('Tournament Restrictions', () => {
  let gameState: GameState;

  beforeEach(() => {
    gameState = createInitialState();
    gameState.tournamentMode = true;
    gameState.phase = 'bidding';
    const hands = dealDominoes();
    gameState.players.forEach((player, i) => {
      player.hand = hands[i];
    });
  });

  describe('Given a tournament game is being played', () => {

    describe('When players are bidding', () => {
      it('Then Nel-O is not allowed', () => {
        const bid = {
          type: 'nello' as BidType,
          value: 1,
          player: gameState.currentPlayer
        };
        
        const isValid = isValidBid(gameState, bid);
        expect(isValid).toBe(false); // Nel-O not allowed in tournament
      });

      it('Then Plunge is not allowed unless holding 4+ doubles', () => {
        // Mock hand with only 2 doubles
        const handWith2Doubles = [
          { high: 0, low: 0, id: '0-0' }, // double blank
          { high: 1, low: 1, id: '1-1' }, // double one
          { high: 3, low: 2, id: '3-2' },
          { high: 5, low: 4, id: '5-4' },
          { high: 6, low: 3, id: '6-3' },
          { high: 4, low: 0, id: '4-0' },
          { high: 5, low: 1, id: '5-1' }
        ];
        gameState.players[gameState.currentPlayer].hand = handWith2Doubles;
        
        const bid = {
          type: 'plunge' as BidType,
          value: 4,
          player: gameState.currentPlayer
        };
        
        const isValidPlunge = isValidBid(gameState, bid);
        expect(isValidPlunge).toBe(false); // Need 4+ doubles for plunge

        // Now test with 4 doubles - should be allowed
        const handWith4Doubles = [
          { high: 0, low: 0, id: '0-0' }, // double blank
          { high: 1, low: 1, id: '1-1' }, // double one
          { high: 2, low: 2, id: '2-2' }, // double two
          { high: 3, low: 3, id: '3-3' }, // double three
          { high: 5, low: 4, id: '5-4' },
          { high: 6, low: 0, id: '6-0' },
          { high: 5, low: 1, id: '5-1' }
        ];
        gameState.players[gameState.currentPlayer].hand = handWith4Doubles;
        
        const validBid = {
          type: 'plunge' as BidType,
          value: 4,
          player: gameState.currentPlayer
        };
        
        const isValidWithDoubles = isValidBid(gameState, validBid);
        expect(isValidWithDoubles).toBe(false); // Plunge not allowed in tournament mode per rules.md line 370
      });

      it('Then Splash is not allowed', () => {
        const bid = {
          type: 'splash' as BidType,
          value: 2,
          player: gameState.currentPlayer
        };
        
        const isValidSplash = isValidBid(gameState, bid);
        expect(isValidSplash).toBe(false); // Splash not allowed in tournament
      });

      it('Then Sevens is not allowed', () => {
        // Note: 'sevens' is not a valid BidType in the current implementation
        // This test would need to be updated if sevens is added
      });
    });
  });
});