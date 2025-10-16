import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, BidType } from '../../game/types';
import { createInitialState } from '../../game/core/state';
import { isValidBid } from '../../game/core/rules';
import { dealDominoesWithSeed } from '../../game/core/dominoes';

describe('Tournament Restrictions', () => {
  let gameState: GameState;

  beforeEach(() => {
    gameState = createInitialState();
    // REMOVED: gameState.tournamentMode = true;
    gameState.phase = 'bidding';
    const hands = dealDominoesWithSeed(12345);
    gameState.players.forEach((player, i) => {
      const hand = hands[i];
      if (hand) {
        player.hand = hand;
      }
    });
  });

  describe('Given a base game rules are being applied', () => {

    describe('When players are bidding', () => {
      it('Then Nel-O is allowed in base rules (filtered by tournament variant)', () => {
        const bid = {
          type: 'nello' as BidType,
          value: 1,
          player: gameState.currentPlayer
        };

        const isValid = isValidBid(gameState, bid);
        // Base rules allow Nel-O; tournament variant filters it out at action level
        expect(isValid).toBe(true);
      });

      it('Then Plunge requires 4+ doubles but is allowed in base rules', () => {
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
        const currentPlayer = gameState.players[gameState.currentPlayer];
        if (currentPlayer) {
          currentPlayer.hand = handWith2Doubles;
        }

        const bid = {
          type: 'plunge' as BidType,
          value: 4,
          player: gameState.currentPlayer
        };

        const isValidPlunge = isValidBid(gameState, bid);
        expect(isValidPlunge).toBe(false); // Need 4+ doubles for plunge

        // Now test with 4 doubles - should be allowed in base rules
        const handWith4Doubles = [
          { high: 0, low: 0, id: '0-0' }, // double blank
          { high: 1, low: 1, id: '1-1' }, // double one
          { high: 2, low: 2, id: '2-2' }, // double two
          { high: 3, low: 3, id: '3-3' }, // double three
          { high: 5, low: 4, id: '5-4' },
          { high: 6, low: 0, id: '6-0' },
          { high: 5, low: 1, id: '5-1' }
        ];
        const currentPlayerWith4 = gameState.players[gameState.currentPlayer];
        if (currentPlayerWith4) {
          currentPlayerWith4.hand = handWith4Doubles;
        }

        const validBid = {
          type: 'plunge' as BidType,
          value: 4,
          player: gameState.currentPlayer
        };

        const isValidWithDoubles = isValidBid(gameState, validBid, handWith4Doubles);
        // Base rules allow Plunge with 4+ doubles; tournament variant filters it out
        expect(isValidWithDoubles).toBe(true);
      });

      it('Then Splash is allowed in base rules (filtered by tournament variant)', () => {
        // Splash requires 3+ doubles
        const handWith3Doubles = [
          { high: 0, low: 0, id: '0-0' }, // double blank
          { high: 1, low: 1, id: '1-1' }, // double one
          { high: 2, low: 2, id: '2-2' }, // double two
          { high: 3, low: 2, id: '3-2' },
          { high: 5, low: 4, id: '5-4' },
          { high: 6, low: 3, id: '6-3' },
          { high: 5, low: 1, id: '5-1' }
        ];

        const bid = {
          type: 'splash' as BidType,
          value: 2,
          player: gameState.currentPlayer
        };

        const isValidSplash = isValidBid(gameState, bid, handWith3Doubles);
        // Base rules allow Splash; tournament variant filters it out at action level
        expect(isValidSplash).toBe(true);
      });

      it('Then Sevens is not allowed', () => {
        // Note: 'sevens' is not a valid BidType in the current implementation
        // This test would need to be updated if sevens is added
      });
    });
  });
});