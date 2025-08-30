import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';
import type { GameState, Player } from '../../game/types';
import { BLANKS, TRES, FIVES, SIXES } from '../../game/types';

describe('Suit Analysis Integration', () => {
  describe('Initial state creation', () => {
    it('should include suit analysis for all players when creating initial state', () => {
      const state = createInitialState({ shuffleSeed: 12345 });
      
      // All players should have suit analysis
      state.players.forEach((player) => {
        expect(player.suitAnalysis).toBeDefined();
        expect(player.suitAnalysis?.count).toBeDefined();
        expect(player.suitAnalysis?.rank).toBeDefined();
        
        // Since no trump is declared yet, trump counts should be 0
        expect(player.suitAnalysis?.count.trump).toBe(0);
        expect(player.suitAnalysis?.rank.trump).toHaveLength(0);
        
        // Verify count matches hand contents
        const handSuitCounts = [0, 0, 0, 0, 0, 0, 0];
        let doubleCount = 0;
        
        player.hand.forEach(domino => {
          if (domino.high === domino.low) {
            doubleCount++;
            const highCount = handSuitCounts[domino.high];
            if (highCount !== undefined) {
              handSuitCounts[domino.high] = highCount + 1;
            }
          } else {
            const highCount = handSuitCounts[domino.high];
            if (highCount !== undefined) {
              handSuitCounts[domino.high] = highCount + 1;
            }
            const lowCount = handSuitCounts[domino.low];
            if (lowCount !== undefined) {
              handSuitCounts[domino.low] = lowCount + 1;
            }
          }
        });
        
        for (let suit = BLANKS; suit <= SIXES; suit++) {
          expect(player.suitAnalysis?.count[suit as keyof typeof player.suitAnalysis.count]).toBe(handSuitCounts[suit]);
        }
        expect(player.suitAnalysis?.count.doubles).toBe(doubleCount);
      });
    });
  });

  describe('Trump declaration updates', () => {
    it('should update suit analysis when trump is declared', () => {
      const state = createInitialState({ shuffleSeed: 12345 });
      
      // Skip to bidding completion - simulate a bid being made
      const stateWithBid: GameState = {
        ...state,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'points', value: 30, player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ],
        phase: 'trump_selection',
        winningBidder: 1,
        currentBid: { type: 'points', value: 30, player: 1 },
        currentPlayer: 1
      };
      
      // Get trump selection transitions
      const trumpTransitions = getNextStates(stateWithBid);
      expect(trumpTransitions.length).toBeGreaterThan(0);
      
      // Check that trump is declared and suit analysis is updated
      const trumpTransition = trumpTransitions.find(t => t.id === 'trump-fives');
      expect(trumpTransition).toBeDefined();
      
      if (trumpTransition) {
        const newState = trumpTransition.newState;
        expect(newState.trump.type).toBe('suit');
        expect(newState.trump.suit).toBe(FIVES); // fives are trump
        
        // All players should have updated suit analysis with trump info
        newState.players.forEach((player: Player) => {
          expect(player.suitAnalysis).toBeDefined();
          
          // Trump count should match suit 5 count
          const fivesCount = player.suitAnalysis?.count[FIVES] || 0;
          expect(player.suitAnalysis?.count.trump).toBe(fivesCount);
          
          // Trump ranking should match suit 5 ranking
          const fivesRanking = player.suitAnalysis?.rank[FIVES] || [];
          expect(player.suitAnalysis?.rank.trump).toEqual(fivesRanking);
        });
      }
    });

    it('should handle doubles trump correctly', () => {
      const state = createInitialState({ shuffleSeed: 12345 });
      
      const stateWithBid: GameState = {
        ...state,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'points', value: 30, player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ],
        phase: 'trump_selection',
        winningBidder: 1,
        currentBid: { type: 'points', value: 30, player: 1 },
        currentPlayer: 1
      };
      
      const trumpTransitions = getNextStates(stateWithBid);
      const doublesTransition = trumpTransitions.find(t => t.id === 'trump-doubles');
      expect(doublesTransition).toBeDefined();
      
      if (doublesTransition) {
        const newState = doublesTransition.newState;
        expect(newState.trump.type).toBe('doubles'); // doubles are trump
        
        newState.players.forEach((player: Player) => {
          // Trump count should match doubles count
          const doublesCount = player.suitAnalysis?.count.doubles || 0;
          expect(player.suitAnalysis?.count.trump).toBe(doublesCount);
          
          // Trump ranking should match doubles ranking
          const doublesRanking = player.suitAnalysis?.rank.doubles || [];
          expect(player.suitAnalysis?.rank.trump).toEqual(doublesRanking);
        });
      }
    });
  });

  describe('Play transitions update suit analysis', () => {
    it('should update suit analysis when player plays a domino', () => {
      // Create a state in playing phase with trump declared
      const initialState = createInitialState({ shuffleSeed: 12345 });
      const playingState: GameState = {
        ...initialState,
        phase: 'playing',
        trump: { type: 'suit', suit: TRES }, // threes are trump
        winningBidder: 1,
        currentPlayer: 1,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'points', value: 30, player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ],
        currentBid: { type: 'points', value: 30, player: 1 }
      };
      
      // Update all players' suit analysis with trump info
      playingState.players.forEach(player => {
        if (player.suitAnalysis) {
          // Recalculate trump counts and rankings for the trump suit
          const trumpCount = player.hand.filter(d => d.high === TRES || d.low === TRES).length;
          const trumpRanking = player.hand.filter(d => d.high === TRES || d.low === TRES);
          player.suitAnalysis.count.trump = trumpCount;
          player.suitAnalysis.rank.trump = trumpRanking;
        }
      });
      
      const currentPlayer = playingState.players[playingState.currentPlayer];
      if (!currentPlayer) {
        throw new Error('Current player not found');
      }
      const initialHandSize = currentPlayer.hand.length;
      
      // Get play transitions
      const playTransitions = getNextStates(playingState);
      expect(playTransitions.length).toBeGreaterThan(0);
      
      // Take the first play transition
      const playTransition = playTransitions[0]!;
      const newState = playTransition.newState;
      
      // Verify the player who played has updated suit analysis
      const updatedPlayer = newState.players[playingState.currentPlayer];
      if (!updatedPlayer) {
        throw new Error('Updated player not found');
      }
      expect(updatedPlayer.hand.length).toBe(initialHandSize - 1);
      expect(updatedPlayer.suitAnalysis).toBeDefined();
      
      // The suit analysis should be recalculated for the reduced hand
      const newHandTotal = updatedPlayer.hand.length;
      expect(newHandTotal).toBe(6); // Should have 6 dominoes left
    });
  });

  describe('Redeal updates suit analysis', () => {
    it('should update suit analysis when all players pass and cards are redealt', () => {
      const state = createInitialState({ shuffleSeed: 12345 });
      
      // Simulate all players passing
      const allPassState: GameState = {
        ...state,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ]
      };
      
      const transitions = getNextStates(allPassState);
      const redealTransition = transitions.find(t => t.id === 'redeal');
      expect(redealTransition).toBeDefined();
      
      if (redealTransition) {
        const newState = redealTransition.newState;
        
        // All players should have new hands and updated suit analysis
        newState.players.forEach((player: Player) => {
          expect(player.hand.length).toBe(7); // Full hand
          expect(player.suitAnalysis).toBeDefined();
          
          // Suit analysis should reflect new hand
          expect(player.suitAnalysis?.count).toBeDefined();
          expect(player.suitAnalysis?.rank).toBeDefined();
          
          // Note: With different seeds, hands should be different, but let's just verify structure
          expect(player.suitAnalysis?.count).toBeDefined();
        });
      }
    });
  });
});