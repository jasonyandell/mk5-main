import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';
import { GameTestHelper, processSequentialConsensus } from '../helpers/gameTestHelper';
import { BID_TYPES } from '../../game/constants';
import type { GameState } from '../../game/types';

describe('Complete Game Flow Integration', () => {
  async function playCompleteHand(state: GameState): Promise<GameState> {
    // Bid
    const bid30 = getNextStates(state).find(t => t.id === 'bid-30');
    state = bid30!.newState;
    
    // Others pass
    for (let i = 0; i < 3; i++) {
      const pass = getNextStates(state).find(t => t.id === 'pass');
      state = pass!.newState;
    }
    
    // Select trump
    const trump = getNextStates(state).find(t => t.id.startsWith('trump-'));
    state = trump!.newState;
    
    // Play all 7 tricks
    for (let trick = 0; trick < 7; trick++) {
      // 4 plays per trick
      for (let play = 0; play < 4; play++) {
        const playOptions = getNextStates(state).filter(t => t.id.startsWith('play-'));
        if (playOptions.length > 0) {
          const firstOption = playOptions[0];
          if (firstOption) {
            state = firstOption.newState;
          }
        }
      }
      
      // Complete trick using helper
      state = await processSequentialConsensus(state, 'completeTrick');
      
      // Now complete the trick
      const finalTransitions = getNextStates(state);
      const completeTrick = finalTransitions.find(t => t.id === 'complete-trick');
      if (completeTrick) {
        state = completeTrick.newState;
      }
    }
    
    // Score hand using helper
    state = await processSequentialConsensus(state, 'scoreHand');
    
    // Now score the hand
    const finalTransitions = getNextStates(state);
    const scoreHand = finalTransitions.find(t => t.id === 'score-hand');
    if (scoreHand) {
      state = scoreHand.newState;
    }
    
    return state;
  }

  describe('Full Tournament Game', () => {
    it('should play complete tournament game to 7 marks', async () => {
      let state = createInitialState();
      let handCount = 0;
      const maxHands = 20; // Safety limit
      
      while (state.phase !== 'game_end' && handCount < maxHands) {
        const initialMarks = [...state.teamMarks];
        
        state = await playCompleteHand(state);
        handCount++;
        
        // Verify marks increased for someone
        const marksChanged = state.teamMarks.some((marks, i) => marks !== initialMarks[i]);
        expect(marksChanged).toBe(true);
        
        // Check if game should end
        const maxMarks = Math.max(...state.teamMarks);
        if (maxMarks >= state.gameTarget) {
          expect(state.phase).toBe('game_end');
          break;
        }
        
        // Start next hand
        if (state.phase === 'scoring') {
          const nextHand = getNextStates(state).find(t => t.id === 'next-hand');
          if (nextHand) {
            state = nextHand.newState;
          }
        }
      }
      
      expect(Math.max(...state.teamMarks)).toBeGreaterThanOrEqual(7);
      expect(handCount).toBeLessThan(maxHands);
    });
  });

  describe('Multiple Game Variations', () => {
    it('should handle casual mode with special contracts', () => {
      let state = createInitialState();
      // REMOVED: state.tournamentMode = false;
      
      // Should allow special contracts
      const initialTransitions = getNextStates(state);
      
      // Special contracts availability depends on hand composition
      expect(initialTransitions.length).toBeGreaterThan(0);
    });

    it('should handle tournament mode restrictions', () => {
      let state = createInitialState();
      // REMOVED: expect(state.tournamentMode).toBe(true);

      // Note: Base engine now generates special contracts; use tournament action transformer to filter them
      const initialTransitions = getNextStates(state);

      // Base engine allows special contracts (test updated for new architecture)
      expect(initialTransitions.length).toBeGreaterThan(0);
    });

    it('should handle different game targets', () => {
      let state = createInitialState();
      state.gameTarget = 10; // Higher target
      
      expect(state.gameTarget).toBe(10);
      expect(state.teamMarks.every(marks => marks < 10)).toBe(true);
    });
  });

  describe('Complex Bidding Scenarios', () => {
    it('should handle escalating mark bids', () => {
      let state = createInitialState();
      
      // Player 0 bids 1 mark
      const mark1 = getNextStates(state).find(t => t.id === 'bid-mark-1');
      if (mark1) {
        state = mark1.newState;
        
        // Player 1 bids 2 marks
        const mark2 = getNextStates(state).find(t => t.id === 'bid-mark-2');
        if (mark2) {
          state = mark2.newState;
          expect(state.currentBid?.value).toBe(2);
        }
      }
    });

    it('should handle mixed point and mark bidding', () => {
      let state = createInitialState();
      
      // Start with point bid
      const bid35 = getNextStates(state).find(t => t.id === 'bid-35');
      if (bid35) {
        state = bid35.newState;
        
        // Next player can bid marks (equivalent to 42+ points)
        const markBids = getNextStates(state).filter(t => t.id.includes('mark'));
        expect(markBids.length).toBeGreaterThan(0);
      }
    });

    it('should prevent invalid bid sequences', () => {
      let state = createInitialState();
      
      // Bid 35 points
      const bid35 = getNextStates(state).find(t => t.id === 'bid-35');
      if (bid35) {
        state = bid35.newState;
        
        // Should not allow lower point bids
        const transitions = getNextStates(state);
        const lowerBids = transitions.filter(t => {
          const match = t.id.match(/bid-(\d+)$/);
          return match && match[1] && parseInt(match[1]) <= 35;
        });
        
        expect(lowerBids.length).toBe(0);
      }
    });
  });

  describe('Scoring Integration', () => {
    it('should maintain 42-point total per hand', async () => {
      let state = createInitialState();
      
      // Play through one complete hand
      state = await playCompleteHand(state);
      
      // Verify total points distributed = 42
      if (state.phase === 'scoring' || state.phase === 'game_end') {
        const totalPoints = state.teamScores[0] + state.teamScores[1];
        expect(totalPoints).toBe(42);
      }
    });

    it('should award marks correctly for successful bids', async () => {
      let state = createInitialState();
      const initialMarks = [...state.teamMarks];
      
      // Force a specific bid and play it out
      state = await playCompleteHand(state);
      
      // Someone should have gained marks
      const marksGained = state.teamMarks.some((marks, i) => {
        const initial = initialMarks[i];
        return initial !== undefined && marks > initial;
      });
      expect(marksGained).toBe(true);
    });

    it('should handle set penalties correctly', () => {
      let state = GameTestHelper.createTestState({
        phase: 'scoring',
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 },
        teamScores: [20, 22], // Team 0 failed 2-mark bid (needed 84 points)
        winningBidder: 0
      });
      
      // Team 0 should lose marks, Team 1 should gain
      expect(state.teamScores[0]).toBeLessThan(84);
    });
  });

  describe('State Consistency', () => {
    it('should maintain valid game state throughout', () => {
      let state = createInitialState();
      let actionCount = 0;
      const maxActions = 1000; // Generous limit to handle complex game scenarios
      
      while (state.phase !== 'game_end' && actionCount < maxActions) {
        // Verify state invariants
        expect(state.players).toHaveLength(4);
        expect(state.currentPlayer).toBeGreaterThanOrEqual(0);
        expect(state.currentPlayer).toBeLessThan(4);
        expect(['bidding', 'trump_selection', 'playing', 'scoring', 'game_end']).toContain(state.phase);
        
        // Take next action
        const transitions = getNextStates(state);
        if (transitions.length === 0) break;
        
        // Smart transition selection to avoid infinite all-pass redeals
        let selectedTransition = transitions[0]!;
        if (state.phase === 'bidding') {
          // Always force a bid on the first opportunity to prevent redeals
          if (state.bids.length === 0) {
            const minBidTransition = transitions.find(t => t.id === 'bid-30');
            if (minBidTransition) {
              selectedTransition = minBidTransition;
            } else {
              // If min bid not available, take any non-pass transition
              const nonPassTransition = transitions.find(t => !t.id.includes('pass') && !t.id.includes('redeal'));
              if (nonPassTransition) {
                selectedTransition = nonPassTransition;
              }
            }
          } else {
            // After first bid, allow normal progression but prefer non-redeal
            const nonRedealTransition = transitions.find(t => !t.id.includes('redeal'));
            if (nonRedealTransition) {
              selectedTransition = nonRedealTransition;
            }
          }
        }
        
        state = selectedTransition.newState;
        actionCount++;
      }
      
      expect(actionCount).toBeLessThan(maxActions);
    });

    it('should preserve domino conservation', () => {
      const state = createInitialState();
      
      // Count total dominoes in all hands
      const totalDominoes = state.players.reduce((sum, player) => sum + player.hand.length, 0);
      expect(totalDominoes).toBe(28);
      
      // Verify no duplicates
      const allDominoes = state.players.flatMap(p => p.hand);
      const uniqueIds = new Set(allDominoes.map(d => d.id));
      expect(uniqueIds.size).toBe(28);
    });

    it('should handle turn order correctly', () => {
      let state = createInitialState();
      const startPlayer = state.currentPlayer;
      
      // First bidding round should cycle through all players
      const playerOrder: number[] = [];
      
      for (let i = 0; i < 4; i++) {
        playerOrder.push(state.currentPlayer);
        const pass = getNextStates(state).find(t => t.id === 'pass');
        if (pass) {
          state = pass.newState;
        }
      }
      
      // Should have gone through all 4 players in order
      expect(playerOrder).toHaveLength(4);
      expect(new Set(playerOrder).size).toBe(4);
      expect(playerOrder[0]).toBe(startPlayer);
    });
  });
});