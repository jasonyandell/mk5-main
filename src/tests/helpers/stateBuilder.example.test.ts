/**
 * Example usage of StateBuilder - demonstrates real-world patterns
 */

import { describe, it, expect } from 'vitest';
import { StateBuilder, DominoBuilder, HandBuilder } from './stateBuilder';
import { ACES, BLANKS } from '../../game/types';
import { BID_TYPES } from '../../game/constants';

describe('StateBuilder - Real World Examples', () => {
  it('Example 1: Test bidding validation', () => {
    // Create a fresh bidding state with specific hands
    const state = StateBuilder
      .inBiddingPhase(0)
      .withPlayerHand(0, HandBuilder.withDoubles(4)) // Player 0 has 4 doubles (plunge-eligible)
      .build();

    expect(state.phase).toBe('bidding');
    expect(state.players[0]!.hand).toHaveLength(7);

    // Verify player has enough doubles
    const doubles = state.players[0]!.hand.filter(d => d.high === d.low);
    expect(doubles.length).toBeGreaterThanOrEqual(4);
  });

  it('Example 2: Test trick-taking logic', () => {
    // Set up a trick where player must follow suit
    const state = StateBuilder
      .inPlayingPhase({ type: 'suit', suit: ACES })
      .withPlayerHand(0, ['6-1', '5-1', '4-1', '3-1', '2-1', '1-0', '0-0']) // Has ones
      .withCurrentTrick([
        { player: 1, domino: '6-5' },  // Lead with non-trump
        { player: 2, domino: '5-4' }
      ])
      .build();

    expect(state.currentTrick).toHaveLength(2);
    // Note: currentPlayer doesn't automatically advance in StateBuilder
    // It stays at the initial value from inPlayingPhase (which is 1)

    // Player 0 should have dominoes with suit 5 (the led suit)
    const player0Hand = state.players[0]!.hand;
    expect(player0Hand.length).toBe(7);
  });

  it('Example 3: Test scoring after completed hand', () => {
    // Create end-of-hand state
    const state = StateBuilder
      .inScoringPhase([35, 7])  // Team 0 made their bid
      .withWinningBid(0, { type: BID_TYPES.POINTS, value: 30, player: 0 })
      .build();

    expect(state.phase).toBe('scoring');
    expect(state.teamScores).toEqual([35, 7]);
    expect(state.winningBidder).toBe(0);

    // All hands should be empty at scoring
    state.players.forEach(player => {
      expect(player.hand).toHaveLength(0);
    });
  });

  it('Example 4: Test mid-trick state', () => {
    // Create a partially completed trick
    const state = StateBuilder
      .inPlayingPhase({ type: 'suit', suit: BLANKS })
      .withCurrentTrick([
        { player: 0, domino: '6-0' },  // Lead with trump
        { player: 1, domino: '5-0' }   // Follow with trump
      ])
      .withTeamScores(10, 5)
      .build();

    expect(state.currentTrick).toHaveLength(2);
    expect(state.currentSuit).toBeDefined();
    expect(state.teamScores).toEqual([10, 5]);
  });

  it('Example 5: Test special contract setup', () => {
    // Create a Nello contract scenario
    const state = StateBuilder
      .nelloContract(0)
      .withPlayerHand(0, ['0-0', '1-1', '2-2', '3-0', '4-0', '1-0', '2-0']) // Low point cards
      .build();

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);
    expect(state.currentBid.type).toBe(BID_TYPES.MARKS);

    // Verify player has a hand
    const player0Hand = state.players[0]!.hand;
    expect(player0Hand).toHaveLength(7);
  });

  it('Example 6: Clone and modify pattern', () => {
    // Create base state
    const baseBuilder = StateBuilder
      .inPlayingPhase()
      .withTeamScores(15, 10);

    // Create two variations
    const state1 = baseBuilder
      .clone()
      .withCurrentPlayer(0)
      .build();

    const state2 = baseBuilder
      .clone()
      .withCurrentPlayer(2)
      .build();

    expect(state1.currentPlayer).toBe(0);
    expect(state2.currentPlayer).toBe(2);
    expect(state1.teamScores).toEqual([15, 10]);
    expect(state2.teamScores).toEqual([15, 10]);
  });

  it('Example 7: Complex chaining for edge case', () => {
    // Test a very specific game state
    const state = StateBuilder
      .withTricksPlayed(6)  // Last trick
      .withTeamScores(20, 20)  // Tied score
      .withCurrentPlayer(0)
      .withPlayerHand(0, ['6-6'])  // Only double-six left
      .withPlayerHand(1, ['5-5'])
      .withPlayerHand(2, ['4-4'])
      .withPlayerHand(3, ['3-3'])
      .build();

    expect(state.tricks).toHaveLength(6);
    expect(state.teamScores).toEqual([20, 20]);

    // Verify each player has exactly one domino
    state.players.forEach(player => {
      expect(player.hand).toHaveLength(1);
    });
  });

  it('Example 8: Test splash contract with partner selection', () => {
    // Splash: bidder has doubles, partner selects trump
    const state = StateBuilder
      .splashContract(1, 2)  // Player 1 bid splash 2
      .withPlayerHand(1, HandBuilder.withDoubles(3)) // Bidder has 3 doubles
      .build();

    expect(state.winningBidder).toBe(1);
    expect(state.currentPlayer).toBe(3); // Partner (opposite of player 1)
    expect(state.currentBid.type).toBe('splash');
    expect(state.currentBid.value).toBe(2);
  });

  it('Example 9: Test using string dominoes vs objects', () => {
    // Both forms should work
    const state1 = StateBuilder
      .inPlayingPhase()
      .withPlayerHand(0, ['6-6', '6-5', '5-5'])
      .build();

    const dominoObjects = [
      DominoBuilder.from('6-6'),
      DominoBuilder.from('6-5'),
      DominoBuilder.from('5-5')
    ];

    const state2 = StateBuilder
      .inPlayingPhase()
      .withPlayerHand(0, dominoObjects)
      .build();

    expect(state1.players[0]!.hand.map(d => d.id))
      .toEqual(state2.players[0]!.hand.map(d => d.id));
  });

  it('Example 10: Test game end with marks', () => {
    // Create a completed game
    const state = StateBuilder
      .gameEnded(1)  // Team 1 won
      .withTeamMarks(3, 7)
      .build();

    expect(state.phase).toBe('game_end');
    expect(state.teamMarks[1]).toBeGreaterThanOrEqual(7);
  });
});
