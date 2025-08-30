import { describe, it, expect } from 'vitest';
import { createTestState } from '../helpers/gameTestHelper';
import { analyzeSuits } from '../../game/core/suit-analysis';
import type { Domino, TrumpSelection } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

describe('Suit Analysis Consistency', () => {
  it('should produce identical suit analysis for same hand and trump', () => {
    // Create a test hand with known dominoes
    const testHand: Domino[] = [
      { id: 'dom1', high: 6, low: 4, points: 10 },
      { id: 'dom2', high: 5, low: 5, points: 10 },
      { id: 'dom3', high: 3, low: 2, points: 5 },
      { id: 'dom4', high: 1, low: 1, points: 0 },
      { id: 'dom5', high: 6, low: 2, points: 0 },
      { id: 'dom6', high: 4, low: 3, points: 0 },
      { id: 'dom7', high: 0, low: 0, points: 0 }
    ];

    // Analyze suits multiple times with same parameters
    const analysis1 = analyzeSuits(testHand, { type: 'suit', suit: SIXES }); // sixes trump
    const analysis2 = analyzeSuits(testHand, { type: 'suit', suit: SIXES }); // sixes trump
    const analysis3 = analyzeSuits(testHand, { type: 'suit', suit: SIXES }); // sixes trump

    // Results should be identical
    expect(analysis1).toEqual(analysis2);
    expect(analysis2).toEqual(analysis3);
    expect(analysis1).toEqual(analysis3);

    // Specifically check suit counts
    expect(analysis1.count).toEqual(analysis2.count);
    expect(analysis1.count).toEqual(analysis3.count);

    // Check trump rankings
    expect(analysis1.rank.trump).toEqual(analysis2.rank.trump);
    expect(analysis1.rank.trump).toEqual(analysis3.rank.trump);
  });

  it('should be deterministic across different sessions', () => {
    // Create the same hand multiple times
    const createHand = () => [
      { id: 'h1', high: 2, low: 1, points: 0 },
      { id: 'h2', high: 4, low: 4, points: 0 },
      { id: 'h3', high: 6, low: 3, points: 0 },
      { id: 'h4', high: 1, low: 0, points: 0 },
      { id: 'h5', high: 5, low: 2, points: 0 },
      { id: 'h6', high: 3, low: 3, points: 0 },
      { id: 'h7', high: 6, low: 5, points: 0 }
    ];

    const hand1 = createHand();
    const hand2 = createHand();
    const hand3 = createHand();

    // Analyze with different trump values
    const trumps: TrumpSelection[] = [
      { type: 'suit', suit: BLANKS }, { type: 'suit', suit: ACES }, { type: 'suit', suit: DEUCES },
      { type: 'suit', suit: TRES }, { type: 'suit', suit: FOURS }, { type: 'suit', suit: FIVES },
      { type: 'suit', suit: SIXES }, { type: 'doubles' }
    ];

    trumps.forEach(trump => {
      const analysis1 = analyzeSuits(hand1, trump);
      const analysis2 = analyzeSuits(hand2, trump);
      const analysis3 = analyzeSuits(hand3, trump);

      expect(analysis1).toEqual(analysis2);
      expect(analysis2).toEqual(analysis3);
    });
  });

  it('should handle state persistence correctly', () => {
    // Create a game state with specific hands
    const initialState = createTestState({
      phase: 'bidding',
      dealer: 0,
      currentPlayer: 1,
      bids: []
    });

    // Manually set hands to ensure consistency
    const player0Hand: Domino[] = [
      { id: 'p0d1', high: 1, low: 0, points: 0 },
      { id: 'p0d2', high: 2, low: 1, points: 0 },
      { id: 'p0d3', high: 3, low: 2, points: 5 },
      { id: 'p0d4', high: 4, low: 3, points: 0 },
      { id: 'p0d5', high: 5, low: 4, points: 0 },
      { id: 'p0d6', high: 6, low: 5, points: 0 },
      { id: 'p0d7', high: 6, low: 6, points: 0 }
    ];

    if (initialState.players[0]) {
      initialState.players[0].hand = player0Hand;
    }
    initialState.hands = { 0: player0Hand, 1: [], 2: [], 3: [] };

    // Analyze suits for this player
    const analysis1 = analyzeSuits(player0Hand, { type: 'not-selected' });
    
    // Simulate what happens after a bid (trump selection phase)
    const afterBidState = createTestState({
      phase: 'trump_selection',
      dealer: 0,
      currentPlayer: 1,
      bids: [{ type: 'points', value: 30, player: 1 }],
      winningBidder: 1
    });

    if (afterBidState.players[0]) {
      afterBidState.players[0].hand = player0Hand;
    }
    afterBidState.hands = { 0: player0Hand, 1: [], 2: [], 3: [] };

    // Analyze suits again - should be identical when trump is none
    const analysis2 = analyzeSuits(player0Hand, { type: 'not-selected' });
    
    expect(analysis1).toEqual(analysis2);

    // Now test with specific trump
    const analysis3 = analyzeSuits(player0Hand, { type: 'suit', suit: SIXES }); // sixes trump
    const analysis4 = analyzeSuits(player0Hand, { type: 'suit', suit: SIXES }); // sixes trump again

    expect(analysis3).toEqual(analysis4);
  });

  it('should maintain consistency after serialization', () => {
    const testHand: Domino[] = [
      { id: 'ser1', high: 0, low: 0, points: 0 },
      { id: 'ser2', high: 1, low: 1, points: 0 },
      { id: 'ser3', high: 2, low: 2, points: 0 },
      { id: 'ser4', high: 6, low: 4, points: 10 },
      { id: 'ser5', high: 5, low: 0, points: 5 },
      { id: 'ser6', high: 3, low: 1, points: 0 },
      { id: 'ser7', high: 4, low: 2, points: 0 }
    ];

    // Original analysis
    const originalAnalysis = analyzeSuits(testHand, { type: 'doubles' }); // doubles trump

    // Serialize and deserialize the hand (simulating URL state persistence)
    const serializedHand = JSON.stringify(testHand);
    const deserializedHand = JSON.parse(serializedHand) as Domino[];

    // Re-analyze
    const deserializedAnalysis = analyzeSuits(deserializedHand, { type: 'doubles' });

    // Should be identical
    expect(originalAnalysis).toEqual(deserializedAnalysis);
  });

  it('should handle edge cases consistently', () => {
    // All doubles hand
    const allDoublesHand: Domino[] = [
      { id: 'ad1', high: 0, low: 0, points: 0 },
      { id: 'ad2', high: 1, low: 1, points: 0 },
      { id: 'ad3', high: 2, low: 2, points: 0 },
      { id: 'ad4', high: 3, low: 3, points: 0 },
      { id: 'ad5', high: 4, low: 4, points: 0 },
      { id: 'ad6', high: 5, low: 5, points: 10 },
      { id: 'ad7', high: 6, low: 6, points: 0 }
    ];

    // Test with doubles trump
    const analysis1 = analyzeSuits(allDoublesHand, { type: 'doubles' });
    const analysis2 = analyzeSuits(allDoublesHand, { type: 'doubles' });
    
    expect(analysis1).toEqual(analysis2);
    expect(analysis1.count.trump).toBe(7); // All should be trump
    expect(analysis1.count.doubles).toBe(7); // All are doubles

    // Test with non-doubles trump
    const analysis3 = analyzeSuits(allDoublesHand, { type: 'suit', suit: TRES });
    const analysis4 = analyzeSuits(allDoublesHand, { type: 'suit', suit: TRES });
    
    expect(analysis3).toEqual(analysis4);
    expect(analysis3.count.trump).toBe(1); // Only 3-3 is trump
    expect(analysis3.count[TRES]).toBe(1); // Only one 3
  });
});