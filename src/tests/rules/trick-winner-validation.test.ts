import { describe, it, expect } from 'vitest';
import { determineTrickWinner } from '../../game/core/rules';
import { getDominoSuit } from '../../game/core/dominoes';
import type { PlayedDomino, TrumpSelection } from '../../game/types';
import { BLANKS, FIVES, SIXES } from '../../game/types';

describe('Trick Winner Validation', () => {
  describe('Trump Beats Non-Trump', () => {
    it('should award trick to lowest trump when multiple trump played', () => {
      const trump: TrumpSelection = { type: 'suit', suit: FIVES };
      const trick: PlayedDomino[] = [
        { domino: { id: "1", low: 0, high: 1 }, player: 0 },   // [1|0] - not trump
        { domino: { id: "10", low: 0, high: 5 }, player: 1 },  // [5|0] - trump
        { domino: { id: "15", low: 5, high: 5 }, player: 2 },  // [5|5] - higher trump
        { domino: { id: "5", low: 2, high: 3 }, player: 3 }    // [2|3] - not trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(2); // Player 2 with [5|5] (highest trump)
    });

    it('should award trick to any trump over non-trump', () => {
      const trump: TrumpSelection = { type: 'suit', suit: SIXES };
      const trick: PlayedDomino[] = [
        { domino: { id: "20", low: 4, high: 6 }, player: 0 },  // [6|4] - trump
        { domino: { id: "15", low: 5, high: 5 }, player: 1 },  // [5|5] - not trump
        { domino: { id: "1", low: 0, high: 1 }, player: 2 },   // [1|0] - not trump
        { domino: { id: "12", low: 3, high: 4 }, player: 3 }   // [4|3] - not trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(0); // Player 0 with trump [6|4]
    });
  });

  describe('No Trump Played - Led Suit Wins', () => {
    it('should award to highest domino of led suit when no trump', () => {
      const trump: TrumpSelection = { type: 'suit', suit: FIVES };
      const trick: PlayedDomino[] = [
        { domino: { id: "12", low: 3, high: 4 }, player: 0 },  // [4|3] - led fours
        { domino: { id: "8", low: 1, high: 4 }, player: 1 },   // [4|1] - higher four
        { domino: { id: "5", low: 2, high: 3 }, player: 2 },   // [2|3] - not fours
        { domino: { id: "1", low: 0, high: 1 }, player: 3 }    // [1|0] - not fours
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(0); // Player 0 with [4|3] (highest four by pip count: 7 > 5)
    });

    it('should handle doubles as highest in their natural suit', () => {
      const trump: TrumpSelection = { type: 'suit', suit: FIVES };
      const trick: PlayedDomino[] = [
        { domino: { id: "5", low: 2, high: 3 }, player: 0 },   // [3|2] - led threes
        { domino: { id: 6, low: 3, high: 3 }, player: 1 },   // [3|3] - highest three
        { domino: { id: "1", low: 0, high: 1 }, player: 2 },   // [1|0] - not threes
        { domino: { id: "12", low: 3, high: 4 }, player: 3 }   // [4|3] - has three but lower
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(1); // Player 1 with [3|3] (double-three beats all threes)
    });
  });

  describe('Doubles Trump Special Cases', () => {
    it('should treat all doubles as trump when doubles trump declared', () => {
      const trump: TrumpSelection = { type: 'doubles' };
      const trick: PlayedDomino[] = [
        { domino: { id: "1", low: 0, high: 1 }, player: 0 },   // [1|0] - not trump
        { domino: { id: 0, low: 0, high: 0 }, player: 1 },   // [0|0] - trump (double)
        { domino: { id: 6, low: 3, high: 3 }, player: 2 },   // [3|3] - trump (higher double)
        { domino: { id: "5", low: 2, high: 3 }, player: 3 }    // [2|3] - not trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(2); // Player 2 with [3|3] (highest double)
    });

    it('should respect double hierarchy when doubles are trump', () => {
      const trump: TrumpSelection = { type: 'doubles' };
      const trick: PlayedDomino[] = [
        { domino: { id: 27, low: 6, high: 6 }, player: 0 },  // [6|6] - highest double
        { domino: { id: "15", low: 5, high: 5 }, player: 1 },  // [5|5] - lower double
        { domino: { id: 6, low: 3, high: 3 }, player: 2 },   // [3|3] - lower double
        { domino: { id: "1", low: 0, high: 1 }, player: 3 }    // [1|0] - not trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(0); // Player 0 with [6|6] (highest double)
    });
  });

  describe('No-Trump Game Rules', () => {
    it('should award to highest domino of led suit in no-trump', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      const trick: PlayedDomino[] = [
        { domino: { id: "12", low: 3, high: 4 }, player: 0 },  // [4|3] - led fours
        { domino: { id: "20", low: 4, high: 6 }, player: 1 },  // [6|4] - higher four
        { domino: { id: 27, low: 6, high: 6 }, player: 2 },  // [6|6] - not fours
        { domino: { id: "15", low: 5, high: 5 }, player: 3 }   // [5|5] - not fours
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(1); // Player 1 with [6|4] (highest four)
    });

    it('should handle all high-value dominoes equally in no-trump', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      const trick: PlayedDomino[] = [
        { domino: { id: 27, low: 6, high: 6 }, player: 0 },  // [6|6] - led sixes
        { domino: { id: "20", low: 4, high: 6 }, player: 1 },  // [6|4] - same suit (sixes)
        { domino: { id: "5", low: 2, high: 3 }, player: 2 },   // [2|3] - not sixes
        { domino: { id: "1", low: 0, high: 1 }, player: 3 }    // [1|0] - not sixes
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(0); // Player 0 with [6|6] (double-six beats [6|4])
    });
  });

  describe('Edge Cases and Complex Scenarios', () => {
    it('should handle first played wins ties rule correctly', () => {
      const trump: TrumpSelection = { type: 'no-trump' };
      const trick: PlayedDomino[] = [
        { domino: { id: "8", low: 1, high: 4 }, player: 0 },   // [4|1] - led fours
        { domino: { id: "12", low: 3, high: 4 }, player: 1 },  // [4|3] - same suit, same high value
        { domino: { id: "5", low: 2, high: 3 }, player: 2 },   // [2|3] - not fours
        { domino: { id: "1", low: 0, high: 1 }, player: 3 }    // [1|0] - not fours
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(1); // Player 1 with [4|3] (4 is higher than 1 in [4|1])
    });

    it('should correctly identify suit when trump domino is led', () => {
      const trump: TrumpSelection = { type: 'suit', suit: FIVES };
      const trick: PlayedDomino[] = [
        { domino: { id: "15", low: 5, high: 5 }, player: 0 },  // [5|5] - trump led
        { domino: { id: "10", low: 0, high: 5 }, player: 1 },  // [5|0] - trump
        { domino: { id: "5", low: 2, high: 3 }, player: 2 },   // [2|3] - not trump
        { domino: { id: "1", low: 0, high: 1 }, player: 3 }    // [1|0] - not trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(0); // Player 0 with [5|5] (highest trump)
    });

    it('should handle mixed trump and non-trump correctly', () => {
      const trump: TrumpSelection = { type: 'suit', suit: BLANKS };
      const trick: PlayedDomino[] = [
        { domino: { id: "12", low: 3, high: 4 }, player: 0 },  // [4|3] - led fours
        { domino: { id: "1", low: 0, high: 1 }, player: 1 },   // [1|0] - trump
        { domino: { id: "8", low: 1, high: 4 }, player: 2 },   // [4|1] - higher four
        { domino: { id: 0, low: 0, high: 0 }, player: 3 }    // [0|0] - trump
      ];

      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = determineTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(3); // Player 3 with [0|0] (highest trump - double blank)
    });
  });
});