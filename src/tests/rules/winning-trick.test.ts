import { describe, it, expect } from 'vitest';
import type { Play, TrumpSelection } from '../../game/types';
import { determineTrickWinner } from '../../game/index';
import { getDominoSuit } from '../../game/core/dominoes';

describe('Winning a Trick', () => {
  describe('Given all players have played to a trick', () => {
    describe('When determining the winner', () => {
      it('Then the highest trump played wins', () => {
        const trump: TrumpSelection = { type: 'suit', suit: 3 }; // threes are trump
        const plays: Play[] = [
          { player: 0, domino: { high: 4, low: 2, id: '4-2' } },
          { player: 1, domino: { high: 3, low: 1, id: '3-1' } }, // trump
          { player: 2, domino: { high: 6, low: 4, id: '6-4' } },
          { player: 3, domino: { high: 3, low: 2, id: '3-2' } }, // higher trump
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(3); // player 3 with higher trump (3-2) wins
      });

      it('And if no trump was played, the highest domino of the led suit wins', () => {
        const trump: TrumpSelection = { type: 'suit', suit: 5 }; // fives are trump (none played)
        const plays: Play[] = [
          { player: 0, domino: { high: 4, low: 2, id: '4-2' } }, // led fours
          { player: 1, domino: { high: 4, low: 1, id: '4-1' } }, // follows fours
          { player: 2, domino: { high: 3, low: 1, id: '3-1' } }, // can't follow
          { player: 3, domino: { high: 4, low: 3, id: '4-3' } }, // highest four
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(3); // player 3 with highest four (4-3) wins
      });

      it('And if no trump was played and all followed suit, the highest of led suit wins', () => {
        const trump: TrumpSelection = { type: 'suit', suit: 2 }; // twos are trump (none played)
        const plays: Play[] = [
          { player: 0, domino: { high: 6, low: 1, id: '6-1' } }, // led sixes
          { player: 1, domino: { high: 6, low: 3, id: '6-3' } }, // follows sixes
          { player: 2, domino: { high: 6, low: 6, id: '6-6' } }, // double six (highest)
          { player: 3, domino: { high: 6, low: 4, id: '6-4' } }, // follows sixes
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(2); // player 2 with double six wins
      });

      it('handles when doubles are trump', () => {
        const trump: TrumpSelection = { type: 'doubles' }; // doubles are trump
        const plays: Play[] = [
          { player: 0, domino: { high: 5, low: 3, id: '5-3' } },
          { player: 1, domino: { high: 2, low: 2, id: '2-2' } }, // low double (trump)
          { player: 2, domino: { high: 6, low: 4, id: '6-4' } },
          { player: 3, domino: { high: 5, low: 5, id: '5-5' } }, // higher double (trump)
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(3); // player 3 with higher double (5-5) wins
      });

      it('handles no-trump (follow-me) games', () => {
        const trump: TrumpSelection = { type: 'no-trump' }; // no-trump
        const plays: Play[] = [
          { player: 0, domino: { high: 3, low: 2, id: '3-2' } }, // led threes
          { player: 1, domino: { high: 3, low: 0, id: '3-0' } }, // follows threes
          { player: 2, domino: { high: 6, low: 6, id: '6-6' } }, // can't follow
          { player: 3, domino: { high: 3, low: 3, id: '3-3' } }, // highest three
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(3); // player 3 with highest three (3-3) wins
      });

      it('handles when only trump is played', () => {
        const trump: TrumpSelection = { type: 'suit', suit: 1 }; // ones are trump
        const plays: Play[] = [
          { player: 0, domino: { high: 1, low: 0, id: '1-0' } }, // trump
          { player: 1, domino: { high: 5, low: 1, id: '5-1' } }, // trump
          { player: 2, domino: { high: 1, low: 1, id: '1-1' } }, // double one (highest trump)
          { player: 3, domino: { high: 4, low: 1, id: '4-1' } }, // trump
        ];
        
        const firstPlay = plays[0];
        if (!firstPlay) throw new Error('First play is undefined');
        const leadSuit = getDominoSuit(firstPlay.domino, trump);
        const winner = determineTrickWinner(plays, trump, leadSuit);
        expect(winner).toBe(2); // player 2 with double one wins
      });
    });
  });

  // Note: The test spec mentions "And in special games like Sevens, the first played wins ties"
  // However, Sevens is not allowed in tournament play per the rules document,
  // so we're not implementing this test case for tournament rules.
});