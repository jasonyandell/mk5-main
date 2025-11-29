import { describe, it, expect } from 'vitest';
import { StateBuilder } from '../helpers';
import { composeRules } from '../../game/layers/compose';
import { baseLayer } from '../../game/layers';
import { canFollowSuit } from '../../game/core/rules';
import { getLedSuit } from '../../game/core/dominoes';
import type { Domino, TrumpSelection } from '../../game/types';
import { ACES, DEUCES, TRES, FIVES, NO_LEAD_SUIT } from '../../game/types';

const rules = composeRules([baseLayer]);

/**
 * Why these tests use exact hands instead of dealConstraints:
 *
 * dealConstraints operates on pip values (e.g., "must have pip 2") but is trump-agnostic.
 * Renege rules require game-suit awareness: "can player follow suit X given trump Y?"
 *
 * Example: With ACES trump and DEUCES led:
 * - A 2-1 domino contains pip 2, but it's TRUMP (contains pip 1), so it can't follow twos
 * - A 2-4 domino contains pip 2 and is NOT trump, so it CAN follow twos
 *
 * dealConstraints cannot express "has pip 2 but NOT pip 1" cleanly, and even if it could,
 * the constraint would be specific to one trump choice. Renege tests need precise control
 * over which dominoes are trump vs. followers vs. neither, so exact hands are appropriate.
 *
 * See: mk5-tailwind-911, mk5-tailwind-lfy for context.
 */
describe('Renege Detection and Prevention', () => {
  describe('Must Follow Suit When Able', () => {
    it('requires following suit when player has matching suit', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 4, points: 0 }, // 2-4 (twos suit - must play)
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 }   // 3-5 (threes suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      // Can only play twos suit when able to follow
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');

      // Validate each domino individually
      const domino0 = playerHand[0];
      const domino1 = playerHand[1];
      const domino2 = playerHand[2];
      if (!domino0 || !domino1 || !domino2) {
        throw new Error('Missing dominoes in player hand');
      }
      expect(rules.isValidPlay(state, domino0, 1)).toBe(true);  // Following suit
      expect(rules.isValidPlay(state, domino1, 1)).toBe(false); // Trump when can follow
      expect(rules.isValidPlay(state, domino2, 1)).toBe(false); // Other suit when can follow
    });

    it('allows any play when cannot follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 },  // 3-5 (threes suit)
        { id: 'other-2', high: 4, low: 6, points: 0 }   // 4-6 (fours suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      // All plays valid when can't follow suit
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3);

      // All dominoes should be valid when can't follow suit
      playerHand.forEach(domino => {
        expect(rules.isValidPlay(state, domino, 1)).toBe(true);
      });
    });

    it('correctly identifies trump vs non-trump suits', () => {
      const trump: TrumpSelection = { type: 'suit', suit: DEUCES }; // Twos trump

      const trumpDomino: Domino = { id: 'trump', high: 2, low: 5, points: 0 }; // 2-5
      const nonTrumpDomino: Domino = { id: 'non-trump', high: 3, low: 4, points: 0 }; // 3-4

      expect(getLedSuit(trumpDomino, trump)).toBe(DEUCES); // Should be suit 2 (trump)
      expect(getLedSuit(nonTrumpDomino, trump)).not.toBe(DEUCES); // Should not be suit 2
    });
  });

  describe('Trump Play Rules', () => {
    it('allows trump when cannot follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 }   // 3-5 (threes suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      expect(canFollowSuit(state.players[1]!, DEUCES, state.trump)).toBe(false);
      expect(rules.isValidPlay(state, playerHand[0]!, 1)).toBe(true); // Trump play allowed
      expect(rules.isValidPlay(state, playerHand[1]!, 1)).toBe(true); // Any play allowed
    });

    it('prevents trump when can follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 4, points: 0 }, // 2-4 (twos suit - must play)
        { id: 'trump-1', high: 1, low: 3, points: 0 }   // 1-3 (trump - cannot play)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      expect(canFollowSuit(state.players[1]!, DEUCES, state.trump)).toBe(true);
      expect(rules.isValidPlay(state, playerHand[0]!, 1)).toBe(true);  // Following suit
      expect(rules.isValidPlay(state, playerHand[1]!, 1)).toBe(false); // Trump when can follow
    });

    it('correctly identifies trump dominoes as not able to follow non-trump suits', () => {
      // Regression test for pip-value vs game-suit bug (mk5-tailwind-lfy)
      // Player has a domino containing pip 2 (the led suit), but it's also trump (contains pip 1)
      // Trump dominoes cannot follow non-trump suits, so canFollowSuit should return false
      const playerHand: Domino[] = [
        { id: 'trump-with-led-pip', high: 2, low: 1, points: 0 }, // 2-1 contains pip 2 BUT is trump (contains 1)
        { id: 'other-1', high: 3, low: 5, points: 0 }              // 3-5 (not trump, not led suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES }) // Aces (1s) are trump
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      // Key assertion: 2-1 contains pip 2, but it's trump so it can't follow twos suit
      // canFollowSuit should return false because player has no non-trump dominoes with pip 2
      expect(canFollowSuit(state.players[1]!, DEUCES, state.trump)).toBe(false);

      // Since player can't follow suit, any play is valid
      expect(rules.isValidPlay(state, playerHand[0]!, 1)).toBe(true);
      expect(rules.isValidPlay(state, playerHand[1]!, 1)).toBe(true);
    });
  });

  describe('Doubles Trump Special Cases', () => {
    it('handles doubles trump correctly', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'doubles' })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }])
        .with({ currentSuit: TRES }) // Threes were led
        .build();

      // Must follow threes suit, not doubles trump
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
    });

    it('allows doubles trump when cannot follow natural suit', () => {
      const playerHand: Domino[] = [
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'double-2', high: 2, low: 2, points: 0 }, // 2-2 (double/trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'doubles' })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }])
        .with({ currentSuit: TRES }) // Threes were led
        .build();

      // Cannot follow threes suit, all plays valid
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3);

      playerHand.forEach(domino => {
        expect(rules.isValidPlay(state, domino, 1)).toBe(true);
      });
    });
  });

  describe('No Trump (Follow-Me) Special Cases', () => {
    it('enforces natural suit following in no-trump', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (ones suit in no-trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (fours suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'no-trump' })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (higher pip = threes)
        }])
        .with({ currentSuit: TRES }) // Threes were led
        .build();

      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
    });
  });

  describe('First Play of Trick', () => {
    it('allows any domino on opening lead', () => {
      const playerHand: Domino[] = [
        { id: 'any-1', high: 2, low: 3, points: 0 },
        { id: 'any-2', high: 1, low: 4, points: 0 }, // Trump
        { id: 'any-3', high: 5, low: 6, points: 0 }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(0)
        .withPlayerHand(0, playerHand)
        .withCurrentTrick([]) // Empty trick
        .with({ currentSuit: NO_LEAD_SUIT })
        .build();

      const validPlays = rules.getValidPlays(state, 0);
      expect(validPlays).toHaveLength(3);

      playerHand.forEach(domino => {
        expect(rules.isValidPlay(state, domino, 0)).toBe(true);
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('handles invalid game phases', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .build();

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(rules.isValidPlay(state, domino, 0)).toBe(false);
    });

    it('handles missing trump declaration', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .with({ phase: 'playing', trump: { type: 'not-selected' } }) // No trump set
        .build();

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(rules.isValidPlay(state, domino, 0)).toBe(false);
    });

    it('prevents playing dominoes not in hand', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentTrick([])
        .build();

      const notInHand: Domino = { id: 'not-owned', high: 1, low: 2, points: 0 };
      expect(rules.isValidPlay(state, notInHand, 0)).toBe(false);
    });

    it('validates player bounds', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentTrick([])
        .build();

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(rules.isValidPlay(state, domino, -1)).toBe(false); // Invalid player
      expect(rules.isValidPlay(state, domino, 4)).toBe(false);  // Invalid player
    });
  });

  describe('Complex Renege Scenarios', () => {
    it('handles multiple trumps in trick with suit following', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 5, points: 0 }, // 2-5 (twos suit - must play)
        { id: 'trump-3', high: 1, low: 6, points: 0 }   // 1-6 (trump)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentPlayer(3)
        .withPlayerHand(3, playerHand)
        .withCurrentTrick([
          { player: 0, domino: { id: 'lead', high: 2, low: 2, points: 0 } }, // 2-2 (twos suit)
          { player: 1, domino: { id: 'trump1', high: 1, low: 3, points: 0 } }, // 1-3 (trump)
          { player: 2, domino: { id: 'trump2', high: 1, low: 4, points: 0 } }  // 1-4 (trump)
        ])
        .with({ currentSuit: DEUCES }) // Twos were led
        .build();

      // Must still follow original suit despite trumps played
      const validPlays = rules.getValidPlays(state, 3);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
    });

    it('prevents renege in complex trick scenarios', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 5, low: 6, points: 0 }, // 5-6 (fives suit)
        { id: 'follow-2', high: 0, low: 5, points: 5 }, // 0-5 (fives suit, counting)
        { id: 'trump-1', high: 3, low: 6, points: 0 },  // 3-6 (trump)
        { id: 'other-1', high: 1, low: 2, points: 0 }   // 1-2 (ones suit)
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: TRES })
        .withCurrentPlayer(1)
        .withPlayerHand(1, playerHand)
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'lead', high: 4, low: 5, points: 0 } // 4-5 (fives suit)
        }])
        .with({ currentSuit: FIVES })
        .build();

      const validPlays = rules.getValidPlays(state, 1);

      // Must play one of the fives suit dominoes
      expect(validPlays).toHaveLength(2);
      expect(validPlays.map(d => d.id)).toContain('follow-1');
      expect(validPlays.map(d => d.id)).toContain('follow-2');
      expect(validPlays.map(d => d.id)).not.toContain('trump-1');
      expect(validPlays.map(d => d.id)).not.toContain('other-1');
    });
  });
});
