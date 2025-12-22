/**
 * Rule Contract Tests - Guardrails for base + special contract semantics
 *
 * These tests enforce the Crystal Palace contract: all rule functions must
 * follow predictable patterns across base and special contracts.
 *
 * Contract coverage:
 * - getLedSuit: What suit does a domino lead?
 * - suitsWithTrump: What suits does a domino belong to?
 * - canFollow: Can a domino follow a led suit?
 * - rankInTrick: What is a domino's trick-taking rank?
 * - calculateTrickWinner: Who won the trick?
 * - isTrump: Is a domino trump?
 */

import { describe, it, expect } from 'vitest';
import { composeRules } from '../../game/layers/compose';
import { baseLayer } from '../../game/layers/base';
import { nelloLayer } from '../../game/layers/nello';
import { sevensLayer } from '../../game/layers/sevens';
import { StateBuilder, DominoBuilder } from '../helpers';
import { ACES, TRES, SIXES, DOUBLES_AS_TRUMP } from '../../game/types';
import type { LedSuit, Play } from '../../game/types';

describe('Rule Contracts: Base Layer', () => {
  const rules = composeRules([baseLayer]);

  describe('getLedSuit contract', () => {
    it('non-trump dominoes lead their higher pip', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES }).build();

      expect(rules.getLedSuit(state, DominoBuilder.from('6-2'))).toBe(6);
      expect(rules.getLedSuit(state, DominoBuilder.from('5-3'))).toBe(5);
      expect(rules.getLedSuit(state, DominoBuilder.from('4-0'))).toBe(4);
    });

    it('trump dominoes lead the trump suit', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      // Dominoes containing trump suit lead trump
      expect(rules.getLedSuit(state, DominoBuilder.from('6-3'))).toBe(TRES);
      expect(rules.getLedSuit(state, DominoBuilder.from('3-0'))).toBe(TRES);
      expect(rules.getLedSuit(state, DominoBuilder.from('3-3'))).toBe(TRES);
    });

    it('doubles-trump: doubles lead suit 7', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' }).build();

      expect(rules.getLedSuit(state, DominoBuilder.from('6-6'))).toBe(DOUBLES_AS_TRUMP);
      expect(rules.getLedSuit(state, DominoBuilder.from('0-0'))).toBe(DOUBLES_AS_TRUMP);
      // Non-doubles still lead higher pip
      expect(rules.getLedSuit(state, DominoBuilder.from('6-2'))).toBe(6);
    });
  });

  describe('suitsWithTrump contract', () => {
    it('non-trump dominoes belong to both their pip suits', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES }).build();

      const suits = rules.suitsWithTrump(state, DominoBuilder.from('6-2'));
      expect(suits).toContain(6);
      expect(suits).toContain(2);
      expect(suits).toHaveLength(2);
    });

    it('trump dominoes belong ONLY to trump suit (trump absorption)', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      // Domino with trump pip belongs only to trump
      const suits = rules.suitsWithTrump(state, DominoBuilder.from('6-3'));
      expect(suits).toEqual([TRES]);

      // Double of trump belongs only to trump
      const doubleSuits = rules.suitsWithTrump(state, DominoBuilder.from('3-3'));
      expect(doubleSuits).toEqual([TRES]);
    });

    it('doubles-trump: doubles belong only to suit 7', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' }).build();

      expect(rules.suitsWithTrump(state, DominoBuilder.from('6-6'))).toEqual([7 as LedSuit]);
      expect(rules.suitsWithTrump(state, DominoBuilder.from('0-0'))).toEqual([7 as LedSuit]);

      // Non-doubles belong to both pips
      expect(rules.suitsWithTrump(state, DominoBuilder.from('6-2'))).toContain(6);
      expect(rules.suitsWithTrump(state, DominoBuilder.from('6-2'))).toContain(2);
    });
  });

  describe('canFollow contract', () => {
    it('domino can follow if it belongs to led suit', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES }).build();

      // 6-2 can follow sixes (higher pip) and deuces (lower pip)
      expect(rules.canFollow(state, 6 as LedSuit, DominoBuilder.from('6-2'))).toBe(true);
      expect(rules.canFollow(state, 2 as LedSuit, DominoBuilder.from('6-2'))).toBe(true);

      // 6-2 cannot follow fours
      expect(rules.canFollow(state, 4 as LedSuit, DominoBuilder.from('6-2'))).toBe(false);
    });

    it('trump dominoes can only follow trump suit', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      // 6-3 is trump, can only follow threes (trump)
      expect(rules.canFollow(state, TRES as LedSuit, DominoBuilder.from('6-3'))).toBe(true);
      expect(rules.canFollow(state, 6 as LedSuit, DominoBuilder.from('6-3'))).toBe(false);
    });

    it('doubles-trump: doubles can only follow suit 7', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' }).build();

      // Doubles can follow suit 7
      expect(rules.canFollow(state, DOUBLES_AS_TRUMP, DominoBuilder.from('4-4'))).toBe(true);

      // Doubles cannot follow regular suits
      expect(rules.canFollow(state, 4 as LedSuit, DominoBuilder.from('4-4'))).toBe(false);
    });
  });

  describe('rankInTrick contract', () => {
    it('trump beats following suit beats slough', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES })
        .with({ currentSuit: SIXES })
        .build();

      const trumpRank = rules.rankInTrick(state, SIXES as LedSuit, DominoBuilder.from('5-3'));
      const followRank = rules.rankInTrick(state, SIXES as LedSuit, DominoBuilder.from('6-2'));
      const sloughRank = rules.rankInTrick(state, SIXES as LedSuit, DominoBuilder.from('5-4'));

      expect(trumpRank).toBeGreaterThan(followRank);
      expect(followRank).toBeGreaterThan(sloughRank);
    });

    it('higher pip sum beats lower within same tier', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .with({ currentSuit: SIXES })
        .build();

      // Both follow sixes, 6-5 (11) beats 6-2 (8)
      const high = rules.rankInTrick(state, SIXES as LedSuit, DominoBuilder.from('6-5'));
      const low = rules.rankInTrick(state, SIXES as LedSuit, DominoBuilder.from('6-2'));

      expect(high).toBeGreaterThan(low);
    });
  });

  describe('isTrump contract', () => {
    it('regular suit: dominoes with trump pip are trump', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      expect(rules.isTrump(state, DominoBuilder.from('6-3'))).toBe(true);
      expect(rules.isTrump(state, DominoBuilder.from('3-0'))).toBe(true);
      expect(rules.isTrump(state, DominoBuilder.from('3-3'))).toBe(true);
      expect(rules.isTrump(state, DominoBuilder.from('6-2'))).toBe(false);
    });

    it('doubles-trump: all doubles are trump', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' }).build();

      expect(rules.isTrump(state, DominoBuilder.from('6-6'))).toBe(true);
      expect(rules.isTrump(state, DominoBuilder.from('0-0'))).toBe(true);
      expect(rules.isTrump(state, DominoBuilder.from('6-5'))).toBe(false);
    });
  });

  describe('calculateTrickWinner contract', () => {
    it('highest trump wins over followers', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES })
        .with({ currentSuit: SIXES })
        .build();

      const trick: Play[] = [
        { player: 0, domino: DominoBuilder.from('6-5') }, // follows
        { player: 1, domino: DominoBuilder.from('5-3') }, // trump
        { player: 2, domino: DominoBuilder.from('6-4') }, // follows
        { player: 3, domino: DominoBuilder.from('6-3') }, // highest trump
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(3);
    });

    it('highest follower wins when no trump played', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .with({ currentSuit: SIXES })
        .build();

      const trick: Play[] = [
        { player: 0, domino: DominoBuilder.from('6-2') },
        { player: 1, domino: DominoBuilder.from('6-5') }, // highest
        { player: 2, domino: DominoBuilder.from('6-0') },
        { player: 3, domino: DominoBuilder.from('5-4') }, // slough
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(1);
    });
  });
});

describe('Rule Contracts: Nello Layer', () => {
  const rules = composeRules([baseLayer, nelloLayer]);

  describe('getLedSuit override', () => {
    it('doubles lead suit 7 in nello', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      expect(rules.getLedSuit(state, DominoBuilder.from('6-6'))).toBe(7);
      expect(rules.getLedSuit(state, DominoBuilder.from('0-0'))).toBe(7);
      // Non-doubles still lead higher pip
      expect(rules.getLedSuit(state, DominoBuilder.from('6-2'))).toBe(6);
    });

    it('passes through to base when not nello', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      // Should use base layer behavior
      expect(rules.getLedSuit(state, DominoBuilder.from('6-3'))).toBe(TRES);
    });
  });

  describe('suitsWithTrump override', () => {
    it('doubles belong only to suit 7 in nello', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      expect(rules.suitsWithTrump(state, DominoBuilder.from('4-4'))).toEqual([7 as LedSuit]);

      // Non-doubles belong to both pips
      expect(rules.suitsWithTrump(state, DominoBuilder.from('6-2'))).toContain(6);
      expect(rules.suitsWithTrump(state, DominoBuilder.from('6-2'))).toContain(2);
    });
  });

  describe('canFollow override', () => {
    it('doubles can only follow suit 7 in nello', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      // Doubles can follow suit 7
      expect(rules.canFollow(state, 7 as LedSuit, DominoBuilder.from('4-4'))).toBe(true);

      // Doubles cannot follow regular suits
      expect(rules.canFollow(state, 4 as LedSuit, DominoBuilder.from('4-4'))).toBe(false);
    });

    it('non-doubles can follow their pip suits', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      expect(rules.canFollow(state, 6 as LedSuit, DominoBuilder.from('6-2'))).toBe(true);
      expect(rules.canFollow(state, 2 as LedSuit, DominoBuilder.from('6-2'))).toBe(true);
    });
  });

  describe('rankInTrick override', () => {
    it('no trump ranking in nello - only suit following matters', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      // When suit 7 (doubles) is led, only doubles follow
      const doubleRank = rules.rankInTrick(state, 7 as LedSuit, DominoBuilder.from('4-4'));
      const sloughRank = rules.rankInTrick(state, 7 as LedSuit, DominoBuilder.from('6-5'));

      expect(doubleRank).toBeGreaterThan(sloughRank);
    });
  });

  describe('isTrump override', () => {
    it('nothing is trump in nello', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' }).build();

      // isTrump should return false for all dominoes in nello
      // (nello layer doesn't override isTrump, so base returns false for nello trump type)
      expect(rules.isTrump(state, DominoBuilder.from('6-6'))).toBe(false);
      expect(rules.isTrump(state, DominoBuilder.from('6-3'))).toBe(false);
    });
  });

  describe('isTrickComplete override', () => {
    it('tricks complete at 3 plays in nello (partner sits out)', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'nello' })
        .withCurrentTrick([
          { player: 0, domino: DominoBuilder.from('6-5') },
          { player: 1, domino: DominoBuilder.from('5-4') },
          { player: 2, domino: DominoBuilder.from('4-3') },
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('tricks require 4 plays for base contract', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentTrick([
          { player: 0, domino: DominoBuilder.from('6-5') },
          { player: 1, domino: DominoBuilder.from('5-4') },
          { player: 2, domino: DominoBuilder.from('4-3') },
        ])
        .build();

      expect(rules.isTrickComplete(state)).toBe(false);
    });
  });
});

describe('Rule Contracts: Sevens Layer', () => {
  const rules = composeRules([baseLayer, sevensLayer]);

  describe('calculateTrickWinner override', () => {
    it('closest to 7 pips wins in sevens', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();

      const trick: Play[] = [
        { player: 0, domino: DominoBuilder.from('6-5') }, // 11 pips, distance 4
        { player: 1, domino: DominoBuilder.from('4-3') }, // 7 pips, distance 0 - WINS
        { player: 2, domino: DominoBuilder.from('6-6') }, // 12 pips, distance 5
        { player: 3, domino: DominoBuilder.from('5-4') }, // 9 pips, distance 2
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(1);
    });

    it('first player wins ties in sevens', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' }).build();

      const trick: Play[] = [
        { player: 0, domino: DominoBuilder.from('5-2') }, // 7 pips, distance 0 - WINS (first)
        { player: 1, domino: DominoBuilder.from('4-3') }, // 7 pips, distance 0 (second)
        { player: 2, domino: DominoBuilder.from('6-1') }, // 7 pips, distance 0 (third)
        { player: 3, domino: DominoBuilder.from('5-5') }, // 10 pips, distance 3
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(0);
    });
  });

  describe('isValidPlay override', () => {
    it('only dominoes closest to 7 are valid in sevens', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' })
        .withPlayerHand(0, ['6-5', '4-3', '6-6'])
        .build();

      // 4-3 (7 pips) is closest to 7
      expect(rules.isValidPlay(state, DominoBuilder.from('4-3'), 0)).toBe(true);
      // 6-5 (11 pips) is not closest
      expect(rules.isValidPlay(state, DominoBuilder.from('6-5'), 0)).toBe(false);
      // 6-6 (12 pips) is not closest
      expect(rules.isValidPlay(state, DominoBuilder.from('6-6'), 0)).toBe(false);
    });
  });

  describe('getValidPlays override', () => {
    it('returns only dominoes closest to 7 in sevens', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' })
        .withPlayerHand(0, ['6-5', '4-3', '5-2', '6-6'])
        .build();

      const validPlays = rules.getValidPlays(state, 0);

      // 4-3 and 5-2 both have 7 pips
      expect(validPlays).toHaveLength(2);
      expect(validPlays.some(d => d.high === 4 && d.low === 3)).toBe(true);
      expect(validPlays.some(d => d.high === 5 && d.low === 2)).toBe(true);
    });
  });

  describe('checkHandOutcome override', () => {
    it('hand ends early if bidding team loses a trick in sevens', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'sevens' })
        .with({ winningBidder: 0 })
        .withTricks([{
          plays: [
            { player: 0, domino: DominoBuilder.from('6-5') },
            { player: 1, domino: DominoBuilder.from('4-3') }, // opponent wins
            { player: 2, domino: DominoBuilder.from('6-4') },
            { player: 3, domino: DominoBuilder.from('5-4') },
          ],
          winner: 1, // opponent
          points: 0
        }])
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
    });
  });
});

describe('Rule Contract Invariants', () => {
  describe('contract consistency across layers', () => {
    it('all layers implement the same GameRules interface methods', () => {
      const base = composeRules([baseLayer]);
      const withNello = composeRules([baseLayer, nelloLayer]);
      const withSevens = composeRules([baseLayer, sevensLayer]);

      // All should have the 6 core rule functions
      const coreMethods = [
        'getLedSuit',
        'suitsWithTrump',
        'canFollow',
        'rankInTrick',
        'calculateTrickWinner',
        'isTrump'
      ] as const;

      for (const method of coreMethods) {
        expect(typeof base[method]).toBe('function');
        expect(typeof withNello[method]).toBe('function');
        expect(typeof withSevens[method]).toBe('function');
      }
    });

    it('layer overrides only activate for their trump type', () => {
      const withNello = composeRules([baseLayer, nelloLayer]);
      const withSevens = composeRules([baseLayer, sevensLayer]);

      // Nello override should not affect base trump
      const baseState = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();
      expect(withNello.getLedSuit(baseState, DominoBuilder.from('3-3'))).toBe(TRES);

      // Sevens override should not affect base trump
      expect(withSevens.getValidPlays(baseState, 0)).toBeDefined();
    });
  });

  describe('suitsWithTrump and canFollow consistency', () => {
    it('canFollow returns true iff led suit is in suitsWithTrump result', () => {
      const rules = composeRules([baseLayer]);
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      const domino = DominoBuilder.from('6-3');
      const suits = rules.suitsWithTrump(state, domino);

      // For each possible led suit 0-7, canFollow should match suitsWithTrump
      for (let led = 0; led <= 7; led++) {
        const canFollow = rules.canFollow(state, led as LedSuit, domino);
        const suitInList = suits.includes(led as LedSuit);
        expect(canFollow).toBe(suitInList);
      }
    });
  });

  describe('getLedSuit and suitsWithTrump consistency', () => {
    it('getLedSuit result is always in suitsWithTrump result', () => {
      const rules = composeRules([baseLayer]);
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: TRES }).build();

      const testDominoes = ['6-5', '6-3', '3-3', '0-0', '6-0'].map(id => DominoBuilder.from(id));

      for (const domino of testDominoes) {
        const ledSuit = rules.getLedSuit(state, domino);
        const suits = rules.suitsWithTrump(state, domino);
        expect(suits).toContain(ledSuit);
      }
    });
  });
});
