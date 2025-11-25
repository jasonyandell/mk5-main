import { describe, test, expect } from 'vitest';
import { composeRules } from '../../game/layers/compose';
import { baseLayer } from '../../game/layers';
import { StateBuilder } from '../helpers';
import type { Domino, GameState, TrumpSelection, LedSuitOrNone } from '../../game/types';
import { FIVES, SIXES, DOUBLES_AS_TRUMP, NO_LEAD_SUIT } from '../../game/types';

const rules = composeRules([baseLayer]);

describe('Doubles Trump Renege Validation', () => {
  const doublesAreTrump: TrumpSelection = { type: 'doubles' };
  const fivesAreTrump: TrumpSelection = { type: 'suit', suit: FIVES };
  const sixesAreTrump: TrumpSelection = { type: 'suit', suit: SIXES };

  function createTestStateWithHand(hand: Domino[], currentTrick: { player: number; domino: Domino }[], trump: TrumpSelection, currentSuit: LedSuitOrNone = NO_LEAD_SUIT): GameState {
    const state = StateBuilder.inPlayingPhase(trump)
      .withCurrentTrick(currentTrick)
      .withCurrentPlayer(1)
      .withPlayerHand(1, hand)
      .with({ currentSuit })
      .build();
    return state;
  }

  describe('When doubles are trump', () => {
    test('6-6 leads, suit is trumps (doubles)', () => {
      const currentTrick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      
      const handWithDouble: Domino[] = [
        { high: 2, low: 2, id: "2-2" }, // Must play this double
        { high: 6, low: 4, id: "6-4" },
        { high: 5, low: 0, id: "5-0" }
      ];
      
      const state = createTestStateWithHand(handWithDouble, currentTrick, doublesAreTrump, DOUBLES_AS_TRUMP); // Doubles led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe("2-2");
    });

    test('6-5 leads with 6s trump, but doubles are trump - suit is 6s', () => {
      // If someone leads 6-5 and 6s are trump, the suit led is 6s
      // But since doubles are trump, only 6s (not trump) can follow
      const currentTrick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      
      const handWith6sAndDoubles: Domino[] = [
        { high: 6, low: 2, id: "6-2" }, // Can follow 6s
        { high: 3, low: 3, id: "3-3" }, // Double - cannot follow 6s since doubles are trump
        { high: 4, low: 1, id: "4-1" }  // Cannot follow
      ];
      
      const state = createTestStateWithHand(handWith6sAndDoubles, currentTrick, doublesAreTrump, SIXES); // 6s led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe("6-2");
    });

    test('6-5 leads when 5s are trump, suit is 5s', () => {
      const currentTrick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      
      const handWith5sAndDoubles: Domino[] = [
        { high: 5, low: 2, id: "5-2" }, // Can follow 5s (trump)
        { high: 3, low: 3, id: "3-3" }, // Double - cannot follow 5s trump
        { high: 6, low: 4, id: "6-4" }  // Cannot follow
      ];
      
      const state = createTestStateWithHand(handWith5sAndDoubles, currentTrick, fivesAreTrump, FIVES); // 5s led (trump)
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe("5-2");
    });

    test('Renege detection - playing non-double when double is available', () => {
      const currentTrick = [{ player: 0, domino: { high: 4, low: 4, id: "4-4" } }];
      
      const hand: Domino[] = [
        { high: 1, low: 1, id: "1-1" }, // Available double - must play
        { high: 6, low: 3, id: "6-3" }  // Invalid play when double available
      ];
      
      const state = createTestStateWithHand(hand, currentTrick, doublesAreTrump, DOUBLES_AS_TRUMP); // Doubles led
      expect(rules.isValidPlay(state, { high: 6, low: 3, id: "6-3" }, 1)).toBe(false);
      expect(rules.isValidPlay(state, { high: 1, low: 1, id: "1-1" }, 1)).toBe(true);
    });

    test('Multiple doubles available - all are valid', () => {
      const currentTrick = [{ player: 0, domino: { high: 0, low: 0, id: "0-0" } }];
      
      const handWithMultipleDoubles: Domino[] = [
        { high: 2, low: 2, id: "2-2" },
        { high: 5, low: 5, id: "5-5" },
        { high: 6, low: 1, id: "6-1" }
      ];
      
      const state = createTestStateWithHand(handWithMultipleDoubles, currentTrick, doublesAreTrump, DOUBLES_AS_TRUMP); // Doubles led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(2);
      expect(validPlays.every(d => d.high === d.low)).toBe(true);
    });
  });

  describe('When specific suit is trump (not doubles)', () => {
    test('6-6 leads when 6s are trump - suit is 6s, double follows as 6', () => {
      const currentTrick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      
      const handWith6s: Domino[] = [
        { high: 6, low: 2, id: "6-2" },  // Can follow 6s
        { high: 5, low: 5, id: "5-5" },  // Cannot follow (different suit)
        { high: 4, low: 1, id: "4-1" }   // Cannot follow
      ];
      
      const state = createTestStateWithHand(handWith6s, currentTrick, sixesAreTrump, SIXES); // 6s led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe("6-2");
    });

    test('5-5 leads when 6s are trump - suit is 5s, must follow with 5s', () => {
      const currentTrick = [{ player: 0, domino: { high: 5, low: 5, id: "5-5" } }];
      
      const handWith5sAnd6s: Domino[] = [
        { high: 5, low: 3, id: "5-3" },  // Can follow 5s
        { high: 6, low: 4, id: "6-4" },  // Cannot follow (different suit, even though 6s are trump)
        { high: 2, low: 1, id: "2-1" }   // Cannot follow
      ];
      
      const state = createTestStateWithHand(handWith5sAnd6s, currentTrick, sixesAreTrump, FIVES); // 5s led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);  
      expect(validPlays[0]?.id).toBe("5-3");
    });

    test('Complex scenario - 6-5 leads when 6s are trump', () => {
      // 6-5 with 6s trump means trump was led (6s)
      const currentTrick = [{ player: 0, domino: { high: 6, low: 5, id: "6-5" } }];
      
      const hand: Domino[] = [
        { high: 6, low: 6, id: "6-6" },  // Can follow trump (6s)
        { high: 6, low: 3, id: "6-3" },  // Can follow trump (6s)
        { high: 5, low: 4, id: "5-4" },  // Cannot follow trump
        { high: 2, low: 1, id: "2-1" }   // Cannot follow trump
      ];
      
      const state = createTestStateWithHand(hand, currentTrick, sixesAreTrump, SIXES); // 6s led (trump)
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(2);
      expect(validPlays.every(d => d.high === 6 || d.low === 6)).toBe(true);
    });
  });

  describe('Edge cases and comprehensive scenarios', () => {
    test('No trump (follow-me) - doubles belong to natural suit', () => {
      const noTrump: TrumpSelection = { type: 'no-trump' };
      const currentTrick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      
      const hand: Domino[] = [
        { high: 6, low: 3, id: "6-3" },  // Can follow 6s
        { high: 5, low: 4, id: "5-4" },  // Cannot follow
        { high: 2, low: 2, id: "2-2" }   // Cannot follow (different suit)
      ];
      
      const state = createTestStateWithHand(hand, currentTrick, noTrump, SIXES); // 6s led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe("6-3");
    });

    test('Leading first domino - all plays valid', () => {
      const emptyTrick: { player: number; domino: Domino }[] = [];
      
      const hand: Domino[] = [
        { high: 6, low: 6, id: "6-6" },
        { high: 5, low: 3, id: "5-3" },
        { high: 2, low: 1, id: "2-1" }
      ];
      
      const state = createTestStateWithHand(hand, emptyTrick, doublesAreTrump, NO_LEAD_SUIT); // No suit led yet
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3);
    });

    test('Cannot follow suit at all - all plays valid', () => {
      const currentTrick = [{ player: 0, domino: { high: 6, low: 6, id: "6-6" } }];
      
      const handWithoutDoubles: Domino[] = [
        { high: 5, low: 3, id: "5-3" },
        { high: 4, low: 2, id: "4-2" },
        { high: 1, low: 0, id: "1-0" }
      ];
      
      const state = createTestStateWithHand(handWithoutDoubles, currentTrick, doublesAreTrump, DOUBLES_AS_TRUMP); // Doubles led
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3); // All valid when can't follow
    });

    test('Comprehensive renege validation - mixed scenarios', () => {
      // Test multiple scenarios in sequence
      const scenarios = [
        {
          name: "Doubles trump, 3-3 led, must play double",
          currentTrick: [{ player: 0, domino: { high: 3, low: 3, id: "3-3" } }],
          hand: [
            { high: 1, low: 1, id: "1-1" },
            { high: 5, low: 2, id: "5-2" }
          ],
          trump: doublesAreTrump,
          currentSuit: DOUBLES_AS_TRUMP, // Doubles led
          expectedValid: ["1-1"],
          expectedInvalid: ["5-2"]
        },
        {
          name: "5s trump, 5-2 led, must follow 5s", 
          currentTrick: [{ player: 0, domino: { high: 5, low: 2, id: "5-2" } }],
          hand: [
            { high: 5, low: 4, id: "5-4" },
            { high: 6, low: 6, id: "6-6" },
            { high: 3, low: 1, id: "3-1" }
          ],
          trump: fivesAreTrump,
          currentSuit: FIVES, // 5s led
          expectedValid: ["5-4"],
          expectedInvalid: ["6-6", "3-1"]
        }
      ];

      scenarios.forEach(scenario => {
        const state = createTestStateWithHand(scenario.hand, scenario.currentTrick, scenario.trump, scenario.currentSuit);
        
        scenario.expectedValid.forEach(dominoId => {
          const domino = scenario.hand.find(d => d.id === dominoId);
          if (!domino) throw new Error(`Domino ${dominoId} not found in hand`);
          expect(rules.isValidPlay(state, domino, 1)).toBe(true);
        });

        scenario.expectedInvalid.forEach(dominoId => {
          const domino = scenario.hand.find(d => d.id === dominoId);
          if (!domino) throw new Error(`Domino ${dominoId} not found in hand`);
          expect(rules.isValidPlay(state, domino, 1)).toBe(false);
        });
      });
    });
  });
});