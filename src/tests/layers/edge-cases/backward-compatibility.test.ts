import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { composeRules, baseLayer } from '../../../game/layers';
import { createInitialState } from '../../../game/core/state';
import { StateBuilder } from '../../helpers';
import { createTestContext } from '../../helpers/executionContext';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../../game/types';


/**
 * Backward Compatibility - Ensures layer system doesn't break existing functionality:
 * - Layer interface stability
 * - Rule method signatures
 * - GameRules interface (composeRules)
 * - Action structure
 * - GameState structure
 * - Public API exports
 */
describe('Backward Compatibility', () => {
  describe('Layer Interface Stability', () => {
    it('should have expected layer properties', () => {
      expect(baseLayer).toHaveProperty('name');
      expect(baseLayer).toHaveProperty('rules');
      expect(baseLayer.name).toBe('base');
      expect(typeof baseLayer.rules).toBe('object');
    });

    it('should compose rules correctly', () => {
      const rules = composeRules([baseLayer]);

      // Verify GameRules interface
      expect(rules).toHaveProperty('isValidBid');
      expect(rules).toHaveProperty('isValidTrump');
      expect(rules).toHaveProperty('isValidPlay');
      expect(rules).toHaveProperty('isTrickComplete');
      expect(rules).toHaveProperty('checkHandOutcome');

      // All should be functions
      expect(typeof rules.isValidBid).toBe('function');
      expect(typeof rules.isValidTrump).toBe('function');
      expect(typeof rules.isValidPlay).toBe('function');
      expect(typeof rules.isTrickComplete).toBe('function');
      expect(typeof rules.checkHandOutcome).toBe('function');
    });
  });

  describe('Rule Method Signatures', () => {
    it('should accept correct parameters for isTrickComplete', () => {
      const rules = composeRules([baseLayer]);
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .build();

      // 3 plays - incomplete
      state.currentTrick = [
        { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
        { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } },
        { player: 2, domino: { id: '3', high: TRES, low: BLANKS } }
      ];
      expect(rules.isTrickComplete(state)).toBe(false);

      // 4 plays - complete
      state.currentTrick.push({ player: 3, domino: { id: '4', high: FOURS, low: BLANKS } });
      expect(rules.isTrickComplete(state)).toBe(true);
    });

    it('should accept correct parameters for checkHandOutcome', () => {
      const rules = composeRules([baseLayer]);
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withTricks([
          { plays: [], winner: 0, points: 10, ledSuit: BLANKS },
          { plays: [], winner: 1, points: 5, ledSuit: ACES },
          { plays: [], winner: 2, points: 10, ledSuit: DEUCES },
          { plays: [], winner: 3, points: 5, ledSuit: TRES },
          { plays: [], winner: 0, points: 10, ledSuit: FOURS },
          { plays: [], winner: 1, points: 0, ledSuit: FIVES },
          { plays: [], winner: 2, points: 5, ledSuit: SIXES }
        ])
        .build();

      const outcome = rules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toBe('All tricks played');
    });
  });

  describe('Action Structure', () => {
    it('should accept all action types with correct shape', () => {
      // Trump selection actions
      let state = StateBuilder.inTrumpSelection(0, 35).build();
      let result = executeAction(state, {
        type: 'select-trump',
        player: 0,
        trump: { type: 'suit', suit: ACES }
      });
      expect(result.trump).toMatchObject({ type: 'suit', suit: ACES });

      result = executeAction(state, {
        type: 'select-trump',
        player: 0,
        trump: { type: 'doubles' }
      });
      expect(result.trump).toMatchObject({ type: 'doubles' });

      // Play action
      state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .withPlayerHand(0, ['1-0'])
        .build();
      result = executeAction(state, { type: 'play', player: 0, dominoId: '1-0' });
      expect(result.currentTrick.length).toBe(1);
      expect(result.currentTrick[0]).toMatchObject({
        player: 0,
        domino: expect.objectContaining({ id: '1-0' })
      });
    });
  });

  describe('GameState Structure & API', () => {
    it('should maintain critical structures and return valid actions', () => {
      const state = createInitialState();

      // Core gameplay fields
      expect(state).toHaveProperty('phase');
      expect(state).toHaveProperty('players');
      expect(state).toHaveProperty('currentPlayer');
      expect(state).toHaveProperty('trump');
      expect(state).toHaveProperty('tricks');
      expect(state).toHaveProperty('actionHistory');
      expect(state).toHaveProperty('bids');
      expect(state).toHaveProperty('teamScores');
      expect(state).toHaveProperty('consensus');

      // Nested structures
      expect(state.players[0]).toHaveProperty('id');
      expect(state.players[0]).toHaveProperty('hand');
      expect(state.players[0]).toHaveProperty('teamId');

      // getNextStates returns valid actions for all phases
      const ctx = createTestContext();
      let transitions = getNextStates(state, ctx);
      expect(transitions.length).toBeGreaterThan(0);

      // Trump selection phase
      const trumpState = StateBuilder.inTrumpSelection(0, 35).build();
      transitions = getNextStates(trumpState, ctx);
      const trumpActions = transitions.filter((a: { id: string }) => a.id.startsWith('trump-'));
      expect(trumpActions.length).toBe(9); // 7 suits + doubles + no-trump

      // Playing phase - verify we get play actions
      const playState = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .build();
      transitions = getNextStates(playState, ctx);
      const playActions = transitions.filter((a: { id: string }) => a.id.startsWith('play-'));
      expect(playActions.length).toBeGreaterThan(0); // Should have play actions based on hand
    });
  });

  describe('API Stability & Regression', () => {
    it('should maintain all public exports and game flow integrity', () => {
      // Verify all exports exist
      expect(executeAction).toBeDefined();
      expect(typeof executeAction).toBe('function');
      expect(createInitialState).toBeDefined();
      expect(getNextStates).toBeDefined();
      expect(composeRules).toBeDefined();
      expect(baseLayer).toBeDefined();

      // executeAction signature compatibility (2 or 3 parameters)
      let state = StateBuilder.inTrumpSelection(0, 35).build();
      const action = {
        type: 'select-trump' as const,
        player: 0,
        trump: { type: 'suit' as const, suit: ACES }
      };

      // 2 parameters (state + action)
      let result = executeAction(state, action);
      expect(result.phase).toBe('playing');

      // 3 parameters (state + action + rules)
      const rules = composeRules([baseLayer]);
      result = executeAction(state, action, rules);
      expect(result.phase).toBe('playing');

      // Action history tracking - use play action
      state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withPlayerHand(0, ['1-0'])
        .build();
      const initialLength = state.actionHistory.length;
      state = executeAction(state, { type: 'play', player: 0, dominoId: '1-0' });
      expect(state.actionHistory.length).toBe(initialLength + 1);
      expect(state.actionHistory[state.actionHistory.length - 1]).toMatchObject({
        type: 'play',
        player: 0,
        dominoId: '1-0'
      });
    });
  });
});
