import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { composeRules, baseRuleSet } from '../../../game/rulesets';
import { createInitialState } from '../../../game/core/state';
import { dealDominoesWithSeed } from '../../../game/core/dominoes';
import { StateBuilder } from '../../helpers';
import { createTestContext } from '../../helpers/executionContext';
import type { GameState } from '../../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../../game/types';

// Helper function to deal dominoes to a state and set to bidding phase
function dealHands(state: GameState): GameState {
  const hands = dealDominoesWithSeed(state.shuffleSeed || 12345);
  return {
    ...state,
    phase: 'bidding', // Move from setup to bidding
    players: state.players.map((player, index) => ({
      ...player,
      hand: hands[index] || []
    }))
  };
}

/**
 * Backward Compatibility - Regression tests ensuring ruleset system doesn't break existing functionality:
 * - executeAction works without rules parameter (uses defaultRules)
 * - Standard 42 gameplay unaffected
 * - All existing bid types work (points, marks, pass)
 * - All existing trump types work (suit, doubles, no-trump)
 * - GameState structure unchanged
 * - Capability system still works correctly
 * - No breaking changes to public API
 */
describe('Backward Compatibility', () => {
  describe('executeAction Default Rules', () => {
    it('should work without rules parameter (uses defaultRules)', () => {
      let state = createInitialState();

      // Deal hands to players so bids are valid
      state = dealHands(state);

      // Execute action without passing rules parameter (using current player 0)
      state = executeAction(state, {
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 30
      });

      expect(state.bids.length).toBe(1);
      expect(state.bids[0]?.type).toBe('points');
      expect(state.bids[0]?.value).toBe(30);
      expect(state.currentPlayer).toBe(1);
    });

    it('should handle full bidding round without rules parameter', () => {
      let state = createInitialState();

      // Deal hands to players
      state = dealHands(state);

      // Player 0 (left of dealer) bids first
      state = executeAction(state, {
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 30
      });

      // Others pass
      state = executeAction(state, { type: 'pass', player: 1 });
      state = executeAction(state, { type: 'pass', player: 2 });
      state = executeAction(state, { type: 'pass', player: 3 });

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
    });

    it('should handle trump selection without rules parameter', () => {
      let state = StateBuilder.inTrumpSelection(0, 35).build();

      state = executeAction(state, {
        type: 'select-trump',
        player: 0,
        trump: { type: 'suit', suit: ACES }
      });

      expect(state.phase).toBe('playing');
      expect(state.trump.type).toBe('suit');
      expect(state.trump.suit).toBe(ACES);
    });

    it('should handle playing without rules parameter', () => {
      let state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .withPlayerHand(0, ['1-0'])
        .withPlayerHand(1, ['2-0'])
        .withPlayerHand(2, ['3-0'])
        .withPlayerHand(3, ['4-0'])
        .build();

      state = executeAction(state, {
        type: 'play',
        player: 0,
        dominoId: '1-0'
      });

      expect(state.currentTrick.length).toBe(1);
      expect(state.currentPlayer).toBe(1);
    });
  });

  describe('Standard 42 Gameplay Unaffected', () => {
    it('should play standard 42 hand without special contracts', () => {
      let state = createInitialState();

      // Deal hands to players
      state = dealHands(state);

      // Standard bidding - player 0 (left of dealer) bids first
      state = executeAction(state, {
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 35
      });
      state = executeAction(state, { type: 'pass', player: 1 });
      state = executeAction(state, { type: 'pass', player: 2 });
      state = executeAction(state, { type: 'pass', player: 3 });

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);

      // Standard trump selection
      state = executeAction(state, {
        type: 'select-trump',
        player: 0,
        trump: { type: 'suit', suit: SIXES }
      });

      expect(state.phase).toBe('playing');
      expect(state.trump.type).toBe('suit');
      expect(state.trump.suit).toBe(SIXES);
      expect(state.currentPlayer).toBe(0); // Bidder leads
    });

    it('should complete standard trick with 4 plays', () => {
      const baseRules = composeRules([baseRuleSet]);
      let state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .build();

      // Add some plays
      state.currentTrick = [
        { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
        { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } },
        { player: 2, domino: { id: '3', high: TRES, low: BLANKS } }
      ];

      // Not complete with 3 plays
      expect(baseRules.isTrickComplete(state)).toBe(false);

      // Add 4th play
      state.currentTrick.push({ player: 3, domino: { id: '4', high: FOURS, low: BLANKS } });

      // Complete with 4 plays
      expect(baseRules.isTrickComplete(state)).toBe(true);
    });

    it('should not have early termination in standard game', () => {
      const baseRules = composeRules([baseRuleSet]);
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: DEUCES })
        .withWinningBid(0, { type: 'points', value: 35, player: 0 })
        .withCurrentPlayer(0)
        .withTricks([
          { plays: [], winner: 1, points: 10, ledSuit: ACES }, // Opponent won
          { plays: [], winner: 3, points: 5, ledSuit: DEUCES }, // Opponent won
          { plays: [], winner: 1, points: 0, ledSuit: TRES }   // Opponent won
        ])
        .build();

      // Standard game should not terminate early
      const outcome = baseRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(false); // Continue playing all 7 tricks
    });

    it('should play all 7 tricks in standard game', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .withTricks([
          { plays: [], winner: 0, points: 0, ledSuit: BLANKS },
          { plays: [], winner: 1, points: 5, ledSuit: ACES },
          { plays: [], winner: 2, points: 10, ledSuit: DEUCES },
          { plays: [], winner: 3, points: 5, ledSuit: TRES },
          { plays: [], winner: 0, points: 10, ledSuit: FOURS },
          { plays: [], winner: 1, points: 0, ledSuit: FIVES },
          { plays: [], winner: 2, points: 5, ledSuit: SIXES }
        ])
        .build();

      // All 7 tricks played
      expect(state.tricks.length).toBe(7);

      // Base ruleset returns determined when all 7 tricks played
      const baseRules = composeRules([baseRuleSet]);
      const outcome = baseRules.checkHandOutcome(state);
      expect(outcome.isDetermined).toBe(true);
      expect((outcome as { isDetermined: true; reason: string }).reason).toBe('All tricks played');
    });
  });

  describe('Existing Bid Types Still Work', () => {
    it('should accept points bids', () => {
      let state = createInitialState();
      state = dealHands(state);

      for (let points = 30; points <= 41; points++) {
        const testState = { ...state, currentPlayer: 1, bids: [] };

        const result = executeAction(testState, {
          type: 'bid',
          player: 1,
          bid: 'points',
          value: points
        });

        expect(result.bids.length).toBe(1);
        expect(result.bids[0]?.type).toBe('points');
        expect(result.bids[0]?.value).toBe(points);
      }
    });

    it('should accept marks bids', () => {
      let state = createInitialState();
      state = dealHands(state);

      // Test marks 1 and 2 as opening bids (valid)
      for (let marks = 1; marks <= 2; marks++) {
        const testState = { ...state, currentPlayer: 0, bids: [] };

        const result = executeAction(testState, {
          type: 'bid',
          player: 0,
          bid: 'marks',
          value: marks
        });

        expect(result.bids.length).toBe(1);
        expect(result.bids[0]?.type).toBe('marks');
        expect(result.bids[0]?.value).toBe(marks);
      }

      // Test marks 3-4 as subsequent bids (requires previous bid)
      for (let marks = 3; marks <= 4; marks++) {
        const testState = {
          ...state,
          currentPlayer: 1,
          bids: [{ type: 'marks' as const, value: marks - 1, player: 0 }]
        };

        const result = executeAction(testState, {
          type: 'bid',
          player: 1,
          bid: 'marks',
          value: marks
        });

        expect(result.bids.length).toBe(2);
        expect(result.bids[1]?.type).toBe('marks');
        expect(result.bids[1]?.value).toBe(marks);
      }
    });

    it('should accept pass bid', () => {
      let state = createInitialState();
      state = dealHands(state);

      // Player 0 (left of dealer) passes first
      state = executeAction(state, {
        type: 'pass',
        player: 0
      });

      expect(state.bids.length).toBe(1);
      expect(state.bids[0]?.type).toBe('pass');
      expect(state.currentPlayer).toBe(1);
    });

    it('should handle all-pass scenario (redeal)', () => {
      let state = createInitialState();
      state = dealHands(state);

      // All players pass in order starting with player 0 (left of dealer)
      state = executeAction(state, { type: 'pass', player: 0 });
      state = executeAction(state, { type: 'pass', player: 1 });
      state = executeAction(state, { type: 'pass', player: 2 });
      state = executeAction(state, { type: 'pass', player: 3 });

      expect(state.bids.length).toBe(4);
      expect(state.phase).toBe('bidding'); // Still in bidding, needs redeal
    });
  });

  describe('Existing Trump Types Still Work', () => {
    it('should accept suit trump (all 7 suits)', () => {
      for (let suit = 0; suit <= 6; suit++) {
        const state = StateBuilder.inTrumpSelection(0, 35).build();

        const result = executeAction(state, {
          type: 'select-trump',
          player: 0,
          trump: { type: 'suit', suit: suit as 0 | 1 | 2 | 3 | 4 | 5 | 6 }
        });

        expect(result.phase).toBe('playing');
        expect(result.trump.type).toBe('suit');
        expect(result.trump.suit).toBe(suit);
      }
    });

    it('should accept doubles trump', () => {
      const state = StateBuilder.inTrumpSelection(1, 30).build();

      const result = executeAction(state, {
        type: 'select-trump',
        player: 1,
        trump: { type: 'doubles' }
      });

      expect(result.phase).toBe('playing');
      expect(result.trump.type).toBe('doubles');
    });

    it('should accept no-trump', () => {
      const state = StateBuilder.inTrumpSelection(2, 41).build();

      const result = executeAction(state, {
        type: 'select-trump',
        player: 2,
        trump: { type: 'no-trump' }
      });

      expect(result.phase).toBe('playing');
      expect(result.trump.type).toBe('no-trump');
    });
  });

  describe('GameState Structure Unchanged', () => {
    it('should maintain all required GameState fields', () => {
      const state = createInitialState();

      // Core fields
      expect(state).toHaveProperty('phase');
      expect(state).toHaveProperty('players');
      expect(state).toHaveProperty('currentPlayer');
      expect(state).toHaveProperty('dealer');
      expect(state).toHaveProperty('bids');
      expect(state).toHaveProperty('currentBid');
      expect(state).toHaveProperty('winningBidder');
      expect(state).toHaveProperty('trump');
      expect(state).toHaveProperty('tricks');
      expect(state).toHaveProperty('currentTrick');
      expect(state).toHaveProperty('currentSuit');
      expect(state).toHaveProperty('teamScores');
      expect(state).toHaveProperty('teamMarks');
      expect(state).toHaveProperty('gameTarget');
      expect(state).toHaveProperty('shuffleSeed');
      expect(state).toHaveProperty('playerTypes');
      expect(state).toHaveProperty('consensus');
      expect(state).toHaveProperty('actionHistory');
      expect(state).toHaveProperty('initialConfig');
      expect(state).toHaveProperty('theme');
      expect(state).toHaveProperty('colorOverrides');
    });

    it('should maintain Player structure', () => {
      const state = createInitialState();
      const player = state.players[0];

      expect(player).toHaveProperty('id');
      expect(player).toHaveProperty('name');
      expect(player).toHaveProperty('hand');
      expect(player).toHaveProperty('teamId');
      expect(player).toHaveProperty('marks');
    });

    it('should maintain Trick structure', () => {
      const state = StateBuilder.inBiddingPhase()
        .withTricks([
          {
            plays: [
              { player: 0, domino: { id: '1', high: ACES, low: BLANKS } },
              { player: 1, domino: { id: '2', high: DEUCES, low: BLANKS } },
              { player: 2, domino: { id: '3', high: TRES, low: BLANKS } },
              { player: 3, domino: { id: '4', high: FOURS, low: BLANKS } }
            ],
            winner: 0,
            points: 0,
            ledSuit: ACES
          }
        ])
        .build();

      const trick = state.tricks[0];
      if (!trick) throw new Error('Expected at least one trick');
      expect(trick).toHaveProperty('plays');
      expect(trick).toHaveProperty('winner');
      expect(trick).toHaveProperty('points');
      expect(trick).toHaveProperty('ledSuit');
      expect(trick.plays).toHaveLength(4);
    });

    it('should maintain consensus structure', () => {
      const state = createInitialState();

      expect(state.consensus).toHaveProperty('completeTrick');
      expect(state.consensus).toHaveProperty('scoreHand');
      expect(state.consensus.completeTrick).toBeInstanceOf(Set);
      expect(state.consensus.scoreHand).toBeInstanceOf(Set);
    });
  });

  describe('getValidActions Compatibility', () => {
    it('should return valid actions for bidding phase', () => {
      const ctx = createTestContext();
      const state = createInitialState();
      const transitions = getNextStates(state, ctx);

      expect(transitions.length).toBeGreaterThan(0);

      // Should have pass action
      const passAction = transitions.find((a: { id: string }) => a.id === 'pass');
      expect(passAction).toBeDefined();

      // Should have some bid actions
      const bidActions = transitions.filter((a: { id: string }) => a.id.startsWith('bid-'));
      expect(bidActions.length).toBeGreaterThan(0);
    });

    it('should return valid actions for trump_selection phase', () => {
      const ctx = createTestContext();
      const state = StateBuilder.inTrumpSelection(0, 35).build();

      const transitions = getNextStates(state, ctx);

      expect(transitions.length).toBeGreaterThan(0);

      // Should have trump selection actions
      const trumpActions = transitions.filter((a: { id: string }) => a.id.startsWith('trump-'));
      expect(trumpActions.length).toBe(9); // 7 suits + doubles + no-trump
    });

    it('should return valid actions for playing phase', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .withPlayerHand(0, ['1-0', '2-0'])
        .withPlayerHand(1, [])
        .withPlayerHand(2, [])
        .withPlayerHand(3, [])
        .build();

      const ctx = createTestContext();
      const transitions = getNextStates(state, ctx);

      // Should have play actions
      const playActions = transitions.filter((a: { id: string }) => a.id.startsWith('play-'));
      expect(playActions.length).toBe(2); // Player has 2 dominoes
    });

    it('should return valid actions for scoring phase', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: ACES })
        .withWinningBid(0, { type: 'points', value: 30, player: 0 })
        .withCurrentPlayer(0)
        .withTeamScores(30, 12)
        .withTricks([
          { plays: [], winner: 0, points: 10, ledSuit: ACES },
          { plays: [], winner: 2, points: 10, ledSuit: DEUCES },
          { plays: [], winner: 0, points: 5, ledSuit: TRES },
          { plays: [], winner: 1, points: 5, ledSuit: FOURS },
          { plays: [], winner: 3, points: 0, ledSuit: FIVES },
          { plays: [], winner: 2, points: 5, ledSuit: SIXES },
          { plays: [], winner: 1, points: 7, ledSuit: BLANKS }
        ])
        .with({ phase: 'scoring' })
        .build();

      const ctx = createTestContext();
      const transitions = getNextStates(state, ctx);

      // Should have agree-complete-trick action for current player
      const agreeAction = transitions.find((a: { id: string }) =>
        a.id.startsWith('agree-')
      );
      expect(agreeAction).toBeDefined();
    });
  });

  describe('No Breaking Changes to Public API', () => {
    it('should export all expected functions from core', () => {
      // Using static imports at top of file, just verify they exist
      expect(executeAction).toBeDefined();
      expect(typeof executeAction).toBe('function');
      expect(createInitialState).toBeDefined();
      expect(typeof createInitialState).toBe('function');
      expect(getNextStates).toBeDefined();
      expect(typeof getNextStates).toBe('function');
    });

    it('should export all expected ruleset functions', () => {
      // Using static imports at top of file, just verify they exist
      expect(composeRules).toBeDefined();
      expect(typeof composeRules).toBe('function');
      expect(baseRuleSet).toBeDefined();
      expect(typeof baseRuleSet).toBe('object');
    });

    it('should maintain executeAction signature', () => {
      let state = createInitialState();
      state = dealHands(state);

      // Should work with 2 parameters (action only) - player 0 bids first
      state = executeAction(state, {
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 30
      });
      expect(state.bids.length).toBe(1);

      // Should work with 3 parameters (action + rules) - player 1 passes
      const rules = composeRules([baseRuleSet]);
      state = executeAction(state, {
        type: 'pass',
        player: 1
      }, rules);
      expect(state.bids.length).toBe(2);
    });
  });

  describe('Action History Tracking', () => {
    it('should continue tracking action history', () => {
      let state = createInitialState();
      state = dealHands(state);
      const initialHistoryLength = state.actionHistory.length;

      // Player 0 (left of dealer) bids first
      state = executeAction(state, {
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 30
      });

      expect(state.actionHistory.length).toBe(initialHistoryLength + 1);
      expect(state.actionHistory[state.actionHistory.length - 1]).toEqual({
        type: 'bid',
        player: 0,
        bid: 'points',
        value: 30
      });
    });

    it('should track all actions in a full round', () => {
      let state = createInitialState();
      state = dealHands(state);
      const startLength = state.actionHistory.length;

      // Player 0 (left of dealer) bids first, others pass in order
      state = executeAction(state, { type: 'bid', player: 0, bid: 'points', value: 30 });
      state = executeAction(state, { type: 'pass', player: 1 });
      state = executeAction(state, { type: 'pass', player: 2 });
      state = executeAction(state, { type: 'pass', player: 3 });
      state = executeAction(state, { type: 'select-trump', player: 0, trump: { type: 'suit', suit: ACES } });

      expect(state.actionHistory.length).toBe(startLength + 5);
    });
  });
});
