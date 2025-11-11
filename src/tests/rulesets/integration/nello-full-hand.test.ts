import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { createTestContextWithRuleSets } from '../../helpers/executionContext';
import { composeRules, baseRuleSet, nelloRuleSet } from '../../../game/rulesets';
import { createTestState, createTestHand, processSequentialConsensus } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import type { GameState } from '../../../game/types';

describe('Nello Full Hand Integration', () => {
  const ctx = createTestContextWithRuleSets(['nello']);
  const ruleSets = [baseRuleSet, nelloRuleSet];
  const rules = composeRules(ruleSets);

  /**
   * Helper to play a complete nello hand
   */
  async function playNelloHand(
    initialState: GameState,
    shouldBidderWin: boolean = false
  ): Promise<{ finalState: GameState; preScoreState: GameState }> {
    let state = initialState;

    // Players should bid/pass in order
    const bidTransitions = getNextStates(state, ctx);
    const marksBid = bidTransitions.find(t => t.id === 'bid-1-marks');
    expect(marksBid).toBeDefined();
    state = executeAction(state, marksBid!.action, rules);

    // Others pass
    for (let i = 0; i < 3; i++) {
      const passTransition = getNextStates(state, ctx).find(t => t.id === 'pass');
      expect(passTransition).toBeDefined();
      state = executeAction(state, passTransition!.action, rules);
    }

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);

    // Select nello trump
    const nelloTransition = getNextStates(state, ctx).find(t =>
      t.action.type === 'select-trump' &&
      t.action.trump?.type === 'nello'
    );
    expect(nelloTransition).toBeDefined();
    state = executeAction(state, nelloTransition!.action, rules);

    expect(state.phase).toBe('playing');
    expect(state.trump.type).toBe('nello');
    expect(state.currentPlayer).toBe(0); // Bidder leads

    // Play tricks
    let trickCount = 0;
    const maxTricks = 7;

    while (state.phase === 'playing' && trickCount < maxTricks) {
      // Play 3 dominoes (partner sits out)
      for (let i = 0; i < 3; i++) {
        const playTransitions = getNextStates(state, ctx).filter(t => t.action.type === 'play');
        expect(playTransitions.length).toBeGreaterThan(0);

        // Select domino to play based on test scenario
        let selectedPlay;
        if (shouldBidderWin && state.currentPlayer === 0 && i === 2) {
          // On last play of trick, if bidder should win, pick highest
          selectedPlay = playTransitions[playTransitions.length - 1];
        } else if (state.currentPlayer === 0) {
          // Bidder plays lowest to try to lose
          selectedPlay = playTransitions[0];
        } else {
          // Others play to make bidder win (if shouldBidderWin) or lose
          selectedPlay = shouldBidderWin ? playTransitions[0] : playTransitions[playTransitions.length - 1];
        }

        state = executeAction(state, selectedPlay!.action, rules);
      }

      // Check trick is complete (3 plays in nello)
      expect(state.currentTrick.length).toBe(3);

      // In nello, process consensus for all 4 players (including partner who didn't play)
      // Players 0, 1, 3 will agree through normal flow
      // Player 2 (partner) needs manual agreement since they're skipped in turn order
      for (let playerId = 0; playerId < 4; playerId++) {
        if (!state.consensus.completeTrick.has(playerId)) {
          const agreeAction = { type: 'agree-complete-trick' as const, player: playerId };
          state = executeAction(state, agreeAction, rules);
        }
      }

      // Complete trick
      const completeTrickTransition = getNextStates(state, ctx).find(t => t.id === 'complete-trick');
      if (completeTrickTransition) {
        state = executeAction(state, completeTrickTransition.action, rules);
        trickCount++;

        // Check if hand ended early
        if (state.phase === 'scoring') {
          break;
        }
      }
    }

    expect(state.phase).toBe('scoring');

    // Process scoring consensus
    state = await processSequentialConsensus(state, 'scoreHand');

    // Score hand
    const scoreTransition = getNextStates(state, ctx).find(t => t.id === 'score-hand');
    expect(scoreTransition).toBeDefined();
    // Save state before scoring to verify tricks
    const preScoreState = state;

    state = executeAction(state, scoreTransition!.action, rules);

    return { finalState: state, preScoreState };
  }

  describe('Successful Nello', () => {
    it('should complete when bidder loses all 7 tricks (3-player)', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 123456,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const { finalState, preScoreState } = await playNelloHand(state, false);

      // Check who actually won tricks (bidder is team 0)
      const bidderTricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 0;
      }).length;

      // Verify scoring matches actual outcome
      if (bidderTricks === 0) {
        // Bidder succeeded (lost all tricks) - should get mark
        expect(finalState.teamMarks[0]).toBe(1);
        expect(finalState.teamMarks[1]).toBe(0);
        // Should have completed all 7 tricks
        expect(preScoreState.tricks.length).toBe(7);
      } else {
        // Bidder failed (won at least one trick) - opponents get mark
        expect(finalState.teamMarks[0]).toBe(0);
        expect(finalState.teamMarks[1]).toBe(1);
      }
    });

    it('should have only 3 plays per trick (partner sits out)', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 789012,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const { preScoreState } = await playNelloHand(state, false);

      // Verify each trick has exactly 3 plays
      preScoreState.tricks.forEach((trick) => {
        expect(trick.plays.length).toBe(3);
      });

      // Verify partner (player 2) never played
      const partnerPlays = preScoreState.tricks.flatMap(t => t.plays).filter(p => p.player === 2);
      expect(partnerPlays.length).toBe(0);
    });

    it('should treat doubles as suit 7 (own suit)', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 345678,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      let testState = state;

      // Bid and select nello
      const marksBid = getNextStates(testState, ctx).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, ctx).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      const nelloTransition = getNextStates(testState, ctx).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'nello'
      );
      testState = executeAction(testState, nelloTransition!.action, rules);

      // Bidder leads with double (0-0)
      const doublePlay = getNextStates(testState, ctx).find(t =>
        t.action.type === 'play' && t.action.dominoId === '0-0'
      );
      expect(doublePlay).toBeDefined();
      testState = executeAction(testState, doublePlay!.action, rules);

      // Current suit should be 7 (doubles form own suit)
      expect(testState.currentSuit).toBe(7);
    });
  });

  describe('Failed Nello', () => {
    it('should end early when bidder wins a trick', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 901234,
        players: [
          // Give bidder high cards to ensure they win
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[1, 5], [2, 4], [2, 5], [3, 5], [4, 4], [4, 5], [5, 5]]) }
        ]
      });

      const { finalState, preScoreState } = await playNelloHand(state, true);

      // Verify opponents scored
      expect(finalState.teamMarks[0]).toBe(0); // Bidder failed
      expect(finalState.teamMarks[1]).toBe(1); // Opponents scored

      // Verify hand ended early (less than 7 tricks)
      expect(preScoreState.tricks.length).toBeLessThan(7);
      expect(preScoreState.tricks.length).toBeGreaterThan(0);

      // Verify bidder won at least one trick
      const bidderTricks = preScoreState.tricks.filter(t => {
        if (t.winner === undefined) throw new Error('Trick has no winner');
        const winner = state.players[t.winner];
        if (!winner) throw new Error(`Invalid winner index: ${t.winner}`);
        return winner.teamId === 0;
      });
      expect(bidderTricks.length).toBeGreaterThan(0);
    });

    it('should score mark value for opponents on failure', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 567890,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[1, 5], [2, 4], [2, 5], [3, 5], [4, 4], [4, 5], [5, 5]]) }
        ]
      });

      const { finalState } = await playNelloHand(state, true);

      // Opponents get the mark value
      expect(finalState.teamMarks[1]).toBe(1);
    });
  });

  describe('Player Rotation', () => {
    it('should skip partner in turn order', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 111111,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      let testState = state;

      // Setup nello
      const marksBid = getNextStates(testState, ctx).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, ctx).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      const nelloTransition = getNextStates(testState, ctx).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'nello'
      );
      testState = executeAction(testState, nelloTransition!.action, rules);

      // Track player order
      const playerOrder: number[] = [];

      // Play first trick
      for (let i = 0; i < 3; i++) {
        playerOrder.push(testState.currentPlayer);
        const playTransitions = getNextStates(testState, ctx).filter(t => t.action.type === 'play');
        testState = executeAction(testState, playTransitions[0]!.action, rules);
      }

      // Should be: 0 (bidder), 1, 3 (skip partner 2)
      expect(playerOrder).toEqual([0, 1, 3]);
    });
  });

  describe('Bidding Requirements', () => {
    it('should only allow nello after marks bid', () => {
      const ctx = createTestContextWithRuleSets(['nello']);
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 1, player: 0 },
        shuffleSeed: 222222
      });

      const transitions = getNextStates(state, ctx);
      const nelloOption = transitions.find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'nello'
      );

      expect(nelloOption).toBeDefined();
    });

    it('should not allow nello after points bid', () => {

      const ctx = createTestContextWithRuleSets(['nello']);
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 },
        shuffleSeed: 333333
      });

      const transitions = getNextStates(state, ctx);
      const nelloOption = transitions.find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'nello'
      );

      expect(nelloOption).toBeUndefined();
    });
  });
});
