import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/gameEngine';
import { composeRules, baseLayer, plungeLayer } from '../../../game/layers';
import { createTestState, createTestHand, processSequentialConsensus, createHandWithDoubles } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import type { GameState } from '../../../game/types';

describe('Plunge Full Hand Integration', () => {
  const layers = [baseLayer, plungeLayer];
  const rules = composeRules(layers);

  /**
   * Helper to play a complete plunge hand
   */
  async function playPlungeHand(
    initialState: GameState,
    shouldBiddingTeamWinAll: boolean = true
  ): Promise<GameState> {
    let state = initialState;

    // Player 0 should have 4+ doubles and bid plunge
    const bidTransitions = getNextStates(state, layers, rules);
    const plungeBid = bidTransitions.find(t =>
      t.action.type === 'bid' && t.action.bid === 'plunge'
    );
    expect(plungeBid).toBeDefined();
    state = executeAction(state, plungeBid!.action, rules);

    // Others pass
    for (let i = 0; i < 3; i++) {
      const passTransition = getNextStates(state, layers, rules).find(t => t.id === 'pass');
      expect(passTransition).toBeDefined();
      state = executeAction(state, passTransition!.action, rules);
    }

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);

    // Partner (player 2) selects trump
    expect(state.currentPlayer).toBe(2); // Partner selects trump

    // Select a suit trump
    const trumpTransition = getNextStates(state, layers, rules).find(t =>
      t.action.type === 'select-trump' &&
      t.action.trump?.type === 'suit'
    );
    expect(trumpTransition).toBeDefined();
    state = executeAction(state, trumpTransition!.action, rules);

    expect(state.phase).toBe('playing');
    expect(state.currentPlayer).toBe(2); // Partner leads

    // Play tricks
    let trickCount = 0;
    const maxTricks = 7;

    while (state.phase === 'playing' && trickCount < maxTricks) {
      // Play 4 dominoes
      for (let i = 0; i < 4; i++) {
        const playTransitions = getNextStates(state, layers, rules).filter(t => t.action.type === 'play');
        expect(playTransitions.length).toBeGreaterThan(0);

        // Select domino based on test scenario
        let selectedPlay;
        const isBiddingTeam = state.players[state.currentPlayer]?.teamId === 0;

        if (shouldBiddingTeamWinAll) {
          // Bidding team plays high, opponents play low
          selectedPlay = isBiddingTeam
            ? playTransitions[playTransitions.length - 1]
            : playTransitions[0];
        } else {
          // Opponents play high to win trick
          selectedPlay = isBiddingTeam
            ? playTransitions[0]
            : playTransitions[playTransitions.length - 1];
        }

        state = executeAction(state, selectedPlay!.action, rules);
      }

      // Check trick is complete
      expect(state.currentTrick.length).toBe(4);

      // Process consensus to complete trick
      state = await processSequentialConsensus(state, 'completeTrick');

      // Complete trick
      const completeTrickTransition = getNextStates(state, layers, rules).find(t => t.id === 'complete-trick');
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
    const scoreTransition = getNextStates(state, layers, rules).find(t => t.id === 'score-hand');
    expect(scoreTransition).toBeDefined();
    state = executeAction(state, scoreTransition!.action, rules);

    return state;
  }

  describe('Successful Plunge', () => {
    it('should complete when bidding team wins all 7 tricks', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 123456,
        players: [
          // Player 0 has 4 doubles to qualify for plunge
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [5, 6], [4, 5], [4, 6], [3, 5], [3, 6], [6, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      const finalState = await playPlungeHand(state, true);

      // Verify bidding team scored
      expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4); // Plunge is worth 4+ marks
      expect(finalState.teamMarks[1]).toBe(0);

      // Verify all 7 tricks were played (successful plunge)
      expect(finalState.tricks.length).toBe(7);

      // Verify bidding team won all tricks
      const biddingTeamTricks = finalState.tricks.filter(t => {
        if (t.winner === undefined) throw new Error('Trick has no winner');
        const winner = state.players[t.winner];
        if (!winner) throw new Error(`Invalid winner index: ${t.winner}`);
        return winner.teamId === 0;
      });
      expect(biddingTeamTricks.length).toBe(7);
    });

    it('should award 4+ marks for successful plunge', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 789012,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(5) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [5, 6], [4, 5], [4, 6], [3, 5], [3, 6], [6, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      const finalState = await playPlungeHand(state, true);

      // Plunge value is automatic: max(4, highest marks bid + 1)
      expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
    });
  });

  describe('Failed Plunge', () => {
    it('should end early when opponents win a trick', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 345678,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[5, 5], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5], [4, 4]]) }
        ]
      });

      const finalState = await playPlungeHand(state, false);

      // Verify opponents scored
      expect(finalState.teamMarks[0]).toBe(0); // Bidding team failed
      expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4); // Opponents get the marks

      // Verify hand ended early (less than 7 tricks)
      expect(finalState.tricks.length).toBeLessThan(7);
      expect(finalState.tricks.length).toBeGreaterThan(0);

      // Verify opponents won at least one trick
      const opponentTricks = finalState.tricks.filter(t => {
        if (t.winner === undefined) throw new Error('Trick has no winner');
        const winner = state.players[t.winner];
        if (!winner) throw new Error(`Invalid winner index: ${t.winner}`);
        return winner.teamId === 1;
      });
      expect(opponentTricks.length).toBeGreaterThan(0);
    });

    it('should award marks to opponents on failure', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 901234,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[5, 5], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5], [4, 4]]) }
        ]
      });

      const finalState = await playPlungeHand(state, false);

      // Opponents get the plunge value
      expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4);
    });
  });

  describe('Trump Selection and Leading', () => {
    it('should have partner select trump', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 567890,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [5, 6], [4, 5], [4, 6], [3, 5], [3, 6], [6, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      let testState = state;

      // Bid plunge
      const plungeBid = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      testState = executeAction(testState, plungeBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      // Verify partner (player 2) is trump selector
      expect(testState.phase).toBe('trump_selection');
      expect(testState.currentPlayer).toBe(2);
    });

    it('should have partner lead first trick', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 111111,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [5, 6], [4, 5], [4, 6], [3, 5], [3, 6], [6, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      let testState = state;

      // Bid plunge
      const plungeBid = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      testState = executeAction(testState, plungeBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      // Partner selects trump
      const trumpTransition = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'select-trump'
      );
      testState = executeAction(testState, trumpTransition!.action, rules);

      // Verify partner leads
      expect(testState.phase).toBe('playing');
      expect(testState.currentPlayer).toBe(2);
    });
  });

  describe('Bidding Requirements', () => {
    it('should require 4+ doubles to bid plunge', () => {
      // With 4 doubles
      const stateWith4 = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 222222,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const transitionsWith4 = getNextStates(stateWith4, layers, rules);
      const plungeOptionWith4 = transitionsWith4.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      expect(plungeOptionWith4).toBeDefined();

      // With only 3 doubles
      const stateWith3 = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 333333,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(3) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const transitionsWith3 = getNextStates(stateWith3, layers, rules);
      const plungeOptionWith3 = transitionsWith3.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      expect(plungeOptionWith3).toBeUndefined();
    });

    it('should have automatic bid value of 4+ marks', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 444444,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(5) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const transitions = getNextStates(state, layers, rules);
      const plungeOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );

      expect(plungeOption).toBeDefined();
      const action = plungeOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBeGreaterThanOrEqual(4);
    });

    it('should jump over existing marks bids', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        shuffleSeed: 555555,
        bids: [{ type: BID_TYPES.MARKS, value: 2, player: 0 }],
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createHandWithDoubles(5) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 6], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [4, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[3, 5], [3, 6], [4, 5], [4, 6], [5, 5], [5, 6], [6, 6]]) }
        ]
      });

      const transitions = getNextStates(state, layers, rules);
      const plungeOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );

      expect(plungeOption).toBeDefined();
      // Should be at least 3 (2 + 1), but minimum 4
      const action = plungeOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBeGreaterThanOrEqual(4);
    });
  });
});
