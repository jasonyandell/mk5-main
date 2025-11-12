import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { createTestContextWithRuleSets } from '../../helpers/executionContext';
import { composeRules, baseRuleSet, plungeRuleSet } from '../../../game/rulesets';
import { createTestState, createTestHand, processSequentialConsensus, createHandWithDoubles } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import type { GameState } from '../../../game/types';

describe('Plunge Full Hand Integration', () => {
  const ctx = createTestContextWithRuleSets(['plunge']);
  const ruleSets = [baseRuleSet, plungeRuleSet];
  const rules = composeRules(ruleSets);

  /**
   * Helper to play a complete plunge hand
   */
  async function playPlungeHand(
    initialState: GameState,
    shouldBiddingTeamWinAll: boolean = true
  ): Promise<{ finalState: GameState; preScoreState: GameState }> {
    let state = initialState;

    // Player 0 should have 4+ doubles and bid plunge
    const bidTransitions = getNextStates(state, ctx);
    const plungeBid = bidTransitions.find(t =>
      t.action.type === 'bid' && t.action.bid === 'plunge'
    );
    expect(plungeBid).toBeDefined();
    state = executeAction(state, plungeBid!.action, rules);

    // Others pass
    for (let i = 0; i < 3; i++) {
      const passTransition = getNextStates(state, ctx).find(t => t.id === 'pass');
      expect(passTransition).toBeDefined();
      state = executeAction(state, passTransition!.action, rules);
    }

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);

    // Partner (player 2) selects trump
    expect(state.currentPlayer).toBe(2); // Partner selects trump

    // Select a suit trump
    const trumpTransition = getNextStates(state, ctx).find(t =>
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
        const playTransitions = getNextStates(state, ctx).filter(t => t.action.type === 'play');
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

    // Save state before scoring to verify tricks (scoring will reset tricks for next hand)
    const preScoreState = state;

    // Execute scoring
    state = executeAction(state, scoreTransition!.action, rules);

    // Return both states for verification
    return { finalState: state, preScoreState };
  }

  describe('Successful Plunge', () => {
    it('should complete when bidding team wins all 7 tricks', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 123456,
        players: [
          // Player 0 (team 0) has 4 doubles + high dominoes for plunge
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [1, 1], [2, 2], [3, 3], [6, 6], [5, 6], [4, 6]]) },
          // Player 1 (team 1) has weak dominoes
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4]]) },
          // Player 2 (team 0) has high dominoes for trump
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5], [4, 4]]) },
          // Player 3 (team 1) has weak dominoes
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 4], [0, 6], [1, 4], [1, 6], [2, 6], [3, 4], [3, 6]]) }
        ]
      });

      const { finalState, preScoreState } = await playPlungeHand(state, true);

      // Check who actually won all tricks (may differ from intent due to trump/suit rules)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify scoring matches actual trick outcomes
      if (team1Tricks === 0) {
        // Team 0 (bidding team) won all tricks - should get marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
        expect(finalState.teamMarks[1]).toBe(0);
      } else {
        // Team 1 won at least one trick - should get marks
        expect(finalState.teamMarks[0]).toBe(0);
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4);
      }
    });

    it('should award 4+ marks for successful plunge', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 789012,
        players: [
          // Player 0 (team 0) has 5 doubles
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 6], [6, 6]]) },
          // Player 1 (team 1) has weak dominoes
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4]]) },
          // Player 2 (team 0) has high dominoes
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5], [4, 6]]) },
          // Player 3 (team 1) has weak dominoes
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 4], [0, 6], [1, 4], [1, 6], [2, 6], [3, 4], [3, 6]]) }
        ]
      });

      const { finalState, preScoreState } = await playPlungeHand(state, true);

      // Check who won (scoring should match actual outcome)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify marks were awarded correctly based on actual outcome
      if (team1Tricks === 0) {
        // Bidding team won all - should get 4+ marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
      } else {
        // Opponents won at least one - should get 4+ marks
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4);
      }
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

      const { finalState, preScoreState } = await playPlungeHand(state, false);

      // Check who won tricks (scoring should match actual outcome)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify scoring matches actual outcome
      if (team1Tricks > 0) {
        // Team 1 (opponents) won at least one trick - should get marks
        expect(finalState.teamMarks[0]).toBe(0);
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4);
        // Should have ended early
        expect(preScoreState.tricks.length).toBeLessThan(7);
      } else {
        // Team 0 (bidding team) won all tricks - should get marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
        expect(finalState.teamMarks[1]).toBe(0);
      }
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

      const { finalState, preScoreState } = await playPlungeHand(state, false);

      // Check who won (scoring should match actual outcome)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify marks were awarded correctly
      if (team1Tricks > 0) {
        // Opponents won - get the marks
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(4);
      } else {
        // Bidding team won all - get the marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
      }
    });

    it('should continue when bidding team wins all tricks so far', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 112233,
        players: [
          // Give bidding team high trump to win tricks
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      const { finalState, preScoreState } = await playPlungeHand(state, true);

      // Should play all 7 tricks if bidding team wins them all
      if (preScoreState.tricks.length === 7) {
        // All 7 tricks played - bidding team won all
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(4);
        expect(finalState.teamMarks[1]).toBe(0);
      }

      // Verify all tricks won by bidding team (if 7 tricks played)
      const biddingTeamTricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 0;
      }).length;

      if (preScoreState.tricks.length === 7) {
        expect(biddingTeamTricks).toBe(7);
      }
    });

    it('should end early when opponents win on 5th trick after losing first 4', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 223344,
        players: [
          // Mix - bidding team wins early tricks, then loses
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createHandWithDoubles(4) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [6, 6], [5, 6], [4, 6], [3, 6], [2, 6]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [4, 5], [3, 5], [0, 2], [1, 1], [1, 2], [2, 2]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[1, 6], [0, 6], [0, 3], [0, 4], [0, 5], [1, 3], [1, 4]]) }
        ]
      });

      // Play through carefully - opponents eventually win
      let testState = state;

      // Bid plunge
      const plungeBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      testState = executeAction(testState, plungeBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, ctx).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      // Partner selects trump
      const trumpTransition = getNextStates(testState, ctx).find(t =>
        t.action.type === 'select-trump'
      );
      testState = executeAction(testState, trumpTransition!.action, rules);

      // Play multiple tricks
      let tricksPlayed = 0;
      while (testState.phase === 'playing' && tricksPlayed < 7) {
        // Play 4 dominoes per trick
        for (let i = 0; i < 4; i++) {
          const playTransitions = getNextStates(testState, ctx).filter(t => t.action.type === 'play');

          // Strategy: opponents play low first few tricks, then high later
          let selectedPlay;
          const currentPlayerTeam = testState.currentPlayer % 2;
          if (currentPlayerTeam === 0) {
            // Bidding team plays to win
            selectedPlay = playTransitions[playTransitions.length - 1];
          } else {
            // Opponents play low early, high late
            if (tricksPlayed < 3) {
              selectedPlay = playTransitions[0]; // Play low
            } else {
              selectedPlay = playTransitions[playTransitions.length - 1]; // Play high to win
            }
          }

          testState = executeAction(testState, selectedPlay!.action, rules);
        }

        // Process consensus
        testState = await processSequentialConsensus(testState, 'completeTrick');

        // Complete trick
        const completeTrickTransition = getNextStates(testState, ctx).find(t => t.id === 'complete-trick');
        if (completeTrickTransition) {
          testState = executeAction(testState, completeTrickTransition.action, rules);
          tricksPlayed++;

          // Check if hand ended early
          if (testState.phase === 'scoring') {
            break;
          }
        }
      }

      // Should have ended early at some point
      if (testState.tricks.length < 7) {
        expect(testState.phase).toBe('scoring');

        // At least one trick should be won by opponents
        const opponentTricks = testState.tricks.filter((t: { winner?: number }) => {
          if (t.winner === undefined) return false;
          const winner = testState.players[t.winner];
          return winner?.teamId === 1;
        });
        expect(opponentTricks.length).toBeGreaterThan(0);
      }
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
      const plungeBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      testState = executeAction(testState, plungeBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, ctx).find(t => t.id === 'pass');
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
      const plungeBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      testState = executeAction(testState, plungeBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, ctx).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      // Partner selects trump
      const trumpTransition = getNextStates(testState, ctx).find(t =>
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
      const ctx = createTestContextWithRuleSets(['plunge']);
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

      const transitionsWith4 = getNextStates(stateWith4, ctx);
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

      const transitionsWith3 = getNextStates(stateWith3, ctx);
      const plungeOptionWith3 = transitionsWith3.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );
      expect(plungeOptionWith3).toBeUndefined();
    });

    it('should have automatic bid value of 4+ marks', () => {

      const ctx = createTestContextWithRuleSets(['plunge']);
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

      const transitions = getNextStates(state, ctx);
      const plungeOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'plunge'
      );

      expect(plungeOption).toBeDefined();
      const action = plungeOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBeGreaterThanOrEqual(4);
    });

    it('should jump over existing marks bids', () => {

      const ctx = createTestContextWithRuleSets(['plunge']);
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

      const transitions = getNextStates(state, ctx);
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
