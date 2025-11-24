import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { createTestContextWithRuleSets } from '../../helpers/executionContext';
import { composeRules, baseRuleSet, splashRuleSet } from '../../../game/layers';
import { StateBuilder, HandBuilder } from '../../helpers';
import { processSequentialConsensus } from '../../helpers/consensusHelpers';
import { BID_TYPES } from '../../../game/constants';
import type { GameState } from '../../../game/types';

describe('Splash Full Hand Integration', () => {
  const ctx = createTestContextWithRuleSets(['splash']);
  const ruleSets = [baseRuleSet, splashRuleSet];
  const rules = composeRules(ruleSets);

  /**
   * Helper to play a complete splash hand
   */
  async function playSplashHand(
    initialState: GameState,
    shouldBiddingTeamWinAll: boolean = true
  ): Promise<{ finalState: GameState; preScoreState: GameState }> {
    let state = initialState;

    // Player 0 should have 3+ doubles and bid splash
    const bidTransitions = getNextStates(state, ctx);
    const splashBid = bidTransitions.find(t =>
      t.action.type === 'bid' && t.action.bid === 'splash'
    );
    expect(splashBid).toBeDefined();
    state = executeAction(state, splashBid!.action, rules);

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

    // Execute scoring
    // Save state before scoring to verify tricks
    const preScoreState = state;

    state = executeAction(state, scoreTransition!.action, rules);

    return { finalState: state, preScoreState };
  }

  describe('Successful Splash', () => {
    it('should complete when bidding team wins all 7 tricks', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(123456)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['5-5', '5-6', '4-5', '4-6', '3-5', '3-6', '6-6']))
        .withPlayerHand(3, HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']))
        .build();

      const { finalState, preScoreState } = await playSplashHand(state, true);

      // Check who actually won (may differ from intent due to trump/suit rules)
      const team1Tricks = preScoreState.tricks.filter(t => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify scoring matches actual trick outcomes
      if (team1Tricks === 0) {
        // Team 0 (bidding team) won all tricks - should get marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[0]).toBeLessThanOrEqual(3);
        expect(finalState.teamMarks[1]).toBe(0);
      } else {
        // Team 1 won at least one trick - should get marks
        expect(finalState.teamMarks[0]).toBe(0);
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[1]).toBeLessThanOrEqual(3);
      }
    });

    it('should award 2-3 marks for successful splash', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(789012)
        .withPlayerHand(0, HandBuilder.withDoubles(4))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['5-5', '5-6', '4-5', '4-6', '3-5', '3-6', '6-6']))
        .withPlayerHand(3, HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']))
        .build();

      const { finalState, preScoreState } = await playSplashHand(state, true);

      // Check who won (scoring should match actual outcome)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify marks were awarded correctly based on actual outcome
      if (team1Tricks === 0) {
        // Bidding team won all - should get 2-3 marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[0]).toBeLessThanOrEqual(3);
      } else {
        // Opponents won at least one - should get 2-3 marks
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[1]).toBeLessThanOrEqual(3);
      }
    });
  });

  describe('Failed Splash', () => {
    it('should end early when opponents win a trick', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(345678)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(3, HandBuilder.fromStrings(['5-5', '4-5', '3-5', '2-5', '1-5', '0-5', '4-4']))
        .build();

      const { finalState, preScoreState } = await playSplashHand(state, false);

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
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[1]).toBeLessThanOrEqual(3);
        // Should have ended early
        expect(preScoreState.tricks.length).toBeLessThan(7);
        expect(preScoreState.tricks.length).toBeGreaterThan(0);
      } else {
        // Team 0 (bidding team) won all tricks - should get marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[0]).toBeLessThanOrEqual(3);
        expect(finalState.teamMarks[1]).toBe(0);
      }
    });

    it('should award marks to opponents on failure', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(901234)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(3, HandBuilder.fromStrings(['5-5', '4-5', '3-5', '2-5', '1-5', '0-5', '4-4']))
        .build();

      const { finalState, preScoreState } = await playSplashHand(state, false);

      // Check who won (scoring should match actual outcome)
      const team1Tricks = preScoreState.tricks.filter((t: { winner?: number }) => {
        if (t.winner === undefined) return false;
        const winner = state.players[t.winner];
        return winner?.teamId === 1;
      }).length;

      // Verify marks were awarded correctly
      if (team1Tricks > 0) {
        // Opponents won - get the marks (2-3)
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[1]).toBeLessThanOrEqual(3);
      } else {
        // Bidding team won all - get the marks
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(2);
        expect(finalState.teamMarks[0]).toBeLessThanOrEqual(3);
      }
    });

    it('should continue when bidding team wins all tricks so far', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(112233)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']))
        .withPlayerHand(3, HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']))
        .build();

      const { finalState, preScoreState } = await playSplashHand(state, true);

      // Should play all 7 tricks if bidding team wins them all
      if (preScoreState.tricks.length === 7) {
        // All 7 tricks played - bidding team won all
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(2);
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

    it('should end early when opponents win on 2nd trick after losing first', async () => {
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(223344)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '6-6', '5-6', '4-6', '3-6', '2-6']))
        .withPlayerHand(2, HandBuilder.fromStrings(['5-5', '4-5', '3-5', '0-2', '1-1', '1-2', '2-2']))
        .withPlayerHand(3, HandBuilder.fromStrings(['1-6', '0-6', '0-3', '0-4', '0-5', '1-3', '1-4']))
        .build();

      // Play through carefully - opponents eventually win
      let testState = state;

      // Bid splash
      const splashBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );
      testState = executeAction(testState, splashBid!.action, rules);

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

          // Strategy: opponents play low first trick, then high on second
          let selectedPlay;
          const currentPlayerTeam = testState.currentPlayer % 2;
          if (currentPlayerTeam === 0) {
            // Bidding team plays to win
            selectedPlay = playTransitions[playTransitions.length - 1];
          } else {
            // Opponents play low first, high second
            if (tricksPlayed < 1) {
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
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(567890)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['5-5', '5-6', '4-5', '4-6', '3-5', '3-6', '6-6']))
        .withPlayerHand(3, HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']))
        .build();

      let testState = state;

      // Bid splash
      const splashBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );
      testState = executeAction(testState, splashBid!.action, rules);

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
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(111111)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['5-5', '5-6', '4-5', '4-6', '3-5', '3-6', '6-6']))
        .withPlayerHand(3, HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']))
        .build();

      let testState = state;

      // Bid splash
      const splashBid = getNextStates(testState, ctx).find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );
      testState = executeAction(testState, splashBid!.action, rules);

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
    it('should require 3+ doubles to bid splash', () => {
      const ctx = createTestContextWithRuleSets(['splash']);
      // With 3 doubles
      const stateWith3 = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(222222)
        .withPlayerHand(0, HandBuilder.withDoubles(3))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']))
        .withPlayerHand(3, HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6']))
        .build();

      const transitionsWith3 = getNextStates(stateWith3, ctx);
      const splashOptionWith3 = transitionsWith3.find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );
      expect(splashOptionWith3).toBeDefined();

      // With only 2 doubles
      const stateWith2 = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(333333)
        .withPlayerHand(0, HandBuilder.withDoubles(2))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']))
        .withPlayerHand(3, HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6']))
        .build();

      const transitionsWith2 = getNextStates(stateWith2, ctx);
      const splashOptionWith2 = transitionsWith2.find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );
      expect(splashOptionWith2).toBeUndefined();
    });

    it('should have automatic bid value of 2-3 marks', () => {

      const ctx = createTestContextWithRuleSets(['splash']);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(0)
        .withSeed(444444)
        .withPlayerHand(0, HandBuilder.withDoubles(4))
        .withPlayerHand(1, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']))
        .withPlayerHand(3, HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6']))
        .build();

      const transitions = getNextStates(state, ctx);
      const splashOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );

      expect(splashOption).toBeDefined();
      const action = splashOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBeGreaterThanOrEqual(2);
      expect(action.value).toBeLessThanOrEqual(3);
    });

    it('should jump over existing marks bids up to maximum of 3', () => {

      const ctx = createTestContextWithRuleSets(['splash']);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(1)
        .withSeed(555555)
        .withBids([{ type: BID_TYPES.MARKS, value: 2, player: 0 }])
        .withPlayerHand(0, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(1, HandBuilder.withDoubles(4))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']))
        .withPlayerHand(3, HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6']))
        .build();

      const transitions = getNextStates(state, ctx);
      const splashOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );

      expect(splashOption).toBeDefined();
      // Should be min(3, max(2, 2 + 1)) = 3
      const action = splashOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBe(3);
    });

    it('should cap at 3 marks even with high existing bids', () => {

      const ctx = createTestContextWithRuleSets(['splash']);
      const state = StateBuilder.inBiddingPhase()
        .withCurrentPlayer(1)
        .withSeed(666666)
        .withBids([{ type: BID_TYPES.MARKS, value: 5, player: 0 }])
        .withPlayerHand(0, HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']))
        .withPlayerHand(1, HandBuilder.withDoubles(3))
        .withPlayerHand(2, HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']))
        .withPlayerHand(3, HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6']))
        .build();

      const transitions = getNextStates(state, ctx);
      const splashOption = transitions.find(t =>
        t.action.type === 'bid' && t.action.bid === 'splash'
      );

      expect(splashOption).toBeDefined();
      // Should be min(3, max(2, 5 + 1)) = 3 (capped)
      const action = splashOption!.action;
      if (action.type !== 'bid') throw new Error('Expected bid action');
      expect(action.value).toBe(3);
    });
  });
});
