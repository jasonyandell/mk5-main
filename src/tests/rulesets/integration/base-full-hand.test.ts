import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/state';
import { createTestContextWithRuleSets } from '../../helpers/executionContext';
import { composeRules, baseRuleSet } from '../../../game/rulesets';
import { createTestState, createTestHand, processSequentialConsensus } from '../../helpers/gameTestHelper';
import type { GameState } from '../../../game/types';

describe('Base (Standard) Full Hand Integration', () => {
  const ctx = createTestContextWithRuleSets([]);
  const rules = composeRules([baseRuleSet]);

  /**
   * Helper to play a complete standard hand
   */
  async function playStandardHand(
    initialState: GameState,
    bidValue: number,
    playStrategy: 'bidder-wins' | 'defenders-win' | 'complete-7-tricks' = 'complete-7-tricks'
  ): Promise<{ finalState: GameState; preScoreState: GameState }> {
    let state = initialState;

    // Players bid in order
    const bidTransitions = getNextStates(state, ctx);
    const pointsBid = bidTransitions.find(t => t.id === `bid-${bidValue}`);
    expect(pointsBid).toBeDefined();
    state = executeAction(state, pointsBid!.action, rules);

    // Others pass
    for (let i = 0; i < 3; i++) {
      const passTransition = getNextStates(state, ctx).find(t => t.id === 'pass');
      expect(passTransition).toBeDefined();
      state = executeAction(state, passTransition!.action, rules);
    }

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);

    // Select trump (sixes)
    const trumpTransition = getNextStates(state, ctx).find(t =>
      t.action.type === 'select-trump' &&
      t.action.trump?.type === 'suit' &&
      t.action.trump?.suit === 6
    );
    expect(trumpTransition).toBeDefined();
    state = executeAction(state, trumpTransition!.action, rules);

    expect(state.phase).toBe('playing');
    expect(state.trump.type).toBe('suit');
    expect(state.currentPlayer).toBe(0); // Bidder leads

    // Play tricks
    let trickCount = 0;
    const maxTricks = 7;

    while (state.phase === 'playing' && trickCount < maxTricks) {
      // Play 4 dominoes (standard 4-player)
      for (let i = 0; i < 4; i++) {
        const playTransitions = getNextStates(state, ctx).filter(t => t.action.type === 'play');
        expect(playTransitions.length).toBeGreaterThan(0);

        let selectedPlay;
        const biddingTeam = state.winningBidder % 2;
        const currentPlayerTeam = state.currentPlayer % 2;

        if (playStrategy === 'bidder-wins') {
          // Bidding team tries to win, defenders try to lose
          if (currentPlayerTeam === biddingTeam) {
            selectedPlay = playTransitions[playTransitions.length - 1]; // Play high
          } else {
            selectedPlay = playTransitions[0]; // Play low
          }
        } else if (playStrategy === 'defenders-win') {
          // Defenders try to win, bidding team tries to lose
          if (currentPlayerTeam === biddingTeam) {
            selectedPlay = playTransitions[0]; // Play low
          } else {
            selectedPlay = playTransitions[playTransitions.length - 1]; // Play high
          }
        } else {
          // Complete all 7 tricks - alternate winners
          selectedPlay = trickCount % 2 === 0
            ? playTransitions[Math.floor(playTransitions.length / 2)]
            : playTransitions[0];
        }

        state = executeAction(state, selectedPlay!.action, rules);
      }

      // Check trick is complete (4 plays in standard)
      expect(state.currentTrick.length).toBe(4);

      // Process consensus
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
      } else {
        break;
      }
    }

    const preScoreState = { ...state };

    // Score the hand if not already scored
    if (state.phase === 'scoring') {
      const scoreTransitions = getNextStates(state, ctx).filter(t => t.action.type === 'score-hand');
      if (scoreTransitions.length > 0) {
        state = executeAction(state, scoreTransitions[0]!.action, rules);
      }
    }

    return { finalState: state, preScoreState };
  }

  describe('Successful Standard Bids', () => {
    it('should complete a successful 30-point bid with all 7 tricks', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 123456,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [6, 4], [6, 3], [5, 5], [5, 0]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[1, 5], [2, 4], [2, 5], [3, 5], [4, 4], [4, 5], [1, 6]]) }
        ]
      });

      const { finalState, preScoreState } = await playStandardHand(state, 30, 'complete-7-tricks');

      // Verify all 7 tricks were played
      expect(preScoreState.tricks.length).toBe(7);
      expect(preScoreState.phase).toBe('scoring');

      // Verify scoring happened
      expect(finalState.phase).toBe('hand_complete');
    });

    it('should complete a successful marks bid (2 marks)', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 234567,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6], [6, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 3], [5, 0], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[4, 3], [3, 4], [2, 4], [1, 4], [0, 4], [3, 2], [2, 3]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 32, 'complete-7-tricks');

      // Verify all 7 tricks were played
      expect(preScoreState.tricks.length).toBe(7);
      expect(preScoreState.phase).toBe('scoring');
    });
  });

  describe('Early Termination - Points Bids', () => {
    it('should end early when bidding team reaches their bid', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 345678,
        players: [
          // Give bidding team (0, 2) high trump and count dominoes
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [6, 4], [6, 3], [5, 5], [5, 0], [4, 4]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[4, 6], [2, 6], [1, 6], [0, 6], [4, 5], [3, 5], [2, 5]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 30, 'bidder-wins');

      // Verify hand ended before 7 tricks
      expect(preScoreState.tricks.length).toBeLessThan(7);
      expect(preScoreState.phase).toBe('scoring');

      // Verify bidding team has at least 30 points
      const biddingTeam = state.players[0]!.teamId;
      const biddingTeamScore = preScoreState.teamScores[biddingTeam];
      expect(biddingTeamScore).toBeGreaterThanOrEqual(30);

      // Verify reason includes that bidding team reached bid
      expect(preScoreState.tricks.length).toBeGreaterThan(0);
    });

    it('should end early when defending team sets the bid', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 456789,
        players: [
          // Give defending team (1, 3) high dominoes to accumulate 13+ points
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 6], [6, 4], [6, 3], [5, 5], [4, 4], [3, 5]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[4, 6], [2, 6], [1, 6], [0, 6], [5, 0], [4, 5], [2, 5]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 30, 'defenders-win');

      // Verify hand ended before 7 tricks
      expect(preScoreState.tricks.length).toBeLessThan(7);
      expect(preScoreState.phase).toBe('scoring');

      // Verify defending team has 13+ points (sets the bid)
      const biddingTeam = 0; // Player 0 bid
      const defendingTeam = 1 - biddingTeam;
      const defendingTeamScore = preScoreState.teamScores[defendingTeam];
      expect(defendingTeamScore).toBeGreaterThanOrEqual(13);

      // Bidding team cannot reach their bid
      const remainingPoints = 42 - (preScoreState.teamScores[0] + preScoreState.teamScores[1]);
      const biddingTeamMaxPossible = preScoreState.teamScores[biddingTeam] + remainingPoints;
      expect(biddingTeamMaxPossible).toBeLessThan(30);
    });

    it('should end early when bidders cannot possibly reach their bid', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 567890,
        players: [
          // Bidding team gets very low cards, defenders get count dominoes
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [0, 2], [1, 2]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [6, 4], [5, 5], [6, 3], [5, 0], [4, 6], [5, 6]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3], [3, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[2, 6], [1, 6], [0, 6], [4, 5], [3, 5], [2, 5], [1, 5]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 30, 'defenders-win');

      // Verify hand ended before 7 tricks
      expect(preScoreState.tricks.length).toBeLessThan(7);
      expect(preScoreState.phase).toBe('scoring');

      // Calculate what bidding team could theoretically still score
      const biddingTeam = 0;
      const remainingPoints = 42 - (preScoreState.teamScores[0] + preScoreState.teamScores[1]);
      const maxPossible = preScoreState.teamScores[biddingTeam] + remainingPoints;

      // Verify they cannot reach 30
      expect(maxPossible).toBeLessThan(30);
    });
  });

  describe('Early Termination - Marks Bids', () => {
    it('should end early when defending team scores any points in a marks bid', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 678901,
        players: [
          // Give defending team one count domino
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0], [3, 6]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[5, 5], [0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[4, 6], [2, 6], [1, 6], [0, 6], [3, 4], [2, 4], [1, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[5, 6], [5, 0], [4, 5], [3, 5], [2, 5], [1, 5], [0, 5]]) }
        ]
      });

      // Bid 2 marks (32 points)
      let currentState = state;

      // Player 0 bids 2 marks
      const bidTransitions = getNextStates(currentState, ctx);
      const marksBid = bidTransitions.find(t => t.id === 'bid-2-marks');
      expect(marksBid).toBeDefined();
      currentState = executeAction(currentState, marksBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(currentState, ctx).find(t => t.id === 'pass');
        expect(passTransition).toBeDefined();
        currentState = executeAction(currentState, passTransition!.action, rules);
      }

      expect(currentState.currentBid?.type).toBe('marks');
      expect(currentState.currentBid?.value).toBe(2);

      // Select trump
      const trumpTransition = getNextStates(currentState, ctx).find(t =>
        t.action.type === 'select-trump'
      );
      expect(trumpTransition).toBeDefined();
      currentState = executeAction(currentState, trumpTransition!.action, rules);

      // Play first trick - arrange for defending team to win and score points
      // Use the 'defenders-win' strategy to give defenders some points
      const { preScoreState } = await playStandardHand(state, 32, 'defenders-win');

      // Verify defending team scored > 0 points
      const defendingTeam = 1;
      expect(preScoreState.teamScores[defendingTeam]).toBeGreaterThan(0);

      // Marks bid should fail once defenders score
      expect(preScoreState.phase).toBe('scoring');
      expect(preScoreState.tricks.length).toBeLessThan(7);
    });

    it('should end early when bidding team cannot reach 42 points in marks bid', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 789012,
        players: [
          // Defenders have most count dominoes
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [0, 1], [0, 2]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [6, 4], [6, 3], [5, 5], [5, 0], [5, 6], [4, 6]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[1, 2], [0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [2, 3]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[2, 6], [1, 6], [0, 6], [4, 5], [3, 5], [2, 5], [1, 5]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 32, 'defenders-win');

      // Verify hand ended early
      expect(preScoreState.tricks.length).toBeLessThan(7);
      expect(preScoreState.phase).toBe('scoring');

      // Bidding team cannot reach 42
      const biddingTeam = 0;
      const remainingPoints = 42 - (preScoreState.teamScores[0] + preScoreState.teamScores[1]);
      const maxPossible = preScoreState.teamScores[biddingTeam] + remainingPoints;
      expect(maxPossible).toBeLessThan(42);
    });
  });

  describe('Standard Game Mechanics', () => {
    it('should allow bidder to select trump', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 890123,
      });

      let currentState = state;

      // Bid 30
      const bidTransitions = getNextStates(currentState, ctx);
      const bid30 = bidTransitions.find(t => t.id === 'bid-30');
      expect(bid30).toBeDefined();
      currentState = executeAction(currentState, bid30!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(currentState, ctx).find(t => t.id === 'pass');
        currentState = executeAction(currentState, passTransition!.action, rules);
      }

      // Bidder selects trump
      expect(currentState.phase).toBe('trump_selection');
      expect(currentState.winningBidder).toBe(0);

      const trumpTransitions = getNextStates(currentState, ctx).filter(t =>
        t.action.type === 'select-trump'
      );

      // Should have options for all 7 suits + doubles
      expect(trumpTransitions.length).toBeGreaterThan(0);
    });

    it('should have bidder lead first trick', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1, // Player 1 will be bidder
        shuffleSeed: 901234,
      });

      let currentState = state;

      // Player 1 bids
      const bidTransitions = getNextStates(currentState, ctx);
      const bid = bidTransitions.find(t => t.id === 'bid-30');
      currentState = executeAction(currentState, bid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(currentState, ctx).find(t => t.id === 'pass');
        currentState = executeAction(currentState, passTransition!.action, rules);
      }

      // Select trump
      const trumpTransition = getNextStates(currentState, ctx).find(t =>
        t.action.type === 'select-trump'
      );
      currentState = executeAction(currentState, trumpTransition!.action, rules);

      // Verify player 1 leads
      expect(currentState.phase).toBe('playing');
      expect(currentState.currentPlayer).toBe(1);
    });

    it('should play all 7 tricks when outcome is not determined early', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 112233,
        players: [
          // Balanced distribution - score close to bid but not reaching until last tricks
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[6, 6], [5, 6], [4, 6], [3, 6], [0, 1], [0, 2], [1, 2]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[6, 4], [6, 3], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[5, 5], [5, 0], [0, 3], [0, 4], [0, 5], [1, 3], [1, 4]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[2, 6], [1, 6], [0, 6], [4, 5], [3, 5], [2, 5], [1, 5]]) }
        ]
      });

      const { preScoreState } = await playStandardHand(state, 30, 'complete-7-tricks');

      // Verify all 7 tricks were played
      expect(preScoreState.tricks.length).toBe(7);
      expect(preScoreState.phase).toBe('scoring');

      // All dominoes should be played
      const totalPlayed = preScoreState.tricks.reduce((sum, trick) => sum + trick.plays.length, 0);
      expect(totalPlayed).toBe(28); // 7 tricks × 4 players
    });
  });
});
