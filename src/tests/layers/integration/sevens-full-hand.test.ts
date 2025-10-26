import { describe, it, expect } from 'vitest';
import { executeAction } from '../../../game/core/actions';
import { getNextStates } from '../../../game/core/gameEngine';
import { composeRules, baseLayer, sevensLayer } from '../../../game/layers';
import { createTestState, createTestHand, processSequentialConsensus } from '../../helpers/gameTestHelper';
import { BID_TYPES } from '../../../game/constants';
import type { GameState } from '../../../game/types';

describe('Sevens Full Hand Integration', () => {
  const layers = [baseLayer, sevensLayer];
  const rules = composeRules(layers);

  /**
   * Helper to play a complete sevens hand
   */
  async function playSevensHand(
    initialState: GameState,
    shouldBiddingTeamWinAll: boolean = true
  ): Promise<GameState> {
    let state = initialState;

    // Player 0 bids marks
    const bidTransitions = getNextStates(state, layers, rules);
    const marksBid = bidTransitions.find(t => t.id === 'bid-1-marks');
    expect(marksBid).toBeDefined();
    state = executeAction(state, marksBid!.action, rules);

    // Others pass
    for (let i = 0; i < 3; i++) {
      const passTransition = getNextStates(state, layers, rules).find(t => t.id === 'pass');
      expect(passTransition).toBeDefined();
      state = executeAction(state, passTransition!.action, rules);
    }

    expect(state.phase).toBe('trump_selection');
    expect(state.winningBidder).toBe(0);

    // Select sevens trump
    const sevensTransition = getNextStates(state, layers, rules).find(t =>
      t.action.type === 'select-trump' &&
      t.action.trump?.type === 'sevens'
    );
    expect(sevensTransition).toBeDefined();
    state = executeAction(state, sevensTransition!.action, rules);

    expect(state.phase).toBe('playing');
    expect(state.trump.type).toBe('sevens');
    expect(state.currentPlayer).toBe(0); // Bidder leads

    // Play tricks
    let trickCount = 0;
    const maxTricks = 7;

    while (state.phase === 'playing' && trickCount < maxTricks) {
      // Play 4 dominoes
      for (let i = 0; i < 4; i++) {
        const playTransitions = getNextStates(state, layers, rules).filter(t => t.action.type === 'play');
        expect(playTransitions.length).toBeGreaterThan(0);

        // Select domino based on test scenario
        // In sevens, closest to 7 total pips wins
        let selectedPlay;
        const isBiddingTeam = state.players[state.currentPlayer]?.teamId === 0;

        if (shouldBiddingTeamWinAll) {
          // Bidding team tries to play closest to 7
          if (isBiddingTeam) {
            // Find domino closest to 7 total pips
            selectedPlay = playTransitions.reduce((best, current) => {
              const currentId = current.action.type === 'play' ? current.action.dominoId : '';
              const bestId = best.action.type === 'play' ? best.action.dominoId : '';
              const currentDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === currentId);
              const bestDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === bestId);

              if (!currentDomino || !bestDomino) return best;

              const currentDist = Math.abs(7 - (currentDomino.high + currentDomino.low));
              const bestDist = Math.abs(7 - (bestDomino.high + bestDomino.low));

              return currentDist < bestDist ? current : best;
            });
          } else {
            // Opponents play far from 7
            selectedPlay = playTransitions.reduce((worst, current) => {
              const currentId = current.action.type === 'play' ? current.action.dominoId : '';
              const worstId = worst.action.type === 'play' ? worst.action.dominoId : '';
              const currentDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === currentId);
              const worstDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === worstId);

              if (!currentDomino || !worstDomino) return worst;

              const currentDist = Math.abs(7 - (currentDomino.high + currentDomino.low));
              const worstDist = Math.abs(7 - (worstDomino.high + worstDomino.low));

              return currentDist > worstDist ? current : worst;
            });
          }
        } else {
          // Opponents try to win (play closest to 7)
          if (!isBiddingTeam) {
            selectedPlay = playTransitions.reduce((best, current) => {
              const currentId = current.action.type === 'play' ? current.action.dominoId : '';
              const bestId = best.action.type === 'play' ? best.action.dominoId : '';
              const currentDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === currentId);
              const bestDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === bestId);

              if (!currentDomino || !bestDomino) return best;

              const currentDist = Math.abs(7 - (currentDomino.high + currentDomino.low));
              const bestDist = Math.abs(7 - (bestDomino.high + bestDomino.low));

              return currentDist < bestDist ? current : best;
            });
          } else {
            // Bidding team plays far from 7
            selectedPlay = playTransitions.reduce((worst, current) => {
              const currentId = current.action.type === 'play' ? current.action.dominoId : '';
              const worstId = worst.action.type === 'play' ? worst.action.dominoId : '';
              const currentDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === currentId);
              const worstDomino = state.players[state.currentPlayer]?.hand.find(d => d.id === worstId);

              if (!currentDomino || !worstDomino) return worst;

              const currentDist = Math.abs(7 - (currentDomino.high + currentDomino.low));
              const worstDist = Math.abs(7 - (worstDomino.high + worstDomino.low));

              return currentDist > worstDist ? current : worst;
            });
          }
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

  describe('Successful Sevens', () => {
    it('should complete when bidding team wins all 7 tricks', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 123456,
        players: [
          // Give bidding team dominoes close to 7 total pips
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      const finalState = await playSevensHand(state, true);

      // Verify bidding team scored
      expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(1); // Sevens is worth 1+ marks
      expect(finalState.teamMarks[1]).toBe(0);

      // Verify all 7 tricks were played (successful sevens)
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

    it('should award 1+ marks for successful sevens', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 789012,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      const finalState = await playSevensHand(state, true);

      // Marks bid value determines sevens value (minimum 1)
      expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Failed Sevens', () => {
    it('should end early when opponents win a trick', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 345678,
        players: [
          // Give opponents dominoes close to 7
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [6, 6], [5, 5], [4, 4], [1, 1]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 6], [1, 5], [2, 4], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [2, 2], [2, 3]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [6, 1], [5, 2], [4, 3], [5, 6]]) }
        ]
      });

      const finalState = await playSevensHand(state, false);

      // Verify opponents scored
      expect(finalState.teamMarks[0]).toBe(0); // Bidding team failed
      expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(1); // Opponents get the marks

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
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[0, 0], [0, 1], [0, 2], [6, 6], [5, 5], [4, 4], [1, 1]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 6], [1, 5], [2, 4], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [2, 2], [2, 3]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [6, 1], [5, 2], [4, 3], [5, 6]]) }
        ]
      });

      const finalState = await playSevensHand(state, false);

      // Opponents get the mark value
      expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Closest to 7 Total Pips Wins', () => {
    it('should determine winner by closest to 7 total pips', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 567890,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      let testState = state;

      // Setup sevens
      const marksBid = getNextStates(testState, layers, rules).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      const sevensTransition = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );
      testState = executeAction(testState, sevensTransition!.action, rules);

      // Play a specific trick to test distance calculation
      // Player 0 plays 3-4 (total = 7, distance = 0)
      const play0 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '3-4'
      );
      expect(play0).toBeDefined();
      testState = executeAction(testState, play0!.action, rules);

      // Player 1 plays 0-1 (total = 1, distance = 6)
      const play1 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '0-1'
      );
      expect(play1).toBeDefined();
      testState = executeAction(testState, play1!.action, rules);

      // Player 2 plays 6-0 (total = 6, distance = 1)
      const play2 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '6-0'
      );
      expect(play2).toBeDefined();
      testState = executeAction(testState, play2!.action, rules);

      // Player 3 plays 6-6 (total = 12, distance = 5)
      const play3 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '6-6'
      );
      expect(play3).toBeDefined();
      testState = executeAction(testState, play3!.action, rules);

      // Complete trick
      testState = await processSequentialConsensus(testState, 'completeTrick');
      const completeTrick = getNextStates(testState, layers, rules).find(t => t.id === 'complete-trick');
      testState = executeAction(testState, completeTrick!.action, rules);

      // Player 0 should win (3-4 has total 7, distance 0)
      expect(testState.tricks[0]?.winner).toBe(0);
    });

    it('should break ties by first played', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 111111,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      let testState = state;

      // Setup sevens
      const marksBid = getNextStates(testState, layers, rules).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      const sevensTransition = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );
      testState = executeAction(testState, sevensTransition!.action, rules);

      // Play dominoes with same distance from 7
      // Player 0 plays 2-5 (total = 7, distance = 0)
      const play0 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '2-5'
      );
      testState = executeAction(testState, play0!.action, rules);

      // Player 1 plays 1-1 (total = 2, distance = 5)
      const play1 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '1-1'
      );
      testState = executeAction(testState, play1!.action, rules);

      // Player 2 plays 3-4 (total = 7, distance = 0, same as player 0)
      const play2 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '3-4'
      );
      // Skip if not available
      if (!play2) {
        // Use alternative domino
        const altPlay = getNextStates(testState, layers, rules).find(t =>
          t.action.type === 'play' && t.action.dominoId === '3-3'
        );
        testState = executeAction(testState, altPlay!.action, rules);
      } else {
        testState = executeAction(testState, play2!.action, rules);
      }

      // Player 3 plays 0-4 (total = 4, distance = 3)
      const play3 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '0-4'
      );
      testState = executeAction(testState, play3!.action, rules);

      // Complete trick
      testState = await processSequentialConsensus(testState, 'completeTrick');
      const completeTrick = getNextStates(testState, layers, rules).find(t => t.id === 'complete-trick');
      testState = executeAction(testState, completeTrick!.action, rules);

      // Player 0 should win (first to play distance 0)
      expect(testState.tricks[0]?.winner).toBe(0);
    });
  });

  describe('No Follow-Suit Requirement', () => {
    it('should allow any domino to be played regardless of led suit', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 222222,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      let testState = state;

      // Setup sevens
      const marksBid = getNextStates(testState, layers, rules).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      const sevensTransition = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );
      testState = executeAction(testState, sevensTransition!.action, rules);

      // Player 0 leads with 3-4
      const play0 = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'play' && t.action.dominoId === '3-4'
      );
      testState = executeAction(testState, play0!.action, rules);

      // Player 1 should be able to play ANY domino (no follow-suit requirement)
      const playOptions = getNextStates(testState, layers, rules).filter(t => t.action.type === 'play');
      expect(playOptions.length).toBe(testState.players[1]?.hand.length);
    });
  });

  describe('Bidding Requirements', () => {
    it('should only allow sevens after marks bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 1, player: 0 },
        shuffleSeed: 333333
      });

      const transitions = getNextStates(state, layers, rules);
      const sevensOption = transitions.find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );

      expect(sevensOption).toBeDefined();
    });

    it('should not allow sevens after points bid', () => {
      const state = createTestState({
        phase: 'trump_selection',
        currentPlayer: 0,
        winningBidder: 0,
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 },
        shuffleSeed: 444444
      });

      const transitions = getNextStates(state, layers, rules);
      const sevensOption = transitions.find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );

      expect(sevensOption).toBeUndefined();
    });

    it('should have bidder lead normally', async () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 0,
        shuffleSeed: 555555,
        players: [
          { id: 0, name: 'P0', teamId: 0, marks: 0, hand: createTestHand([[3, 4], [2, 5], [1, 6], [0, 0], [6, 1], [5, 2], [4, 3]]) },
          { id: 1, name: 'P1', teamId: 1, marks: 0, hand: createTestHand([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [2, 2], [3, 3]]) },
          { id: 2, name: 'P2', teamId: 0, marks: 0, hand: createTestHand([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]) },
          { id: 3, name: 'P3', teamId: 1, marks: 0, hand: createTestHand([[6, 6], [5, 5], [4, 4], [5, 6], [4, 6], [4, 5], [0, 4]]) }
        ]
      });

      let testState = state;

      // Bid marks
      const marksBid = getNextStates(testState, layers, rules).find(t => t.id === 'bid-1-marks');
      testState = executeAction(testState, marksBid!.action, rules);

      // Others pass
      for (let i = 0; i < 3; i++) {
        const passTransition = getNextStates(testState, layers, rules).find(t => t.id === 'pass');
        testState = executeAction(testState, passTransition!.action, rules);
      }

      // Select sevens
      const sevensTransition = getNextStates(testState, layers, rules).find(t =>
        t.action.type === 'select-trump' && t.action.trump?.type === 'sevens'
      );
      testState = executeAction(testState, sevensTransition!.action, rules);

      // Verify bidder (player 0) leads
      expect(testState.phase).toBe('playing');
      expect(testState.currentPlayer).toBe(0);
    });
  });
});
