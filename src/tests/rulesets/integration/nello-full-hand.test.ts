import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { HandBuilder } from '../../helpers';
import type { Domino, Trick } from '../../../game/types';
import type { GameConfig } from '../../../game/types/config';

/**
 * Nello Full Hand Integration Tests
 *
 * Crystal Palace Principles:
 * - Trust auto-execute for complete-trick and score-hand actions
 * - All 4 players must agree to consensus (including partner who sits out)
 * - Capture tricks BEFORE scoring (executeScoreHand resets tricks: [])
 * - Clean, coherent play mechanics throughout
 */
describe('Nello Full Hand Integration', () => {

  /**
   * Creates a HeadlessRoom with nello enabled and custom hands.
   */
  function createNelloRoom(seed: number, hands: Domino[][]): HeadlessRoom {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      enabledRuleSets: ['base', 'nello'],
      shuffleSeed: seed,
      dealOverrides: { initialHands: hands }
    };

    return new HeadlessRoom(config, seed);
  }

  /**
   * Plays a complete nello hand from bidding through scoring.
   *
   * Flow:
   * 1. Player 0 bids 1-marks
   * 2. Others pass
   * 3. Player 0 selects nello trump
   * 4. Play tricks until hand ends (7 tricks or early termination)
   * 5. All 4 players agree to complete each trick (auto-execute finalizes)
   * 6. All 4 players agree to score hand (auto-execute scores and resets to next hand)
   *
   * @param room - The HeadlessRoom instance
   * @param shouldBidderWin - Strategy: if true, bidder tries to win tricks; if false, tries to lose
   * @returns Tricks array captured BEFORE scoring clears it
   */
  function playNelloHand(room: HeadlessRoom, shouldBidderWin: boolean = false): Trick[] {
    // === BIDDING ===
    const bidAction = room.getValidActions(0).find(
      a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
    );
    expect(bidAction).toBeDefined();
    room.executeAction(0, bidAction!.action);

    // Others pass
    for (let i = 1; i < 4; i++) {
      const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
      expect(passAction).toBeDefined();
      room.executeAction(i, passAction!.action);
    }

    // === TRUMP SELECTION ===
    const nelloAction = room.getValidActions(0).find(
      a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
    );
    expect(nelloAction).toBeDefined();
    room.executeAction(0, nelloAction!.action);

    // === PLAY TRICKS ===
    let tricksPlayed = 0;
    const maxTricks = 7;

    while (room.getState().phase === 'playing' && tricksPlayed < maxTricks) {
      // Play 3 dominoes (nello: partner sits out)
      while (room.getState().currentTrick.length < 3) {
        const state = room.getState();
        const playerIndex = state.currentPlayer;
        const playActions = room.getValidActions(playerIndex).filter(a => a.action.type === 'play');

        expect(playActions.length).toBeGreaterThan(0);

        // Play strategy
        let selectedAction;
        if (shouldBidderWin && playerIndex === 0 && state.currentTrick.length === 2) {
          // Bidder plays high to win on last play
          selectedAction = playActions[playActions.length - 1];
        } else if (playerIndex === 0) {
          // Bidder plays low to lose
          selectedAction = playActions[0];
        } else {
          // Others play to help or hurt bidder
          selectedAction = shouldBidderWin ? playActions[0] : playActions[playActions.length - 1];
        }

        room.executeAction(playerIndex, selectedAction!.action);
      }

      // Consensus to complete trick
      // All 4 players must agree (including partner who sits out!)
      // Auto-execute will complete trick when all have agreed
      while (room.getState().phase === 'playing' && room.getState().tricks.length < tricksPlayed + 1) {
        let anyConsensus = false;

        for (let i = 0; i < 4; i++) {
          const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-complete-trick');
          if (agreeAction) {
            room.executeAction(i, agreeAction.action);
            anyConsensus = true;
          }
        }

        // After all 4 agree, processAutoExecuteActions completes the trick automatically
        if (!anyConsensus) break;
      }

      tricksPlayed++;

      // Check for early termination (bidder won a trick in nello)
      if (room.getState().phase === 'scoring') {
        break;
      }
    }

    // === CAPTURE TRICKS BEFORE SCORING ===
    // CRITICAL: executeScoreHand resets tricks: [] for next hand
    const tricksBeforeScoring = [...room.getState().tricks];

    // === SCORING ===
    // All 4 players must agree to score hand
    // Auto-execute will score when all have agreed
    if (room.getState().phase === 'scoring') {
      let rounds = 0;
      while (room.getState().phase === 'scoring' && rounds < 10) {
        rounds++;
        let anyConsensus = false;

        for (let i = 0; i < 4; i++) {
          const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-score-hand');
          if (agreeAction) {
            room.executeAction(i, agreeAction.action);
            anyConsensus = true;
          }
        }

        // After all 4 agree, processAutoExecuteActions scores automatically
        if (!anyConsensus) break;
      }
    }

    return tricksBeforeScoring;
  }

  // === TEST FIXTURES ===

  const standardHands = [
    HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']),
    HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']),
    HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']),
    HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6'])
  ];

  const bidderHighCards = [
    HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']),
    HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']),
    HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']),
    HandBuilder.fromStrings(['1-5', '2-4', '2-5', '3-5', '4-4', '4-5', '5-5'])
  ];

  const bidderLowCards = [
    HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']),
    HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']),
    HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']),
    HandBuilder.fromStrings(['1-5', '2-4', '2-5', '3-5', '4-4', '4-5', '5-5'])
  ];

  // === SUCCESSFUL NELLO TESTS ===

  describe('Successful Nello', () => {
    it('should complete when bidder loses all 7 tricks (3-player)', () => {
      const room = createNelloRoom(123456, standardHands);
      const tricks = playNelloHand(room, false);
      const finalState = room.getState();

      // Verify tricks (from BEFORE scoring)
      expect(tricks.length).toBe(7);

      const bidderTricks = tricks.filter(t =>
        t.winner !== undefined && finalState.players[t.winner]?.teamId === 0
      ).length;
      expect(bidderTricks).toBe(0);

      // Verify marks (from AFTER scoring)
      expect(finalState.teamMarks[0]).toBe(1); // Bidder succeeded
      expect(finalState.teamMarks[1]).toBe(0);
    });

    it('should have only 3 plays per trick (partner sits out)', () => {
      const room = createNelloRoom(789012, standardHands);
      const tricks = playNelloHand(room, false);

      // All tricks should have exactly 3 plays
      tricks.forEach(trick => {
        expect(trick.plays.length).toBe(3);
      });

      // Partner (player 2) should never play
      const partnerPlays = tricks.flatMap(t => t.plays).filter(p => p.player === 2);
      expect(partnerPlays.length).toBe(0);
    });

    it('should treat doubles as suit 7 (own suit)', () => {
      const room = createNelloRoom(345678, standardHands);

      // Bid and select nello
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
      );
      room.executeAction(0, bidAction!.action);

      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      const nelloAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      room.executeAction(0, nelloAction!.action);

      // Bidder leads with double (0-0)
      const playActions = room.getValidActions(0).filter(a => a.action.type === 'play');
      const doublePlay = playActions.find(a => a.action.type === 'play' && a.action.dominoId === '0-0');
      expect(doublePlay).toBeDefined();
      room.executeAction(0, doublePlay!.action);

      // Current suit should be 7 (doubles form own suit)
      expect(room.getState().currentSuit).toBe(7);
    });
  });

  // === FAILED NELLO TESTS ===

  describe('Failed Nello', () => {
    it('should end early when bidder wins a trick', () => {
      const room = createNelloRoom(901234, bidderHighCards);
      const tricks = playNelloHand(room, true);
      const finalState = room.getState();

      // Verify hand ended early
      expect(tricks.length).toBeLessThan(7);
      expect(tricks.length).toBeGreaterThan(0);

      // Verify bidder won at least one trick
      const bidderTricks = tricks.filter(t => {
        if (t.winner === undefined) return false;
        const winner = finalState.players[t.winner];
        return winner?.teamId === 0;
      });
      expect(bidderTricks.length).toBeGreaterThan(0);

      // Verify opponents scored
      expect(finalState.teamMarks[0]).toBe(0); // Bidder failed
      expect(finalState.teamMarks[1]).toBe(1); // Opponents scored
    });

    it('should score mark value for opponents on failure', () => {
      const room = createNelloRoom(567890, bidderHighCards);
      playNelloHand(room, true);
      const finalState = room.getState();

      // Opponents get the mark value
      expect(finalState.teamMarks[1]).toBe(1);
    });

    it('should continue playing when bidder loses all tricks so far', () => {
      const room = createNelloRoom(678901, bidderLowCards);
      const tricks = playNelloHand(room, false);
      const finalState = room.getState();

      // Verify all 7 tricks played
      expect(tricks.length).toBe(7);

      // Verify bidder scored marks (successful nello)
      expect(finalState.teamMarks[0]).toBeGreaterThan(0); // Bidder succeeded
      expect(finalState.teamMarks[1]).toBe(0);
    });
  });

  // === PLAYER ROTATION TESTS ===

  describe('Player Rotation', () => {
    it('should skip partner in turn order', () => {
      const room = createNelloRoom(111111, standardHands);

      // Bid and select nello
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
      );
      room.executeAction(0, bidAction!.action);

      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      const nelloAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      room.executeAction(0, nelloAction!.action);

      // Track turn order during first trick
      const turnOrder: number[] = [];
      while (room.getState().currentTrick.length < 3) {
        const playerIndex = room.getState().currentPlayer;
        turnOrder.push(playerIndex);

        const playAction = room.getValidActions(playerIndex).find(a => a.action.type === 'play');
        room.executeAction(playerIndex, playAction!.action);
      }

      // Should be 0 -> 1 -> 3 (partner 2 skipped)
      expect(turnOrder).toEqual([0, 1, 3]);
    });
  });

  // === BIDDING REQUIREMENTS TESTS ===

  describe('Bidding Requirements', () => {
    it('should only allow nello after marks bid', () => {
      const room = createNelloRoom(222222, standardHands);

      // Bid marks
      const marksBid = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks'
      );
      room.executeAction(0, marksBid!.action);

      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      // Nello should be available
      const nelloAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      expect(nelloAction).toBeDefined();
    });

    it('should not allow nello after points bid', () => {
      const room = createNelloRoom(333333, standardHands);

      // Bid 30 points
      const pointsBid = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'points' && a.action.value === 30
      );
      room.executeAction(0, pointsBid!.action);

      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      // Nello should NOT be available
      const nelloAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
      );
      expect(nelloAction).toBeUndefined();
    });
  });
});
