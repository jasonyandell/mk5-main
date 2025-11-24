import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { HandBuilder } from '../../helpers';
import type { Domino, Trick } from '../../../game/types';
import type { GameConfig } from '../../../game/types/config';

/**
 * Sevens Full Hand Integration Tests
 *
 * Crystal Palace Principles:
 * - Trust auto-execute for complete-trick and score-hand actions
 * - All 4 players must agree to consensus
 * - Capture tricks BEFORE scoring (executeScoreHand resets tricks: [])
 * - Clean, coherent play mechanics throughout
 *
 * Sevens Rules:
 * - Domino closest to 7 total pips wins each trick
 * - Bidding team must win all 7 tricks (early termination if they lose one)
 * - Must play domino closest to 7 pips (constraint-based play validation)
 */
describe('Sevens Full Hand Integration', () => {

  /**
   * Creates a HeadlessRoom with sevens enabled and custom hands.
   */
  function createSevensRoom(seed: number, hands: Domino[][]): HeadlessRoom {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      enabledLayers: ['base', 'sevens'],
      shuffleSeed: seed,
      dealOverrides: { initialHands: hands }
    };

    return new HeadlessRoom(config, seed);
  }

  /**
   * Plays a complete sevens hand from bidding through scoring.
   *
   * Sevens is completely deterministic - outcome depends solely on dealt hands.
   * Players MUST play domino closest to 7 (no strategy/choice).
   *
   * Flow:
   * 1. Player 0 bids 1 mark
   * 2. Others pass
   * 3. Player 0 selects sevens trump
   * 4. Play tricks until hand ends (7 tricks or early termination)
   * 5. All 4 players agree to complete each trick (auto-execute finalizes)
   * 6. All 4 players agree to score hand (auto-execute scores)
   *
   * @param room - The HeadlessRoom instance
   * @returns Tricks array captured BEFORE scoring clears it
   */
  function playSevensHand(room: HeadlessRoom): Trick[] {
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
    const sevensAction = room.getValidActions(0).find(
      a => a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
    );
    expect(sevensAction).toBeDefined();
    room.executeAction(0, sevensAction!.action);

    // === PLAY TRICKS ===
    const maxTricks = 7;

    while (room.getState().phase === 'playing' && room.getState().tricks.length < maxTricks) {
      // Play 4 dominoes (standard 4-player)
      while (room.getState().currentTrick.length < 4) {
        const state = room.getState();
        const playerIndex = state.currentPlayer;
        const playActions = room.getValidActions(playerIndex).filter(a => a.action.type === 'play');

        expect(playActions.length).toBeGreaterThan(0);

        // Sevens is deterministic - ruleset already enforces "closest to 7"
        // No strategy needed, just play the first valid action
        room.executeAction(playerIndex, playActions[0]!.action);
      }

      // Consensus to complete trick
      // All 4 players must agree
      // Auto-execute will complete trick when all have agreed
      const tricksBeforeConsensus = room.getState().tricks.length;
      while (room.getState().phase === 'playing' && room.getState().tricks.length === tricksBeforeConsensus) {
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

      // Check for early termination (defenders won a trick in sevens)
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
  // All fixtures must have 28 unique dominoes (double-6 set), 7 per player
  // Distributed by distance from 7 to create desired outcomes

  // Bidding team (Team 0: P0, P2) has dominoes closest to 7 → wins all tricks
  const biddingTeamWinsHands = [
    HandBuilder.fromStrings(['3-4', '2-5', '1-6', '3-3', '2-4', '4-4', '1-5']), // P0: dist 0,0,0,1,1,1,1
    HandBuilder.fromStrings(['0-3', '5-6', '1-1', '0-2', '6-6', '0-1', '0-0']), // P1: dist 4,4,5,5,5,6,7 (far)
    HandBuilder.fromStrings(['3-5', '0-6', '2-6', '2-3', '1-4', '0-5', '4-5']), // P2: dist 1,1,1,2,2,2,2
    HandBuilder.fromStrings(['3-6', '2-2', '1-3', '0-4', '5-5', '4-6', '1-2'])  // P3: dist 2,3,3,3,3,3,4 (far)
  ];

  // Defending team (Team 1: P1, P3) has dominoes closest to 7 → bidder loses early
  const defendingTeamWinsHands = [
    HandBuilder.fromStrings(['0-3', '5-6', '1-1', '0-2', '6-6', '0-1', '0-0']), // P0: dist 4,4,5,5,5,6,7 (far)
    HandBuilder.fromStrings(['3-4', '2-5', '1-6', '3-3', '2-4', '4-4', '1-5']), // P1: dist 0,0,0,1,1,1,1 (close)
    HandBuilder.fromStrings(['3-6', '2-2', '1-3', '0-4', '5-5', '4-6', '1-2']), // P2: dist 2,3,3,3,3,3,4 (far)
    HandBuilder.fromStrings(['3-5', '0-6', '2-6', '2-3', '1-4', '0-5', '4-5'])  // P3: dist 1,1,1,2,2,2,2 (close)
  ];

  // === SUCCESSFUL SEVENS TESTS ===

  describe('Successful Sevens', () => {
    it('should complete when bidding team wins all 7 tricks', () => {
      const room = createSevensRoom(334455, biddingTeamWinsHands);
      const tricks = playSevensHand(room);
      const finalState = room.getState();

      // Verify all 7 tricks were played
      expect(tricks.length).toBe(7);

      // Verify all tricks have 4 plays (standard 4-player)
      tricks.forEach(trick => {
        expect(trick.plays.length).toBe(4);
      });

      // Verify bidding team won all tricks
      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && t.winner % 2 === 0 // Team 0
      );
      expect(biddingTeamTricks.length).toBe(7);

      // Verify bidding team scored
      expect(finalState.teamMarks[0]).toBe(1); // 1 mark bid
      expect(finalState.teamMarks[1]).toBe(0);
    });

    it('should continue when partner wins trick and partner leads next', () => {
      // Partner (P2) has dominoes closest to 7 to win tricks
      const partnerWinsHands = [
        HandBuilder.fromStrings(['0-3', '5-6', '1-1', '0-2', '6-6', '0-1', '0-0']), // P0 (bidder): far from 7
        HandBuilder.fromStrings(['3-6', '2-2', '1-3', '0-4', '5-5', '4-6', '1-2']), // P1: medium dist
        HandBuilder.fromStrings(['3-4', '2-5', '1-6', '3-3', '2-4', '4-4', '1-5']), // P2 (partner): closest to 7!
        HandBuilder.fromStrings(['3-5', '0-6', '2-6', '2-3', '1-4', '0-5', '4-5'])  // P3: medium dist
      ];

      const room = createSevensRoom(778899, partnerWinsHands);
      const tricks = playSevensHand(room);
      const finalState = room.getState();

      // Verify hand completed successfully
      expect(tricks.length).toBe(7);

      // Verify partner (P2) won tricks
      const partnerTricks = tricks.filter(t => t.winner === 2);
      expect(partnerTricks.length).toBeGreaterThan(0);

      // Verify bidding team still won (partner counts as bidding team)
      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && t.winner % 2 === 0 // Team 0 (P0 and P2)
      );
      expect(biddingTeamTricks.length).toBe(7);

      // Verify team scored marks
      expect(finalState.teamMarks[0]).toBe(1);
      expect(finalState.teamMarks[1]).toBe(0);
    });
  });

  // === FAILED SEVENS - EARLY TERMINATION TESTS ===

  describe('Failed Sevens - Early Termination', () => {
    it('should end early and award marks to defenders when they win a trick', () => {
      const room = createSevensRoom(556677, defendingTeamWinsHands);
      const tricks = playSevensHand(room);
      const finalState = room.getState();

      // Verify hand ended early (before 7 tricks)
      expect(tricks.length).toBeLessThan(7);
      expect(tricks.length).toBeGreaterThan(0);

      // Verify defending team won at least one trick
      const defendingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && t.winner % 2 === 1 // Team 1
      );
      expect(defendingTeamTricks.length).toBeGreaterThan(0);

      // Verify defending team scored (bid failed)
      expect(finalState.teamMarks[1]).toBe(1); // Defending team gets bid value
      expect(finalState.teamMarks[0]).toBe(0);
    });
  });

  // === SEVENS GAME MECHANICS TESTS ===

  describe('Sevens Game Mechanics', () => {
    it('should calculate trick winner based on closest to 7 pips', () => {
      const room = createSevensRoom(889900, biddingTeamWinsHands);

      // Bid and select sevens trump
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
      );
      room.executeAction(0, bidAction!.action);

      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      const sevensAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      room.executeAction(0, sevensAction!.action);

      // Play one trick
      for (let i = 0; i < 4; i++) {
        const playerIndex = room.getState().currentPlayer;
        const playActions = room.getValidActions(playerIndex).filter(a => a.action.type === 'play');
        room.executeAction(playerIndex, playActions[0]!.action);
      }

      // Complete trick with consensus
      for (let i = 0; i < 4; i++) {
        const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-complete-trick');
        if (agreeAction) {
          room.executeAction(i, agreeAction.action);
        }
      }

      // Verify a trick was completed
      const state = room.getState();
      expect(state.tricks.length).toBe(1);
      expect(state.tricks[0]!.winner).toBeGreaterThanOrEqual(0);
    });

    it('should be available after any marks bid', () => {
      const room = createSevensRoom(990011, biddingTeamWinsHands);

      // Bid 1 mark
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
      );
      expect(bidAction).toBeDefined();
      room.executeAction(0, bidAction!.action);

      // Others pass
      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      // Check sevens is available as a trump option
      const sevensAction = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      expect(sevensAction).toBeDefined();
    });
  });
});
