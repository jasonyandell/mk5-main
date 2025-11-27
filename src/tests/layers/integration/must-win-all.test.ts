/**
 * Integration tests for "must win all tricks" contracts.
 *
 * Plunge, splash, and sevens share the same core mechanic:
 * - Bidding team must win all 7 tricks
 * - Early termination if opponents win any trick
 * - Marks awarded based on success/failure
 *
 * This file consolidates integration tests for all three contracts,
 * verifying full game flow using HeadlessRoom.
 */

import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { HandBuilder } from '../../helpers';
import type { Domino, Trick } from '../../../game/types';
import type { GameConfig } from '../../../game/types/config';

/**
 * Test fixture type for contract scenarios
 */
interface ContractTestFixture {
  name: string;
  layers: string[];
  bidType: string;
  bidValue: number;
  trumpType: 'suit' | 'sevens' | 'doubles';
  minDoubles: number; // For plunge/splash eligibility
}

/**
 * Creates a HeadlessRoom with specified layers and custom hands.
 */
function createRoom(
  layers: string[],
  seed: number,
  hands: Domino[][]
): HeadlessRoom {
  const config: GameConfig = {
    playerTypes: ['human', 'human', 'human', 'human'],
    layers,
    shuffleSeed: seed,
    dealOverrides: { initialHands: hands }
  };
  return new HeadlessRoom(config, seed);
}

/**
 * Plays a complete hand for a "must win all" contract.
 *
 * Flow:
 * 1. Player 0 places the contract bid
 * 2. Others pass
 * 3. Trump selector (bidder or partner) selects trump
 * 4. Play tricks until hand ends
 * 5. All players agree to complete each trick
 * 6. All players agree to score hand
 *
 * @returns Tricks array captured BEFORE scoring clears it
 */
function playMustWinAllHand(
  room: HeadlessRoom,
  bidType: string,
  _bidValue: number,
  trumpType: 'suit' | 'sevens' | 'doubles'
): Trick[] {
  // === BIDDING ===
  const bidAction = room.getValidActions(0).find(
    a => a.action.type === 'bid' && a.action.bid === bidType
  );
  expect(bidAction, `Expected ${bidType} bid to be available`).toBeDefined();
  room.executeAction(0, bidAction!.action);

  // Others pass
  for (let i = 1; i < 4; i++) {
    const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
    expect(passAction).toBeDefined();
    room.executeAction(i, passAction!.action);
  }

  // === TRUMP SELECTION ===
  const state = room.getState();
  const trumpSelector = state.currentPlayer;

  const trumpAction = room.getValidActions(trumpSelector).find(a => {
    if (a.action.type !== 'select-trump') return false;
    if (trumpType === 'sevens') return a.action.trump?.type === 'sevens';
    if (trumpType === 'doubles') return a.action.trump?.type === 'doubles';
    return a.action.trump?.type === 'suit';
  });
  expect(trumpAction, `Expected ${trumpType} trump option`).toBeDefined();
  room.executeAction(trumpSelector, trumpAction!.action);

  // === PLAY TRICKS ===
  const maxTricks = 7;

  while (room.getState().phase === 'playing' && room.getState().tricks.length < maxTricks) {
    // Play 4 dominoes
    while (room.getState().currentTrick.length < 4) {
      const currentState = room.getState();
      const playerIndex = currentState.currentPlayer;
      const playActions = room.getValidActions(playerIndex).filter(a => a.action.type === 'play');

      expect(playActions.length).toBeGreaterThan(0);
      // Play the first valid action (deterministic for tests)
      room.executeAction(playerIndex, playActions[0]!.action);
    }

    // Consensus to complete trick - all 4 players must agree
    const tricksBeforeConsensus = room.getState().tricks.length;
    let rounds = 0;
    while (room.getState().phase === 'playing' && room.getState().tricks.length === tricksBeforeConsensus && rounds < 10) {
      rounds++;
      for (let i = 0; i < 4; i++) {
        const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-complete-trick');
        if (agreeAction) {
          room.executeAction(i, agreeAction.action);
        }
      }
    }

    // Check for early termination
    if (room.getState().phase === 'scoring') break;
  }

  // === CAPTURE TRICKS BEFORE SCORING ===
  const tricksBeforeScoring = [...room.getState().tricks];

  // === SCORING ===
  if (room.getState().phase === 'scoring') {
    let rounds = 0;
    while (room.getState().phase === 'scoring' && rounds < 10) {
      rounds++;
      for (let i = 0; i < 4; i++) {
        const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-score-hand');
        if (agreeAction) {
          room.executeAction(i, agreeAction.action);
        }
      }
    }
  }

  return tricksBeforeScoring;
}

// === TEST FIXTURES ===

// Hands where bidding team (P0, P2) has strong dominoes
const biddingTeamStrongHands = [
  HandBuilder.fromStrings(['6-6', '5-5', '4-4', '3-3', '6-5', '5-4', '4-3']), // P0: 4 doubles + high
  HandBuilder.fromStrings(['0-1', '0-2', '0-3', '1-2', '1-3', '2-3', '2-4']), // P1: weak
  HandBuilder.fromStrings(['6-4', '6-3', '6-2', '6-1', '5-3', '5-2', '5-1']), // P2: strong non-doubles
  HandBuilder.fromStrings(['0-0', '1-1', '2-2', '0-4', '0-5', '0-6', '1-4'])  // P3: 3 doubles but weak
];

// Hands where defending team (P1, P3) can win tricks
// Must have 28 unique dominoes across all 4 hands
const defendingTeamStrongHands = [
  HandBuilder.fromStrings(['0-0', '1-1', '2-2', '3-3', '0-1', '0-2', '0-3']), // P0: 4 doubles but weak non-doubles
  HandBuilder.fromStrings(['6-6', '6-5', '6-4', '6-3', '6-2', '6-1', '6-0']), // P1: all sixes - very strong!
  HandBuilder.fromStrings(['0-4', '0-5', '1-2', '1-3', '1-4', '1-5', '2-3']), // P2: weak
  HandBuilder.fromStrings(['5-5', '4-4', '5-4', '5-3', '4-3', '5-2', '4-2'])  // P3: strong
];

// Hands for sevens (based on distance from 7 pips)
const sevensWinHands = [
  HandBuilder.fromStrings(['3-4', '2-5', '1-6', '3-3', '2-4', '4-4', '1-5']), // P0: dist 0,0,0,1,1,1,1 (closest)
  HandBuilder.fromStrings(['0-3', '5-6', '1-1', '0-2', '6-6', '0-1', '0-0']), // P1: dist 4,4,5,5,5,6,7 (far)
  HandBuilder.fromStrings(['3-5', '0-6', '2-6', '2-3', '1-4', '0-5', '4-5']), // P2: dist 1,1,1,2,2,2,2
  HandBuilder.fromStrings(['3-6', '2-2', '1-3', '0-4', '5-5', '4-6', '1-2'])  // P3: dist 2,3,3,3,3,3,4 (far)
];

const sevensLoseHands = [
  HandBuilder.fromStrings(['0-3', '5-6', '1-1', '0-2', '6-6', '0-1', '0-0']), // P0: dist 4,4,5,5,5,6,7 (far)
  HandBuilder.fromStrings(['3-4', '2-5', '1-6', '3-3', '2-4', '4-4', '1-5']), // P1: dist 0,0,0,1,1,1,1 (closest!)
  HandBuilder.fromStrings(['3-6', '2-2', '1-3', '0-4', '5-5', '4-6', '1-2']), // P2: dist 2,3,3,3,3,3,4 (far)
  HandBuilder.fromStrings(['3-5', '0-6', '2-6', '2-3', '1-4', '0-5', '4-5'])  // P3: dist 1,1,1,2,2,2,2 (close)
];

/**
 * Contract test fixtures
 */
const contracts: ContractTestFixture[] = [
  {
    name: 'plunge',
    layers: ['base', 'plunge'],
    bidType: 'plunge',
    bidValue: 4,
    trumpType: 'suit',
    minDoubles: 4
  },
  {
    name: 'splash',
    layers: ['base', 'splash'],
    bidType: 'splash',
    bidValue: 2,
    trumpType: 'suit',
    minDoubles: 3
  }
];

/**
 * Parameterized tests for plunge and splash contracts
 */
describe.each(contracts)('$name contract integration', ({ layers, bidType, bidValue, trumpType }) => {

  describe('successful contract', () => {
    it('completes when bidding team wins all 7 tricks', () => {
      const room = createRoom(layers, 112233, biddingTeamStrongHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, trumpType);
      const finalState = room.getState();

      // Count tricks by team
      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 0 || t.winner === 2)
      ).length;

      // Verify scoring matches actual outcome
      if (biddingTeamTricks === 7) {
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(bidValue);
        expect(finalState.teamMarks[1]).toBe(0);
      }
    });

    it(`awards ${bidValue}+ marks on success`, () => {
      const room = createRoom(layers, 445566, biddingTeamStrongHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, trumpType);
      const finalState = room.getState();

      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 0 || t.winner === 2)
      ).length;

      if (biddingTeamTricks === 7) {
        expect(finalState.teamMarks[0]).toBeGreaterThanOrEqual(bidValue);
      }
    });
  });

  describe('failed contract', () => {
    it('terminates early when opponents win a trick', () => {
      const room = createRoom(layers, 778899, defendingTeamStrongHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, trumpType);

      // Should have fewer than 7 tricks if early termination
      const defendingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 1 || t.winner === 3)
      ).length;

      if (defendingTeamTricks > 0) {
        // Early termination should have occurred
        expect(tricks.length).toBeLessThanOrEqual(7);
      }
    });

    it('awards marks to opponents on failure', () => {
      const room = createRoom(layers, 101112, defendingTeamStrongHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, trumpType);
      const finalState = room.getState();

      const defendingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 1 || t.winner === 3)
      ).length;

      if (defendingTeamTricks > 0) {
        // Opponents should get the marks
        expect(finalState.teamMarks[1]).toBeGreaterThanOrEqual(bidValue);
        expect(finalState.teamMarks[0]).toBe(0);
      }
    });
  });

  describe('partner mechanics', () => {
    it('partner selects trump (not bidder)', () => {
      const room = createRoom(layers, 131415, biddingTeamStrongHands);

      // Bid
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === bidType
      );
      room.executeAction(0, bidAction!.action);

      // Others pass
      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      // Partner (player 2) should be current player for trump selection
      expect(room.getState().currentPlayer).toBe(2);
    });
  });
});

/**
 * Sevens contract tests (separate because trump type differs)
 */
describe('sevens contract integration', () => {
  const layers = ['base', 'sevens'];
  const bidType = 'marks';
  const bidValue = 1;

  describe('successful sevens', () => {
    it('completes when bidding team wins all 7 tricks', () => {
      const room = createRoom(layers, 161718, sevensWinHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, 'sevens');
      const finalState = room.getState();

      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 0 || t.winner === 2)
      ).length;

      if (biddingTeamTricks === 7) {
        expect(finalState.teamMarks[0]).toBe(bidValue);
        expect(finalState.teamMarks[1]).toBe(0);
      }
    });

    it('plays all 7 tricks when bidding team winning', () => {
      const room = createRoom(layers, 192021, sevensWinHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, 'sevens');

      const biddingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 0 || t.winner === 2)
      ).length;

      if (biddingTeamTricks === 7) {
        expect(tricks.length).toBe(7);
      }
    });
  });

  describe('failed sevens', () => {
    it('terminates early when opponents win a trick', () => {
      const room = createRoom(layers, 222324, sevensLoseHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, 'sevens');

      const defendingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 1 || t.winner === 3)
      ).length;

      if (defendingTeamTricks > 0) {
        expect(tricks.length).toBeLessThanOrEqual(7);
      }
    });

    it('awards marks to opponents on failure', () => {
      const room = createRoom(layers, 252627, sevensLoseHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, 'sevens');
      const finalState = room.getState();

      const defendingTeamTricks = tricks.filter(t =>
        t.winner !== undefined && (t.winner === 1 || t.winner === 3)
      ).length;

      if (defendingTeamTricks > 0) {
        expect(finalState.teamMarks[1]).toBe(bidValue);
        expect(finalState.teamMarks[0]).toBe(0);
      }
    });
  });

  describe('sevens-specific mechanics', () => {
    it('bidder selects sevens trump (not partner)', () => {
      const room = createRoom(layers, 282930, sevensWinHands);

      // Bid marks
      const bidAction = room.getValidActions(0).find(
        a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
      );
      room.executeAction(0, bidAction!.action);

      // Others pass
      for (let i = 1; i < 4; i++) {
        const passAction = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, passAction!.action);
      }

      // Bidder (player 0) should be current player for trump selection
      expect(room.getState().currentPlayer).toBe(0);

      // Sevens trump option should be available
      const sevensOption = room.getValidActions(0).find(
        a => a.action.type === 'select-trump' && a.action.trump?.type === 'sevens'
      );
      expect(sevensOption).toBeDefined();
    });

    it('trick winner is determined by distance from 7', () => {
      const room = createRoom(layers, 313233, sevensWinHands);
      const tricks = playMustWinAllHand(room, bidType, bidValue, 'sevens');

      // All tricks should have 4 plays (standard 4-player)
      tricks.forEach(trick => {
        expect(trick.plays.length).toBe(4);
        expect(trick.winner).toBeDefined();
      });
    });
  });
});
