import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { HandBuilder } from '../../helpers';
import type { Domino, Trick } from '../../../game/types';
import type { GameConfig } from '../../../game/types/config';

/**
 * Nello Three-Player Integration Tests
 *
 * Tests core nello mechanics:
 * - Three-player trick completion (partner sits out)
 * - Successful nello (bidder loses all 7 tricks)
 * - Failed nello (bidder wins a trick)
 * - Nello trump selection after marks bid
 */
describe('Nello Three-Player Integration', () => {

  function createNelloRoom(seed: number, hands: Domino[][]): HeadlessRoom {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      layers: ['base', 'nello'],
      shuffleSeed: seed,
      dealOverrides: { initialHands: hands }
    };
    return new HeadlessRoom(config, seed);
  }

  function playNelloHand(room: HeadlessRoom, shouldBidderWin: boolean = false): Trick[] {
    // Bidding
    room.executeAction(0, room.getValidActions(0).find(
      a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1
    )!.action);
    for (let i = 1; i < 4; i++) {
      room.executeAction(i, room.getValidActions(i).find(a => a.action.type === 'pass')!.action);
    }

    // Trump selection
    room.executeAction(0, room.getValidActions(0).find(
      a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
    )!.action);

    // Play tricks
    let tricksPlayed = 0;
    while (room.getState().phase === 'playing' && tricksPlayed < 7) {
      // Play 3 dominoes per trick
      while (room.getState().currentTrick.length < 3) {
        const state = room.getState();
        const playerIndex = state.currentPlayer;
        const playActions = room.getValidActions(playerIndex).filter(a => a.action.type === 'play');
        const action = (shouldBidderWin && playerIndex === 0 && state.currentTrick.length === 2)
          ? playActions[playActions.length - 1]
          : (playerIndex === 0) ? playActions[0]
          : shouldBidderWin ? playActions[0] : playActions[playActions.length - 1];
        room.executeAction(playerIndex, action!.action);
      }

      // Consensus to complete trick
      while (room.getState().phase === 'playing' && room.getState().tricks.length < tricksPlayed + 1) {
        let anyConsensus = false;
        for (let i = 0; i < 4; i++) {
          const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-complete-trick');
          if (agreeAction) {
            room.executeAction(i, agreeAction.action);
            anyConsensus = true;
          }
        }
        if (!anyConsensus) break;
      }
      tricksPlayed++;
      if (room.getState().phase === 'scoring') break;
    }

    const tricks = [...room.getState().tricks];

    // Consensus to score hand
    if (room.getState().phase === 'scoring') {
      for (let rounds = 0; rounds < 10 && room.getState().phase === 'scoring'; rounds++) {
        for (let i = 0; i < 4; i++) {
          const agreeAction = room.getValidActions(i).find(a => a.action.type === 'agree-score-hand');
          if (agreeAction) room.executeAction(i, agreeAction.action);
        }
      }
    }

    return tricks;
  }

  // Test fixtures
  const bidderLosesAllHands = [
    HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']),
    HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']),
    HandBuilder.fromStrings(['0-6', '1-5', '1-6', '2-4', '2-5', '2-6', '4-4']),
    HandBuilder.fromStrings(['3-5', '3-6', '4-5', '4-6', '5-5', '5-6', '6-6'])
  ];

  const bidderWinsOneHands = [
    HandBuilder.fromStrings(['6-6', '5-6', '4-6', '3-6', '2-6', '1-6', '0-6']),
    HandBuilder.fromStrings(['0-0', '0-1', '0-2', '1-1', '1-2', '2-2', '3-3']),
    HandBuilder.fromStrings(['0-3', '0-4', '0-5', '1-3', '1-4', '2-3', '3-4']),
    HandBuilder.fromStrings(['1-5', '2-4', '2-5', '3-5', '4-4', '4-5', '5-5'])
  ];

  it('should complete tricks with exactly 3 plays (partner sits out)', () => {
    const room = createNelloRoom(123456, bidderLosesAllHands);
    const tricks = playNelloHand(room);

    expect(tricks.length).toBe(7);
    tricks.forEach(trick => {
      expect(trick.plays.length).toBe(3);
    });

    // Partner (player 2) should never play
    const partnerPlays = tricks.flatMap(t => t.plays).filter(p => p.player === 2);
    expect(partnerPlays.length).toBe(0);
  });

  it('should award marks when bidder loses all 7 tricks', () => {
    const room = createNelloRoom(345678, bidderLosesAllHands);
    const tricks = playNelloHand(room);
    const finalState = room.getState();

    expect(tricks.length).toBe(7);

    const bidderTricks = tricks.filter(t =>
      t.winner !== undefined && finalState.players[t.winner]?.teamId === 0
    ).length;
    expect(bidderTricks).toBe(0);

    expect(finalState.teamMarks[0]).toBe(1);
    expect(finalState.teamMarks[1]).toBe(0);
  });

  it('should end early and award marks to opponents when bidder wins a trick', () => {
    const room = createNelloRoom(901234, bidderWinsOneHands);
    const tricks = playNelloHand(room, true);
    const finalState = room.getState();

    expect(tricks.length).toBeLessThan(7);
    expect(tricks.length).toBeGreaterThan(0);

    const bidderTricks = tricks.filter(t =>
      t.winner !== undefined && finalState.players[t.winner]?.teamId === 0
    );
    expect(bidderTricks.length).toBeGreaterThan(0);

    expect(finalState.teamMarks[0]).toBe(0);
    expect(finalState.teamMarks[1]).toBe(1);
  });

  it('should allow nello selection after marks bid only', () => {
    // Marks bid allows nello
    const marksRoom = createNelloRoom(111111, bidderLosesAllHands);
    marksRoom.executeAction(0, marksRoom.getValidActions(0).find(
      a => a.action.type === 'bid' && a.action.bid === 'marks'
    )!.action);
    for (let i = 1; i < 4; i++) {
      marksRoom.executeAction(i, marksRoom.getValidActions(i).find(a => a.action.type === 'pass')!.action);
    }
    expect(marksRoom.getValidActions(0).find(
      a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
    )).toBeDefined();

    // Points bid does not allow nello
    const pointsRoom = createNelloRoom(222222, bidderLosesAllHands);
    pointsRoom.executeAction(0, pointsRoom.getValidActions(0).find(
      a => a.action.type === 'bid' && a.action.bid === 'points' && a.action.value === 30
    )!.action);
    for (let i = 1; i < 4; i++) {
      pointsRoom.executeAction(i, pointsRoom.getValidActions(i).find(a => a.action.type === 'pass')!.action);
    }
    expect(pointsRoom.getValidActions(0).find(
      a => a.action.type === 'select-trump' && a.action.trump?.type === 'nello'
    )).toBeUndefined();
  });
});
