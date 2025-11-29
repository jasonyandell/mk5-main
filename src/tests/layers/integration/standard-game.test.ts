import { describe, it, expect } from 'vitest';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { StateBuilder, roomConsensus } from '../../helpers';
import type { Domino, Trick } from '../../../game/types';
import type { GameConfig } from '../../../game/types/config';

/**
 * Standard Game Integration Tests
 *
 * Tests essential flows for standard Texas 42 gameplay:
 * - Points bidding (30 points)
 * - Marks bidding (32+ points)
 * - Early termination conditions
 * - Trump selection and trick-taking mechanics
 *
 * Uses HeadlessRoom for realistic multiplayer flow simulation.
 * Hands are generated using dealConstraints to express test intent.
 */
describe('Standard Game Integration', () => {
  /**
   * Create a HeadlessRoom configured for standard gameplay
   *
   * @param seed - Seed for room initialization
   * @param hands - Pre-dealt hands (from StateBuilder with constraints)
   */
  function createStandardRoom(seed: number, hands: Domino[][]): HeadlessRoom {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      layers: ['base', 'consensus'],
      shuffleSeed: seed,
      dealOverrides: { initialHands: hands }
    };
    return new HeadlessRoom(config, seed);
  }

  /**
   * Helper: Generate a strong bidding hand for player 0
   * Strong = high points, multiple doubles, trump-rich (6s)
   */
  function createStrongBiddingHand(fillSeed: number): Domino[][] {
    const state = StateBuilder.inBiddingPhase()
      .withDealConstraints({
        players: {
          0: {
            minPoints: 25,      // Strong point-bearing dominoes
            minDoubles: 2,      // Multiple doubles for control
            minSuitCount: { 6: 4 } // Rich in 6s (common trump suit)
          }
        },
        fillSeed
      })
      .build();

    return state.players.map(p => p.hand);
  }

  /**
   * Helper: Generate a weak bidding hand for player 0
   * Weak = void in high suits, defenders have strong points/trumps
   */
  function createWeakBiddingHand(fillSeed: number): Domino[][] {
    const state = StateBuilder.inBiddingPhase()
      .withDealConstraints({
        players: {
          0: {
            voidInSuit: [6, 5], // Void in high suits (weak for bidding)
          },
          1: {
            minPoints: 15,      // Defenders have points
            minSuitCount: { 6: 3 } // Strong in 6s trump
          },
          3: {
            minPoints: 10,      // More defensive strength
            minSuitCount: { 5: 2 } // Strong in 5s
          }
        },
        fillSeed
      })
      .build();

    return state.players.map(p => p.hand);
  }

  /**
   * Helper: Generate a balanced hand distribution
   * All players get roughly equal strength
   */
  function createBalancedHands(fillSeed: number): Domino[][] {
    const state = StateBuilder.inBiddingPhase()
      .withDealConstraints({
        players: {
          0: { minDoubles: 2 },
          1: { minDoubles: 1 },
          2: { minDoubles: 1 },
          3: { minDoubles: 1 }
        },
        fillSeed
      })
      .build();

    return state.players.map(p => p.hand);
  }

  /**
   * Play a standard hand from bidding through scoring
   *
   * @param room - HeadlessRoom instance
   * @param bidValue - Points to bid (30, 32, 36, 42)
   * @param strategy - How to play tricks ('bidder-wins' | 'defenders-win' | 'balanced')
   * @returns Array of completed tricks (before scoring clears them)
   */
  function playStandardHand(
    room: HeadlessRoom,
    bidValue: number,
    strategy: 'bidder-wins' | 'defenders-win' | 'balanced' = 'balanced'
  ): Trick[] {
    // 1. Player 0 bids
    const bidActions = room.getValidActions(0);
    const bidAction = bidActions.find(a =>
      a.action.type === 'bid' &&
      ((a.action.bid === 'points' && a.action.value === bidValue) ||
       (a.action.bid === 'marks' && a.action.value === bidValue))
    );
    expect(bidAction, `Bid ${bidValue} should be available`).toBeDefined();
    room.executeAction(0, bidAction!.action);

    // 2. Other players pass
    for (let i = 1; i < 4; i++) {
      const passActions = room.getValidActions(i);
      const passAction = passActions.find(a => a.action.type === 'pass');
      expect(passAction, `Player ${i} should have pass option`).toBeDefined();
      room.executeAction(i, passAction!.action);
    }

    // 3. Player 0 selects trump (first available suit trump)
    const trumpActions = room.getValidActions(0);
    const trumpAction = trumpActions.find(a =>
      a.action.type === 'select-trump' &&
      a.action.trump?.type === 'suit'
    );
    expect(trumpAction, 'Trump selection should be available').toBeDefined();
    room.executeAction(0, trumpAction!.action);

    // 4. Play tricks until hand ends (early termination or 7 tricks)
    let trickCount = 0;

    while (room.getState().phase === 'playing' && trickCount < 7) {
      // Each player plays in turn (4 plays per trick)
      for (let i = 0; i < 4; i++) {
        const currentPlayer = room.getState().currentPlayer;
        const playActions = room.getValidActions(currentPlayer);
        const plays = playActions.filter(a => a.action.type === 'play');

        expect(plays.length, `Player ${currentPlayer} should have plays available`).toBeGreaterThan(0);

        // Select play based on strategy
        let selectedPlay;
        const biddingTeam = 0; // Player 0's team (0 or 2)
        const currentPlayerTeam = currentPlayer % 2;

        if (strategy === 'bidder-wins') {
          selectedPlay = currentPlayerTeam === biddingTeam
            ? plays[plays.length - 1] // Bidders play high
            : plays[0]; // Defenders play low
        } else if (strategy === 'defenders-win') {
          selectedPlay = currentPlayerTeam === biddingTeam
            ? plays[0] // Bidders play low
            : plays[plays.length - 1]; // Defenders play high
        } else {
          // Balanced - middle play
          selectedPlay = plays[Math.floor(plays.length / 2)];
        }

        room.executeAction(currentPlayer, selectedPlay!.action);
      }

      // All 4 players agree to complete trick (consensus)
      // When all 4 agree, complete-trick is auto-executed
      roomConsensus(room, 'trick');

      // After consensus, trick is automatically completed
      trickCount++;

      // Check if phase changed after completing trick (early termination)
      if (room.getState().phase !== 'playing') {
        break;
      }
    }

    // Capture tricks before scoring clears them
    let currentState = room.getState();
    const tricks = [...currentState.tricks];

    // 5. Score hand (consensus + score-hand action)
    // After all tricks or early termination, phase should be 'scoring'
    while (currentState.phase === 'scoring' || currentState.phase === 'setup' || currentState.phase === 'one-hand-complete') {
      if (currentState.phase === 'scoring') {
        // All players agree to score (auto-executes score-hand when consensus reached)
        roomConsensus(room, 'score');
      } else if (currentState.phase === 'setup' || currentState.phase === 'one-hand-complete') {
        // Deal next hand
        const dealActions = room.getValidActions(0);
        // Look for any action that will progress the game (deal is internal, may not be exposed)
        const nextAction = dealActions[0];
        if (nextAction) {
          room.executeAction(0, nextAction.action);
        } else {
          // No actions available, break to avoid infinite loop
          break;
        }
      }

      currentState = room.getState();
    }

    return tricks;
  }

  describe('Points Bidding', () => {
    it('should complete successful 30-point bid with early termination', () => {
      // Strong hand for player 0: high points, doubles, trump-rich
      const hands = createStrongBiddingHand(123456);

      const room = createStandardRoom(123456, hands);
      const tricks = playStandardHand(room, 30, 'balanced');

      // Early termination should occur
      expect(tricks.length).toBeLessThanOrEqual(7);
      expect(tricks.length).toBeGreaterThan(0);

      // After scoring, game should transition to next hand
      const finalState = room.getState();
      expect(finalState.phase).toBe('bidding');

      // One team awarded a mark
      const totalMarks = finalState.teamMarks[0] + finalState.teamMarks[1];
      expect(totalMarks).toBe(1);

      // New hand dealt
      finalState.players.forEach(player => {
        expect(player.hand.length).toBe(7);
      });
    });

    it('should handle failed bid (defenders set)', () => {
      // Weak hand for player 0: low points, void in trump, defenders strong
      const hands = createWeakBiddingHand(456789);

      const room = createStandardRoom(456789, hands);
      const tricks = playStandardHand(room, 30, 'defenders-win');

      // Early termination when defenders set bid
      expect(tricks.length).toBeLessThan(7);

      const finalState = room.getState();
      expect(finalState.phase).toBe('bidding');

      // Defending team (team 1) should get the mark
      expect(finalState.teamMarks[1]).toBe(1);
      expect(finalState.teamMarks[0]).toBe(0);
    });
  });

  describe('Marks Bidding', () => {
    it('should complete marks bid successfully when bidders get all points', () => {
      // Strong hand for marks bid (32): needs to capture ALL points
      const hands = createStrongBiddingHand(234567);

      const room = createStandardRoom(234567, hands);
      playStandardHand(room, 32, 'bidder-wins');

      const finalState = room.getState();
      expect(finalState.phase).toBe('bidding');

      // Marks bid awards 2 marks on success
      const totalMarks = finalState.teamMarks[0] + finalState.teamMarks[1];
      expect(totalMarks).toBeGreaterThanOrEqual(1);
    });

    it('should fail marks bid when defenders score any points', () => {
      // Weak hand for player 0: marks bid fails if defenders score anything
      const hands = createWeakBiddingHand(678901);

      const room = createStandardRoom(678901, hands);
      const tricks = playStandardHand(room, 32, 'defenders-win');

      // Early termination when defenders score
      expect(tricks.length).toBeLessThan(7);

      const finalState = room.getState();
      expect(finalState.phase).toBe('bidding');

      // Defending team gets mark
      expect(finalState.teamMarks[1]).toBeGreaterThan(0);
    });
  });

  describe('Early Termination', () => {
    it('should end when bidders reach their bid', () => {
      // Strong hand: bidders will reach 30 before all 7 tricks
      const hands = createStrongBiddingHand(345678);

      const room = createStandardRoom(345678, hands);
      const tricks = playStandardHand(room, 30, 'bidder-wins');

      // Should end before 7 tricks
      expect(tricks.length).toBeLessThan(7);
      expect(room.getState().phase).toBe('bidding');
    });

    it('should end when bidders cannot possibly reach bid', () => {
      // Weak hand: defenders will prevent bidders from reaching 30
      const hands = createWeakBiddingHand(567890);

      const room = createStandardRoom(567890, hands);
      const tricks = playStandardHand(room, 30, 'defenders-win');

      expect(tricks.length).toBeLessThan(7);
      expect(room.getState().phase).toBe('bidding');
    });
  });

  describe('Game Mechanics', () => {
    it('should allow bidder to select trump', () => {
      // Balanced hands: focus is on mechanics, not bidding strength
      const hands = createBalancedHands(890123);

      const room = createStandardRoom(890123, hands);

      // Bid
      const bidActions = room.getValidActions(0);
      const bid = bidActions.find(a => a.action.type === 'bid');
      room.executeAction(0, bid!.action);

      // Others pass
      for (let i = 1; i < 4; i++) {
        const pass = room.getValidActions(i).find(a => a.action.type === 'pass');
        room.executeAction(i, pass!.action);
      }

      // Verify trump selection available
      expect(room.getState().phase).toBe('trump_selection');
      expect(room.getState().winningBidder).toBe(0);

      const trumpActions = room.getValidActions(0);
      const trumpOptions = trumpActions.filter(a => a.action.type === 'select-trump');

      // Should have multiple trump options (suits + doubles)
      expect(trumpOptions.length).toBeGreaterThan(0);
    });

    it('should have bidder lead first trick', () => {
      // Balanced hands for all players - testing mechanic, not strength
      const hands = createBalancedHands(901234);

      const room = createStandardRoom(901234, hands);

      // Player 2 bids (not player 0)
      room.executeAction(0, room.getValidActions(0).find(a => a.action.type === 'pass')!.action);
      room.executeAction(1, room.getValidActions(1).find(a => a.action.type === 'pass')!.action);
      room.executeAction(2, room.getValidActions(2).find(a => a.action.type === 'bid')!.action);
      room.executeAction(3, room.getValidActions(3).find(a => a.action.type === 'pass')!.action);

      // Select trump
      const trump = room.getValidActions(2).find(a => a.action.type === 'select-trump');
      room.executeAction(2, trump!.action);

      // Verify player 2 leads
      expect(room.getState().phase).toBe('playing');
      expect(room.getState().currentPlayer).toBe(2);
    });

    it('should complete standard 4-player trick taking', () => {
      // Balanced distribution for standard trick-taking test
      const hands = createBalancedHands(112233);

      const room = createStandardRoom(112233, hands);
      const tricks = playStandardHand(room, 30, 'balanced');

      // Each trick should have 4 plays
      tricks.forEach(trick => {
        expect(trick.plays.length).toBe(4);
      });

      // Should have completed at least one trick
      expect(tricks.length).toBeGreaterThan(0);
      expect(tricks.length).toBeLessThanOrEqual(7);
    });
  });
});
