/**
 * Comprehensive test suite for StateBuilder
 *
 * Tests all factory methods, modifiers, and helper builders
 */

import { describe, it, expect } from 'vitest';
import { StateBuilder, DominoBuilder, HandBuilder } from './stateBuilder';
import { ACES, BLANKS } from '../../game/types';
import { BID_TYPES } from '../../game/constants';

describe('DominoBuilder', () => {
  describe('from()', () => {
    it('should parse valid domino string', () => {
      const domino = DominoBuilder.from('6-5');
      expect(domino.high).toBe(6);
      expect(domino.low).toBe(5);
      expect(domino.id).toBe('6-5');
    });

    it('should normalize order (handle "5-6" as "6-5")', () => {
      const domino = DominoBuilder.from('5-6');
      expect(domino.high).toBe(6);
      expect(domino.low).toBe(5);
      expect(domino.id).toBe('6-5');
    });

    it('should handle doubles', () => {
      const domino = DominoBuilder.from('6-6');
      expect(domino.high).toBe(6);
      expect(domino.low).toBe(6);
      expect(domino.id).toBe('6-6');
    });

    it('should calculate points correctly for 5-5', () => {
      const domino = DominoBuilder.from('5-5');
      expect(domino.points).toBe(10);
    });

    it('should calculate points correctly for 6-4', () => {
      const domino = DominoBuilder.from('6-4');
      expect(domino.points).toBe(10);
    });

    it('should calculate points correctly for 5-0', () => {
      const domino = DominoBuilder.from('5-0');
      expect(domino.points).toBe(5);
    });

    it('should calculate points correctly for 6-5', () => {
      const domino = DominoBuilder.from('6-5');
      expect(domino.points).toBe(5);
    });

    it('should throw on invalid format', () => {
      expect(() => DominoBuilder.from('6')).toThrow('Invalid domino ID');
      expect(() => DominoBuilder.from('6-5-4')).toThrow('Invalid domino ID');
      expect(() => DominoBuilder.from('abc')).toThrow('Invalid domino ID');
    });

    it('should throw on out-of-range pips', () => {
      expect(() => DominoBuilder.from('7-5')).toThrow('Pips must be 0-6');
      expect(() => DominoBuilder.from('6-7')).toThrow('Pips must be 0-6');
      expect(() => DominoBuilder.from('-1-5')).toThrow('Invalid domino ID');
    });
  });

  describe('fromPair()', () => {
    it('should create domino from pair', () => {
      const domino = DominoBuilder.fromPair(6, 5);
      expect(domino.high).toBe(6);
      expect(domino.low).toBe(5);
      expect(domino.id).toBe('6-5');
    });

    it('should throw on invalid pips', () => {
      expect(() => DominoBuilder.fromPair(7, 5)).toThrow('Invalid domino pips');
      expect(() => DominoBuilder.fromPair(-1, 5)).toThrow('Invalid domino pips');
    });
  });

  describe('doubles()', () => {
    it('should create double domino', () => {
      const domino = DominoBuilder.doubles(6);
      expect(domino.high).toBe(6);
      expect(domino.low).toBe(6);
      expect(domino.id).toBe('6-6');
    });

    it('should create all doubles', () => {
      for (let i = 0; i <= 6; i++) {
        const domino = DominoBuilder.doubles(i as 0 | 1 | 2 | 3 | 4 | 5 | 6);
        expect(domino.high).toBe(i);
        expect(domino.low).toBe(i);
      }
    });
  });
});

describe('HandBuilder', () => {
  describe('withDoubles()', () => {
    it('should create hand with specified doubles', () => {
      const hand = HandBuilder.withDoubles(3);
      expect(hand).toHaveLength(7);

      const doubles = hand.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(3);
    });

    it('should handle 0 doubles', () => {
      const hand = HandBuilder.withDoubles(0);
      expect(hand).toHaveLength(7);

      const doubles = hand.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(0);
    });

    it('should handle 7 doubles', () => {
      const hand = HandBuilder.withDoubles(7);
      expect(hand).toHaveLength(7);

      const doubles = hand.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(7);
    });

    it('should throw on invalid count', () => {
      expect(() => HandBuilder.withDoubles(-1)).toThrow('Invalid double count');
      expect(() => HandBuilder.withDoubles(8)).toThrow('Invalid double count');
    });
  });

  describe('fromStrings()', () => {
    it('should parse array of domino strings', () => {
      const hand = HandBuilder.fromStrings(['6-6', '6-5', '5-5']);
      expect(hand).toHaveLength(3);
      expect(hand[0]!.id).toBe('6-6');
      expect(hand[1]!.id).toBe('6-5');
      expect(hand[2]!.id).toBe('5-5');
    });

    it('should handle empty array', () => {
      const hand = HandBuilder.fromStrings([]);
      expect(hand).toHaveLength(0);
    });
  });

  describe('random()', () => {
    it('should generate 7 dominoes', () => {
      const hand = HandBuilder.random();
      expect(hand).toHaveLength(7);
    });

    it('should be deterministic with seed', () => {
      const hand1 = HandBuilder.random(12345);
      const hand2 = HandBuilder.random(12345);

      expect(hand1.map(d => d.id)).toEqual(hand2.map(d => d.id));
    });

    it('should be different without seed', () => {
      const hand1 = HandBuilder.random(12345);
      const hand2 = HandBuilder.random(67890);

      expect(hand1.map(d => d.id)).not.toEqual(hand2.map(d => d.id));
    });
  });
});

describe('StateBuilder - Factory Methods', () => {
  describe('inBiddingPhase()', () => {
    it('should create bidding phase state', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(state.phase).toBe('bidding');
      expect(state.players).toHaveLength(4);
      expect(state.dealer).toBe(0);
      expect(state.currentPlayer).toBe(1); // Left of dealer
    });

    it('should accept custom dealer', () => {
      const state = StateBuilder.inBiddingPhase(2).build();

      expect(state.dealer).toBe(2);
      expect(state.currentPlayer).toBe(3); // Left of dealer
    });

    it('should deal hands to all players', () => {
      const state = StateBuilder.inBiddingPhase().build();

      state.players.forEach(player => {
        expect(player.hand.length).toBe(7);
      });
    });
  });

  describe('inTrumpSelection()', () => {
    it('should create trump selection state', () => {
      const state = StateBuilder.inTrumpSelection(0, 32).build();

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
      expect(state.currentPlayer).toBe(0);
      expect(state.currentBid.type).toBe(BID_TYPES.POINTS);
      expect(state.currentBid.value).toBe(32);
    });

    it('should default to bid of 30', () => {
      const state = StateBuilder.inTrumpSelection(1).build();

      expect(state.currentBid.value).toBe(30);
    });

    it('should create bid history', () => {
      const state = StateBuilder.inTrumpSelection(0, 32).build();

      expect(state.bids).toHaveLength(1);
      expect(state.bids[0]!.player).toBe(0);
      expect(state.bids[0]!.value).toBe(32);
    });
  });

  describe('inPlayingPhase()', () => {
    it('should create playing phase with default trump', () => {
      const state = StateBuilder.inPlayingPhase().build();

      expect(state.phase).toBe('playing');
      expect(state.trump.type).toBe('suit');
      expect(state.trump.suit).toBe(ACES);
    });

    it('should accept custom trump', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'suit', suit: BLANKS }).build();

      expect(state.trump.type).toBe('suit');
      expect(state.trump.suit).toBe(BLANKS);
    });

    it('should handle doubles trump', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'doubles' }).build();

      expect(state.trump.type).toBe('doubles');
    });

    it('should handle no-trump', () => {
      const state = StateBuilder.inPlayingPhase({ type: 'no-trump' }).build();

      expect(state.trump.type).toBe('no-trump');
    });
  });

  describe('withTricksPlayed()', () => {
    it('should create state with N tricks', () => {
      const state = StateBuilder.withTricksPlayed(3).build();

      expect(state.tricks).toHaveLength(3);
      expect(state.phase).toBe('playing');
    });

    it('should reduce hand sizes', () => {
      const state = StateBuilder.withTricksPlayed(3).build();

      state.players.forEach(player => {
        expect(player.hand.length).toBeLessThanOrEqual(4); // 7 - 3 = 4
      });
    });

    it('should throw on invalid count', () => {
      expect(() => StateBuilder.withTricksPlayed(-1).build()).toThrow('Invalid trick count');
      expect(() => StateBuilder.withTricksPlayed(8).build()).toThrow('Invalid trick count');
    });
  });

  describe('inScoringPhase()', () => {
    it('should create scoring phase state', () => {
      const state = StateBuilder.inScoringPhase([30, 12]).build();

      expect(state.phase).toBe('scoring');
      expect(state.teamScores).toEqual([30, 12]);
      expect(state.tricks).toHaveLength(7);
    });

    it('should empty all hands', () => {
      const state = StateBuilder.inScoringPhase([30, 12]).build();

      state.players.forEach(player => {
        expect(player.hand).toHaveLength(0);
      });
    });
  });

  describe('gameEnded()', () => {
    it('should create game_end state for team 0 win', () => {
      const state = StateBuilder.gameEnded(0).build();

      expect(state.phase).toBe('game_end');
      expect(state.teamMarks).toEqual([7, 0]);
    });

    it('should create game_end state for team 1 win', () => {
      const state = StateBuilder.gameEnded(1).build();

      expect(state.phase).toBe('game_end');
      expect(state.teamMarks).toEqual([0, 7]);
    });
  });
});

describe('StateBuilder - Special Contract Presets', () => {
  describe('nelloContract()', () => {
    it('should create nello state', () => {
      const state = StateBuilder.nelloContract(0).build();

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
      expect(state.currentBid.type).toBe(BID_TYPES.MARKS);
    });
  });

  describe('splashContract()', () => {
    it('should create splash state with partner as current player', () => {
      const state = StateBuilder.splashContract(0, 2).build();

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
      expect(state.currentPlayer).toBe(2); // Partner
      expect(state.currentBid.type).toBe('splash');
      expect(state.currentBid.value).toBe(2);
    });

    it('should throw on invalid splash value', () => {
      expect(() => StateBuilder.splashContract(0, 1).build()).toThrow('Invalid splash value');
      expect(() => StateBuilder.splashContract(0, 4).build()).toThrow('Invalid splash value');
    });
  });

  describe('plungeContract()', () => {
    it('should create plunge state with partner as current player', () => {
      const state = StateBuilder.plungeContract(0, 4).build();

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
      expect(state.currentPlayer).toBe(2); // Partner
      expect(state.currentBid.type).toBe('plunge');
      expect(state.currentBid.value).toBe(4);
    });

    it('should throw on invalid plunge value', () => {
      expect(() => StateBuilder.plungeContract(0, 3).build()).toThrow('Invalid plunge value');
      expect(() => StateBuilder.plungeContract(0, 8).build()).toThrow('Invalid plunge value');
    });
  });

  describe('sevensContract()', () => {
    it('should create sevens state', () => {
      const state = StateBuilder.sevensContract(0).build();

      expect(state.phase).toBe('trump_selection');
      expect(state.winningBidder).toBe(0);
      expect(state.currentBid.type).toBe(BID_TYPES.MARKS);
      expect(state.currentBid.value).toBe(1);
    });
  });
});

describe('StateBuilder - Chainable Modifiers', () => {
  describe('withDealer()', () => {
    it('should set dealer', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withDealer(2)
        .build();

      expect(state.dealer).toBe(2);
    });

    it('should throw on invalid dealer', () => {
      expect(() => StateBuilder.inBiddingPhase().withDealer(-1).build()).toThrow('Invalid dealer');
      expect(() => StateBuilder.inBiddingPhase().withDealer(4).build()).toThrow('Invalid dealer');
    });
  });

  describe('withCurrentPlayer()', () => {
    it('should set current player', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(3)
        .build();

      expect(state.currentPlayer).toBe(3);
    });

    it('should throw on invalid player', () => {
      expect(() => StateBuilder.inBiddingPhase().withCurrentPlayer(-1).build()).toThrow('Invalid player');
      expect(() => StateBuilder.inBiddingPhase().withCurrentPlayer(4).build()).toThrow('Invalid player');
    });
  });

  describe('withBids()', () => {
    it('should set bids', () => {
      const bids = [
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.PASS, player: 1 }
      ];

      const state = StateBuilder
        .inBiddingPhase()
        .withBids(bids)
        .build();

      expect(state.bids).toHaveLength(2);
      expect(state.currentBid.type).toBe(BID_TYPES.POINTS);
      expect(state.currentBid.value).toBe(30);
    });

    it('should find highest bid', () => {
      const bids = [
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.POINTS, value: 34, player: 1 },
        { type: BID_TYPES.PASS, player: 2 }
      ];

      const state = StateBuilder
        .inBiddingPhase()
        .withBids(bids)
        .build();

      expect(state.currentBid.value).toBe(34);
    });
  });

  describe('withTrump()', () => {
    it('should set trump', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .withTrump({ type: 'suit', suit: BLANKS })
        .build();

      expect(state.trump.type).toBe('suit');
      expect(state.trump.suit).toBe(BLANKS);
    });
  });

  describe('withWinningBid()', () => {
    it('should set winning bid and bidder', () => {
      const bid = { type: BID_TYPES.POINTS, value: 32, player: 1 };
      const state = StateBuilder
        .inBiddingPhase()
        .withWinningBid(1, bid)
        .build();

      expect(state.winningBidder).toBe(1);
      expect(state.currentBid).toEqual(bid);
      expect(state.bids).toContainEqual(bid);
    });
  });

  describe('withPlayerHand()', () => {
    it('should set player hand from strings', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withPlayerHand(0, ['6-6', '6-5', '5-5'])
        .build();

      expect(state.players[0]!.hand).toHaveLength(3);
      expect(state.players[0]!.hand[0]!.id).toBe('6-6');
    });

    it('should set player hand from Domino objects', () => {
      const dominoes = [
        DominoBuilder.from('6-6'),
        DominoBuilder.from('6-5')
      ];

      const state = StateBuilder
        .inBiddingPhase()
        .withPlayerHand(0, dominoes)
        .build();

      expect(state.players[0]!.hand).toHaveLength(2);
    });

    it('should throw on invalid player index', () => {
      expect(() => StateBuilder.inBiddingPhase().withPlayerHand(-1, []).build()).toThrow('Invalid player index');
      expect(() => StateBuilder.inBiddingPhase().withPlayerHand(4, []).build()).toThrow('Invalid player index');
    });
  });

  describe('withHands()', () => {
    it('should set all hands at once', () => {
      const hands = [
        ['6-6'],
        ['5-5'],
        ['4-4'],
        ['3-3']
      ];

      const state = StateBuilder
        .inBiddingPhase()
        .withHands(hands)
        .build();

      expect(state.players[0]!.hand[0]!.id).toBe('6-6');
      expect(state.players[1]!.hand[0]!.id).toBe('5-5');
      expect(state.players[2]!.hand[0]!.id).toBe('4-4');
      expect(state.players[3]!.hand[0]!.id).toBe('3-3');
    });

    it('should throw on wrong number of hands', () => {
      expect(() => StateBuilder.inBiddingPhase().withHands([[], []]).build()).toThrow('Must provide exactly 4 hands');
    });
  });

  describe('withCurrentTrick()', () => {
    it('should set current trick from strings', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .withCurrentTrick([
          { player: 0, domino: '6-5' },
          { player: 1, domino: '5-4' }
        ])
        .build();

      expect(state.currentTrick).toHaveLength(2);
      expect(state.currentTrick[0]!.domino.id).toBe('6-5');
      expect(state.currentSuit).toBeDefined();
    });

    it('should set current trick from Domino objects', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .withCurrentTrick([
          { player: 0, domino: DominoBuilder.from('6-5') }
        ])
        .build();

      expect(state.currentTrick).toHaveLength(1);
    });
  });

  describe('withTricks()', () => {
    it('should set completed tricks', () => {
      const tricks = [
        {
          plays: [{ player: 0, domino: DominoBuilder.from('6-5') }],
          winner: 0,
          points: 5
        }
      ];

      const state = StateBuilder
        .inPlayingPhase()
        .withTricks(tricks)
        .build();

      expect(state.tricks).toHaveLength(1);
    });
  });

  describe('addTrick()', () => {
    it('should add a completed trick', () => {
      const plays = [
        { player: 0, domino: DominoBuilder.from('6-5') },
        { player: 1, domino: DominoBuilder.from('5-4') },
        { player: 2, domino: DominoBuilder.from('4-3') },
        { player: 3, domino: DominoBuilder.from('3-2') }
      ];

      const state = StateBuilder
        .inPlayingPhase()
        .addTrick(plays, 0, 10)
        .build();

      expect(state.tricks).toHaveLength(1);
      expect(state.tricks[0]!.winner).toBe(0);
      expect(state.tricks[0]!.points).toBe(10);
      expect(state.teamScores[0]).toBe(10); // Team 0 scored
    });
  });

  describe('withTeamScores()', () => {
    it('should set team scores', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .withTeamScores(25, 17)
        .build();

      expect(state.teamScores).toEqual([25, 17]);
    });
  });

  describe('withTeamMarks()', () => {
    it('should set team marks', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .withTeamMarks(3, 2)
        .build();

      expect(state.teamMarks).toEqual([3, 2]);
    });
  });

  describe('withSeed()', () => {
    it('should set shuffle seed', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withSeed(99999)
        .build();

      expect(state.shuffleSeed).toBe(99999);
      expect(state.initialConfig.shuffleSeed).toBe(99999);
    });
  });

  describe('withConfig()', () => {
    it('should merge config', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withConfig({ playerTypes: ['human', 'human', 'ai', 'ai'] })
        .build();

      expect(state.playerTypes).toEqual(['human', 'human', 'ai', 'ai']);
      expect(state.initialConfig.playerTypes).toEqual(['human', 'human', 'ai', 'ai']);
    });
  });

  describe('with()', () => {
    it('should merge arbitrary state overrides', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .with({ phase: 'scoring' })
        .build();

      expect(state.phase).toBe('scoring');
    });
  });
});

describe('StateBuilder - Integration Tests', () => {
  it('should support complex chaining', () => {
    const state = StateBuilder
      .inPlayingPhase({ type: 'suit', suit: ACES })
      .withDealer(2)
      .withCurrentPlayer(0)
      .withPlayerHand(0, ['6-6', '6-5', '5-5'])
      .withTeamScores(15, 8)
      .withTeamMarks(2, 1)
      .build();

    expect(state.phase).toBe('playing');
    expect(state.dealer).toBe(2);
    expect(state.currentPlayer).toBe(0);
    expect(state.players[0]!.hand).toHaveLength(3);
    expect(state.teamScores).toEqual([15, 8]);
    expect(state.teamMarks).toEqual([2, 1]);
  });

  it('should support cloning and modifying', () => {
    const builder1 = StateBuilder.inBiddingPhase().withDealer(0);
    const builder2 = builder1.clone().withDealer(2);

    const state1 = builder1.build();
    const state2 = builder2.build();

    expect(state1.dealer).toBe(0);
    expect(state2.dealer).toBe(2);
  });

  it('should produce immutable states', () => {
    const builder = StateBuilder.inBiddingPhase();
    const state1 = builder.build();
    const state2 = builder.build();

    // Modify state1
    state1.dealer = 3;

    // state2 should be unaffected
    expect(state2.dealer).not.toBe(3);
  });

  it('should handle special contract preset modifications', () => {
    const state = StateBuilder
      .splashContract(0, 2)
      .withPlayerHand(0, HandBuilder.withDoubles(4))
      .build();

    expect(state.currentBid.type).toBe('splash');
    expect(state.players[0]!.hand).toHaveLength(7);

    const doubles = state.players[0]!.hand.filter(d => d.high === d.low);
    expect(doubles).toHaveLength(4);
  });

  it('should handle trick progression correctly', () => {
    const state = StateBuilder
      .inPlayingPhase()
      .withCurrentTrick([
        { player: 0, domino: '6-5' },
        { player: 1, domino: '5-4' }
      ])
      .build();

    expect(state.currentTrick).toHaveLength(2);
    expect(state.currentSuit).not.toBe(-1); // Should have a suit
  });
});

describe('StateBuilder - Edge Cases', () => {
  it('should handle empty player hand', () => {
    const state = StateBuilder
      .inBiddingPhase()
      .withPlayerHand(0, [])
      .build();

    expect(state.players[0]!.hand).toHaveLength(0);
  });

  it('should handle full hand replacement', () => {
    const originalState = StateBuilder.inBiddingPhase().build();
    const originalHand = [...originalState.players[0]!.hand];

    const newState = StateBuilder
      .inBiddingPhase()
      .withPlayerHand(0, ['6-6', '5-5', '4-4', '3-3', '2-2', '1-1', '0-0'])
      .build();

    expect(newState.players[0]!.hand).toHaveLength(7);
    expect(newState.players[0]!.hand.map(d => d.id)).not.toEqual(
      originalHand.map(d => d.id)
    );
  });

  it('should handle 0 tricks played', () => {
    const state = StateBuilder.withTricksPlayed(0).build();

    expect(state.tricks).toHaveLength(0);
    expect(state.phase).toBe('playing');
  });

  it('should handle all 7 tricks played', () => {
    const state = StateBuilder.withTricksPlayed(7).build();

    expect(state.tricks).toHaveLength(7);
  });
});
