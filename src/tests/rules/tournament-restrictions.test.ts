import { describe, it, expect, beforeEach } from 'vitest';
import { Game } from '../../game/Game';
import { Player } from '../../game/Player';
import { BidType } from '../../game/Bid';

describe('Tournament Restrictions', () => {
  let game: Game;
  let players: Player[];

  beforeEach(() => {
    game = new Game();
    players = [
      new Player('North'),
      new Player('East'),
      new Player('South'),
      new Player('West')
    ];
    game.setPlayers(players);
    game.startNewHand();
  });

  describe('Given a tournament game is being played', () => {
    beforeEach(() => {
      game.setTournamentMode(true);
    });

    describe('When players are bidding', () => {
      it('Then Nel-O is not allowed', () => {
        const currentBidder = game.getCurrentBidder();
        
        expect(() => {
          game.makeBid(currentBidder, 1, BidType.NELO);
        }).toThrow('Nel-O is not allowed in tournament play');
      });

      it('Then Plunge is not allowed unless holding 4+ doubles', () => {
        const currentBidder = game.getCurrentBidder();
        
        // Mock hand with only 2 doubles
        const handWith2Doubles = [
          { pips1: 0, pips2: 0 }, // double blank
          { pips1: 1, pips2: 1 }, // double one
          { pips1: 2, pips2: 3 },
          { pips1: 4, pips2: 5 },
          { pips1: 3, pips2: 6 },
          { pips1: 0, pips2: 4 },
          { pips1: 1, pips2: 5 }
        ];
        currentBidder.setHand(handWith2Doubles);
        
        expect(() => {
          game.makeBid(currentBidder, 4, BidType.PLUNGE);
        }).toThrow('Plunge requires holding 4 or more doubles');

        // Now test with 4 doubles - should be allowed
        const handWith4Doubles = [
          { pips1: 0, pips2: 0 }, // double blank
          { pips1: 1, pips2: 1 }, // double one
          { pips1: 2, pips2: 2 }, // double two
          { pips1: 3, pips2: 3 }, // double three
          { pips1: 4, pips2: 5 },
          { pips1: 0, pips2: 6 },
          { pips1: 1, pips2: 5 }
        ];
        currentBidder.setHand(handWith4Doubles);
        
        expect(() => {
          game.makeBid(currentBidder, 4, BidType.PLUNGE);
        }).not.toThrow();
      });

      it('Then Splash is not allowed', () => {
        const currentBidder = game.getCurrentBidder();
        
        expect(() => {
          game.makeBid(currentBidder, 2, BidType.SPLASH);
        }).toThrow('Splash is not allowed in tournament play');
      });

      it('Then Sevens is not allowed', () => {
        const currentBidder = game.getCurrentBidder();
        
        expect(() => {
          game.makeBid(currentBidder, 1, BidType.SEVENS);
        }).toThrow('Sevens is not allowed in tournament play');
      });
    });
  });
});