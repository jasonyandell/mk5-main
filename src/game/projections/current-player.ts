import { Projection } from './base';
import type { GameEvent } from '../events/types';

export class CurrentPlayerProjection extends Projection<number> {
  private dealer: number = 0;
  private passCount: number = 0;
  private highestBidder: number = -1;

  protected getInitialValue(): number {
    return 0;
  }

  protected project(current: number, event: GameEvent): number {
    switch (event.type) {
      case 'GAME_STARTED':
        this.dealer = event.dealer;
        this.passCount = 0;
        this.highestBidder = -1;
        return (event.dealer + 1) % 4;
      
      case 'REDEAL_INITIATED':
        this.passCount = 0;
        this.highestBidder = -1;
        return (this.dealer + 1) % 4;
      
      case 'BID_PLACED':
        this.highestBidder = event.player;
        this.passCount = 0;
        return (event.player + 1) % 4;
      
      case 'PLAYER_PASSED':
        this.passCount++;
        if (this.passCount === 3 && this.highestBidder !== -1) {
          return this.highestBidder;
        }
        return (event.player + 1) % 4;
      
      case 'BIDDING_COMPLETED':
        return event.winner;
      
      case 'TRUMP_SELECTED':
        return (this.dealer + 1) % 4;
      
      case 'DOMINO_PLAYED':
        return (event.player + 1) % 4;
      
      case 'TRICK_COMPLETED':
        return event.winner;
      
      default:
        return current;
    }
  }
}