import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { Bid } from '../types';

export interface BidsState {
  bids: Bid[];
  highestBid: Bid | null;
  highestBidder: number;
  passes: Set<number>;
}

export class BidsProjection extends Projection<BidsState> {
  protected getInitialValue(): BidsState {
    return {
      bids: [],
      highestBid: null,
      highestBidder: -1,
      passes: new Set()
    };
  }

  protected project(current: BidsState, event: GameEvent): BidsState {
    switch (event.type) {
      case 'BID_PLACED': {
        const newBids = [...current.bids, event.bid];
        return {
          bids: newBids,
          highestBid: event.bid,
          highestBidder: event.player,
          passes: new Set()
        };
      }
      
      case 'PLAYER_PASSED': {
        const newPasses = new Set(current.passes);
        newPasses.add(event.player);
        return {
          ...current,
          passes: newPasses
        };
      }
      
      case 'BIDDING_COMPLETED': {
        return {
          ...current,
          highestBid: event.bid,
          highestBidder: event.winner
        };
      }
      
      case 'GAME_STARTED':
      case 'REDEAL_INITIATED':
      case 'ALL_PLAYERS_PASSED': {
        return this.getInitialValue();
      }
      
      default:
        return current;
    }
  }
}