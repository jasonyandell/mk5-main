import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { Trick } from '../types';

export class CompletedTricksProjection extends Projection<Trick[]> {
  protected getInitialValue(): Trick[] {
    return [];
  }

  protected project(current: Trick[], event: GameEvent): Trick[] {
    switch (event.type) {
      case 'TRICK_COMPLETED': {
        const newTrick: Trick = {
          plays: [],
          winner: event.winner,
          points: event.points,
          ledSuit: undefined
        };
        return [...current, newTrick];
      }
      
      case 'TRUMP_SELECTED':
      case 'HAND_SCORED':
      case 'GAME_STARTED':
        return [];
      
      default:
        return current;
    }
  }
}