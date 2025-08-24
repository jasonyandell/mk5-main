import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { TrumpSelection } from '../types';

export class TrumpProjection extends Projection<TrumpSelection> {
  protected getInitialValue(): TrumpSelection {
    return { type: 'none' };
  }

  protected project(current: TrumpSelection, event: GameEvent): TrumpSelection {
    switch (event.type) {
      case 'TRUMP_SELECTED':
        return { ...event.trump };
      
      case 'GAME_STARTED':
      case 'REDEAL_INITIATED':
      case 'HAND_SCORED':
        return { type: 'none' };
      
      default:
        return current;
    }
  }
}