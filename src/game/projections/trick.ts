import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { Play } from '../types';

export class TrickProjection extends Projection<Play[]> {
  protected getInitialValue(): Play[] {
    return [];
  }

  protected project(current: Play[], event: GameEvent): Play[] {
    switch (event.type) {
      case 'DOMINO_PLAYED':
        return [...current, { player: event.player, domino: event.domino }];
      
      case 'TRICK_COMPLETED':
      case 'TRUMP_SELECTED':
      case 'HAND_SCORED':
      case 'GAME_STARTED':
        return [];
      
      default:
        return current;
    }
  }
}