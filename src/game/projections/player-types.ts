import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { PlayerType } from '../types';

export class PlayerTypesProjection extends Projection<PlayerType[]> {
  protected getInitialValue(): PlayerType[] {
    return ['human', 'human', 'human', 'human'];
  }

  protected project(current: PlayerType[], event: GameEvent): PlayerType[] {
    switch (event.type) {
      case 'GAME_STARTED':
        return [...event.playerTypes];
      
      default:
        return current;
    }
  }
}