import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { GamePhase } from '../types';

export class GamePhaseProjection extends Projection<GamePhase> {
  protected getInitialValue(): GamePhase {
    return 'setup';
  }

  protected project(current: GamePhase, event: GameEvent): GamePhase {
    switch (event.type) {
      case 'GAME_STARTED':
      case 'ALL_PLAYERS_PASSED':
      case 'REDEAL_INITIATED':
        return 'bidding';
      
      case 'BIDDING_COMPLETED':
        return 'trump_selection';
      
      case 'TRUMP_SELECTED':
        return 'playing';
      
      case 'HAND_READY_FOR_SCORING':
        return 'scoring';
      
      case 'GAME_ENDED':
        return 'game_end';
      
      default:
        return current;
    }
  }
}