import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { ConsensusAction } from '../types';

export type ConsensusState = Map<ConsensusAction, Set<number>>;

export class ConsensusProjection extends Projection<ConsensusState> {
  protected getInitialValue(): ConsensusState {
    return new Map();
  }

  protected project(current: ConsensusState, event: GameEvent): ConsensusState {
    switch (event.type) {
      case 'CONSENSUS_REQUESTED': {
        const newState = new Map(current);
        newState.set(event.action, new Set());
        return newState;
      }
      
      case 'PLAYER_AGREED': {
        const newState = new Map(current);
        const players = newState.get(event.action) || new Set();
        const newPlayers = new Set(players);
        newPlayers.add(event.player);
        newState.set(event.action, newPlayers);
        return newState;
      }
      
      case 'CONSENSUS_REACHED': {
        const newState = new Map(current);
        newState.delete(event.action);
        return newState;
      }
      
      case 'TRICK_COMPLETED': {
        const newState = new Map(current);
        newState.delete('complete-trick');
        return newState;
      }
      
      case 'HAND_SCORED': {
        const newState = new Map(current);
        newState.delete('score-hand');
        return newState;
      }
      
      case 'GAME_STARTED':
      case 'REDEAL_INITIATED': {
        return new Map();
      }
      
      default:
        return current;
    }
  }
}