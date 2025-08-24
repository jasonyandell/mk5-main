import { Projection } from './base';
import type { GameEvent } from '../events/types';

export interface TeamState {
  trickPoints: [number, number];
  marks: [number, number];
}

export class TeamsProjection extends Projection<TeamState> {
  protected getInitialValue(): TeamState {
    return {
      trickPoints: [0, 0],
      marks: [0, 0]
    };
  }

  protected project(current: TeamState, event: GameEvent): TeamState {
    switch (event.type) {
      case 'TRICK_COMPLETED': {
        const newState = { ...current };
        const team = event.winner % 2 as 0 | 1;
        newState.trickPoints = [...current.trickPoints] as [number, number];
        newState.trickPoints[team] += event.points;
        return newState;
      }
      
      case 'MARKS_AWARDED': {
        const newState = { ...current };
        newState.marks = [...current.marks] as [number, number];
        newState.marks[event.team] += event.marks;
        return newState;
      }
      
      case 'TRUMP_SELECTED':
      case 'HAND_SCORED': {
        return {
          ...current,
          trickPoints: [0, 0]
        };
      }
      
      case 'GAME_STARTED': {
        return this.getInitialValue();
      }
      
      default:
        return current;
    }
  }
}