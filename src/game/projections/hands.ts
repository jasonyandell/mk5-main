import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { Domino } from '../types';

export class HandsProjection extends Projection<Record<number, Domino[]>> {
  protected getInitialValue(): Record<number, Domino[]> {
    return {
      0: [],
      1: [],
      2: [],
      3: []
    };
  }

  protected project(current: Record<number, Domino[]>, event: GameEvent): Record<number, Domino[]> {
    switch (event.type) {
      case 'HANDS_DEALT':
        return { ...event.hands };
      
      case 'DOMINO_PLAYED': {
        const newHands = { ...current };
        const playerHand = [...newHands[event.player]];
        const index = playerHand.findIndex(d => 
          d.high === event.domino.high && d.low === event.domino.low
        );
        if (index !== -1) {
          playerHand.splice(index, 1);
          newHands[event.player] = playerHand;
        }
        return newHands;
      }
      
      default:
        return current;
    }
  }
}