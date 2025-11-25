import type { GameAction } from '../game/types';
import type { GameView } from './types';

// Client → Server
export type ClientMessage =
  | { type: 'EXECUTE_ACTION'; action: GameAction }
  | { type: 'JOIN'; playerIndex: number; name: string }
  | { type: 'SET_CONTROL'; playerIndex: number; controlType: 'human' | 'ai' };

// Server → Client
export type ServerMessage =
  | { type: 'STATE_UPDATE'; view: GameView }
  | { type: 'ERROR'; error: string };
