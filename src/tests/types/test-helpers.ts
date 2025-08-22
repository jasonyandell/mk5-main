import type { GameState } from '../../game/types';

export interface PartialGameState {
  phase?: GameState['phase'];
  currentPlayer?: GameState['currentPlayer'];
  trump?: GameState['trump'];
  currentBid?: GameState['currentBid'];
  winningBidder?: GameState['winningBidder'];
  playerTypes?: GameState['playerTypes'];
  dealer?: GameState['dealer'];
  tournamentMode?: GameState['tournamentMode'];
  shuffleSeed?: GameState['shuffleSeed'];
  gameTarget?: GameState['gameTarget'];
}

export function isPartialGameState(obj: unknown): obj is PartialGameState {
  return obj !== null && typeof obj === 'object';
}