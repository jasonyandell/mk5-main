import { writable, derived } from 'svelte/store';
import type { GameState, StateTransition } from '../../game/types';
import { createInitialState, getNextStates } from '../../game';

// Core game state store
export const gameState = writable<GameState>(createInitialState());

// Available actions store
export const availableActions = derived(
  gameState,
  ($gameState) => getNextStates($gameState)
);

// Current player store
export const currentPlayer = derived(
  gameState,
  ($gameState) => $gameState.players[$gameState.currentPlayer]
);

// Game phase store
export const gamePhase = derived(
  gameState,
  ($gameState) => $gameState.phase
);

// Bidding information
export const biddingInfo = derived(
  gameState,
  ($gameState) => ({
    currentBid: $gameState.currentBid,
    bids: $gameState.bids,
    winningBidder: $gameState.winningBidder
  })
);

// Team scores and marks
export const teamInfo = derived(
  gameState,
  ($gameState) => ({
    scores: $gameState.teamScores,
    marks: $gameState.teamMarks,
    target: $gameState.gameTarget
  })
);

// Current trick information
export const trickInfo = derived(
  gameState,
  ($gameState) => ({
    currentTrick: $gameState.currentTrick,
    completedTricks: $gameState.tricks,
    trump: $gameState.trump
  })
);

// Game actions
export const gameActions = {
  executeAction: (transition: StateTransition) => {
    gameState.set(transition.newState);
  },
  
  resetGame: () => {
    gameState.set(createInitialState());
  },
  
  loadState: (state: GameState) => {
    gameState.set(state);
  }
};