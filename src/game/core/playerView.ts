import type { GameState, PlayerView, PublicPlayer, StateTransition } from '../types';
import { getNextStates } from './gameEngine';

/**
 * Creates a player-specific view of the game state with privacy guarantees.
 * Other players' hands are never included in the type structure.
 */
export function getPlayerView(state: GameState, playerId: number): PlayerView {
  // Get all possible transitions
  const allTransitions = getNextStates(state);
  
  // Filter transitions to only those this player can take
  const validTransitions = allTransitions.filter(transition => {
    const action = transition.action;
    
    // Actions without a player field are available to everyone
    if (!('player' in action)) {
      return true;
    }
    
    // Actions with a player field are only for that player
    return action.player === playerId;
  });
  
  // Create public player info (no hands visible!)
  const publicPlayers: PublicPlayer[] = state.players.map(p => ({
    id: p.id,
    name: p.name,
    teamId: p.teamId,
    marks: p.marks,
    handCount: p.hand.length
  }));
  
  // Get self info (only self has hand visible)
  const selfPlayer = state.players[playerId];
  if (!selfPlayer) {
    throw new Error(`Invalid player ID: ${playerId}`);
  }
  
  const self = {
    id: selfPlayer.id,
    hand: [...selfPlayer.hand]  // Clone to prevent mutations
  };
  
  return {
    playerId,
    phase: state.phase,
    self,
    players: publicPlayers,
    validTransitions,
    consensus: {
      completeTrick: new Set(state.consensus.completeTrick),
      scoreHand: new Set(state.consensus.scoreHand)
    },
    currentTrick: [...state.currentTrick],
    tricks: state.tricks.map(t => ({
      ...t,
      plays: [...t.plays]
    })),
    teamScores: [...state.teamScores] as [number, number],
    teamMarks: [...state.teamMarks] as [number, number],
    trump: state.trump
  };
}

/**
 * Checks if a player is human (for AI response handling)
 */
export function isHumanPlayer(playerId: number): boolean {
  // For now, assume player 0 is human and others are AI
  // This can be configured later based on game settings
  return playerId === 0;
}

/**
 * Simple AI action selection - picks the first available action
 * Can be enhanced with smarter logic later
 */
export function chooseAIAction(transitions: StateTransition[]): StateTransition | null {
  if (transitions.length === 0) {
    return null;
  }
  
  // For consensus actions, always agree immediately
  const consensusAction = transitions.find(t => 
    t.action.type === 'agree-complete-trick' || 
    t.action.type === 'agree-score-hand'
  );
  
  if (consensusAction) {
    return consensusAction;
  }
  
  // Otherwise pick the first available action
  // This can be made smarter with game logic
  return transitions[0];
}