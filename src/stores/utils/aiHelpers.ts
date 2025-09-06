import type { GameState, StateTransition } from '../../game/types';
import { getNextStates } from '../../game';

// Helper function to execute AI moves immediately (for tests/replay)
// Returns the new state and the AI actions taken
export function executeAllAIImmediate(state: GameState): { state: GameState; aiActions: StateTransition[] } {
  let currentState = state;
  const aiActions: StateTransition[] = [];
  
  // Keep executing AI until no AI player needs to move
  while (currentState.playerTypes[currentState.currentPlayer] === 'ai') {
    const availableTransitions = getNextStates(currentState);
    
    // Find the best AI action (using existing AI logic)
    let aiTransition: StateTransition | undefined;
    
    // Try to find a play action first
    aiTransition = availableTransitions.find(t => 
      t.action.type === 'play' && 
      'player' in t.action && 
      t.action.player === currentState.currentPlayer
    );
    
    // If no play action, try bid/pass
    if (!aiTransition) {
      aiTransition = availableTransitions.find(t => 
        (t.action.type === 'bid' || t.action.type === 'pass') && 
        'player' in t.action && 
        t.action.player === currentState.currentPlayer
      );
    }
    
    // If no player action, try trump selection
    if (!aiTransition) {
      aiTransition = availableTransitions.find(t => 
        t.action.type === 'select-trump' && 
        currentState.winningBidder === currentState.currentPlayer
      );
    }
    
    // If no action found, break
    if (!aiTransition) {
      break;
    }
    
    currentState = aiTransition.newState;
    aiActions.push(aiTransition);
  }
  
  return { state: currentState, aiActions };
}

// Inject compact consensus agreements until the boundary action is available.
// Used when replaying URLs/histories that omit agree-* actions.
export function injectConsensusIfNeeded(
  state: GameState,
  boundaryActionId: 'complete-trick' | 'score-hand',
  validActions: StateTransition[]
): GameState {
  let currentState = state;
  const targetAgreeType = boundaryActionId === 'complete-trick'
    ? 'agree-complete-trick'
    : 'agree-score-hand';

  // Keep applying agree-* transitions until the boundary action is available
  // or there are no more agree-* actions to apply.
  for (;;) {
    const transitions = getNextStates(currentState);

    // Stop if the boundary action is now available
    if (transitions.some(t => t.id === boundaryActionId)) {
      break;
    }

    // Find an agree-* transition to apply
    const agree = transitions.find(t => t.action.type === targetAgreeType);
    if (!agree) break;

    validActions.push(agree);
    currentState = agree.newState;
  }

  return currentState;
}
