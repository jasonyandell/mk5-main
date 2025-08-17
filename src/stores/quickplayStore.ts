/* global requestAnimationFrame, cancelAnimationFrame */
import { writable, get } from 'svelte/store';
import { gameState, availableActions, gameActions, actionHistory } from './gameStore';
import type { GameState, StateTransition, Domino, Player } from '../game/types';
import { BID_TYPES } from '../game/constants';
import { getDominoPoints, countDoubles } from '../game/core/dominoes';
import { getStrongestSuits } from '../game/core/suit-analysis';

export interface QuickplayState {
  enabled: boolean;
  speed: 'instant' | 'fast' | 'normal' | 'slow';
  aiPlayers: Set<number>;  // 0-3
  isPaused: boolean;
}

export interface QuickplayError {
  message: string;
  state: GameState;
  availableActions: string[];
  timestamp: string;
}

// Speed delays in milliseconds
// const SPEED_DELAYS = {
//   instant: 0,
//   fast: 200,
//   normal: 500,
//   slow: 1000
// };

// Create the quickplay store
export const quickplayState = writable<QuickplayState>({
  enabled: false,
  speed: 'instant',
  aiPlayers: new Set([0, 1, 2, 3]), // All players AI by default
  isPaused: false
});

// Store for quickplay errors
export const quickplayErrorStore = writable<QuickplayError | null>(null);

// AI Decision Logic
function makeAIDecision(state: GameState, actions: StateTransition[]): StateTransition | null {
  if (actions.length === 0) return null;
  
  const currentPlayer = state.players[state.currentPlayer];
  if (!currentPlayer) return actions[0] || null;
  
  switch (state.phase) {
    case 'bidding':
      return makeAIBidDecision(state, currentPlayer, actions);
    case 'trump_selection':
      return makeAITrumpDecision(state, currentPlayer, actions);
    case 'playing':
      return makeAIPlayDecision(state, currentPlayer, actions);
    default:
      // For other phases (scoring, etc), just take the first available action
      return actions[0] || null;
  }
}

// AI Bidding Logic
function makeAIBidDecision(state: GameState, player: Player, actions: StateTransition[]): StateTransition {
  // Ensure we have actions to work with
  if (actions.length === 0) {
    throw new Error('No actions available for AI bidding decision');
  }
  
  // Calculate hand strength
  const handStrength = calculateHandStrength(player.hand);
  
  // Find pass action
  const passAction = actions.find(a => a.id === 'pass');
  
  // Get current highest bid value
  const currentBidValue = getCurrentBidValue(state);
  
  // Weak hand - pass if can't bid 30
  if (handStrength < 8 && currentBidValue >= 30) {
    return passAction || actions[0]!;
  }
  
  // Calculate bid based on hand strength
  let targetBid = 30;
  if (handStrength >= 12) targetBid = 32;
  if (handStrength >= 15) targetBid = 35;
  
  // Find best available bid
  const bidActions = actions.filter(a => a.id.startsWith('bid-') && !a.id.includes('marks'));
  const validBids = bidActions.filter(a => {
    const parts = a.id.split('-');
    if (parts.length < 2 || !parts[1]) return false;
    const bidValue = parseInt(parts[1]);
    return !isNaN(bidValue) && bidValue >= targetBid && bidValue > currentBidValue;
  });
  
  if (validBids.length > 0) {
    // Sort by bid value and take the lowest valid bid
    validBids.sort((a, b) => {
      const aParts = a.id.split('-');
      const bParts = b.id.split('-');
      const aValue = aParts.length >= 2 && aParts[1] ? parseInt(aParts[1]) : 0;
      const bValue = bParts.length >= 2 && bParts[1] ? parseInt(bParts[1]) : 0;
      return aValue - bValue;
    });
    return validBids[0]!;
  }
  
  // Can't make desired bid - pass
  return passAction || actions[0]!;
}

// Calculate hand strength for bidding
function calculateHandStrength(hand: Domino[]): number {
  let strength = 0;
  
  // Count points
  hand.forEach(domino => {
    strength += getDominoPoints(domino) / 5; // 0-2 points per domino
  });
  
  // Count doubles (adds strength)
  const doubleCount = countDoubles(hand);
  strength += doubleCount * 2;
  
  // High dominoes add strength
  hand.forEach(domino => {
    if (domino.high + domino.low >= 10) {
      strength += 1;
    }
  });
  
  return strength;
}

// Get current highest bid value
function getCurrentBidValue(state: GameState): number {
  const nonPassBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
  if (nonPassBids.length === 0) return 0;
  
  return nonPassBids.reduce((max, bid) => {
    if (bid.type === BID_TYPES.POINTS) {
      return Math.max(max, bid.value || 0);
    }
    if (bid.type === BID_TYPES.MARKS) {
      return Math.max(max, (bid.value || 0) * 42);
    }
    return max;
  }, 0);
}

// AI Trump Selection Logic
function makeAITrumpDecision(_state: GameState, player: Player, actions: StateTransition[]): StateTransition {
  if (actions.length === 0) {
    throw new Error('No actions available for AI trump decision');
  }
  
  if (!player.suitAnalysis) {
    // Fallback - pick first available trump
    return actions[0]!;
  }
  
  // Get strongest suits
  const strongestSuits = getStrongestSuits(player.suitAnalysis);
  
  // Check for 3+ doubles - consider doubles trump
  const doubleCount = countDoubles(player.hand);
  if (doubleCount >= 3) {
    const doublesAction = actions.find(a => a.id === 'trump-doubles');
    if (doublesAction) return doublesAction;
  }
  
  // Pick strongest suit as trump
  for (const suit of strongestSuits) {
    const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
    const trumpAction = actions.find(a => a.id === `trump-${suitNames[suit]}`);
    if (trumpAction) return trumpAction;
  }
  
  // Fallback
  return actions[0]!;
}

// AI Playing Logic
function makeAIPlayDecision(state: GameState, player: Player, actions: StateTransition[]): StateTransition {
  if (actions.length === 0) {
    throw new Error('No actions available for AI play decision');
  }
  
  // If completing a trick, do it
  const completeTrickAction = actions.find(a => a.id === 'complete-trick');
  if (completeTrickAction) return completeTrickAction;
  
  // Extract dominoes from play actions
  const playActions = actions.filter(a => a.id.startsWith('play-'));
  if (playActions.length === 0) return actions[0]!;
  
  // Leading a trick
  if (state.currentTrick.length === 0) {
    // Lead with high-point dominoes
    const scoredActions = playActions.map(action => {
      const dominoId = action.id.split('-')[1];
      const domino = player.hand.find(d => d.id.toString() === dominoId);
      const points = domino ? getDominoPoints(domino) : 0;
      return { action, points };
    });
    
    scoredActions.sort((a, b) => b.points - a.points);
    return scoredActions[0]?.action || actions[0]!;
  }
  
  // Following in a trick
  const myTeam = player.teamId;
  const trickLeader = state.currentTrick[0]?.player;
  if (trickLeader === undefined) return actions[0]!;
  const leaderPlayer = state.players[trickLeader];
  if (!leaderPlayer) return actions[0]!;
  const leaderTeam = leaderPlayer.teamId;
  const partnerLed = myTeam === leaderTeam;
  
  // Simple strategy: play high if opponent led, low if partner led
  const scoredActions = playActions.map(action => {
    const dominoId = action.id.split('-')[1];
    const domino = player.hand.find(d => d.id.toString() === dominoId);
    const value = domino ? domino.high + domino.low : 0;
    return { action, value };
  });
  
  if (partnerLed) {
    // Partner led - play low
    scoredActions.sort((a, b) => a.value - b.value);
  } else {
    // Opponent led - play high to try to win
    scoredActions.sort((a, b) => b.value - a.value);
  }
  
  return scoredActions[0]?.action || actions[0]!;
}

// Subscribe to game state changes
let unsubscribe: (() => void) | null = null;
let animationFrameId: number | null = null;

// Process AI moves continuously when running
function processAIMoves() {
  const $gameState = get(gameState);
  const $quickplayState = get(quickplayState);
  const $availableActions = get(availableActions);
    
  // Check if we should continue
  if (!$quickplayState.enabled || $quickplayState.isPaused) {
    animationFrameId = null;
    return;
  }
  
  // Check if game is complete - start new game automatically
  if ($gameState.phase === 'game_end' && $gameState.isComplete) {
    // Debug logging
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      console.log('[Quickplay] Game complete, resetting...', {
        phase: $gameState.phase,
        isComplete: $gameState.isComplete,
        marks: $gameState.teamMarks,
        actionCount: get(actionHistory).length,
        handsEmpty: $gameState.players.every(p => p.hand.length === 0)
      });
    }
    
    // Reset to new game
    gameActions.resetGame();
    
    // Continue processing after a brief pause to ensure reset completes
    animationFrameId = requestAnimationFrame(() => {
      animationFrameId = requestAnimationFrame(processAIMoves);
    });
    return;
  }
  
  // Check if current player is AI
  if (!$quickplayState.aiPlayers.has($gameState.currentPlayer)) {
    // Schedule next check
    animationFrameId = requestAnimationFrame(processAIMoves);
    return;
  }
  
  // No actions available
  if ($availableActions.length === 0) {
    // Schedule next check
    animationFrameId = requestAnimationFrame(processAIMoves);
    return;
  }
  
  // Make AI decision immediately for instant speed
  if ($quickplayState.speed === 'instant') {
    try {
      const decision = makeAIDecision($gameState, $availableActions);
      if (decision) {
        gameActions.executeAction(decision);
      }
    } catch (error) {
      // Log error details for debugging
      console.error('[Quickplay] AI decision error:', {
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        phase: $gameState.phase,
        currentPlayer: $gameState.currentPlayer,
        availableActions: $availableActions.map(a => a.id),
        actionCount: get(actionHistory).length,
        currentTrick: $gameState.currentTrick,
        hands: $gameState.hands,
        trump: $gameState.trump
      });
      
      // Store error in a new error store for UI display
      quickplayErrorStore.set({
        message: error instanceof Error ? error.message : 'Unknown error in AI decision',
        state: JSON.parse(JSON.stringify($gameState)),
        availableActions: $availableActions.map(a => a.id),
        timestamp: new Date().toISOString()
      });
      
      // Stop quickplay on error
      quickplayState.update(state => ({ ...state, enabled: false }));
      animationFrameId = null;
      return;
    }
    
    // Continue processing
    animationFrameId = requestAnimationFrame(processAIMoves);
  } else {
    // For non-instant speeds, we'll implement delay tracking later
    // For now, just handle instant speed
    animationFrameId = null;
  }
}

export function startQuickplay() {
  if (animationFrameId !== null) return; // Already running
  
  // Start the continuous processing loop
  animationFrameId = requestAnimationFrame(processAIMoves);
  
  // Subscribe to state changes to detect when to stop
  if (!unsubscribe) {
    unsubscribe = quickplayState.subscribe(($quickplayState) => {
      if (!$quickplayState.enabled && animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      } else if ($quickplayState.enabled && animationFrameId === null) {
        animationFrameId = requestAnimationFrame(processAIMoves);
      }
    });
  }
}

export function stopQuickplay() {
  // Cancel animation frame
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  
  // Keep subscription active so we can restart if needed
}

// Quickplay actions
export const quickplayActions = {
  toggle: () => {
    quickplayState.update(state => {
      const newEnabled = !state.enabled;
      if (newEnabled) {
        startQuickplay();
      } else {
        stopQuickplay();
      }
      return { ...state, enabled: newEnabled };
    });
  },
  
  pause: () => {
    quickplayState.update(state => ({ ...state, isPaused: true }));
  },
  
  resume: () => {
    quickplayState.update(state => ({ ...state, isPaused: false }));
  },
  
  setSpeed: (speed: QuickplayState['speed']) => {
    quickplayState.update(state => ({ ...state, speed }));
  },
  
  togglePlayer: (playerId: number) => {
    quickplayState.update(state => {
      const newAiPlayers = new Set(state.aiPlayers);
      if (newAiPlayers.has(playerId)) {
        newAiPlayers.delete(playerId);
      } else {
        newAiPlayers.add(playerId);
      }
      return { ...state, aiPlayers: newAiPlayers };
    });
  },
  
  step: () => {
    const $gameState = get(gameState);
    const $availableActions = get(availableActions);
    const $quickplayState = get(quickplayState);
    
    // Check if current player is AI
    if (!$quickplayState.aiPlayers.has($gameState.currentPlayer)) return;
    
    try {
      const decision = makeAIDecision($gameState, $availableActions);
      if (decision) {
        gameActions.executeAction(decision);
      }
    } catch (error) {
      console.error('[Quickplay] Step error:', {
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        phase: $gameState.phase,
        currentPlayer: $gameState.currentPlayer,
        availableActions: $availableActions.map(a => a.id),
        actionCount: get(actionHistory).length,
        currentTrick: $gameState.currentTrick,
        hands: $gameState.hands
      });
      
      // Store error for UI display
      quickplayErrorStore.set({
        message: error instanceof Error ? error.message : 'Unknown error in AI step',
        state: JSON.parse(JSON.stringify($gameState)),
        availableActions: $availableActions.map(a => a.id),
        timestamp: new Date().toISOString()
      });
      
      return;
    }
  },
  
  playToEndOfHand: () => {
    const runToHandEnd = () => {
      const $gameState = get(gameState);
      const $availableActions = get(availableActions);
      
      // Stop if we've reached scoring phase
      if ($gameState.phase === 'scoring' || $gameState.phase === 'game_end') {
        return;
      }
      
      // Execute one action
      try {
        const decision = makeAIDecision($gameState, $availableActions);
        if (decision) {
          gameActions.executeAction(decision);
          // Continue with next action
          setTimeout(runToHandEnd, 0);
        }
      } catch {
        // Stop on error
        return;
      }
    };
    
    runToHandEnd();
  },
  
  playToEndOfGame: () => {
    const runToGameEnd = () => {
      const $gameState = get(gameState);
      const $availableActions = get(availableActions);
      
      // Stop if game is complete
      if ($gameState.phase === 'game_end' && $gameState.isComplete) {
        return;
      }
      
      // Execute one action
      try {
        const decision = makeAIDecision($gameState, $availableActions);
        if (decision) {
          gameActions.executeAction(decision);
          // Continue with next action
          setTimeout(runToGameEnd, 0);
        }
      } catch {
        // Stop on error
        return;
      }
    };
    
    runToGameEnd();
  }
};