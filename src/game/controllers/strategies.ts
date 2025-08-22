import type { AIStrategy } from './types';
import type { GameState, StateTransition, Domino, Player } from '../types';
import { BID_TYPES } from '../constants';
import { getDominoPoints, countDoubles } from '../core/dominoes';
import { getStrongestSuits } from '../core/suit-analysis';

/**
 * Simple AI strategy - picks first available action
 */
export class SimpleAIStrategy implements AIStrategy {
  chooseAction(_state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Prioritize consensus actions for quick agreement
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // Otherwise pick the first available action
    return transitions[0] || null;
  }
  
  getThinkingTime(actionType: string): number {
    // Instant response for consensus actions
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 0;
    }
    
    // Normal thinking time for game actions
    return 500 + Math.random() * 1500;
  }
}

/**
 * Random AI strategy - picks random action
 */
export class RandomAIStrategy implements AIStrategy {
  chooseAction(_state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Still prioritize consensus for game flow
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // Random choice from available actions
    const index = Math.floor(Math.random() * transitions.length);
    return transitions[index] || null;
  }
  
  getThinkingTime(actionType: string): number {
    // Instant response for consensus actions
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 0;
    }
    
    return 300 + Math.random() * 1000;
  }
}

/**
 * Smart AI strategy - makes intelligent decisions using quickplay logic
 */
export class SmartAIStrategy implements AIStrategy {
  chooseAction(state: GameState, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Prioritize consensus
    const consensusAction = transitions.find(t =>
      t.action.type === 'agree-complete-trick' ||
      t.action.type === 'agree-score-hand'
    );
    
    if (consensusAction) {
      return consensusAction;
    }
    
    // Get current player for context
    const currentPlayer = state.players[state.currentPlayer];
    if (!currentPlayer) return transitions[0] || null;
    
    // Use phase-specific logic
    switch (state.phase) {
      case 'bidding':
        return this.makeBidDecision(state, currentPlayer, transitions);
      case 'trump_selection':
        return this.makeTrumpDecision(state, currentPlayer, transitions);
      case 'playing':
        return this.makePlayDecision(state, currentPlayer, transitions);
      default:
        return transitions[0] || null;
    }
  }
  
  private makeBidDecision(state: GameState, player: Player, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // Calculate hand strength
    const handStrength = this.calculateHandStrength(player.hand);
    
    // Find pass action
    const passAction = transitions.find(t => t.id === 'pass');
    
    // Get current highest bid value
    const currentBidValue = this.getCurrentBidValue(state);
    
    // Weak hand - pass if can't bid 30
    if (handStrength < 8 && currentBidValue >= 30) {
      return passAction || transitions[0] || null;
    }
    
    // Calculate bid based on hand strength
    let targetBid = 30;
    if (handStrength >= 12) targetBid = 32;
    if (handStrength >= 15) targetBid = 35;
    
    // Find best available bid
    const bidActions = transitions.filter(t => t.id.startsWith('bid-') && !t.id.includes('marks'));
    const validBids = bidActions.filter(t => {
      const parts = t.id.split('-');
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
      return validBids[0] || null;
    }
    
    // Can't make desired bid - pass
    return passAction || transitions[0] || null;
  }
  
  private makeTrumpDecision(_state: GameState, player: Player, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    if (!player.suitAnalysis) {
      // Fallback - pick first available trump
      return transitions[0] || null;
    }
    
    // Get strongest suits
    const strongestSuits = getStrongestSuits(player.suitAnalysis);
    
    // Check for 3+ doubles - consider doubles trump
    const doubleCount = countDoubles(player.hand);
    if (doubleCount >= 3) {
      const doublesAction = transitions.find(t => t.id === 'trump-doubles');
      if (doublesAction) return doublesAction;
    }
    
    // Pick strongest suit as trump
    for (const suit of strongestSuits) {
      const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
      const trumpAction = transitions.find(t => t.id === `trump-${suitNames[suit]}`);
      if (trumpAction) return trumpAction;
    }
    
    // Fallback
    return transitions[0] || null;
  }
  
  private makePlayDecision(state: GameState, player: Player, transitions: StateTransition[]): StateTransition | null {
    if (transitions.length === 0) return null;
    
    // If completing a trick, do it
    const completeTrickAction = transitions.find(t => t.id === 'complete-trick');
    if (completeTrickAction) return completeTrickAction;
    
    // Extract dominoes from play actions
    const playActions = transitions.filter(t => t.id.startsWith('play-'));
    if (playActions.length === 0) return transitions[0] || null;
    
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
      return scoredActions[0]?.action || transitions[0] || null;
    }
    
    // Following in a trick
    const myTeam = player.teamId;
    const trickLeader = state.currentTrick[0]?.player;
    if (trickLeader === undefined) return transitions[0] || null;
    const leaderPlayer = state.players[trickLeader];
    if (!leaderPlayer) return transitions[0] || null;
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
    
    return scoredActions[0]?.action || transitions[0] || null;
  }
  
  private calculateHandStrength(hand: Domino[]): number {
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
  
  private getCurrentBidValue(state: GameState): number {
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
  
  getThinkingTime(actionType: string): number {
    // Instant response for consensus actions
    if (actionType === 'agree-complete-trick' || actionType === 'agree-score-hand') {
      return 0;
    }
    
    return 800 + Math.random() * 2000;
  }
}