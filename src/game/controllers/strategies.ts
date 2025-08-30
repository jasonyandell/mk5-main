import type { AIStrategy } from './types';
import type { GameState, StateTransition, Player } from '../types';
import { BID_TYPES } from '../constants';
import { calculateTrickWinner } from '../core/scoring';
import { analyzeHand } from '../ai/utilities';
import { 
  calculateHandStrengthWithTrump, 
  determineBestTrump,
  LAYDOWN_SCORE,
  BID_THRESHOLDS 
} from '../ai/hand-strength';


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

// NOTE: All AI decision-making utilities have been moved to src/game/ai/utilities.ts
// The main analysis function analyzeHand() provides complete context analysis
// including domino values and trump information

/**
 * Beginner AI strategy - makes basic intelligent decisions
 */
export class BeginnerAIStrategy implements AIStrategy {
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
    
    // Calculate hand strength using shared utility
    // Pass undefined to force proper trump analysis for bidding decisions
    const handStrength = calculateHandStrengthWithTrump(player.hand, undefined);
    
    // Find pass action
    const passAction = transitions.find(t => t.id === 'pass');
    
    // Get current highest bid value
    const currentBidValue = this.getCurrentBidValue(state);
    
    // Weak hand - pass if can't bid 30
    if (handStrength < 30 && currentBidValue >= 30) {
      return passAction || transitions[0] || null;
    }
    
    // Calculate bid based on hand strength with non-linear scoring
    
    // Special handling for laydown hands
    if (handStrength === LAYDOWN_SCORE) {
      // Calculate how many points we can actually make
      // Count points in hand + 7 tricks we'll win
      let handCount = 0;
      for (const domino of player.hand) {
        const sum = domino.high + domino.low;
        if (sum % 5 === 0 && sum > 0) {
          handCount += sum === 5 ? 5 : 10;
        }
      }
      const guaranteedScore = handCount + 7; // 7 tricks we'll win
      
      // Find the best bid based on what we can actually make
      const bidActions = transitions.filter(t => t.id.startsWith('bid-'));
      
      // If we can make exactly 42, bid 2 marks (highest bid)
      if (guaranteedScore >= 42) {
        const marksBid = bidActions.find(t => t.id === 'bid-2-marks');
        if (marksBid) return marksBid;
      }
      
      // Otherwise bid our actual score (or highest available under our score)
      const pointBids = bidActions
        .filter(t => !t.id.includes('marks'))
        .map(t => {
          const parts = t.id.split('-');
          const value = parts.length >= 2 && parts[1] ? parseInt(parts[1]) : 0;
          return { value, action: t };
        })
        .filter(item => !isNaN(item.value) && item.value <= guaranteedScore)
        .sort((a, b) => b.value - a.value);
      
      if (pointBids.length > 0 && pointBids[0]) {
        return pointBids[0].action;
      }
      
      // If we can't find a suitable bid, just bid 30 (minimum)
      const bid30 = bidActions.find(t => t.id === 'bid-30');
      if (bid30) return bid30;
      
      // Fallback
      return transitions[0] || null;
    }
    
    // Pass should be the most common bid!
    if (handStrength < BID_THRESHOLDS.PASS) {
      return passAction || transitions[0] || null;
    }
    
    // Determine target bid using shared thresholds
    let targetBid = 30;
    
    if (handStrength >= BID_THRESHOLDS.BID_31) targetBid = 31;
    if (handStrength >= BID_THRESHOLDS.BID_32) targetBid = 32;
    if (handStrength >= BID_THRESHOLDS.BID_33) targetBid = 33;
    if (handStrength >= BID_THRESHOLDS.BID_34) targetBid = 34;
    if (handStrength >= BID_THRESHOLDS.BID_35) targetBid = 35;
    
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
    
    // Use the shared utility to determine best trump
    const bestTrump = determineBestTrump(player.hand, player.suitAnalysis);
    
    if (bestTrump.type === 'doubles') {
      const doublesAction = transitions.find(t => t.id === 'trump-doubles');
      if (doublesAction) return doublesAction;
    } else if (bestTrump.type === 'suit' && typeof bestTrump.suit === 'number') {
      const suitNames = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
      const suitName = suitNames[bestTrump.suit];
      const trumpAction = transitions.find(t => t.id === `trump-${suitName}`);
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
    
    // Analyze hand using simplified interface
    const analysis = analyzeHand(state, player.id);
    
    // Leading a trick
    if (state.currentTrick.length === 0) {
      // Lead with best domino (already sorted by effective position)
      // First playable domino is the strongest
      const playable = analysis.dominoes.filter(d => d.beatenBy !== undefined);
      if (playable.length > 0) {
        const bestLead = playable[0];
        if (bestLead) {
          const action = playActions.find(a => 
            a.id === `play-${bestLead.domino.id}`
          );
          if (action) {
            return action;
          }
        }
      }
      
      // Fallback
      return transitions[0] || null;
    }
    
    // Following in a trick - determine who's currently winning
    const myTeam = player.teamId;
    
    // Find current trick winner player id
    const currentWinnerPlayerId = calculateTrickWinner(state.currentTrick, state.trump, state.currentSuit);
    if (currentWinnerPlayerId === -1) return transitions[0] || null;
    
    // Find the winning player
    const winnerPlayer = state.players[currentWinnerPlayerId];
    if (!winnerPlayer) return transitions[0] || null;
    
    const partnerCurrentlyWinning = winnerPlayer.teamId === myTeam;
    
    // Map play actions to analyzed dominoes
    const scoredActions = playActions.map(action => {
      // Extract domino ID from action id (e.g., 'play-5-5' -> '5-5')
      const dominoId = action.id.replace('play-', '');
      const dominoAnalysis = analysis.dominoes.find(d => d.domino.id.toString() === dominoId);
      
      if (!dominoAnalysis) return null;
      
      return {
        action,
        points: dominoAnalysis.points,
        canBeat: dominoAnalysis.wouldBeatTrick,
      };
    }).filter(Boolean) as Array<{ action: StateTransition; points: number; canBeat: boolean }>;
    
    if (partnerCurrentlyWinning) {
      // Partner currently winning - PLAY COUNT (safe points!)
      // Sort to play highest count first
      scoredActions.sort((a, b) => {
        // Always prioritize playing count (10 > 5 > 0)
        const pointDiff = b.points - a.points;
        if (pointDiff !== 0) {
          return pointDiff; // Play high count
        }
        return 0;
      });
    } else {
      // Opponent currently winning
      const winningPlays = scoredActions.filter(a => a.canBeat);
      
      if (winningPlays.length > 0) {
        // Can win - use lowest count to win
        winningPlays.sort((a, b) => a.points - b.points);
        return winningPlays[0]?.action || transitions[0] || null;
      } else {
        // Can't win - play LOW count (don't give them points!)
        scoredActions.sort((a, b) => {
          // First priority: play low count
          if (Math.abs(a.points - b.points) >= 5) {
            return a.points - b.points; // Play low count
          }
          return 0;
        });
      }
    }
    
    return scoredActions[0]?.action || transitions[0] || null;
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
