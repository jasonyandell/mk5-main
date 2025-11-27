import type { AIStrategy } from './types';
import type { GameState, Player } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import { BID_TYPES } from '../constants';
import { calculateTrickWinner } from '../core/scoring';
import { analyzeHand } from '../ai/utilities';
import {
  determineBestTrump,
  LAYDOWN_SCORE,
  BID_THRESHOLDS
} from '../ai/hand-strength';
import { calculateLexicographicStrength } from '../ai/lexicographic-strength';


/**
 * Random AI strategy - picks random action
 */
export class RandomAIStrategy implements AIStrategy {
  chooseAction(_state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('RandomAIStrategy: No valid actions available to choose from');
    }

    // Still prioritize consensus for game flow
    const consensusAction = validActions.find(va =>
      va.action.type === 'agree-complete-trick' ||
      va.action.type === 'agree-score-hand'
    );

    if (consensusAction) {
      return consensusAction;
    }

    // Random choice from available actions
    const index = Math.floor(Math.random() * validActions.length);
    const chosen = validActions[index];
    if (!chosen) {
      throw new Error(`RandomAIStrategy: Failed to select action at index ${index} from ${validActions.length} actions`);
    }
    return chosen;
  }
}

// NOTE: All AI decision-making utilities have been moved to src/game/ai/utilities.ts
// The main analysis function analyzeHand() provides complete context analysis
// including domino values and trump information

/**
 * Beginner AI strategy - makes basic intelligent decisions
 */
export class BeginnerAIStrategy implements AIStrategy {
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('BeginnerAIStrategy: No valid actions available to choose from');
    }

    // Prioritize consensus
    const consensusAction = validActions.find(va =>
      va.action.type === 'agree-complete-trick' ||
      va.action.type === 'agree-score-hand'
    );

    if (consensusAction) {
      return consensusAction;
    }

    // Get current player for context
    const currentPlayer = state.players[state.currentPlayer];
    if (!currentPlayer) {
      throw new Error(`BeginnerAIStrategy: Current player ${state.currentPlayer} not found in state`);
    }

    // Use phase-specific logic
    switch (state.phase) {
      case 'bidding':
        return this.makeBidDecision(state, currentPlayer, validActions);
      case 'trump_selection':
        return this.makeTrumpDecision(state, currentPlayer, validActions);
      case 'playing':
        return this.makePlayDecision(state, currentPlayer, validActions);
      default: {
        const fallback = validActions[0];
        if (!fallback) {
          throw new Error(`BeginnerAIStrategy: No fallback action available for phase ${state.phase}`);
        }
        return fallback;
      }
    }
  }
  
  private makeBidDecision(state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('BeginnerAIStrategy.makeBidDecision: No valid actions available');
    }

    // Calculate hand strength using shared utility
    // Pass undefined to force proper trump analysis for bidding decisions
    const handStrength = calculateLexicographicStrength(player.hand, state.trump, state);

    // Find pass action
    const passAction = validActions.find(va => va.action.type === 'pass');

    // Get current highest bid value
    const currentBidValue = this.getCurrentBidValue(state);

    // Weak hand - pass if can't bid 30
    if (handStrength < 30 && currentBidValue >= 30) {
      if (passAction) return passAction;
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makeBidDecision: No pass action or fallback available for weak hand');
      }
      return fallback;
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
      const bidActions = validActions.filter(va =>
        va.action.type === 'bid'
      );

      // If we can make exactly 42, bid 2 marks (highest bid)
      if (guaranteedScore >= 42) {
        const marksBid = bidActions.find(va =>
          va.action.type === 'bid' &&
          'bid' in va.action &&
          va.action.bid === 'marks' &&
          va.action.value === 2
        );
        if (marksBid) return marksBid;
      }

      // Otherwise bid our actual score (or highest available under our score)
      const pointBids = bidActions
        .filter(va =>
          va.action.type === 'bid' &&
          'bid' in va.action &&
          va.action.bid === 'points'
        )
        .map(va => {
          const value = va.action.type === 'bid' && 'value' in va.action ? va.action.value || 0 : 0;
          return { value, action: va };
        })
        .filter(item => !isNaN(item.value) && item.value <= guaranteedScore)
        .sort((a, b) => b.value - a.value);

      if (pointBids.length > 0 && pointBids[0]) {
        return pointBids[0].action;
      }

      // If we can't find a suitable bid, just bid 30 (minimum)
      const bid30 = bidActions.find(va =>
        va.action.type === 'bid' &&
        'value' in va.action &&
        va.action.value === 30
      );
      if (bid30) return bid30;

      // Fallback
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makeBidDecision: No fallback available for laydown hand');
      }
      return fallback;
    }

    // Pass should be the most common bid!
    if (handStrength < BID_THRESHOLDS.PASS) {
      if (passAction) return passAction;
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makeBidDecision: No pass action or fallback available');
      }
      return fallback;
    }

    // Determine target bid using shared thresholds
    let targetBid = 30;

    if (handStrength >= BID_THRESHOLDS.BID_31) targetBid = 31;
    if (handStrength >= BID_THRESHOLDS.BID_32) targetBid = 32;
    if (handStrength >= BID_THRESHOLDS.BID_33) targetBid = 33;
    if (handStrength >= BID_THRESHOLDS.BID_34) targetBid = 34;
    if (handStrength >= BID_THRESHOLDS.BID_35) targetBid = 35;

    // Find best available bid
    const bidActions = validActions.filter(va =>
      va.action.type === 'bid' &&
      'bid' in va.action &&
      va.action.bid === 'points'
    );
    const validBids = bidActions.filter(va => {
      const value = va.action.type === 'bid' && 'value' in va.action ? va.action.value || 0 : 0;
      return !isNaN(value) && value >= targetBid && value > currentBidValue;
    });

    if (validBids.length > 0) {
      // Sort by bid value and take the lowest valid bid
      validBids.sort((a, b) => {
        const aValue = a.action.type === 'bid' && 'value' in a.action ? a.action.value || 0 : 0;
        const bValue = b.action.type === 'bid' && 'value' in b.action ? b.action.value || 0 : 0;
        return aValue - bValue;
      });
      const chosen = validBids[0];
      if (!chosen) {
        throw new Error('BeginnerAIStrategy.makeBidDecision: No valid bid found after sorting');
      }
      return chosen;
    }

    // Can't make desired bid - pass
    if (passAction) return passAction;
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('BeginnerAIStrategy.makeBidDecision: Cannot make desired bid and no pass/fallback available');
    }
    return fallback;
  }
  
  private makeTrumpDecision(_state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('BeginnerAIStrategy.makeTrumpDecision: No valid actions available');
    }

    // Use the shared utility to determine best trump
    const bestTrump = determineBestTrump(player.hand, player.suitAnalysis);

    if (bestTrump.type === 'doubles') {
      const doublesAction = validActions.find(va =>
        va.action.type === 'select-trump' &&
        'trump' in va.action &&
        va.action.trump.type === 'doubles'
      );
      if (doublesAction) return doublesAction;
    } else if (bestTrump.type === 'suit' && typeof bestTrump.suit === 'number') {
      const trumpAction = validActions.find(va =>
        va.action.type === 'select-trump' &&
        'trump' in va.action &&
        va.action.trump.type === 'suit' &&
        va.action.trump.suit === bestTrump.suit
      );
      if (trumpAction) return trumpAction;
    }

    // Fallback
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('BeginnerAIStrategy.makeTrumpDecision: No trump selection available');
    }
    return fallback;
  }
  
  private makePlayDecision(state: GameState, player: Player, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('BeginnerAIStrategy.makePlayDecision: No valid actions available');
    }

    // If completing a trick, do it
    const completeTrickAction = validActions.find(va => va.action.type === 'complete-trick');
    if (completeTrickAction) return completeTrickAction;

    // Extract dominoes from play actions
    const playActions = validActions.filter(va => va.action.type === 'play');
    if (playActions.length === 0) {
      throw new Error('BeginnerAIStrategy.makePlayDecision: No play actions or fallback available');
    }

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
          const action = playActions.find(va =>
            va.action.type === 'play' &&
            'dominoId' in va.action &&
            va.action.dominoId === bestLead.domino.id
          );
          if (action) {
            return action;
          }
        }
      }

      // Fallback when no suitable lead found
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makePlayDecision: No suitable lead play or fallback available');
      }
      return fallback;
    }

    // Following in a trick - determine who's currently winning
    const myTeam = player.teamId;

    // Find current trick winner player id
    const currentWinnerPlayerId = calculateTrickWinner(state.currentTrick, state.trump, state.currentSuit);
    if (currentWinnerPlayerId === -1) {
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makePlayDecision: Cannot determine trick winner and no fallback available');
      }
      return fallback;
    }

    // Find the winning player
    const winnerPlayer = state.players[currentWinnerPlayerId];
    if (!winnerPlayer) {
      throw new Error(`BeginnerAIStrategy.makePlayDecision: Winner player ${currentWinnerPlayerId} not found in state`);
    }

    const partnerCurrentlyWinning = winnerPlayer.teamId === myTeam;

    // Map play actions to analyzed dominoes
    const scoredActions = playActions.map(va => {
      // Extract domino ID from action
      if (va.action.type !== 'play' || !('dominoId' in va.action)) {
        return null;
      }
      const dominoId = va.action.dominoId;
      const dominoAnalysis = analysis.dominoes.find(d => d.domino.id === dominoId);

      if (!dominoAnalysis) {
        // Skip actions that don't have analysis (should not normally happen)
        return null;
      }

      return {
        action: va,
        points: dominoAnalysis.points,
        canBeat: dominoAnalysis.wouldBeatTrick,
      };
    }).filter((item): item is { action: ValidAction; points: number; canBeat: boolean } => item !== null);

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
        const chosen = winningPlays[0];
        if (!chosen?.action) {
          throw new Error('BeginnerAIStrategy.makePlayDecision: No winning play found after filtering');
        }
        return chosen.action;
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

    const chosen = scoredActions[0];
    if (!chosen?.action) {
      const fallback = validActions[0];
      if (!fallback) {
        throw new Error('BeginnerAIStrategy.makePlayDecision: No scored action or fallback available');
      }
      return fallback;
    }
    return chosen.action;
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
}
