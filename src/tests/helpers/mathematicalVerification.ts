import { createInitialState } from '../../game/core/state';
import { createDominoes, getDominoPoints } from '../../game/core/dominoes';
import type { GameState, Domino } from '../../game/types';

/**
 * Mathematical verification utilities for Texas 42 game constants
 */

export class MathematicalVerification {
  /**
   * Verify that the total points in a domino set equals 35 (mk4 rules)
   */
  static verifyTotalPoints(): boolean {
    const dominoSet = createDominoes();
    let totalPoints = 0;
    
    dominoSet.forEach((domino: Domino) => {
      totalPoints += getDominoPoints(domino);
    });
    
    return totalPoints === 35; // 3×5 + 2×10 = 35 in mk4 rules
  }

  /**
   * Verify that exactly 28 dominoes exist in a complete set
   */
  static verifyDominoCount(): boolean {
    const dominoSet = createDominoes();
    return dominoSet.length === 28;
  }

  /**
   * Verify no duplicate dominoes in the set
   */
  static verifyNoDuplicates(): boolean {
    const dominoSet = createDominoes();
    const dominoStrings = dominoSet.map((d: Domino) => `${Math.min(d.high, d.low)}-${Math.max(d.high, d.low)}`);
    const uniqueStrings = new Set(dominoStrings);
    
    return uniqueStrings.size === dominoSet.length;
  }

  /**
   * Verify that all dominoes from 0-0 to 6-6 exist
   */
  static verifyCompleteSet(): boolean {
    const dominoSet = createDominoes();
    const expected = new Set<string>();
    
    // Generate expected domino set
    for (let i = 0; i <= 6; i++) {
      for (let j = i; j <= 6; j++) {
        expected.add(`${i}-${j}`);
      }
    }
    
    // Check actual set
    const actual = new Set(
      dominoSet.map((d: Domino) => `${Math.min(d.high, d.low)}-${Math.max(d.high, d.low)}`)
    );
    
    return expected.size === actual.size && 
           [...expected].every(domino => actual.has(domino));
  }

  /**
   * Verify point distribution follows Texas 42 rules
   */
  static verifyPointDistribution(): { [points: number]: number } {
    const dominoSet = createDominoes();
    const distribution: { [points: number]: number } = {};
    
    dominoSet.forEach((domino: Domino) => {
      const points = getDominoPoints(domino);
      distribution[points] = (distribution[points] || 0) + 1;
    });
    
    return distribution;
  }

  /**
   * Verify that exactly 5 dominoes have points (mk4 rules)
   */
  static verifyPointDominoCount(): boolean {
    const distribution = this.verifyPointDistribution();
    const pointDominoes = Object.entries(distribution)
      .filter(([points]) => parseInt(points) > 0)
      .reduce((sum, [, count]) => sum + count, 0);
    
    return pointDominoes === 5;
  }

  /**
   * Verify specific high-value dominoes exist
   */
  static verifyHighValueDominoes(): boolean {
    const dominoSet = createDominoes();
    
    const actualDominoes = new Set(
      dominoSet.map((d: Domino) => `${d.high}-${d.low}`)
    );
    
    // Check for 6-6, 6-4, 5-5 specifically
    return actualDominoes.has('6-6') && 
           actualDominoes.has('6-4') && 
           actualDominoes.has('5-5');
  }

  /**
   * Verify game state mathematical consistency
   */
  static verifyGameStateConsistency(state: GameState): boolean {
    // All players should have 7 dominoes initially
    const totalDominoes = state.players.reduce((sum, player) => sum + player.hand.length, 0);
    if (totalDominoes !== 28) return false;
    
    // Team scores should not exceed 42
    if (state.teamScores[0] + state.teamScores[1] > 42) return false;
    
    // Current player should be valid
    if (state.currentPlayer < 0 || state.currentPlayer >= 4) return false;
    
    // Dealer should be valid
    if (state.dealer < 0 || state.dealer >= 4) return false;
    
    // Team assignments should be correct (0&2 vs 1&3)
    if (state.players[0].teamId !== 0) return false;
    if (state.players[1].teamId !== 1) return false;
    if (state.players[2].teamId !== 0) return false;
    if (state.players[3].teamId !== 1) return false;
    
    return true;
  }

  /**
   * Verify trick mathematical consistency
   */
  static verifyTrickConsistency(state: GameState): boolean {
    // Each completed trick should have exactly 4 plays
    for (const trick of state.tricks) {
      if (trick.plays.length !== 4) return false;
    }
    
    // Current trick should have 0-3 plays
    if (state.currentTrick.length > 3) return false;
    
    // Total tricks played plus current trick plays should not exceed 28
    const totalPlays = state.tricks.reduce((sum, trick) => sum + trick.plays.length, 0) + 
                      state.currentTrick.length;
    if (totalPlays > 28) return false;
    
    return true;
  }

  /**
   * Verify bidding mathematical constraints
   */
  static verifyBiddingConstraints(state: GameState): boolean {
    // Bid values should be within valid ranges
    for (const bid of state.bids) {
      switch (bid.type) {
        case 'points':
          if (bid.value! < 30 || bid.value! > 42) return false;
          break;
        case 'marks':
          if (bid.value! < 1 || bid.value! > 6) return false;
          break;
        case 'nello':
          if (bid.value! < 1 || bid.value! > 4) return false;
          break;
        case 'splash':
          if (bid.value! < 2 || bid.value! > 6) return false;
          break;
        case 'plunge':
          if (bid.value! < 4 || bid.value! > 8) return false;
          break;
        case 'pass':
          // Pass bids don't have values
          break;
      }
    }
    
    return true;
  }

  /**
   * Run comprehensive mathematical verification
   */
  static runFullVerification(): {
    totalPoints: boolean;
    dominoCount: boolean;
    noDuplicates: boolean;
    completeSet: boolean;
    pointDominoCount: boolean;
    highValueDominoes: boolean;
    gameStateConsistency: boolean;
    trickConsistency: boolean;
    biddingConstraints: boolean;
  } {
    const state = createInitialState();
    
    return {
      totalPoints: this.verifyTotalPoints(),
      dominoCount: this.verifyDominoCount(),
      noDuplicates: this.verifyNoDuplicates(),
      completeSet: this.verifyCompleteSet(),
      pointDominoCount: this.verifyPointDominoCount(),
      highValueDominoes: this.verifyHighValueDominoes(),
      gameStateConsistency: this.verifyGameStateConsistency(state),
      trickConsistency: this.verifyTrickConsistency(state),
      biddingConstraints: this.verifyBiddingConstraints(state)
    };
  }
}