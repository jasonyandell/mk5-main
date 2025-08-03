import { describe, it } from 'vitest';
import type { GameState, Domino, TrumpSelection } from '../../game/types';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';
import { testLog } from '../helpers/testConsole';

describe('Laydown Detection', () => {
  const ALL_DOMINOES: [number, number][] = [
    [6,6], [6,5], [6,4], [6,3], [6,2], [6,1], [6,0],
    [5,5], [5,4], [5,3], [5,2], [5,1], [5,0],
    [4,4], [4,3], [4,2], [4,1], [4,0],
    [3,3], [3,2], [3,1], [3,0],
    [2,2], [2,1], [2,0],
    [1,1], [1,0],
    [0,0]
  ];

  function createDomino(high: number, low: number): Domino {
    return {
      high: Math.max(high, low),
      low: Math.min(high, low),
      id: `${Math.max(high, low)}-${Math.min(high, low)}`,
      points: 0
    };
  }

  function numberToTrumpSelection(trump: number): TrumpSelection {
    if (trump === 7) {
      return { type: 'doubles' };
    } else if (trump === 8) {
      return { type: 'no-trump' };
    } else if (trump >= 0 && trump <= 6) {
      return { type: 'suit', suit: trump as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
    } else {
      return { type: 'none' };
    }
  }

  function setupGameWithHand(hand: [number, number][], trump: TrumpSelection): GameState {
    const state = createInitialState();
    
    state.phase = 'playing';
    state.trump = trump;
    state.winningBidder = 0;
    state.currentPlayer = 0;
    state.currentBid = { type: 'marks', value: 1, player: 0 };
    
    const allDominoes: Domino[] = [];
    for (let i = 0; i <= 6; i++) {
      for (let j = i; j <= 6; j++) {
        allDominoes.push(createDomino(j, i));
      }
    }
    
    const bidderHand = hand.map(([h, l]) => {
      const domino = allDominoes.find(d => 
        (d.high === h && d.low === l) || (d.high === l && d.low === h)
      );
      if (!domino) throw new Error(`Domino ${h}-${l} not found`);
      return domino;
    });
    
    const usedIds = new Set(bidderHand.map(d => d.id));
    const remainingDominoes = allDominoes.filter(d => !usedIds.has(d.id));
    
    state.players[0].hand = bidderHand;
    state.players[1].hand = remainingDominoes.slice(0, 7);
    state.players[2].hand = remainingDominoes.slice(7, 14);
    state.players[3].hand = remainingDominoes.slice(14, 21);
    
    return state;
  }

  function determineBestTrump(hand: [number, number][]): number {
    // For laydowns, we should check ALL possible trumps and pick the one
    // that actually makes it a laydown. If multiple work, prefer the one
    // with the most dominoes.
    
    // First, try each suit (0-6) as trump
    for (let trump = 0; trump <= 6; trump++) {
      if (isLaydownWithTrump(hand, trump)) {
        return trump;
      }
    }
    
    // Then try doubles
    if (isLaydownWithTrump(hand, 7)) {
      return 7;
    }
    
    // If no trump makes it a laydown, return -1 to indicate failure
    return -1;
  }
  
  function isLaydownWithTrump(hand: [number, number][], trump: number): boolean {
    const trumpCount = trump === 7 
      ? hand.filter(d => d[0] === d[1]).length
      : hand.filter(d => d[0] === trump || d[1] === trump).length;
    
    const opponentTrumps = 7 - trumpCount;
    
    // Must have more trumps than opponents
    if (trumpCount <= opponentTrumps) return false;
    
    // Must have the highest trump
    const hasHighestTrump = trump === 7
      ? hand.some(d => d[0] === 6 && d[1] === 6)
      : hand.some(d => d[0] === trump && d[1] === trump);
    
    if (!hasHighestTrump) return false;
    
    // Check if we have enough consecutive top trumps
    const allPossibleTrumps = getAllTrumpsInOrder(trump);
    const ourTrumpSet = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    
    let consecutiveTopTrumps = 0;
    for (const t of allPossibleTrumps) {
      if (ourTrumpSet.has(`${t[0]}-${t[1]}`) || ourTrumpSet.has(`${t[1]}-${t[0]}`)) {
        consecutiveTopTrumps++;
      } else {
        break;
      }
    }
    
    // Need more consecutive top trumps than opponents have total
    if (consecutiveTopTrumps <= opponentTrumps) return false;
    
    // Check that all non-trumps are winners
    const handSet = new Set(hand.map(d => `${d[0]}-${d[1]}`));
    const nonTrumps = hand.filter(d => 
      trump === 7 ? d[0] !== d[1] : (d[0] !== trump && d[1] !== trump)
    );
    
    for (const nt of nonTrumps) {
      if (!isNonTrumpWinner(nt, trump, handSet)) {
        return false;
      }
    }
    
    // NEW: Check for void situations
    // If opponents have any trumps left after we exhaust ours,
    // they can use them to trump our non-trumps if void in a suit
    const opponentTrumpsRemaining = Math.max(0, opponentTrumps - trumpCount);
    if (opponentTrumpsRemaining > 0) {
      // Count non-trumps by suit
      const nonTrumpSuits: { [suit: number]: number } = {};
      for (const nt of nonTrumps) {
        const suit = Math.max(nt[0], nt[1]);
        nonTrumpSuits[suit] = (nonTrumpSuits[suit] || 0) + 1;
      }
      
      // If any suit has multiple non-trumps, opponents could be void
      // and trump those extras
      for (const count of Object.values(nonTrumpSuits)) {
        if (count > 1) {
          // Opponents could be void and trump all but one
          const vulnerableCount = count - 1;
          if (vulnerableCount >= 1) {
            return false; // They can trump enough to beat us
          }
        }
      }
    }
    
    return true;
  }

  function getPlayOrder(hand: [number, number][], trump: number): [number, number][] {
    // Separate trumps and non-trumps
    const trumps: [number, number][] = [];
    const nonTrumps: [number, number][] = [];
    
    if (trump === 7) {
      // Doubles are trump
      trumps.push(...hand.filter(d => d[0] === d[1]));
      nonTrumps.push(...hand.filter(d => d[0] !== d[1]));
      
      // Sort trumps: highest double first
      trumps.sort((a, b) => b[0] - a[0]);
      
      // Sort non-trumps: highest pip value first
      nonTrumps.sort((a, b) => {
        const aMax = Math.max(a[0], a[1]);
        const bMax = Math.max(b[0], b[1]);
        return bMax - aMax;
      });
    } else {
      // Regular suit is trump
      for (const domino of hand) {
        if (domino[0] === trump || domino[1] === trump) {
          trumps.push(domino);
        } else {
          nonTrumps.push(domino);
        }
      }
      
      // Sort trumps: double first, then by other pip (highest to lowest)
      trumps.sort((a, b) => {
        const aIsDouble = a[0] === a[1] && a[0] === trump;
        const bIsDouble = b[0] === b[1] && b[0] === trump;
        
        if (aIsDouble && !bIsDouble) return -1;
        if (!aIsDouble && bIsDouble) return 1;
        
        // Both non-doubles or both doubles
        const aOther = a[0] === trump ? a[1] : a[0];
        const bOther = b[0] === trump ? b[1] : b[0];
        return bOther - aOther;
      });
      
      // Sort non-trumps: doubles first (by value), then non-doubles
      nonTrumps.sort((a, b) => {
        const aIsDouble = a[0] === a[1];
        const bIsDouble = b[0] === b[1];
        
        if (aIsDouble && !bIsDouble) return -1;
        if (!aIsDouble && bIsDouble) return 1;
        
        if (aIsDouble && bIsDouble) {
          return b[0] - a[0]; // Higher double first
        }
        
        // Both non-doubles
        const aMax = Math.max(a[0], a[1]);
        const bMax = Math.max(b[0], b[1]);
        return bMax - aMax;
      });
    }
    
    // Play order: all trumps first (highest to lowest), then non-trumps
    return [...trumps, ...nonTrumps];
  }


  function getAllTrumpsInOrder(trump: number): [number, number][] {
    if (trump === 7) {
      // Doubles as trump, highest to lowest
      return [[6,6], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]];
    }
    
    const trumps: [number, number][] = [];
    // Double is highest
    trumps.push([trump, trump]);
    
    // Then by other pip, highest to lowest
    for (let i = 6; i >= 0; i--) {
      if (i !== trump) {
        if (i > trump) {
          trumps.push([i, trump]);
        } else {
          trumps.push([trump, i]);
        }
      }
    }
    
    return trumps;
  }

  function isNonTrumpWinner(domino: [number, number], trump: number, ourHandSet: Set<string>): boolean {
    // For laydowns, a non-trump is only safe if no opponent can beat it
    // This assumes opponents are void in trump (we've drawn them all out)
    
    const suit = Math.max(domino[0], domino[1]);
    
    for (const opp of ALL_DOMINOES) {
      if (ourHandSet.has(`${opp[0]}-${opp[1]}`)) continue;
      
      // Skip if opponent domino is trump
      if (trump === 7) {
        if (opp[0] === opp[1]) continue; // It's a trump (double)
      } else {
        if (opp[0] === trump || opp[1] === trump) continue; // It's a trump
      }
      
      // Check if opponent domino follows suit and beats ours
      if (opp[0] === suit || opp[1] === suit) {
        if (compareDominosInSuit(opp, domino, suit) > 0) {
          return false; // Opponent has higher
        }
      }
    }
    
    return true;
  }

  function compareDominosInSuit(d1: [number, number], d2: [number, number], suit: number): number {
    const d1FollowsSuit = d1[0] === suit || d1[1] === suit;
    const d2FollowsSuit = d2[0] === suit || d2[1] === suit;
    
    if (!d1FollowsSuit || !d2FollowsSuit) {
      throw new Error('Both dominoes must follow suit');
    }
    
    const d1IsDouble = d1[0] === d1[1] && d1[0] === suit;
    const d2IsDouble = d2[0] === d2[1] && d2[0] === suit;
    
    if (d1IsDouble && !d2IsDouble) return 1;
    if (!d1IsDouble && d2IsDouble) return -1;
    
    const d1OtherPip = d1[0] === suit ? d1[1] : d1[0];
    const d2OtherPip = d2[0] === suit ? d2[1] : d2[0];
    
    return d1OtherPip - d2OtherPip; // Higher pip wins
  }

  function generateAllCombinations<T>(
    arr: T[], 
    k: number, 
    callback: (combination: T[]) => void
  ): void {
    const n = arr.length;
    if (k > n) return;
    
    const indices = Array.from({ length: k }, (_, i) => i);
    
    while (true) {
      const combination = indices.map(i => arr[i]);
      callback(combination);
      
      let i = k - 1;
      while (i >= 0 && indices[i] === n - k + i) {
        i--;
      }
      
      if (i < 0) break;
      
      indices[i]++;
      
      for (let j = i + 1; j < k; j++) {
        indices[j] = indices[j - 1] + 1;
      }
    }
  }

  it('should find all true laydowns by checking all possible trumps', () => {
    testLog('\n=== Laydown Detection ===\n');
    
    const laydowns: Array<{
      hand: [number, number][],
      trump: number,
      playOrder: [number, number][]
    }> = [];
    
    let handsChecked = 0;
    
    // Generate all possible 7-domino hands
    generateAllCombinations(ALL_DOMINOES, 7, (hand) => {
      handsChecked++;
      
      // Determine best trump for this hand
      const bestTrump = determineBestTrump(hand);
      
      // Only add if we found a trump that makes it a laydown
      if (bestTrump !== -1) {
        const playOrder = getPlayOrder(hand, bestTrump);
        // Store the hand in play order
        laydowns.push({ hand: playOrder, trump: bestTrump, playOrder });
      }
    });
    
    testLog(`Hands checked: ${handsChecked}`);
    testLog(`Laydowns found: ${laydowns.length}\n`);
    
    // Show some examples, including some with 0 as trump
    testLog('Example laydowns with correct play order:\n');
      
      // First show any with 0 as trump
      const zeroTrumpExamples = laydowns.filter(l => l.trump === 0).slice(0, 2);
      zeroTrumpExamples.forEach((laydown, i) => {
        testLog(`Example ${i + 1} (0s trump):`);
        testLog(`Hand: ${laydown.hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
        testLog(`Best trump: ${laydown.trump}`);
        testLog(`Play order: ${laydown.playOrder.map(d => `${d[0]}-${d[1]}`).join(' → ')}`);
        
        // Count trumps
        const trumpCount = laydown.hand.filter(d => d[0] === 0 || d[1] === 0).length;
        testLog(`Trumps: ${trumpCount}, Non-trumps: ${7 - trumpCount}`);
        testLog('---\n');
      });
      
      // Then show other examples
      laydowns.slice(0, 3).forEach((laydown, i) => {
        testLog(`Example ${i + 3}:`);
        testLog(`Hand: ${laydown.hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
        testLog(`Best trump: ${laydown.trump === 7 ? 'Doubles' : laydown.trump}`);
        testLog(`Play order: ${laydown.playOrder.map(d => `${d[0]}-${d[1]}`).join(' → ')}`);
        
        // Count trumps
        const trumpCount = laydown.trump === 7
          ? laydown.hand.filter(d => d[0] === d[1]).length
          : laydown.hand.filter(d => d[0] === laydown.trump || d[1] === laydown.trump).length;
        testLog(`Trumps: ${trumpCount}, Non-trumps: ${7 - trumpCount}`);
        testLog('---\n');
      });
      
      // Group by trump
      const byTrump: { [key: string]: number } = {};
      laydowns.forEach(l => {
        const key = l.trump === 7 ? 'Doubles' : `${l.trump}s`;
        byTrump[key] = (byTrump[key] || 0) + 1;
      });
      
      testLog('Laydowns by trump suit:');
      Object.entries(byTrump)
        .sort(([,a], [,b]) => b - a)
        .forEach(([trump, count]) => {
          testLog(`${trump}: ${count}`);
        });
  });

  it('should verify detected laydowns with game engine', () => {
    testLog('\n=== Engine Verification of Detected Laydowns ===\n');
    
    // Find a few laydowns with optimal trump
    const testHands: Array<{ hand: [number, number][], trump: number }> = [];
    
    generateAllCombinations(ALL_DOMINOES, 7, (hand) => {
      if (testHands.length >= 10) return;
      
      const bestTrump = determineBestTrump(hand);
      if (bestTrump !== -1) {
        testHands.push({ hand: [...hand], trump: bestTrump });
      }
    });
    
    let verified = 0;
    testHands.forEach((test, i) => {
      const gameState = setupGameWithHand(test.hand, numberToTrumpSelection(test.trump));
      const isLaydown = exhaustiveCheck(gameState);
      
      testLog(`${i + 1}. ${test.hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
      testLog(`   Trump: ${test.trump === 7 ? 'Doubles' : test.trump}`);
      testLog(`   Engine: ${isLaydown ? '✓ LAYDOWN' : '✗ NOT LAYDOWN'}`);
      
      if (isLaydown) verified++;
    });
    
    testLog('Testing first 10 corrected laydowns:\n');
    testLog(`\nVerified: ${verified}/${testHands.length}`);
  });

  function exhaustiveCheck(state: GameState): boolean {
    const cache = new Map<string, boolean>();
    const maxDepth = 50; // Prevent infinite recursion
    
    function getCacheKey(state: GameState): string {
      const tricksByPlayer = [0, 0, 0, 0];
      state.tricks.forEach(t => {
        if (t.winner !== undefined) {
          tricksByPlayer[t.winner]++;
        }
      });
      const cardsLeft = state.players.map(p => p.hand.length).join('-');
      const currentTrick = state.currentTrick ? 
        state.currentTrick.map(p => p.domino.id).join(',') : '';
      return `${tricksByPlayer.join('-')}|${cardsLeft}|P${state.currentPlayer}|T${currentTrick}`;
    }
    
    function canBidderWinAll(state: GameState, depth = 0): boolean {
      // Prevent infinite recursion
      if (depth > maxDepth) {
        return false;
      }
      
      // If bidder already lost a trick, they can't win all
      if (state.tricks.some(t => t.winner !== 0)) {
        return false;
      }
      
      // If game is over, check if bidder won all 7 tricks
      if (state.phase !== 'playing') {
        return state.tricks.length === 7 && state.tricks.every(t => t.winner === 0);
      }
      
      const key = getCacheKey(state);
      if (cache.has(key)) {
        return cache.get(key)!;
      }
      
      const transitions = getNextStates(state);
      if (transitions.length === 0) {
        cache.set(key, false);
        return false;
      }
      
      let result: boolean;
      if (state.currentPlayer === 0) {
        // Bidder's turn: they need at least one winning move
        result = transitions.some(t => canBidderWinAll(t.newState, depth + 1));
      } else {
        // Opponent's turn: all their moves must lead to bidder winning
        result = transitions.every(t => canBidderWinAll(t.newState, depth + 1));
      }
      
      cache.set(key, result);
      return result;
    }
    
    return canBidderWinAll(state);
  }
});