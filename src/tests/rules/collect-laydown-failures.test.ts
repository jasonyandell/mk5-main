import { describe, it } from 'vitest';
import type { GameState, Domino, TrumpSelection } from '../../game/types';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/gameEngine';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { testLog } from '../helpers/testConsole';

describe('Collect Laydown Failures', () => {
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
    
    // Initialize suit analysis for all players
    state.players.forEach(player => {
      player.suitAnalysis = analyzeSuits(player.hand, trump);
    });
    
    return state;
  }

  interface FailureDetails {
    hand: [number, number][];
    trump: TrumpSelection;
    testedTrump: TrumpSelection;
    losingPath: string[];
    lastMoves: string[];
    bidderLostAt: string;
  }

  function findLosingPath(initialState: GameState): FailureDetails | null {
    const hand = initialState.players[0].hand.map(d => [d.high, d.low] as [number, number]);
    const trumpValue = initialState.trump.type === 'doubles' ? 7 : 
                      initialState.trump.type === 'suit' ? initialState.trump.suit! : -1;
    
    // Track the path that leads to bidder losing
    let losingPath: string[] = [];
    let lastMoves: string[] = [];
    
    function explorePath(state: GameState, path: string[], depth: number = 0): boolean {
      if (depth > 50) return false;
      
      // Check if bidder already lost a trick
      const bidderLostTrick = state.tricks.find(t => t.winner !== 0);
      if (bidderLostTrick) {
        // Found a losing path! Capture the last few moves
        const recentMoves = path.slice(-8); // Get last 8 moves (about 2 tricks)
        
        losingPath = path;
        lastMoves = recentMoves;
        
        return true;
      }
      
      // If game is over and bidder won all tricks, this path doesn't work
      if (state.phase !== 'playing') {
        return false;
      }
      
      const transitions = getNextStates(state);
      
      // For bidder: try all moves
      if (state.currentPlayer === 0) {
        for (const t of transitions) {
          const moveDesc = extractMoveDescription(t.label, state.currentPlayer);
          if (explorePath(t.newState, [...path, moveDesc], depth + 1)) {
            return true;
          }
        }
      } else {
        // For opponents: find any move that leads to bidder losing
        for (const t of transitions) {
          const moveDesc = extractMoveDescription(t.label, state.currentPlayer);
          if (explorePath(t.newState, [...path, moveDesc], depth + 1)) {
            return true;
          }
        }
      }
      
      return false;
    }
    
    if (explorePath(initialState, [])) {
      // Find where bidder lost
      let testState = initialState;
      for (const move of losingPath) {
        if (move.includes('Complete trick')) {
          const lastTrick = testState.tricks[testState.tricks.length - 1];
          if (lastTrick && lastTrick.winner !== 0) {
            return {
              hand,
              trump: trumpValue === 7 ? { type: 'doubles' } : trumpValue === -1 ? { type: 'no-trump' } : { type: 'suit', suit: trumpValue as 0 | 1 | 2 | 3 | 4 | 5 | 6 },
              testedTrump: trumpValue === 7 ? { type: 'doubles' } : trumpValue === -1 ? { type: 'no-trump' } : { type: 'suit', suit: trumpValue as 0 | 1 | 2 | 3 | 4 | 5 | 6 },
              losingPath,
              lastMoves,
              bidderLostAt: `Trick ${testState.tricks.length}, won by Player ${lastTrick.winner}`
            };
          }
        }
        // Would need to replay moves to track exact state, but we have the path
      }
      
      return {
        hand,
        trump: trumpValue === 7 ? { type: 'doubles' } : trumpValue === -1 ? { type: 'no-trump' } : { type: 'suit', suit: trumpValue as 0 | 1 | 2 | 3 | 4 | 5 | 6 },
        testedTrump: trumpValue === 7 ? { type: 'doubles' } : trumpValue === -1 ? { type: 'no-trump' } : { type: 'suit', suit: trumpValue as 0 | 1 | 2 | 3 | 4 | 5 | 6 },
        losingPath,
        lastMoves,
        bidderLostAt: 'Unknown'
      };
    }
    
    return null;
  }

  function extractMoveDescription(label: string, player: number): string {
    if (label.includes('Play')) {
      const match = label.match(/Play (\d+-\d+)/);
      if (match) {
        return `P${player}: ${match[1]}`;
      }
    } else if (label.includes('Complete trick')) {
      return 'Complete trick';
    } else if (label.includes('wins trick')) {
      const match = label.match(/P(\d) wins trick \((\d+) points\)/);
      if (match) {
        return `P${match[1]} wins trick (${match[2]} points)`;
      }
    }
    return label;
  }

  function exhaustiveLaydownCheck(initialState: GameState): boolean {
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
    
    return canBidderWinAll(initialState);
  }

  // Functions to find potential laydowns (using corrected logic)
  function findBestTrump(hand: [number, number][]): number {
    // Try each suit (0-6) as trump
    for (let trump = 0; trump <= 6; trump++) {
      if (isLaydownWithTrump(hand, trump)) {
        return trump;
      }
    }
    
    // Then try doubles
    if (isLaydownWithTrump(hand, 7)) {
      return 7;
    }
    
    return -1; // No trump makes it a laydown
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
    const suit = Math.max(domino[0], domino[1]);
    
    for (const opp of ALL_DOMINOES) {
      if (ourHandSet.has(`${opp[0]}-${opp[1]}`)) continue;
      
      // Skip if opponent domino is trump
      if (trump === 7) {
        if (opp[0] === opp[1]) continue;
      } else {
        if (opp[0] === trump || opp[1] === trump) continue;
      }
      
      // Check if opponent domino follows suit and beats ours
      if (opp[0] === suit || opp[1] === suit) {
        if (compareDominosInSuit(opp, domino, suit) > 0) {
          return false;
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

  it('should collect and analyze laydown failures', { timeout: 600000 }, () => {
    testLog('\n=== Collecting Laydown Failures ===\n');
    
    const failures: FailureDetails[] = [];
    let potentialLaydowns = 0;
    let trueLaydowns = 0;
    
    // Generate all hands and test each
    generateAllCombinations(ALL_DOMINOES, 7, (hand) => {
      // Find best trump for this hand
      const bestTrump = findBestTrump(hand);
      
      if (bestTrump !== -1) {
        potentialLaydowns++;
        
        // Get play order and use that as the hand
        const playOrder = getPlayOrder(hand, bestTrump);
        
        // Verify with engine using play order
        const trumpSelection: TrumpSelection = bestTrump === 7 ? { type: 'doubles' } : { type: 'suit', suit: bestTrump as 0|1|2|3|4|5|6 };
        const gameState = setupGameWithHand(playOrder, trumpSelection);
        const isLaydown = exhaustiveLaydownCheck(gameState);
        
        if (isLaydown) {
          trueLaydowns++;
        } else {
          // Find the losing path
          const failure = findLosingPath(gameState);
          if (failure) {
            // Update failure to show original hand in play order
            failure.hand = playOrder;
            failures.push(failure);
          }
        }
      }
    });
    
    testLog(`Potential laydowns (by mathematical analysis): ${potentialLaydowns}`);
    testLog(`True laydowns (verified by engine): ${trueLaydowns}`);
    testLog(`Failures: ${failures.length}\n`);
    
    // Display ALL failures with key info and play order
    testLog('=== All Failure Details ===\n');
      failures.forEach((f, i) => {
        const handStr = f.hand.map(d => `${d[0]}-${d[1]}`).join(',');
        const lastPlay = f.lastMoves.filter(m => m.includes('P0:')).pop() || 'unknown';
        const beatenBy = f.lastMoves.filter(m => m.includes('P') && !m.includes('P0:')).pop() || 'unknown';
        const trumpStr = f.trump.type === 'doubles' ? 'D' : f.trump.type === 'suit' ? f.trump.suit!.toString() : '?';
        
        // Show the play order for this hand
        const trumpNum = f.trump.type === 'doubles' ? 7 : f.trump.type === 'suit' ? f.trump.suit! : -1;
        const playOrder = getPlayOrder(f.hand, trumpNum);
        const orderStr = playOrder.map(d => `${d[0]}-${d[1]}`).join(' → ');
        
        testLog(`${i+1}. [${handStr}] T:${trumpStr}`);
        testLog(`   Order: ${orderStr}`);
        testLog(`   Last: ${lastPlay} beaten by ${beatenBy}`);
      });
      
      // Group failures by pattern
      testLog('=== Failure Patterns ===\n');
      const patterns: { [key: string]: number } = {};
      
      failures.forEach(f => {
        const trumpNum = f.trump.type === 'doubles' ? 7 : f.trump.type === 'suit' ? f.trump.suit! : -1;
        const trumpCount = f.hand.filter(d => 
          f.trump.type === 'doubles' ? d[0] === d[1] : (d[0] === trumpNum || d[1] === trumpNum)
        ).length;
        const trumpStr = f.trump.type === 'doubles' ? 'doubles' : f.trump.type === 'suit' ? f.trump.suit!.toString() : 'none';
        const pattern = `${trumpCount} trumps (trump: ${trumpStr})`;
        patterns[pattern] = (patterns[pattern] || 0) + 1;
      });
      
      Object.entries(patterns)
        .sort(([,a], [,b]) => b - a)
        .forEach(([pattern, count]) => {
          testLog(`${pattern}: ${count} failures`);
        });
      
      // Show specific examples of different failure types
      testLog('\n=== Example Failures by Type ===\n');
      
      // Find examples of different trump counts
      const examplesByTrumpCount: { [key: number]: FailureDetails } = {};
      failures.forEach(f => {
        const trumpNum = f.trump.type === 'doubles' ? 7 : f.trump.type === 'suit' ? f.trump.suit! : -1;
        const trumpCount = f.hand.filter(d => 
          f.trump.type === 'doubles' ? d[0] === d[1] : (d[0] === trumpNum || d[1] === trumpNum)
        ).length;
        if (!examplesByTrumpCount[trumpCount]) {
          examplesByTrumpCount[trumpCount] = f;
        }
      });
      
      Object.entries(examplesByTrumpCount)
        .sort(([a], [b]) => Number(b) - Number(a))
        .forEach(([trumpCount, failure]) => {
          testLog(`Example with ${trumpCount} trumps:`);
          testLog(`Hand: ${failure.hand.map(d => `${d[0]}-${d[1]}`).join(', ')}`);
          const trumpDisplay = failure.trump.type === 'doubles' ? 'doubles' : failure.trump.type === 'suit' ? failure.trump.suit!.toString() : 'none';
          testLog(`Trump: ${trumpDisplay}`);
          const lastPlay = failure.lastMoves.filter(m => m.includes('P0:')).pop() || 'unknown';
          const beatenBy = failure.lastMoves.filter(m => m.includes('P') && !m.includes('P0:')).pop() || 'unknown';
          testLog(`Bidder ${lastPlay} beaten by ${beatenBy}`);
          testLog('');
        });
      
      // Should have no failures with corrected logic
      testLog(`\nExpected failures: 0 (all potential laydowns should verify)`);
      testLog(`Actual failures: ${failures.length}`);
      
      if (failures.length > 0) {
        testLog('\n⚠️  WARNING: The laydown detection logic may still have issues!');
        testLog('These hands were identified as laydowns but fail engine verification.');
      }
  });
});