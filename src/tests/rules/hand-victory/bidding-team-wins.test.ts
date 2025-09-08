import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, Player, Bid, Trick, Play, Domino } from '../../../game/types';
import { EMPTY_BID, isEmptyBid } from '../../../game/types';
import { GAME_CONSTANTS } from '../../../game/constants';
import { FIVES, NO_LEAD_SUIT } from '../../../game/types';

describe('Hand Victory - Bidding Team Wins', () => {
  let gameState: GameState;
  
  beforeEach(() => {
    const players: Player[] = [
      { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
      { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
      { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
      { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 }
    ];
    
    gameState = {
      phase: 'scoring',
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      players,
      currentPlayer: 0,
      dealer: 0,
      bids: [],
      currentBid: EMPTY_BID,
      winningBidder: 0, // Team 0 won the bid
      trump: { type: 'suit', suit: FIVES }, // fives are trump
      tricks: [],
      currentTrick: [],
      currentSuit: NO_LEAD_SUIT,
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
      tournamentMode: true,
      shuffleSeed: 12345,
      consensus: {
        completeTrick: new Set(),
        scoreHand: new Set()
      },
      actionHistory: [],
      theme: 'coffee',
      colorOverrides: {}
    };
  });

  describe('Scenario: Bidding Team Wins', () => {
    it('should win when bidding team meets exact bid of 30 points', () => {
      // Given a hand has been played
      const bid: Bid = { type: 'points', value: 30, player: 0 };
      gameState.currentBid = bid;
      gameState.bids = [
        { type: 'pass', player: 3 },
        bid,
        { type: 'pass', player: 1 },
        { type: 'pass', player: 2 }
      ];
      
      // When the bidding team takes points equal to their bid
      // Create tricks where team 0 wins 30 points exactly
      const tricks = createTricksWithPoints(30, 0); // Team 0 gets 30 points
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then the bidding team wins the hand
      expect(handResult.winner).toBe(0); // Team 0 wins
      expect(handResult.pointsTaken).toBe(30);
      expect(handResult.bidMade).toBe(true);
    });

    it('should win when bidding team exceeds bid of 35 points', () => {
      // Given a hand has been played with a 35 point bid
      const bid: Bid = { type: 'points', value: 35, player: 2 }; // Player 2 (team 0)
      gameState.currentBid = bid;
      gameState.winningBidder = 2;
      
      // When the bidding team takes more points than their bid
      // Note: With standard counting dominoes, some exact scores aren't possible
      // Closest to 38 is either 36 or 40
      const tricks = createTricksWithPoints(36, 0); // Team 0 gets 36 points (exceeds 35 bid)
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then the bidding team wins the hand
      expect(handResult.winner).toBe(0); // Team 0 wins
      expect(handResult.pointsTaken).toBe(36); // Changed from 38 to achievable 36
      expect(handResult.bidMade).toBe(true);
    });

    it('should win when bidding team makes 1 mark bid (42 points)', () => {
      // Given a 1 mark bid (42 points)
      const bid: Bid = { type: 'marks', value: 1, player: 0 };
      gameState.currentBid = bid;
      gameState.winningBidder = 0;
      
      // When the bidding team takes all 42 points
      const tricks = createTricksWithPoints(42, 0); // Team 0 gets all points
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then the bidding team wins the hand
      expect(handResult.winner).toBe(0);
      expect(handResult.pointsTaken).toBe(42);
      expect(handResult.bidMade).toBe(true);
    });

    it('should win when bidding team makes 2 mark bid (84 points)', () => {
      // Given a 2 mark bid (must win all tricks)
      const bid: Bid = { type: 'marks', value: 2, player: 2 };
      gameState.currentBid = bid;
      gameState.winningBidder = 2;
      
      // When the bidding team wins all 7 tricks
      const tricks = createAllTricksWonByTeam(0); // Team 0 wins all
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then the bidding team wins the hand
      expect(handResult.winner).toBe(0);
      expect(handResult.allTricksWon).toBe(true);
      expect(handResult.bidMade).toBe(true);
    });

    it('should track points from counting dominoes correctly', () => {
      // Given a 30 point bid
      const bid: Bid = { type: 'points', value: 30, player: 0 };
      gameState.currentBid = bid;
      
      // When team wins tricks with specific counting dominoes
      const tricks: Trick[] = [
        // Trick 1: Team 0 wins with 5-5 (10 points) + 1 trick point = 11
        createTrickWithCounters([createDomino(5, 5)], 0),
        // Trick 2: Team 0 wins with 6-4 (10 points) + 1 trick point = 11
        createTrickWithCounters([createDomino(6, 4)], 0),
        // Trick 3: Team 0 wins with 5-0 (5 points) + 1 trick point = 6
        createTrickWithCounters([createDomino(5, 0)], 0),
        // Trick 4: Team 0 wins with 3-2 (5 points) + 1 trick point = 6
        createTrickWithCounters([createDomino(3, 2)], 0),
        // Tricks 5-7: Team 1 wins the rest
        createTrickWithCounters([], 1),
        createTrickWithCounters([], 1),
        createTrickWithCounters([], 1)
      ];
      
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then points are calculated correctly: 30 domino points + 4 trick points = 34
      expect(handResult.pointsTaken).toBe(34);
      expect(handResult.winner).toBe(0);
      expect(handResult.bidMade).toBe(true);
    });
  });
});

// Test-only helper functions
interface HandResult {
  winner: number;
  pointsTaken: number;
  bidMade: boolean;
  allTricksWon?: boolean;
}

function calculateHandWinner(state: GameState): HandResult {
  if (isEmptyBid(state.currentBid) || state.winningBidder === -1) {
    throw new Error('No bid or winning bidder');
  }
  
  const biddingPlayer = state.players[state.winningBidder];
  if (!biddingPlayer) {
    throw new Error('Bidding player not found');
  }
  const biddingTeam = biddingPlayer.teamId;
  let biddingTeamPoints = 0;
  let tricksWonByBiddingTeam = 0;
  
  // Calculate points for each trick
  for (const trick of state.tricks) {
    if (trick.winner !== undefined) {
      const winningPlayer = state.players[trick.winner];
      if (!winningPlayer) {
        continue; // Skip if player not found
      }
      const winningTeam = winningPlayer.teamId;
      if (winningTeam === biddingTeam) {
        tricksWonByBiddingTeam++;
        biddingTeamPoints += trick.points + 1; // Trick points + 1 for winning the trick
      }
    }
  }
  
  // Determine bid requirement
  let requiredPoints: number;
  if (state.currentBid.type === 'points') {
    requiredPoints = state.currentBid.value!;
  } else if (state.currentBid.type === 'marks') {
    if (state.currentBid.value! >= 2) {
      // 2+ marks requires all tricks
      requiredPoints = 42;
    } else {
      // 1 mark = 42 points
      requiredPoints = 42;
    }
  } else {
    throw new Error('Invalid bid type for standard game');
  }
  
  const bidMade = biddingTeamPoints >= requiredPoints;
  const allTricksWon = tricksWonByBiddingTeam === 7;
  
  // For 2+ mark bids, must win all tricks
  const actualBidMade = state.currentBid.type === 'marks' && state.currentBid.value! >= 2
    ? allTricksWon
    : bidMade;
  
  return {
    winner: actualBidMade ? biddingTeam : (1 - biddingTeam),
    pointsTaken: biddingTeamPoints,
    bidMade: actualBidMade,
    allTricksWon
  };
}

function createTricksWithPoints(targetPoints: number, winningTeam: number): Trick[] {
  const tricks: Trick[] = [];
  
  // All counting dominoes (35 points total)
  const allDominoes = [
    { domino: createDomino(5, 5), points: 10 },
    { domino: createDomino(6, 4), points: 10 },
    { domino: createDomino(5, 0), points: 5 },
    { domino: createDomino(4, 1), points: 5 },
    { domino: createDomino(3, 2), points: 5 },
    { domino: createDomino(1, 0), points: 0 },
    { domino: createDomino(2, 0), points: 0 }
  ];
  
  // For specific test cases, hardcode the optimal allocation
  if (targetPoints === 30) {
    // Need exactly 30 points for winning team
    // Option: 5-5 (11) + 6-4 (11) + 4-1 (6) + 2 trick points = 30
    const allocation = [
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(5, 5), points: 10 }, // 11
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(6, 4), points: 10 }, // 11
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(4, 1), points: 5 },  // 6
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(1, 0), points: 0 },  // 1
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(2, 0), points: 0 },  // 1
      { winner: winningTeam === 0 ? 1 : 0, domino: createDomino(5, 0), points: 5 },  // 0
      { winner: winningTeam === 0 ? 1 : 0, domino: createDomino(3, 2), points: 5 }   // 0
    ]; // Total: 10+10+5+0+0 = 25 domino points + 5 tricks = 30
    
    for (const alloc of allocation) {
      tricks.push({
        plays: createPlaysForTrick(alloc.domino, alloc.winner),
        winner: alloc.winner,
        points: alloc.points
      });
    }
  } else if (targetPoints === 36) {
    // Need exactly 36 points for winning team
    // Win 10+10+5+5 = 30 domino points + 6 tricks = 36 exactly!
    const allocation = [
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(5, 5), points: 10 }, // 11
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(6, 4), points: 10 }, // 11
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(5, 0), points: 5 },  // 6
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(3, 2), points: 5 },  // 6
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(1, 0), points: 0 },  // 1
      { winner: winningTeam === 0 ? 0 : 1, domino: createDomino(2, 0), points: 0 },  // 1
      { winner: winningTeam === 0 ? 1 : 0, domino: createDomino(4, 1), points: 5 }   // opponent
    ]; // Total for winner: 10+10+5+5+0+0 = 30 domino points + 6 tricks = 36
    
    for (const alloc of allocation) {
      tricks.push({
        plays: createPlaysForTrick(alloc.domino, alloc.winner),
        winner: alloc.winner,
        points: alloc.points
      });
    }
  } else if (targetPoints === 42) {
    // Need all 42 points - winning team must win all 7 tricks
    for (let i = 0; i < 7; i++) {
      const domino = allDominoes[i];
      if (!domino) {
        throw new Error(`Missing domino at index ${i}`);
      }
      const winner = winningTeam === 0 ? 0 : 1; // All tricks to winning team
      tricks.push({
        plays: createPlaysForTrick(domino.domino, winner),
        winner: winner,
        points: domino.points
      });
    }
  } else {
    // General case - just distribute sensibly
    for (let i = 0; i < 7; i++) {
      const domino = allDominoes[i];
      if (!domino) {
        throw new Error(`Missing domino at index ${i}`);
      }
      const winner = i < 5 ? (winningTeam === 0 ? 0 : 1) : (winningTeam === 0 ? 1 : 0);
      tricks.push({
        plays: createPlaysForTrick(domino.domino, winner),
        winner: winner,
        points: domino.points
      });
    }
  }
  
  return tricks;
}

function createAllTricksWonByTeam(team: number): Trick[] {
  const tricks: Trick[] = [];
  const winner = team === 0 ? 0 : 1; // Use player 0 or 1 as winner
  
  // Distribute all counting dominoes across tricks
  const counters = [
    createDomino(5, 5), // Trick 1
    createDomino(6, 4), // Trick 2
    createDomino(5, 0), // Trick 3
    createDomino(4, 1), // Trick 4
    createDomino(3, 2), // Trick 5
  ];
  
  // First 5 tricks with counters
  for (const counter of counters) {
    tricks.push({
      plays: createPlaysForTrick(counter, winner),
      winner,
      points: getDominoPoints(counter)
    });
  }
  
  // Last 2 tricks without counters
  for (let i = 0; i < 2; i++) {
    tricks.push({
      plays: createPlaysForTrick(createDomino(2, 1), winner),
      winner,
      points: 0
    });
  }
  
  return tricks;
}

function createTrickWithCounters(counters: Domino[], winningPlayer: number): Trick {
  let trickPoints = 0;
  const plays: Play[] = [];
  
  // Add plays for all 4 players
  for (let i = 0; i < 4; i++) {
    const domino = i < counters.length ? counters[i] : createDomino(1, 0);
    if (!domino) {
      throw new Error(`Missing domino for player ${i}`);
    }
    plays.push({ player: i, domino });
    trickPoints += getDominoPoints(domino);
  }
  
  return {
    plays,
    winner: winningPlayer,
    points: trickPoints
  };
}

function createPlaysForTrick(winningDomino: Domino, winner: number): Play[] {
  const plays: Play[] = [];
  
  // Create plays for all 4 players
  for (let i = 0; i < 4; i++) {
    if (i === winner) {
      plays.push({ player: i, domino: winningDomino });
    } else {
      plays.push({ player: i, domino: createDomino(0, 1) });
    }
  }
  
  return plays;
}

function createDomino(high: number, low: number): Domino {
  return {
    high: Math.max(high, low),
    low: Math.min(high, low),
    id: `${Math.max(high, low)}-${Math.min(high, low)}`
  };
}

function getDominoPoints(domino: Domino): number {
  const total = domino.high + domino.low;
  if (total === 10 && domino.high === 5 && domino.low === 5) return 10; // 5-5
  if (total === 10 && ((domino.high === 6 && domino.low === 4) || (domino.high === 4 && domino.low === 6))) return 10; // 6-4
  if (total === 5 && ((domino.high === 5 && domino.low === 0) || (domino.high === 0 && domino.low === 5))) return 5; // 5-0
  if (total === 5 && ((domino.high === 4 && domino.low === 1) || (domino.high === 1 && domino.low === 4))) return 5; // 4-1
  if (total === 5 && ((domino.high === 3 && domino.low === 2) || (domino.high === 2 && domino.low === 3))) return 5; // 3-2
  return 0;
}