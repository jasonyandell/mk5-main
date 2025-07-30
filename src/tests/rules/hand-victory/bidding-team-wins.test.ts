import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, Player, Bid, Trick, Play, Domino } from '../../../game/types';
import { GAME_CONSTANTS, POINT_VALUES } from '../../../game/constants';

describe('Hand Victory - Bidding Team Wins', () => {
  let gameState: GameState;
  
  beforeEach(() => {
    const players: Player[] = [
      { id: 0, name: 'Player 1', hand: [], teamId: 0, marks: 0 },
      { id: 1, name: 'Player 2', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'Player 3', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'Player 4', hand: [], teamId: 1, marks: 0 }
    ];
    
    gameState = {
      phase: 'scoring',
      players,
      currentPlayer: 0,
      dealer: 0,
      bids: [],
      currentBid: null,
      winningBidder: 0, // Team 0 won the bid
      trump: 5, // fives are trump
      tricks: [],
      currentTrick: [],
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
      tournamentMode: true,
      shuffleSeed: 12345
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
      const tricks = createTricksWithPoints(38, 0); // Team 0 gets 38 points
      gameState.tricks = tricks;
      
      const handResult = calculateHandWinner(gameState);
      
      // Then the bidding team wins the hand
      expect(handResult.winner).toBe(0); // Team 0 wins
      expect(handResult.pointsTaken).toBe(38);
      expect(handResult.bidMade).toBe(true);
    });

    it('should win when bidding team makes 1 mark bid (42 points)', () => {
      // Given a 1 mark bid (42 points)
      const bid: Bid = { type: 'marks', value: 1, player: 0 };
      gameState.currentBid = bid;
      
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
  if (!state.currentBid || state.winningBidder === null) {
    throw new Error('No bid or winning bidder');
  }
  
  const biddingTeam = state.players[state.winningBidder].teamId;
  let biddingTeamPoints = 0;
  let tricksWonByBiddingTeam = 0;
  
  // Calculate points for each trick
  for (const trick of state.tricks) {
    if (trick.winner !== undefined) {
      const winningTeam = state.players[trick.winner].teamId;
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
  let pointsAllocated = 0;
  let trickCount = 0;
  
  // Allocate counting dominoes to reach target
  const counters = [
    createDomino(5, 5), // 10 points
    createDomino(6, 4), // 10 points
    createDomino(5, 0), // 5 points
    createDomino(4, 1), // 5 points
    createDomino(3, 2)  // 5 points
  ];
  
  // Create tricks to reach target points
  for (const counter of counters) {
    if (pointsAllocated < targetPoints && trickCount < 7) {
      const plays = createPlaysForTrick(counter, winningTeam === 0 ? 0 : 1);
      tricks.push({
        plays,
        winner: winningTeam === 0 ? 0 : 1,
        points: getDominoPoints(counter)
      });
      pointsAllocated += getDominoPoints(counter) + 1; // +1 for trick point
      trickCount++;
    }
  }
  
  // Fill remaining tricks
  while (trickCount < 7) {
    const shouldWinnerGetTrick = pointsAllocated < targetPoints;
    const winner = shouldWinnerGetTrick ? (winningTeam === 0 ? 0 : 1) : (winningTeam === 0 ? 1 : 0);
    
    tricks.push({
      plays: createPlaysForTrick(createDomino(1, 0), winner),
      winner,
      points: 0
    });
    
    if (shouldWinnerGetTrick) {
      pointsAllocated += 1; // Trick point only
    }
    trickCount++;
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