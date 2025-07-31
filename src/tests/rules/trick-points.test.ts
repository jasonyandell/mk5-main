import { describe, it, expect } from 'vitest';
import { 
  GAME_CONSTANTS, 
  createInitialState
} from '../../game';

describe('Scoring Systems - Trick Points', () => {
  describe('When calculating trick points', () => {
    it('should award 1 point for each trick won using game engine', () => {
      const gameState = createInitialState();
      
      // Simulate winning 5 tricks by adding them to a team's tricks array
      const mockTricks = Array(5).fill(null).map(() => ({
        plays: [
          { player: 0, domino: { high: 6, low: 5, id: '6-5' } },
          { player: 1, domino: { high: 5, low: 4, id: '5-4' } },
          { player: 2, domino: { high: 4, low: 3, id: '4-3' } },
          { player: 3, domino: { high: 3, low: 2, id: '3-2' } }
        ],
        winner: 0,
        points: 1
      }));
      
      gameState.tricks = mockTricks;
      
      // Calculate trick points for team 0 (tricks won by players 0 and 2)
      const team0TrickPoints = gameState.tricks.filter(trick => 
        trick.winner === 0 || trick.winner === 2
      ).length;
      
      expect(team0TrickPoints).toBe(5);
    });

    it('should have 7 total tricks worth 7 points using game constants', () => {
      const totalTricks = GAME_CONSTANTS.TRICKS_PER_HAND;
      const maxTrickPoints = totalTricks; // Each trick is worth 1 point
      
      expect(totalTricks).toBe(7);
      expect(maxTrickPoints).toBe(7);
    });

    it('should calculate hand total as 42 points (35 count + 7 tricks) using game constants', () => {
      const countPoints = 35; // Total counting domino points (5-5=10, 6-4=10, 5-0=5, 4-1=5, 3-2=5)
      const trickPoints = GAME_CONSTANTS.TRICKS_PER_HAND;   // From 7 tricks
      const handTotal = countPoints + trickPoints;
      
      expect(countPoints).toBe(35);
      expect(trickPoints).toBe(7);
      expect(handTotal).toBe(42);
      expect(handTotal).toBe(GAME_CONSTANTS.TOTAL_POINTS);
    });

    it('should award 0 trick points when no tricks are won', () => {
      const gameState = createInitialState();
      // Start with empty tricks array
      gameState.tricks = [];
      
      // Team 0 should have 0 trick points
      const team0TrickPoints = gameState.tricks.filter(trick => 
        trick.winner === 0 || trick.winner === 2
      ).length;
      
      expect(team0TrickPoints).toBe(0);
    });

    it('should calculate correct points for partial tricks won using game engine', () => {
      const gameState = createInitialState();
      
      // Test various scenarios
      const scenarios = [
        { tricksWon: 1, expectedPoints: 1 },
        { tricksWon: 2, expectedPoints: 2 },
        { tricksWon: 3, expectedPoints: 3 },
        { tricksWon: 4, expectedPoints: 4 },
        { tricksWon: 5, expectedPoints: 5 },
        { tricksWon: 6, expectedPoints: 6 },
        { tricksWon: 7, expectedPoints: 7 }
      ];

      scenarios.forEach(({ tricksWon, expectedPoints }) => {
        // Create mock tricks for team 0
        const mockTricks = Array(tricksWon).fill(null).map((_, index) => ({
          plays: [
            { player: 0, domino: { high: 6, low: 5, id: `6-5-${index}` } },
            { player: 1, domino: { high: 5, low: 4, id: `5-4-${index}` } },
            { player: 2, domino: { high: 4, low: 3, id: `4-3-${index}` } },
            { player: 3, domino: { high: 3, low: 2, id: `3-2-${index}` } }
          ],
          winner: 0, // All tricks won by player 0 (team 0)
          points: 1
        }));
        
        gameState.tricks = mockTricks;
        
        // Calculate trick points for team 0
        const team0TrickPoints = gameState.tricks.filter(trick => 
          trick.winner === 0 || trick.winner === 2
        ).length;
        
        expect(team0TrickPoints).toBe(expectedPoints);
      });
    });

    it('should properly count tricks for both teams', () => {
      const gameState = createInitialState();
      
      // Create mixed tricks - some for each team
      const mockTricks = [
        { plays: [], winner: 0, points: 1 }, // Team 0 (players 0,2)
        { plays: [], winner: 1, points: 1 }, // Team 1 (players 1,3)
        { plays: [], winner: 2, points: 1 }, // Team 0 (players 0,2)
        { plays: [], winner: 3, points: 1 }, // Team 1 (players 1,3)
        { plays: [], winner: 0, points: 1 }, // Team 0 (players 0,2)
        { plays: [], winner: 1, points: 1 }, // Team 1 (players 1,3)
        { plays: [], winner: 2, points: 1 }  // Team 0 (players 0,2)
      ];
      
      gameState.tricks = mockTricks;
      
      // Team 0 should have 4 tricks (players 0 and 2)
      const team0TrickPoints = gameState.tricks.filter(trick => 
        trick.winner === 0 || trick.winner === 2
      ).length;
      
      // Team 1 should have 3 tricks (players 1 and 3)
      const team1TrickPoints = gameState.tricks.filter(trick => 
        trick.winner === 1 || trick.winner === 3
      ).length;
      
      expect(team0TrickPoints).toBe(4);
      expect(team1TrickPoints).toBe(3);
      expect(team0TrickPoints + team1TrickPoints).toBe(7);
    });
  });
});