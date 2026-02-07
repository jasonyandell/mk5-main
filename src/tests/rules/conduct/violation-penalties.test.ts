import { describe, it, expect } from 'vitest';
import type { GameState } from '../../../game/types';

interface Violation {
  player: number;
  type: string;
  description: string;
  timestamp: number;
  penalty?: string;
  severity?: string;
  penaltyApplied?: boolean;
}

interface GameStateWithViolations extends Partial<GameState> {
  violations: Violation[];
  applied?: boolean;
  type?: string;
}

describe('Feature: Tournament Conduct', () => {
  describe('Scenario: Violation Penalties', () => {
    // Test setup for tracking violations
    const setupGameWithViolations = (): GameStateWithViolations => {
      return {
        phase: 'playing',
        currentPlayer: 0,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 }
        ],
        teamMarks: [3, 2],
        violations: []
      };
    };

    it('should result in a warning for first offense', () => {
      // Given a player has violated communication rules
      const gameState = setupGameWithViolations();
      
      // When penalties are assessed
      const firstViolation = {
        player: 0,
        type: 'communication',
        severity: 'minor',
        description: 'voice inflection during bid',
        timestamp: Date.now()
      };
      
      gameState.violations.push(firstViolation);
      
      // Then first offense results in a warning
      const penalty = assessPenalty(gameState.violations, 0);
      expect(penalty.type).toBe('warning');
      expect(penalty.marksAwarded).toBe(0);
      expect(penalty.message).toContain('warning');
    });

    it('should result in a mark to opponents for second offense', () => {
      // Given a player has already received a warning
      const gameState = setupGameWithViolations();
      
      const firstViolation = {
        player: 1,
        type: 'communication',
        severity: 'minor',
        description: 'gesture during play',
        timestamp: Date.now() - 10000,
        penaltyApplied: true
      };
      
      const secondViolation = {
        player: 1,
        type: 'communication',
        severity: 'minor',
        description: 'table talk',
        timestamp: Date.now()
      };
      
      gameState.violations.push(firstViolation, secondViolation);
      
      // When penalties are assessed
      // Then second offense results in a mark to opponents
      const penalty = assessPenalty(gameState.violations, 1);
      expect(penalty.type).toBe('mark_penalty');
      expect(penalty.marksAwarded).toBe(1);
      expect(penalty.awardedToTeam).toBe(0); // Opposing team (player 1 is on team 1)
    });

    it('should result in ejection for severe violations', () => {
      // Given a player has committed a severe violation
      const gameState = setupGameWithViolations();
      
      const severeViolation = {
        player: 2,
        type: 'communication',
        severity: 'severe',
        description: 'deliberate signaling system',
        timestamp: Date.now()
      };
      
      gameState.violations.push(severeViolation);
      
      // When penalties are assessed
      // Then severe violations result in ejection from game/tournament
      const penalty = assessPenalty(gameState.violations, 2);
      expect(penalty.type).toBe('ejection');
      expect(penalty.gameTerminated).toBe(true);
      expect(penalty.reason).toContain('severe violation');
    });

    it('should track violations across the entire game', () => {
      // Given multiple violations have occurred during a game
      const gameState = setupGameWithViolations();
      
      const violations = [
        { player: 0, type: 'communication', severity: 'minor', phase: 'bidding', description: 'Violation 1', timestamp: Date.now() },
        { player: 1, type: 'communication', severity: 'minor', phase: 'playing', description: 'Violation 2', timestamp: Date.now() },
        { player: 0, type: 'communication', severity: 'minor', phase: 'playing', description: 'Violation 3', timestamp: Date.now() },
        { player: 3, type: 'communication', severity: 'moderate', phase: 'bidding', description: 'Violation 4', timestamp: Date.now() }
      ];
      
      violations.forEach(v => gameState.violations.push(v));
      
      // Then the system should track all violations
      expect(gameState.violations.length).toBe(4);
      
      // And should correctly count violations per player
      const player0Violations = gameState.violations.filter((v: Violation) => v.player === 0);
      expect(player0Violations.length).toBe(2);
      
      // And should identify repeat offenders
      const repeatOffenders = identifyRepeatOffenders(gameState.violations);
      expect(repeatOffenders).toContain(0);
      expect(repeatOffenders).not.toContain(1);
      expect(repeatOffenders).not.toContain(3);
    });

    it('should differentiate between violation severities', () => {
      // Given violations of different severities
      setupGameWithViolations();
      
      const minorViolation = {
        player: 0,
        type: 'communication',
        severity: 'minor',
        description: 'slight hesitation',
        timestamp: Date.now()
      };
      
      const moderateViolation = {
        player: 1,
        type: 'communication',
        severity: 'moderate',
        description: 'obvious gesture to partner',
        timestamp: Date.now()
      };
      
      const severeViolation = {
        player: 2,
        type: 'communication',
        severity: 'severe',
        description: 'pre-arranged signal system',
        timestamp: Date.now()
      };
      
      // Then penalties should scale with severity
      const minorPenalty = assessPenaltyForViolation(minorViolation, []);
      const moderatePenalty = assessPenaltyForViolation(moderateViolation, []);
      const severePenalty = assessPenaltyForViolation(severeViolation, []);
      
      expect(minorPenalty.type).toBe('warning');
      expect(moderatePenalty.type).toBe('mark_penalty');
      expect(severePenalty.type).toBe('ejection');
    });

    it('should apply penalties immediately when detected', () => {
      // Given a violation is detected during play
      const gameState = setupGameWithViolations();
      if (!gameState.teamMarks) {
        throw new Error('teamMarks should be defined for this test');
      }
      const initialMarks = [...gameState.teamMarks];
      
      // When a second offense is committed
      gameState.violations.push(
        { player: 0, type: 'communication', severity: 'minor', penaltyApplied: true, description: 'First violation', timestamp: Date.now() },
        { player: 0, type: 'communication', severity: 'minor', description: 'Second violation', timestamp: Date.now() }
      );
      
      // Then the penalty should be applied immediately
      const penalty = assessAndApplyPenalty(gameState, 0);
      
      expect(penalty.applied).toBe(true);
      expect(penalty.type).toBe('mark_penalty');
      
      // And team marks should be updated
      if (penalty.type === 'mark_penalty' && gameState.teamMarks && initialMarks) {
        const expectedMarks = [...initialMarks];
        const team1Mark = expectedMarks[1];
        if (team1Mark !== undefined) {
          expectedMarks[1] = team1Mark + 1; // Team 1 gets a mark (player 0 is on team 0)
        }
        expect(gameState.teamMarks).toEqual(expectedMarks);
      }
    });
  });
});

// Helper functions for test implementation
function assessPenalty(violations: Violation[], playerId: number): { type: string; marksAwarded?: number; message?: string; gameTerminated?: boolean; reason?: string; awardedToTeam?: number } {
  const playerViolations = violations.filter(v => v.player === playerId);
  const priorPenalties = playerViolations.filter(v => v.penaltyApplied).length;
  
  if (playerViolations.some(v => v.severity === 'severe')) {
    return {
      type: 'ejection',
      gameTerminated: true,
      reason: 'severe violation'
    };
  }
  
  if (priorPenalties === 0) {
    return {
      type: 'warning',
      marksAwarded: 0,
      message: 'First offense - warning issued'
    };
  }
  
  if (priorPenalties === 1) {
    return {
      type: 'mark_penalty',
      marksAwarded: 1,
      awardedToTeam: playerId === 0 || playerId === 2 ? 1 : 0
    };
  }
  
  return {
    type: 'ejection',
    gameTerminated: true,
    reason: 'multiple violations'
  };
}

function identifyRepeatOffenders(violations: Violation[]): number[] {
  const violationCounts = violations.reduce((acc, v) => {
    acc[v.player] = (acc[v.player] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  
  return Object.entries(violationCounts)
    .filter(([_, count]) => (count as number) > 1)
    .map(([player]) => parseInt(player));
}

function assessPenaltyForViolation(violation: Violation, priorViolations: Violation[]): { type: string; marksAwarded?: number; awardedToTeam?: number } {
  const playerPriors = priorViolations.filter(v => v.player === violation.player);
  
  if (violation.severity === 'severe') {
    return { type: 'ejection' };
  }
  
  if (violation.severity === 'moderate' && playerPriors.length === 0) {
    return { type: 'mark_penalty' };
  }
  
  if (playerPriors.length === 0) {
    return { type: 'warning' };
  }
  
  return { type: 'mark_penalty' };
}

function assessAndApplyPenalty(gameState: GameStateWithViolations, playerId: number): { type: string; applied: boolean; marksAwarded?: number } {
  const penalty = assessPenalty(gameState.violations, playerId);
  
  if (penalty.type === 'mark_penalty' && gameState.teamMarks && penalty.marksAwarded && gameState.players) {
    const player = gameState.players.find((p) => p.id === playerId);
    if (!player) {
      throw new Error(`Player ${playerId} not found`);
    }
    const opposingTeam = player.teamId === 0 ? 1 : 0;
    gameState.teamMarks[opposingTeam] += penalty.marksAwarded;
  }
  
  return { type: penalty.type, applied: true, marksAwarded: penalty.marksAwarded ?? 0 };
}