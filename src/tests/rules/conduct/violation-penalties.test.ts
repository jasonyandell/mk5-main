import { describe, it, expect } from 'vitest';
import type { GameState } from '../../../game/types';

describe('Feature: Tournament Conduct', () => {
  describe('Scenario: Violation Penalties', () => {
    // Test setup for tracking violations
    const setupGameWithViolations = (): Partial<GameState> & { violations: any[] } => {
      return {
        phase: 'playing',
        currentPlayer: 0,
        tournamentMode: true,
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 }
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
        penaltyApplied: { type: 'warning', marksAwarded: 0 }
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
        { player: 0, type: 'communication', severity: 'minor', phase: 'bidding' },
        { player: 1, type: 'communication', severity: 'minor', phase: 'playing' },
        { player: 0, type: 'communication', severity: 'minor', phase: 'playing' },
        { player: 3, type: 'communication', severity: 'moderate', phase: 'bidding' }
      ];
      
      violations.forEach(v => gameState.violations.push(v));
      
      // Then the system should track all violations
      expect(gameState.violations.length).toBe(4);
      
      // And should correctly count violations per player
      const player0Violations = gameState.violations.filter((v: any) => v.player === 0);
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
        description: 'slight hesitation'
      };
      
      const moderateViolation = {
        player: 1,
        type: 'communication',
        severity: 'moderate',
        description: 'obvious gesture to partner'
      };
      
      const severeViolation = {
        player: 2,
        type: 'communication',
        severity: 'severe',
        description: 'pre-arranged signal system'
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
      const initialMarks = [...gameState.teamMarks!];
      
      // When a second offense is committed
      gameState.violations.push(
        { player: 0, type: 'communication', severity: 'minor', penaltyApplied: { type: 'warning' } },
        { player: 0, type: 'communication', severity: 'minor' }
      );
      
      // Then the penalty should be applied immediately
      const penalty = assessAndApplyPenalty(gameState, 0);
      
      expect(penalty.applied).toBe(true);
      expect(penalty.type).toBe('mark_penalty');
      
      // And team marks should be updated
      if (penalty.type === 'mark_penalty') {
        const expectedMarks = [...initialMarks];
        expectedMarks[1] += 1; // Team 1 gets a mark (player 0 is on team 0)
        expect(gameState.teamMarks).toEqual(expectedMarks);
      }
    });
  });
});

// Helper functions for test implementation
function assessPenalty(violations: any[], playerId: number): any {
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

function identifyRepeatOffenders(violations: any[]): number[] {
  const violationCounts = violations.reduce((acc, v) => {
    acc[v.player] = (acc[v.player] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  
  return Object.entries(violationCounts)
    .filter(([_, count]) => (count as number) > 1)
    .map(([player]) => parseInt(player));
}

function assessPenaltyForViolation(violation: any, priorViolations: any[]): any {
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

function assessAndApplyPenalty(gameState: any, playerId: number): any {
  const penalty = assessPenalty(gameState.violations, playerId);
  
  if (penalty.type === 'mark_penalty' && gameState.teamMarks) {
    const playerTeam = gameState.players.find((p: any) => p.id === playerId)?.teamId;
    const opposingTeam = playerTeam === 0 ? 1 : 0;
    gameState.teamMarks[opposingTeam] += penalty.marksAwarded;
  }
  
  return { ...penalty, applied: true };
}