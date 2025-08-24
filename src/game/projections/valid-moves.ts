import { Projection } from './base';
import type { GameEvent } from '../events/types';
import type { Domino, GameState, Play, TrumpSelection } from '../types';
import { getValidPlays } from '../core/rules';
import type { ProjectionManager } from './manager';

export class ValidMovesProjection extends Projection<Map<number, Domino[]>> {
  private manager?: ProjectionManager;

  setManager(manager: ProjectionManager): void {
    this.manager = manager;
  }

  protected getInitialValue(): Map<number, Domino[]> {
    return new Map();
  }

  protected project(current: Map<number, Domino[]>, event: GameEvent): Map<number, Domino[]> {
    if (!this.manager) return current;

    const newMoves = new Map<number, Domino[]>();
    
    for (let playerId = 0; playerId < 4; playerId++) {
      const moves = this.getValidMovesForPlayer(playerId);
      newMoves.set(playerId, moves);
    }
    
    return newMoves;
  }

  private getValidMovesForPlayer(playerId: number): Domino[] {
    if (!this.manager) return [];

    const phase = this.manager.getProjection('gamePhase').getCurrentValue();
    if (phase !== 'playing') return [];

    const currentPlayer = this.manager.getProjection('currentPlayer').getCurrentValue();
    if (currentPlayer !== playerId) return [];

    const hands = this.manager.getProjection('hands').getCurrentValue();
    const playerHand = hands[playerId] || [];
    const trick = this.manager.getProjection('trick').getCurrentValue();
    const trump = this.manager.getProjection('trump').getCurrentValue();

    const minimalState = this.buildMinimalState(playerId, hands, trick, trump);
    
    return getValidPlays(minimalState, playerId);
  }

  private buildMinimalState(
    playerId: number,
    hands: Record<number, Domino[]>,
    trick: Play[],
    trump: TrumpSelection
  ): GameState {
    const players = [0, 1, 2, 3].map(id => ({
      id,
      name: `Player ${id + 1}`,
      hand: hands[id] || [],
      teamId: (id % 2) as 0 | 1,
      marks: 0
    }));

    let currentSuit = -1;
    if (trick.length > 0) {
      const leadDomino = trick[0].domino;
      if (trump.type === 'doubles' && leadDomino.high === leadDomino.low) {
        currentSuit = 7;
      } else if (trump.type === 'suit' && trump.suit !== undefined) {
        if (leadDomino.high === trump.suit || leadDomino.low === trump.suit) {
          currentSuit = trump.suit;
        } else {
          currentSuit = leadDomino.high;
        }
      } else {
        currentSuit = leadDomino.high;
      }
    }

    return {
      phase: 'playing',
      players,
      currentPlayer: playerId,
      dealer: 0,
      bids: [],
      currentBid: { type: 'pass', player: -1 },
      winningBidder: -1,
      trump,
      tricks: [],
      currentTrick: trick,
      currentSuit,
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: 7,
      tournamentMode: false,
      shuffleSeed: 0,
      playerTypes: ['human', 'human', 'human', 'human'],
      consensus: {
        completeTrick: new Set(),
        scoreHand: new Set()
      },
      actionHistory: [],
      aiSchedule: {},
      currentTick: 0
    };
  }
}