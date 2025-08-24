import type { EventEnvelope } from '../events/types';
import type { Projection } from './base';
import type { Domino } from '../types';
import { GamePhaseProjection } from './game-phase';
import { PlayerTypesProjection } from './player-types';
import { HandsProjection } from './hands';
import { TrickProjection } from './trick';
import { TrumpProjection } from './trump';
import { CurrentPlayerProjection } from './current-player';
import { TeamsProjection } from './teams';
import { BidsProjection } from './bids';
import { CompletedTricksProjection } from './completed-tricks';
import { ConsensusProjection } from './consensus';
import { ValidMovesProjection } from './valid-moves';

export class ProjectionManager {
  private projections: Map<string, Projection<any>> = new Map();

  constructor() {
    this.registerProjections();
  }

  private registerProjections(): void {
    // Create projections with their own initial values
    const gamePhase = new GamePhaseProjection('setup');
    const playerTypes = new PlayerTypesProjection(['human', 'human', 'human', 'human']);
    const hands = new HandsProjection({});
    const trick = new TrickProjection([]);
    const trump = new TrumpProjection({ type: 'none' });
    const currentPlayer = new CurrentPlayerProjection(0);
    const teams = new TeamsProjection({ trickPoints: [0, 0], marks: [0, 0] });
    const bids = new BidsProjection({ bids: [], highestBid: null, highestBidder: -1, passes: new Set() });
    const completedTricks = new CompletedTricksProjection([]);
    const consensus = new ConsensusProjection(new Map());
    const validMoves = new ValidMovesProjection(new Map());
    
    validMoves.setManager(this);
    
    this.projections.set('gamePhase', gamePhase);
    this.projections.set('playerTypes', playerTypes);
    this.projections.set('hands', hands);
    this.projections.set('trick', trick);
    this.projections.set('trump', trump);
    this.projections.set('currentPlayer', currentPlayer);
    this.projections.set('teams', teams);
    this.projections.set('bids', bids);
    this.projections.set('completedTricks', completedTricks);
    this.projections.set('consensus', consensus);
    this.projections.set('validMoves', validMoves);
  }

  handleEvent(event: EventEnvelope): void {
    for (const projection of this.projections.values()) {
      projection.handleEvent(event);
    }
  }

  getProjection<T>(name: string): Projection<T> {
    const projection = this.projections.get(name);
    if (!projection) {
      throw new Error(`Projection '${name}' not found`);
    }
    return projection;
  }

  getValidMovesForPlayer(playerId: number): Domino[] {
    const validMoves = this.getProjection<Map<number, Domino[]>>('validMoves');
    const moves = validMoves.getCurrentValue();
    return moves.get(playerId) || [];
  }

  getAllProjections(): Map<string, Projection<any>> {
    return new Map(this.projections);
  }
}