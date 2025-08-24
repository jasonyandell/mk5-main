import { derived, get } from 'svelte/store';
import { EventStore } from '../game/events/store';
import { ProjectionManager } from '../game/projections/manager';
import { CommandProcessor } from '../game/commands/processor';
import { EffectHandler } from '../game/effects/handler';
import type { Command } from '../game/commands/types';
import type { GamePhase, Domino, Play, TrumpSelection, Bid, PlayerType, ConsensusAction } from '../game/types';
import type { TeamState } from '../game/projections/teams';
import type { BidsState } from '../game/projections/bids';
import type { ConsensusState } from '../game/projections/consensus';

class EventGameStore {
  private eventStore: EventStore;
  private projectionManager: ProjectionManager;
  private commandProcessor: CommandProcessor;
  private effectHandler: EffectHandler;

  constructor() {
    this.eventStore = EventStore.fromURL();
    this.projectionManager = new ProjectionManager();
    this.commandProcessor = new CommandProcessor(this.eventStore, this.projectionManager);
    this.effectHandler = new EffectHandler(this.projectionManager, this.commandProcessor);

    this.eventStore.subscribe((event) => {
      this.projectionManager.handleEvent(event);
      this.effectHandler.handleEvent(event);
    });

    // Replay existing events to initialize projections
    const existingEvents = this.eventStore.getEvents();
    for (const event of existingEvents) {
      this.projectionManager.handleEvent(event);
    }

    if (this.eventStore.getEventCount() === 0) {
      this.sendCommand({ type: 'START_GAME' });
    }
  }

  get events() {
    return derived([], () => this.eventStore.getEvents());
  }

  get gamePhase() {
    return this.projectionManager.getProjection<GamePhase>('gamePhase').value;
  }

  get currentPlayer() {
    return this.projectionManager.getProjection<number>('currentPlayer').value;
  }

  get playerTypes() {
    return this.projectionManager.getProjection<PlayerType[]>('playerTypes').value;
  }

  get hands() {
    return this.projectionManager.getProjection<Record<number, Domino[]>>('hands').value;
  }

  get trick() {
    return this.projectionManager.getProjection<Play[]>('trick').value;
  }

  get teams() {
    return this.projectionManager.getProjection<TeamState>('teams').value;
  }

  get bids() {
    return this.projectionManager.getProjection<BidsState>('bids').value;
  }

  get trump() {
    return this.projectionManager.getProjection<TrumpSelection>('trump').value;
  }

  get consensus() {
    return this.projectionManager.getProjection<ConsensusState>('consensus').value;
  }

  get validMoves() {
    return this.projectionManager.getProjection<Map<number, Domino[]>>('validMoves').value;
  }

  getValidMovesForPlayer(playerId: number): Domino[] {
    return this.projectionManager.getValidMovesForPlayer(playerId);
  }

  sendCommand(command: Command): void {
    const result = this.commandProcessor.execute(command);
    if (!result.success) {
      console.error('Command failed:', result.error);
    }
  }

  timeTravel(index: number): void {
    this.eventStore.timeTravel(index);
  }

  clear(): void {
    this.eventStore.clear();
    this.sendCommand({ type: 'START_GAME' });
  }

  enableQuickplay(speed?: 'instant' | 'fast' | 'normal'): void {
    this.sendCommand({ type: 'ENABLE_QUICKPLAY', speed });
  }

  disableQuickplay(): void {
    this.sendCommand({ type: 'DISABLE_QUICKPLAY' });
  }

  placeBid(player: number, bid: Bid): void {
    this.sendCommand({ type: 'PLACE_BID', player, bid });
  }

  pass(player: number): void {
    this.sendCommand({ type: 'PASS', player });
  }

  selectTrump(player: number, trump: TrumpSelection): void {
    this.sendCommand({ type: 'SELECT_TRUMP', player, trump });
  }

  playDomino(player: number, domino: Domino): void {
    this.sendCommand({ type: 'PLAY_DOMINO', player, domino });
  }

  agreeToAction(player: number, action: ConsensusAction): void {
    this.sendCommand({ type: 'AGREE_TO_ACTION', player, action });
  }

  getGameState() {
    try {
      const phase = this.projectionManager.getProjection('gamePhase').getCurrentValue();
      const currentPlayer = this.projectionManager.getProjection('currentPlayer').getCurrentValue();
      const playerTypes = this.projectionManager.getProjection('playerTypes').getCurrentValue();
      const hands = this.projectionManager.getProjection('hands').getCurrentValue();
      const trick = this.projectionManager.getProjection('trick').getCurrentValue();
      const teams = this.projectionManager.getProjection('teams').getCurrentValue();
      const bids = this.projectionManager.getProjection('bids').getCurrentValue();
      const trump = this.projectionManager.getProjection('trump').getCurrentValue();
      const consensusMap = this.projectionManager.getProjection('consensus').getCurrentValue();
      
      return {
        phase,
        currentPlayer,
        playerTypes,
        hands,
        trick,
        teams,
        bids,
        trump,
        consensus: {
          has: (key: string) => consensusMap.has(key as any),
          get: (key: string) => consensusMap.get(key as any)
        }
      };
    } catch (e) {
      console.error('[EventGameStore] Error in getGameState:', e);
      return null;
    }
  }
}

export const eventGame = new EventGameStore();