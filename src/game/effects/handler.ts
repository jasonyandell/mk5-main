import type { EventEnvelope } from '../events/types';
import type { Command } from '../commands/types';
import type { ProjectionManager } from '../projections/manager';
import type { CommandProcessor } from '../commands/processor';
import type { Effect, QuickplaySettings, ScheduledEffect } from './types';
import { getAIAction } from '../core/ai-action';

export class EffectHandler {
  private scheduled: Map<string, ScheduledEffect> = new Map();
  private quickplay: QuickplaySettings = {
    enabled: false,
    speed: 'normal'
  };

  constructor(
    private projections: ProjectionManager,
    private commandProcessor: CommandProcessor
  ) {}

  handleEvent(event: EventEnvelope): void {
    if (event.payload.type === 'RESET') {
      this.cancelAll();
      return;
    }

    switch (event.payload.type) {
      case 'QUICKPLAY_ENABLED':
        this.quickplay.enabled = true;
        break;
      case 'QUICKPLAY_DISABLED':
        this.quickplay.enabled = false;
        this.cancelAll();
        break;
      case 'QUICKPLAY_SPEED_SET':
        this.quickplay.speed = event.payload.speed;
        break;
    }

    if (this.quickplay.enabled) {
      this.scheduleEffectsForEvent(event);
    }
  }

  private scheduleEffectsForEvent(event: EventEnvelope): void {
    const delay = this.getDelay();
    
    switch (event.payload.type) {
      case 'GAME_STARTED':
      case 'REDEAL_INITIATED':
        this.scheduleAIIfNeeded(1, delay);
        break;
      
      case 'BID_PLACED':
      case 'PLAYER_PASSED':
        const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
        this.scheduleAIIfNeeded(currentPlayer, delay);
        break;
      
      case 'BIDDING_COMPLETED':
        this.scheduleAIIfNeeded(event.payload.winner, delay);
        break;
      
      case 'TRUMP_SELECTED':
        const dealer = this.getDealer();
        this.scheduleAIIfNeeded((dealer + 1) % 4, delay);
        break;
      
      case 'DOMINO_PLAYED':
        const trick = this.projections.getProjection('trick').getCurrentValue();
        if (trick.length < 4) {
          const nextPlayer = (event.payload.player + 1) % 4;
          this.scheduleAIIfNeeded(nextPlayer, delay);
        }
        break;
      
      case 'CONSENSUS_REQUESTED':
        for (let i = 0; i < 4; i++) {
          if (this.isAI(i)) {
            this.scheduleConsensusAI(i, event.payload.action, delay);
          }
        }
        break;
      
      case 'TRICK_COMPLETED':
        const phase = this.projections.getProjection('gamePhase').getCurrentValue();
        if (phase === 'playing') {
          this.scheduleAIIfNeeded(event.payload.winner, delay);
        }
        break;
      
      case 'HAND_SCORED':
        const gamePhase = this.projections.getProjection('gamePhase').getCurrentValue();
        if (gamePhase !== 'game_end') {
          setTimeout(() => {
            const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
            this.scheduleAIIfNeeded(currentPlayer, delay);
          }, delay);
        }
        break;
    }
  }

  private scheduleAIIfNeeded(playerId: number, delay: number): void {
    if (!this.isAI(playerId)) return;
    
    const effectId = `ai-${playerId}-${Date.now()}`;
    const effect: Effect = {
      id: effectId,
      type: 'ai-action',
      data: { playerId },
      executeAt: Date.now() + delay
    };
    
    this.scheduleEffect(effect);
  }

  private scheduleConsensusAI(playerId: number, action: string, delay: number): void {
    const consensus = this.projections.getProjection('consensus').getCurrentValue();
    const players = consensus.get(action as any);
    if (players && players.has(playerId)) return;
    
    const effectId = `consensus-${playerId}-${action}-${Date.now()}`;
    const effect: Effect = {
      id: effectId,
      type: 'ai-action',
      data: {
        playerId,
        command: {
          type: 'AGREE_TO_ACTION',
          player: playerId,
          action: action as any
        }
      },
      executeAt: Date.now() + delay
    };
    
    this.scheduleEffect(effect);
  }

  private scheduleEffect(effect: Effect): void {
    const delay = effect.executeAt - Date.now();
    
    const timeoutId = setTimeout(() => {
      this.executeEffect(effect);
      this.scheduled.delete(effect.id);
    }, Math.max(0, delay)) as unknown as number;
    
    this.scheduled.set(effect.id, {
      effect,
      timeoutId
    });
  }

  private executeEffect(effect: Effect): void {
    if (effect.type === 'ai-action') {
      const command = effect.data.command || this.getAICommand(effect.data.playerId);
      if (command) {
        this.commandProcessor.execute(command);
      }
    }
  }

  private getAICommand(playerId: number): Command | null {
    const consensus = this.projections.getProjection('consensus').getCurrentValue();
    
    for (const [action, players] of consensus.entries()) {
      if (!players.has(playerId)) {
        return {
          type: 'AGREE_TO_ACTION',
          player: playerId,
          action
        };
      }
    }
    
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
    
    if (currentPlayer !== playerId) return null;
    
    return getAIAction(this.buildGameState(), playerId);
  }

  private buildGameState(): any {
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    const hands = this.projections.getProjection('hands').getCurrentValue();
    const trump = this.projections.getProjection('trump').getCurrentValue();
    const trick = this.projections.getProjection('trick').getCurrentValue();
    const bids = this.projections.getProjection('bids').getCurrentValue();
    const teams = this.projections.getProjection('teams').getCurrentValue();
    
    return {
      phase,
      players: [0, 1, 2, 3].map(id => ({
        id,
        hand: hands[id] || [],
        teamId: (id % 2) as 0 | 1
      })),
      trump,
      currentTrick: trick,
      bids: bids.bids,
      currentBid: bids.highestBid || { type: 'pass', player: -1 },
      teamScores: teams.trickPoints,
      validTransitions: []
    };
  }

  private isAI(playerId: number): boolean {
    const playerTypes = this.projections.getProjection('playerTypes').getCurrentValue();
    return playerTypes[playerId] === 'ai';
  }

  private getDelay(): number {
    switch (this.quickplay.speed) {
      case 'instant': return 0;
      case 'fast': return 100;
      case 'normal': return 600;
      default: return 600;
    }
  }

  private getDealer(): number {
    return 0;
  }

  private cancelAll(): void {
    for (const scheduled of this.scheduled.values()) {
      if (scheduled.timeoutId !== undefined) {
        clearTimeout(scheduled.timeoutId);
      }
    }
    this.scheduled.clear();
  }

  cancelEffect(effectId: string): void {
    const scheduled = this.scheduled.get(effectId);
    if (scheduled && scheduled.timeoutId !== undefined) {
      clearTimeout(scheduled.timeoutId);
      this.scheduled.delete(effectId);
    }
  }
}