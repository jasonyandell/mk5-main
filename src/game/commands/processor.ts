import type { Command, CommandResult } from './types';
import type { EventStore } from '../events/store';
import type { ProjectionManager } from '../projections/manager';
import type { GameEvent } from '../events/types';
import type { Bid, Domino } from '../types';
import { isValidBid, isValidPlay } from '../core/rules';
import { dealDominoesWithSeed } from '../core/dominoes';
import { calculateTrickPoints } from '../core/scoring';
import { determineTrickWinner } from '../core/rules';
import { checkHandOutcome } from '../core/handOutcome';
import { createHash } from '../core/random';

export class CommandProcessor {
  constructor(
    private store: EventStore,
    private projections: ProjectionManager
  ) {}

  execute(command: Command): CommandResult {
    try {
      const events = this.processCommand(command);
      if (events.length === 0) {
        return { success: false, events: [], error: 'Invalid command' };
      }

      const correlationId = this.generateCorrelationId();
      const envelopes = this.store.appendMultiple(events, correlationId);
      
      return { success: true, events: envelopes };
    } catch (error) {
      return { 
        success: false, 
        events: [], 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  private processCommand(command: Command): GameEvent[] {
    switch (command.type) {
      case 'START_GAME':
        return this.handleStartGame(command);
      case 'PLACE_BID':
        return this.handlePlaceBid(command);
      case 'PASS':
        return this.handlePass(command);
      case 'SELECT_TRUMP':
        return this.handleSelectTrump(command);
      case 'PLAY_DOMINO':
        return this.handlePlayDomino(command);
      case 'AGREE_TO_ACTION':
        return this.handleAgreeToAction(command);
      case 'ENABLE_QUICKPLAY':
        return this.handleEnableQuickplay(command);
      case 'DISABLE_QUICKPLAY':
        return this.handleDisableQuickplay();
      case 'REQUEST_REDEAL':
        return this.handleRequestRedeal();
      default:
        return [];
    }
  }

  private handleStartGame(command: typeof Command & { type: 'START_GAME' }): GameEvent[] {
    const seed = command.seed || Date.now();
    const playerTypes = command.playerTypes || ['human', 'human', 'human', 'human'];
    const hands = dealDominoesWithSeed(seed);
    
    return [
      {
        type: 'GAME_STARTED',
        seed,
        dealer: 0,
        playerTypes
      },
      {
        type: 'HANDS_DEALT',
        hands
      }
    ];
  }

  private handlePlaceBid(command: typeof Command & { type: 'PLACE_BID' }): GameEvent[] {
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    if (phase !== 'bidding') return [];

    const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
    if (currentPlayer !== command.player) return [];

    const bidsState = this.projections.getProjection('bids').getCurrentValue();
    const bidWithPlayer = { ...command.bid, player: command.player };
    
    if (!isValidBid(this.buildMinimalState(), bidWithPlayer)) return [];

    const events: GameEvent[] = [{
      type: 'BID_PLACED',
      player: command.player,
      bid: bidWithPlayer
    }];

    const passes = bidsState.passes;
    if (passes.size === 3 || this.isBiddingComplete(bidWithPlayer)) {
      events.push({
        type: 'BIDDING_COMPLETED',
        winner: command.player,
        bid: bidWithPlayer
      });
    }

    return events;
  }

  private handlePass(command: typeof Command & { type: 'PASS' }): GameEvent[] {
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    if (phase !== 'bidding') return [];

    const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
    if (currentPlayer !== command.player) return [];

    const events: GameEvent[] = [{
      type: 'PLAYER_PASSED',
      player: command.player
    }];

    const bidsState = this.projections.getProjection('bids').getCurrentValue();
    const newPasses = new Set(bidsState.passes);
    newPasses.add(command.player);

    if (newPasses.size === 4) {
      const dealer = this.getDealer();
      const newSeed = this.generateNewSeed();
      events.push(
        { type: 'ALL_PLAYERS_PASSED', dealer },
        { type: 'REDEAL_INITIATED', newSeed },
        { type: 'HANDS_DEALT', hands: dealDominoesWithSeed(newSeed) }
      );
    } else if (newPasses.size === 3 && bidsState.highestBid) {
      events.push({
        type: 'BIDDING_COMPLETED',
        winner: bidsState.highestBidder,
        bid: bidsState.highestBid
      });
    }

    return events;
  }

  private handleSelectTrump(command: typeof Command & { type: 'SELECT_TRUMP' }): GameEvent[] {
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    if (phase !== 'trump_selection') return [];

    const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
    if (currentPlayer !== command.player) return [];

    return [{
      type: 'TRUMP_SELECTED',
      player: command.player,
      trump: command.trump
    }];
  }

  private handlePlayDomino(command: typeof Command & { type: 'PLAY_DOMINO' }): GameEvent[] {
    const phase = this.projections.getProjection('gamePhase').getCurrentValue();
    if (phase !== 'playing') return [];

    const currentPlayer = this.projections.getProjection('currentPlayer').getCurrentValue();
    if (currentPlayer !== command.player) return [];

    const hands = this.projections.getProjection('hands').getCurrentValue();
    const playerHand = hands[command.player] || [];
    const dominoInHand = playerHand.find(d => 
      d.high === command.domino.high && d.low === command.domino.low
    );
    if (!dominoInHand) return [];

    const validMoves = this.projections.getValidMovesForPlayer(command.player);
    const isValid = validMoves.some(d => 
      d.high === command.domino.high && d.low === command.domino.low
    );
    if (!isValid) return [];

    const events: GameEvent[] = [{
      type: 'DOMINO_PLAYED',
      player: command.player,
      domino: command.domino
    }];

    const trick = this.projections.getProjection('trick').getCurrentValue();
    if (trick.length === 3) {
      events.push({
        type: 'CONSENSUS_REQUESTED',
        action: 'complete-trick'
      });
    }

    return events;
  }

  private handleAgreeToAction(command: typeof Command & { type: 'AGREE_TO_ACTION' }): GameEvent[] {
    const consensus = this.projections.getProjection('consensus').getCurrentValue();
    const players = consensus.get(command.action);
    if (!players) return [];
    if (players.has(command.player)) return [];

    const events: GameEvent[] = [{
      type: 'PLAYER_AGREED',
      player: command.player,
      action: command.action
    }];

    const newPlayers = new Set(players);
    newPlayers.add(command.player);

    if (newPlayers.size === 4) {
      events.push({
        type: 'CONSENSUS_REACHED',
        action: command.action
      });

      if (command.action === 'complete-trick') {
        events.push(...this.completeTrick());
      } else if (command.action === 'score-hand') {
        events.push(...this.scoreHand());
      }
    }

    return events;
  }

  private completeTrick(): GameEvent[] {
    const trick = this.projections.getProjection('trick').getCurrentValue();
    if (trick.length !== 4) return [];

    const trump = this.projections.getProjection('trump').getCurrentValue();
    const leadSuit = this.getDominoSuit(trick[0].domino, trump);
    const winner = determineTrickWinner(trick, trump, leadSuit);
    const points = calculateTrickPoints(trick);

    const events: GameEvent[] = [{
      type: 'TRICK_COMPLETED',
      winner,
      points
    }];

    const completedTricks = this.projections.getProjection('completedTricks').getCurrentValue();
    const trickCount = completedTricks.length + 1;

    if (trickCount === 7 || this.isHandDeterminedEarly()) {
      events.push(
        {
          type: 'HAND_READY_FOR_SCORING',
          reason: trickCount === 7 ? 'all-tricks' : 'determined-early'
        },
        {
          type: 'CONSENSUS_REQUESTED',
          action: 'score-hand'
        }
      );
    }

    return events;
  }

  private scoreHand(): GameEvent[] {
    const teams = this.projections.getProjection('teams').getCurrentValue();
    const bids = this.projections.getProjection('bids').getCurrentValue();
    
    const events: GameEvent[] = [{
      type: 'HAND_SCORED',
      teamScores: teams.trickPoints
    }];

    if (bids.highestBid) {
      const bidderTeam = (bids.highestBidder % 2) as 0 | 1;
      const bidValue = bids.highestBid.value || 0;
      const marks = this.calculateMarks(teams.trickPoints, bidderTeam, bidValue);
      
      events.push({
        type: 'MARKS_AWARDED',
        team: marks.team,
        marks: marks.amount
      });

      const newMarks = [...teams.marks] as [number, number];
      newMarks[marks.team] += marks.amount;

      if (newMarks[0] >= 7 || newMarks[1] >= 7) {
        const winningTeam = newMarks[0] >= 7 ? 0 : 1;
        events.push(
          {
            type: 'GAME_TARGET_REACHED',
            team: winningTeam,
            finalMarks: newMarks
          },
          {
            type: 'GAME_ENDED',
            winningTeam
          }
        );
      } else {
        const newSeed = this.generateNewSeed();
        events.push(
          { type: 'REDEAL_INITIATED', newSeed },
          { type: 'HANDS_DEALT', hands: dealDominoesWithSeed(newSeed) }
        );
      }
    }

    return events;
  }

  private handleEnableQuickplay(command: typeof Command & { type: 'ENABLE_QUICKPLAY' }): GameEvent[] {
    const events: GameEvent[] = [{ type: 'QUICKPLAY_ENABLED' }];
    if (command.speed) {
      events.push({
        type: 'QUICKPLAY_SPEED_SET',
        speed: command.speed
      });
    }
    return events;
  }

  private handleDisableQuickplay(): GameEvent[] {
    return [{ type: 'QUICKPLAY_DISABLED' }];
  }

  private handleRequestRedeal(): GameEvent[] {
    const newSeed = this.generateNewSeed();
    return [
      { type: 'REDEAL_INITIATED', newSeed },
      { type: 'HANDS_DEALT', hands: dealDominoesWithSeed(newSeed) }
    ];
  }

  private buildMinimalState(): any {
    const bids = this.projections.getProjection('bids').getCurrentValue();
    return {
      phase: this.projections.getProjection('gamePhase').getCurrentValue(),
      currentPlayer: this.projections.getProjection('currentPlayer').getCurrentValue(),
      bids: bids.bids,
      currentBid: bids.highestBid || { type: 'pass', player: -1 },
      players: [0, 1, 2, 3].map(id => ({ id, teamId: id % 2 }))
    };
  }

  private isBiddingComplete(bid: Bid): boolean {
    return bid.type === 'marks' && bid.value === 2;
  }

  private isHandDeterminedEarly(): boolean {
    const teams = this.projections.getProjection('teams').getCurrentValue();
    const bids = this.projections.getProjection('bids').getCurrentValue();
    
    if (!bids.highestBid) return false;
    
    const bidderTeam = (bids.highestBidder % 2) as 0 | 1;
    const outcome = checkHandOutcome(
      teams.trickPoints,
      bidderTeam,
      bids.highestBid
    );
    
    return outcome !== null;
  }

  private getDominoSuit(domino: Domino, trump: any): number {
    if (trump.type === 'doubles' && domino.high === domino.low) {
      return 7;
    }
    if (trump.type === 'suit' && trump.suit !== undefined) {
      if (domino.high === trump.suit || domino.low === trump.suit) {
        return trump.suit;
      }
    }
    return domino.high;
  }

  private calculateMarks(trickPoints: [number, number], bidderTeam: 0 | 1, bidValue: number): { team: 0 | 1; amount: number } {
    const bidderPoints = trickPoints[bidderTeam];
    const defenderTeam = (1 - bidderTeam) as 0 | 1;
    
    if (bidderPoints >= bidValue) {
      return { team: bidderTeam, amount: 1 };
    } else {
      return { team: defenderTeam, amount: 1 };
    }
  }

  private getDealer(): number {
    const events = this.store.getEvents();
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].payload.type === 'GAME_STARTED') {
        return (events[i].payload as any).dealer;
      }
    }
    return 0;
  }

  private generateNewSeed(): number {
    const events = this.store.getEvents();
    let prevSeed = Date.now();
    let handIndex = 0;
    
    for (let i = events.length - 1; i >= 0; i--) {
      const event = events[i].payload;
      if (event.type === 'GAME_STARTED') {
        prevSeed = (event as any).seed;
        break;
      } else if (event.type === 'REDEAL_INITIATED') {
        prevSeed = (event as any).newSeed;
        handIndex++;
      }
    }
    
    const hash = createHash(`${prevSeed}:${handIndex}:redeal`);
    return Math.abs(hash);
  }

  private generateCorrelationId(): string {
    return Math.random().toString(36).substring(2, 15);
  }
}