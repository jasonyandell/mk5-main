import type { Domino, Bid, PlayerType, TrumpSelection, ConsensusAction } from '../types';

export interface EventEnvelope<T = GameEvent> {
  id: string;
  idx: number;
  timestamp: number;
  correlationId: string;
  causationId?: string;
  payload: T;
}

export type GameEvent = 
  | GameStartedEvent
  | HandsDealtEvent
  | RedealInitiatedEvent
  | GameEndedEvent
  | BidPlacedEvent
  | PlayerPassedEvent
  | BiddingCompletedEvent
  | AllPlayersPassedEvent
  | TrumpSelectedEvent
  | DominoPlayedEvent
  | TrickCompletedEvent
  | HandReadyForScoringEvent
  | HandScoredEvent
  | MarksAwardedEvent
  | GameTargetReachedEvent
  | ConsensusRequestedEvent
  | PlayerAgreedEvent
  | ConsensusReachedEvent
  | AIScheduledEvent
  | AIThinkingEvent
  | AIDecidedEvent
  | QuickplayEnabledEvent
  | QuickplayDisabledEvent
  | QuickplaySpeedSetEvent
  | EffectScheduledEvent
  | AnimationStartedEvent
  | AnimationCompletedEvent
  | ResetEvent;

export interface GameStartedEvent {
  type: 'GAME_STARTED';
  seed: number;
  dealer: number;
  playerTypes: PlayerType[];
}

export interface HandsDealtEvent {
  type: 'HANDS_DEALT';
  hands: Record<number, Domino[]>;
}

export interface RedealInitiatedEvent {
  type: 'REDEAL_INITIATED';
  newSeed: number;
}

export interface GameEndedEvent {
  type: 'GAME_ENDED';
  winningTeam: 0 | 1;
}

export interface BidPlacedEvent {
  type: 'BID_PLACED';
  player: number;
  bid: Bid;
}

export interface PlayerPassedEvent {
  type: 'PLAYER_PASSED';
  player: number;
}

export interface BiddingCompletedEvent {
  type: 'BIDDING_COMPLETED';
  winner: number;
  bid: Bid;
}

export interface AllPlayersPassedEvent {
  type: 'ALL_PLAYERS_PASSED';
  dealer: number;
}

export interface TrumpSelectedEvent {
  type: 'TRUMP_SELECTED';
  player: number;
  trump: TrumpSelection;
}

export interface DominoPlayedEvent {
  type: 'DOMINO_PLAYED';
  player: number;
  domino: Domino;
}

export interface TrickCompletedEvent {
  type: 'TRICK_COMPLETED';
  winner: number;
  points: number;
}

export interface HandReadyForScoringEvent {
  type: 'HAND_READY_FOR_SCORING';
  reason: 'all-tricks' | 'determined-early';
}

export interface HandScoredEvent {
  type: 'HAND_SCORED';
  teamScores: [number, number];
}

export interface MarksAwardedEvent {
  type: 'MARKS_AWARDED';
  team: 0 | 1;
  marks: number;
}

export interface GameTargetReachedEvent {
  type: 'GAME_TARGET_REACHED';
  team: 0 | 1;
  finalMarks: [number, number];
}

export interface ConsensusRequestedEvent {
  type: 'CONSENSUS_REQUESTED';
  action: ConsensusAction;
}

export interface PlayerAgreedEvent {
  type: 'PLAYER_AGREED';
  player: number;
  action: ConsensusAction;
}

export interface ConsensusReachedEvent {
  type: 'CONSENSUS_REACHED';
  action: ConsensusAction;
}

export interface AIScheduledEvent {
  type: 'AI_SCHEDULED';
  player: number;
  executeAt: number;
}

export interface AIThinkingEvent {
  type: 'AI_THINKING';
  player: number;
}

export interface AIDecidedEvent {
  type: 'AI_DECIDED';
  player: number;
  action: Command;
}

export interface QuickplayEnabledEvent {
  type: 'QUICKPLAY_ENABLED';
}

export interface QuickplayDisabledEvent {
  type: 'QUICKPLAY_DISABLED';
}

export interface QuickplaySpeedSetEvent {
  type: 'QUICKPLAY_SPEED_SET';
  speed: 'instant' | 'fast' | 'normal';
}

export interface EffectScheduledEvent {
  type: 'EFFECT_SCHEDULED';
  effect: Effect;
  executeAt: number;
}

export interface AnimationStartedEvent {
  type: 'ANIMATION_STARTED';
  animation: string;
  duration: number;
}

export interface AnimationCompletedEvent {
  type: 'ANIMATION_COMPLETED';
  animation: string;
  duration?: number;
}

export interface ResetEvent {
  type: 'RESET';
}

export type Command = 
  | StartGameCommand
  | PlaceBidCommand
  | PassCommand
  | SelectTrumpCommand
  | PlayDominoCommand
  | AgreeToActionCommand
  | EnableQuickplayCommand
  | DisableQuickplayCommand
  | RequestRedealCommand;

export interface StartGameCommand {
  type: 'START_GAME';
  seed?: number;
  playerTypes?: PlayerType[];
}

export interface PlaceBidCommand {
  type: 'PLACE_BID';
  player: number;
  bid: Bid;
}

export interface PassCommand {
  type: 'PASS';
  player: number;
}

export interface SelectTrumpCommand {
  type: 'SELECT_TRUMP';
  player: number;
  trump: TrumpSelection;
}

export interface PlayDominoCommand {
  type: 'PLAY_DOMINO';
  player: number;
  domino: Domino;
}

export interface AgreeToActionCommand {
  type: 'AGREE_TO_ACTION';
  player: number;
  action: ConsensusAction;
}

export interface EnableQuickplayCommand {
  type: 'ENABLE_QUICKPLAY';
  speed?: 'instant' | 'fast' | 'normal';
}

export interface DisableQuickplayCommand {
  type: 'DISABLE_QUICKPLAY';
}

export interface RequestRedealCommand {
  type: 'REQUEST_REDEAL';
}

export interface Effect {
  id: string;
  type: 'ai-action' | 'animation' | 'sound';
  data: any;
}

export interface CommandResult {
  success: boolean;
  events: EventEnvelope[];
  error?: string;
}