/**
 * Multiplayer Module - Public API
 *
 * This module provides the complete multiplayer infrastructure for Texas 42.
 *
 * Architecture:
 * - Socket: Minimal transport interface (WebSocket, postMessage, etc.)
 * - GameClient: Client class for sending actions and receiving updates
 * - Room: Server-side game authority (transport-agnostic)
 * - Protocol: Wire format (ClientMessage, ServerMessage)
 * - Types: Shared types (GameView, ValidAction, Capability, etc.)
 */

// Types
export type {
  Capability,
  PlayerSession,
  MultiplayerGameState,
  ActionRequest,
  GameView,
  ValidAction,
  ViewTransition,
  PlayerInfo,
  Result,
  PlayerIndex
} from './types';

export { ok, err, hasCapability, hasCapabilityType, isPlayerIndex, assertPlayerIndex } from './types';

// Protocol
export type { ClientMessage, ServerMessage } from './protocol';

// Transport
export type { Socket } from './Socket';

// Client
export { GameClient } from './GameClient';

// Capabilities
export {
  playerCapabilities,
  spectatorCapabilities,
  buildBaseCapabilities,
  buildCapabilities,
  CapabilityBuilder,
  filterActionForSession,
  filterActionsForSession,
  getVisibleStateForSession,
  resolveSessionForAction
} from './capabilities';

// Authorization
export { canPlayerExecuteAction, authorizeAndExecute } from './authorization';

// State Lifecycle
export {
  createMultiplayerGame,
  addPlayer,
  removePlayer,
  updatePlayerSession,
  type CreateMultiplayerGameOptions
} from './stateLifecycle';

// Local Game Wiring
export { createLocalGame, attachAIBehavior, type LocalGame, type LocalGameOptions } from './local';
