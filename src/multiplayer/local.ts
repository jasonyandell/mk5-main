import type { GameConfig } from '../game/types/config';
import type { Socket } from './Socket';
import { GameClient } from './GameClient';
import type { ServerMessage } from './protocol';
import { Room } from '../server/Room';
import { selectAIAction } from '../game/ai/actionSelector';

export { attachAIBehavior };

/**
 * Attach AI behavior to a GameClient.
 * Subscribes to view updates and sends actions when it's the AI's turn.
 */
function attachAIBehavior(client: GameClient, playerIndex: number): void {
  client.subscribe((view) => {
    const chosen = selectAIAction(view.state, playerIndex, view.validActions);
    if (!chosen) return;

    client.send({ type: 'EXECUTE_ACTION', action: chosen.action });
  });
}

/**
 * Options for createLocalGame.
 */
export interface LocalGameOptions {
  /** Player indexes that should have AI behavior attached (default: [1, 2, 3]) */
  aiPlayerIndexes?: number[];
  /** If true, skip attaching AI behavior (useful when replaying actions first) */
  skipAIBehavior?: boolean;
}

/**
 * Result of createLocalGame - provides access to client, room, and socket factory.
 */
export interface LocalGame {
  /** GameClient for player-0 (human player) */
  client: GameClient;
  /** Room instance for direct access (e.g., action replay) */
  room: Room;
  /** Factory to create additional sockets for this room */
  createSocket: (clientId: string) => Socket;
  /** Attach AI behavior to all AI clients (call after replaying actions if skipAIBehavior was true) */
  attachAI: () => void;
}

/**
 * Create a local game with human player and AI opponents.
 * Returns an object with client, room, and socket factory for advanced use cases.
 */
export function createLocalGame(config: GameConfig, options?: LocalGameOptions): LocalGame {
  const aiPlayerIndexes = options?.aiPlayerIndexes ?? [1, 2, 3];
  const skipAIBehavior = options?.skipAIBehavior ?? false;

  // Message routing: clientId → handler
  const handlers = new Map<string, (data: string) => void>();

  // Room sends via handlers map
  const room = new Room(config, (clientId: string, message: ServerMessage) => {
    const handler = handlers.get(clientId);
    if (handler) handler(JSON.stringify(message));
  });

  // Factory to create sockets for this room
  function createSocket(clientId: string): Socket {
    return {
      send: (data) => {
        // Client → Room (use queueMicrotask to match async behavior)
        queueMicrotask(() => room.handleMessage(clientId, JSON.parse(data)));
      },
      onMessage: (handler) => {
        handlers.set(clientId, handler);
      },
      close: () => {
        handlers.delete(clientId);
        room.handleDisconnect(clientId);
      }
    };
  }

  // Create human client
  // Order matters: create socket, create client (registers handler), then connect
  const humanSocket = createSocket('player-0');
  const humanClient = new GameClient(humanSocket);  // registers onMessage handler
  room.handleConnect('player-0');  // sends initial state via handler

  // Create AI clients for specified indexes
  // Order matters: create socket, create client (registers handler), then connect
  const aiClients: { client: GameClient; playerIndex: number }[] = [];
  for (const i of aiPlayerIndexes) {
    const aiSocket = createSocket(`ai-${i}`);
    const aiClient = new GameClient(aiSocket);  // registers onMessage handler
    room.handleConnect(`ai-${i}`);  // sends initial state via handler
    // AI needs to JOIN to associate with a player
    aiClient.send({ type: 'JOIN', playerIndex: i, name: `AI Player ${i + 1}` });
    aiClients.push({ client: aiClient, playerIndex: i });
    if (!skipAIBehavior) {
      attachAIBehavior(aiClient, i);
    }
  }

  // Function to attach AI behavior after the fact
  const attachAI = () => {
    for (const { client, playerIndex } of aiClients) {
      attachAIBehavior(client, playerIndex);
    }
  };

  return { client: humanClient, room, createSocket, attachAI };
}
