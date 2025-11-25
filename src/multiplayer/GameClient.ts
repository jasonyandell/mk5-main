import type { Socket } from './Socket';
import type { ClientMessage, ServerMessage } from './protocol';
import type { GameView } from '../shared/multiplayer/protocol';

export class GameClient {
  view: GameView | null = null;
  private listeners = new Set<(view: GameView) => void>();
  private socket: Socket;

  constructor(socket: Socket) {
    this.socket = socket;
    socket.onMessage((data) => this.handleMessage(JSON.parse(data)));
  }

  /** Send a message to the server. Fire-and-forget - result comes via subscription. */
  send(message: ClientMessage): void {
    this.socket.send(JSON.stringify(message));
  }

  /** Subscribe to state updates. Returns unsubscribe function. */
  subscribe(callback: (view: GameView) => void): () => void {
    if (this.view) callback(this.view);
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /** Disconnect from the game. */
  disconnect(): void {
    this.socket.close();
  }

  private handleMessage(message: ServerMessage): void {
    switch (message.type) {
      case 'STATE_UPDATE':
        this.view = message.view;
        this.listeners.forEach(cb => cb(this.view!));
        break;
      case 'ERROR':
        console.error('Server error:', message.error);
        break;
    }
  }
}
