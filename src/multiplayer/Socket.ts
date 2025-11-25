/**
 * Minimal transport interface for bidirectional messaging.
 * Matches WebSocket, postMessage, and any bidirectional channel.
 */
export interface Socket {
  send(data: string): void;
  onMessage(handler: (data: string) => void): void;
  close(): void;
}
