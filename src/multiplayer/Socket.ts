// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

/**
 * Minimal transport interface for bidirectional messaging.
 * Matches WebSocket, postMessage, and any bidirectional channel.
 */
export interface Socket {
  send(data: string): void;
  onMessage(handler: (data: string) => void): void;
  close(): void;
}
