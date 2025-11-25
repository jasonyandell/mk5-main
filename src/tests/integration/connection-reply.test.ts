/**
 * Test for Room send callback pattern.
 *
 * The new architecture passes a send callback to Room's constructor.
 * This ensures Room can send messages immediately without initialization order issues.
 */

import { describe, it, expect, vi } from 'vitest';
import { Room } from '../../server/Room';
import type { GameConfig } from '../../game/types/config';
import type { ServerMessage } from '../../multiplayer/protocol';

describe('Room send callback pattern', () => {
  it('should send messages via callback without errors', () => {
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    const sentMessages: Map<string, ServerMessage[]> = new Map();

    // Spy on console.error to catch any errors
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room with send callback - no Transport needed
    const room = new Room(config, (clientId, message) => {
      const messages = sentMessages.get(clientId) ?? [];
      messages.push(message);
      sentMessages.set(clientId, messages);
    });

    // Connect clients and verify messages are sent
    room.handleConnect('client-1');

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    consoleErrorSpy.mockRestore();

    // No errors should occur
    expect(errorCalls.length).toBe(0);

    // Client should have received initial state
    const messages = sentMessages.get('client-1') ?? [];
    expect(messages.length).toBe(1);
    expect(messages[0]?.type).toBe('STATE_UPDATE');
  });

  it('should handle multiple clients connecting simultaneously', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human'],
      shuffleSeed: 99999
    };

    const sentMessages: Map<string, ServerMessage[]> = new Map();

    // Spy on console.error
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room with send callback
    const room = new Room(config, (clientId, message) => {
      const messages = sentMessages.get(clientId) ?? [];
      messages.push(message);
      sentMessages.set(clientId, messages);
    });

    // Connect all 4 players
    for (let i = 0; i < 4; i++) {
      room.handleConnect(`client-${i}`);
    }

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    consoleErrorSpy.mockRestore();

    // No errors should occur
    expect(errorCalls.length).toBe(0);

    // All clients should have received initial state
    for (let i = 0; i < 4; i++) {
      const messages = sentMessages.get(`client-${i}`) ?? [];
      expect(messages.length).toBe(1);
      expect(messages[0]?.type).toBe('STATE_UPDATE');
    }
  });
});
