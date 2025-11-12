/**
 * Test for Connection.reply() pattern.
 *
 * This test verifies that AI clients can receive SUBSCRIBE responses without
 * requiring Room.transport to be set. The bug: when AI clients send SUBSCRIBE
 * in their constructor, Room tries to reply via this.transport which is null.
 *
 * Solution: Each Connection should know how to reply to itself via reply() method.
 */

import { describe, it, expect, vi } from 'vitest';
import { Room } from '../../server/Room';
import { aiCapabilities } from '../../game/multiplayer/capabilities';
import type { GameConfig } from '../../game/types/config';
import type { PlayerSession } from '../../game/multiplayer/types';

describe('Connection.reply() pattern', () => {
  it('should allow AI clients to receive SUBSCRIBE response without Room.transport', () => {
    // Setup: Create game with AI players only
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    const gameId = 'test-connection-reply';

    // Create player sessions
    const sessions: PlayerSession[] = config.playerTypes!.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = aiCapabilities(idx);

      return {
        playerId: `ai-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `Player ${index + 1}`,
        capabilities
      };
    });

    // Spy on console.error to catch "sendMessage called before transport was set"
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room but NEVER call setTransport
    // AI clients will send SUBSCRIBE in their constructor
    new Room(gameId, config, sessions);

    // Check for the specific error we're trying to fix
    const errorCalls = consoleErrorSpy.mock.calls;
    const sendMessageErrors = errorCalls.filter(call =>
      call.some(arg => String(arg).includes('sendMessage called before transport was set'))
    );

    // Restore console.error
    consoleErrorSpy.mockRestore();

    // TEST: AI clients should receive SUBSCRIBE response without error
    // This will FAIL initially, proving we replicate the bug
    // After implementing connection.reply(), this should PASS
    expect(sendMessageErrors.length).toBe(0);
  });

  it('should handle multiple AI clients subscribing simultaneously', () => {
    // Setup: Create game with AI players only
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 99999
    };

    const gameId = 'test-multi-subscribe';

    // Create player sessions
    const sessions: PlayerSession[] = config.playerTypes!.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = aiCapabilities(idx);

      return {
        playerId: `ai-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `AI Player ${index + 1}`,
        capabilities
      };
    });

    // Spy on console.error
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room - all 4 AIs will subscribe in constructor
    new Room(gameId, config, sessions);

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    const sendMessageErrors = errorCalls.filter(call =>
      call.some(arg => String(arg).includes('sendMessage called before transport was set'))
    );

    // Restore console.error
    consoleErrorSpy.mockRestore();

    // TEST: All AI clients should receive responses without error
    expect(sendMessageErrors.length).toBe(0);
  });
});
