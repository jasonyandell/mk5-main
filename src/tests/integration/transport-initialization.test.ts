/**
 * Test for AI Transport decoupling.
 *
 * Architecture: AIs always use dedicated InProcessTransport (internal to Room),
 * independent of Room's transport for human clients. This eliminates initialization
 * ordering issues and ensures AIs are always local/server-side.
 */

import { describe, it, expect, vi } from 'vitest';
import { Room } from '../../server/Room';
import { InProcessTransport } from '../../server/transports/InProcessTransport';
import { humanCapabilities, aiCapabilities } from '../../game/multiplayer/capabilities';
import type { GameConfig } from '../../game/types/config';
import type { PlayerSession } from '../../game/multiplayer/types';

describe('AI Transport Decoupling', () => {
  it('should spawn AI clients immediately in constructor', () => {
    // Setup: Create game with AI players
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    const gameId = 'test-game';

    // Create player sessions
    const sessions: PlayerSession[] = config.playerTypes!.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = type === 'human'
        ? humanCapabilities(idx)
        : aiCapabilities(idx);

      return {
        playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `Player ${index + 1}`,
        capabilities
      };
    });

    // Spy on console.error to verify no errors
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room (AI clients should spawn immediately with internal transport)
    new Room(gameId, config, sessions);

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    const transportErrors = errorCalls.filter(call =>
      call.some(arg => String(arg).includes('InProcessTransport.send: Room not set'))
    );

    // Restore console.error
    consoleErrorSpy.mockRestore();

    // TEST: AI clients should work without any transport errors
    expect(transportErrors.length).toBe(0);
  });

  it('should allow any order for Room transport wiring', () => {
    // Setup: Create game with AI players
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    const gameId = 'test-game-2';

    // Create player sessions
    const sessions: PlayerSession[] = config.playerTypes!.map((type, index) => {
      const idx = index as 0 | 1 | 2 | 3;
      const capabilities = type === 'human'
        ? humanCapabilities(idx)
        : aiCapabilities(idx);

      return {
        playerId: `${type === 'human' ? 'player' : 'ai'}-${index}`,
        playerIndex: idx,
        controlType: type,
        isConnected: true,
        name: `Player ${index + 1}`,
        capabilities
      };
    });

    // Spy on console.error
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // 1. Create Room (AIs spawn with internal transport)
    const room = new Room(gameId, config, sessions);

    // 2. Create Transport for humans (optional)
    const transport = new InProcessTransport();

    // 3. Wire transport to room (one-way now)
    transport.setRoom(room);

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    const transportErrors = errorCalls.filter(call =>
      call.some(arg => String(arg).includes('InProcessTransport.send: Room not set'))
    );

    // Restore console.error
    consoleErrorSpy.mockRestore();

    // Should NOT have any errors
    expect(transportErrors.length).toBe(0);
  });

  it('should allow AI to work even if Room transport is never set', () => {
    // Setup: Create game with AI players only
    const config: GameConfig = {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    const gameId = 'test-game-3';

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

    // Spy on console.error
    const consoleErrorSpy = vi.spyOn(console, 'error');

    // Create Room but NEVER call setTransport
    new Room(gameId, config, sessions);

    // AI clients should still work (they use internal transport)
    // No need to call room.setTransport() for AIs

    // Check for errors
    const errorCalls = consoleErrorSpy.mock.calls;
    const transportErrors = errorCalls.filter(call =>
      call.some(arg => String(arg).includes('InProcessTransport.send: Room not set'))
    );

    // Restore console.error
    consoleErrorSpy.mockRestore();

    // TEST: AIs should work without Room transport being set
    expect(transportErrors.length).toBe(0);
  });
});
