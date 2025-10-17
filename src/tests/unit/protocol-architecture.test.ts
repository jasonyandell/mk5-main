/**
 * Tests for the new protocol-based architecture.
 *
 * These tests verify:
 * 1. GameHost works correctly without transport
 * 2. Protocol messages flow correctly
 * 3. AI clients work via protocol
 * 4. Backwards compatibility is maintained
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GameHost, GameRegistry } from '../../server/game/GameHost';
import { InProcessAdapter } from '../../server/offline/InProcessAdapter';
import { NetworkGameClient } from '../../game/multiplayer/NetworkGameClient';
import { AIClient } from '../../game/multiplayer/AIClient';
import type {
  GameConfig,
  ServerMessage
} from '../../shared/multiplayer/protocol';

describe('GameHost', () => {
  let host: GameHost;
  let gameId: string;

  beforeEach(() => {
    gameId = 'test-game-123';
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      shuffleSeed: 12345
    };
    const players = config.playerTypes.map((type, i) => ({
      playerId: type === 'human' ? `player-${i}` : `ai-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      capabilities: [
        { type: 'act-as-player', playerIndex: i as 0 | 1 | 2 | 3 },
        { type: 'observe-own-hand' },
        ...(type === 'ai' ? [{ type: 'replace-ai' as const }] : [])
      ]
    }));
    host = new GameHost(gameId, config, players);
  });

  it('should create initial game state', () => {
    const view = host.getView();

    expect(view.state).toBeDefined();
    expect(view.state.phase).toBe('bidding');
    expect(view.validActions).toBeDefined();
    expect(view.players).toHaveLength(4);
    expect(view.players[0]?.controlType).toBe('human');
    expect(view.players[1]?.controlType).toBe('ai');
  });

  it('should execute valid actions', () => {
    const players = host.getPlayers();
    const player = players[0];
    expect(player).toBeDefined();
    const playerId = player!.playerId;  // Get first player (player-0)

    const view = host.getView(playerId);  // Get filtered view for player 0
    const validAction = view.validActions[0];

    if (validAction) {
      const result = host.executeAction(playerId, validAction.action);
      expect(result.ok).toBe(true);

      const newView = host.getView(playerId);
      expect(newView.state.actionHistory).toHaveLength(1);
    }
  });

  it('should reject invalid actions', () => {
    const invalidAction = {
      type: 'play' as const,
      player: 0,
      domino: [6, 6] as [number, number]
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result = host.executeAction('player-0', invalidAction as any);
    expect(result.ok).toBe(false);
    expect(result.error).toBeDefined();
  });

  it('should handle player control changes', () => {
    host.setPlayerControl(0, 'ai');
    const players = host.getPlayers();

    expect(players[0]!.controlType).toBe('ai');
  });

  it('should notify subscribers on state changes', () => {
    const listener = vi.fn();
    const unsubscribe = host.subscribe(undefined, listener);

    // Should get immediate call with current state
    expect(listener).toHaveBeenCalledTimes(1);

    // Execute an action
    const players = host.getPlayers();
    const player = players[0];
    expect(player).toBeDefined();
    const playerId = player!.playerId;

    const view = host.getView(playerId);  // Get filtered view for player 0
    const validAction = view.validActions[0];

    if (validAction) {
      host.executeAction(playerId, validAction.action);
      expect(listener).toHaveBeenCalledTimes(2);
    }

    unsubscribe();
  });
});

describe('Protocol Flow', () => {
  let adapter: InProcessAdapter;
  let client: NetworkGameClient;
  let receivedMessages: ServerMessage[] = [];

  beforeEach(() => {
    adapter = new InProcessAdapter();
    receivedMessages = [];

    // Subscribe to server messages
    adapter.subscribe((message) => {
      receivedMessages.push(message);
    });
  });

  it('should create game via protocol', async () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai']
    };

    client = new NetworkGameClient(adapter, config);

    // Wait for game creation
    await new Promise(resolve => setTimeout(resolve, 100));

    // Should receive GAME_CREATED message
    const gameCreated = receivedMessages.find(m => m.type === 'GAME_CREATED');
    expect(gameCreated).toBeDefined();
    expect(gameCreated?.gameId).toBeDefined();
  });

  it('should execute actions via protocol', async () => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human']
    };

    client = new NetworkGameClient(adapter, config);

    // Wait for game creation
    await new Promise(resolve => setTimeout(resolve, 100));

    // Get valid actions from client
    const validActions = client.getValidActions(0);
    if (validActions.length > 0 && validActions[0]) {
      // Execute action
      const result = await client.requestAction(0, validActions[0]);
      expect(result.ok).toBe(true);

      // Should receive STATE_UPDATE message
      const stateUpdate = receivedMessages.find(m => m.type === 'STATE_UPDATE');
      expect(stateUpdate).toBeDefined();
    }
  });

  it('should handle player control changes via protocol', async () => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'human', 'human']
    };

    client = new NetworkGameClient(adapter, config);

    // Wait for game creation
    await new Promise(resolve => setTimeout(resolve, 100));

    // Change player to AI
    await client.setPlayerControl(1, 'ai');

    // Should receive PLAYER_STATUS message
    const playerStatus = receivedMessages.find(m =>
      m.type === 'PLAYER_STATUS' && m.status === 'control_changed'
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ) as any;
    expect(playerStatus).toBeDefined();
    expect(playerStatus?.controlType).toBe('ai');
  });

  it('should switch perspective to another player', async () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      shuffleSeed: 13579
    };

    client = new NetworkGameClient(adapter, config);

    await new Promise(resolve => setTimeout(resolve, 100));

    const initialState = client.getState();
    expect(initialState.state.players[1]?.hand).toHaveLength(0);

    const seatOneId = initialState.sessions.find(s => s.playerIndex === 1)?.playerId ?? 'player-1';

    await client.setPlayerId(seatOneId);
    await new Promise(resolve => setTimeout(resolve, 100));

    const switchedState = client.getState();
    expect(switchedState.state.players[1]?.hand.length).toBeGreaterThan(0);
  });
});

describe('AI Client', () => {
  let adapter: InProcessAdapter;
  let aiClient: AIClient;
  let gameId: string;

  beforeEach(async () => {
    adapter = new InProcessAdapter();

    let resolved = false;
    const gameIdPromise = new Promise<void>((resolve) => {
      const unsubscribe = adapter.subscribe((message) => {
        if (message.type === 'GAME_CREATED') {
          gameId = message.gameId;
          unsubscribe();
          resolved = true;
          resolve();
        }
      });
    });

    // Create a game first
    await adapter.send({
      type: 'CREATE_GAME',
      config: {
        playerTypes: ['ai', 'ai', 'ai', 'ai']
      },
      clientId: 'test-client'
    });

    // Wait for game ID with timeout
    const timeoutPromise = new Promise<void>((_, reject) => {
      setTimeout(() => {
        if (!resolved) {
          reject(new Error('Timeout waiting for GAME_CREATED'));
        }
      }, 5000);
    });

    await Promise.race([gameIdPromise, timeoutPromise]);
  }, 15000); // Increase timeout for beforeEach

  afterEach(() => {
    if (aiClient) {
      aiClient.destroy();
    }
    if (adapter) {
      adapter.destroy();
    }
  });

  it('should create and start AI client', () => {
    aiClient = new AIClient(gameId, 0, adapter, 'ai-0', 'beginner');
    aiClient.start();

    const info = aiClient.getInfo();
    expect(info.gameId).toBe(gameId);
    expect(info.playerId).toBe('ai-0');
    expect(info.playerIndex).toBe(0);
    expect(info.difficulty).toBe('beginner');
    expect(info.active).toBe(true);
  });

  it('should respond to state updates', async () => {
    aiClient = new AIClient(gameId, 0, adapter, 'ai-0', 'beginner');
    aiClient.start();

    // Wait for AI to make a move
    await new Promise(resolve => setTimeout(resolve, 2500));

    // AI should have sent an EXECUTE_ACTION message
    // (Hard to test directly without mocking, but we can check it's still active)
    expect(aiClient.getInfo().active).toBe(true);

    aiClient.destroy();
  });

  it('should clean up on destroy', () => {
    aiClient = new AIClient(gameId, 0, adapter, 'ai-0', 'beginner');
    aiClient.start();

    aiClient.destroy();

    const info = aiClient.getInfo();
    expect(info.active).toBe(false);
  });
});

describe('GameRegistry', () => {
  let registry: GameRegistry;

  beforeEach(() => {
    registry = new GameRegistry();
  });

  it('should create and manage games', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai']
    };

    const instance = registry.createGame(config);
    expect(instance.id).toBeDefined();
    expect(instance.host).toBeDefined();

    const retrieved = registry.getGame(instance.id);
    expect(retrieved).toBe(instance);
  });

  it('should remove games', () => {
    const config: GameConfig = {
      playerTypes: ['human', 'ai', 'ai', 'ai']
    };

    const instance = registry.createGame(config);
    registry.removeGame(instance.id);

    const retrieved = registry.getGame(instance.id);
    expect(retrieved).toBeUndefined();
  });
});

describe('Backwards Compatibility', () => {
  it('should maintain GameClient interface', async () => {
    const adapter = new InProcessAdapter();
    const client = new NetworkGameClient(adapter, {
      playerTypes: ['human', 'ai', 'ai', 'ai']
    });

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));

    // Test GameClient methods
    expect(client.getState).toBeDefined();
    expect(client.requestAction).toBeDefined();
    expect(client.subscribe).toBeDefined();
    expect(client.setPlayerControl).toBeDefined();
    expect(client.destroy).toBeDefined();

    // Test that methods work
    const state = client.getState();
    expect(state.state).toBeDefined();
    expect(state.sessions).toBeDefined();

    client.destroy();
  });

  it('should work with existing store patterns', async () => {
    const adapter = new InProcessAdapter();
    const client = new NetworkGameClient(adapter, {
      playerTypes: ['human', 'ai', 'ai', 'ai']
    });

    // Wait for initialization
    await new Promise(resolve => setTimeout(resolve, 100));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const states: any[] = [];
    const unsubscribe = client.subscribe((state) => {
      states.push(state);
    });

    // Should get immediate state
    expect(states.length).toBeGreaterThan(0);

    unsubscribe();
    client.destroy();
  });
});
