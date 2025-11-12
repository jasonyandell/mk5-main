import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Room } from '../../server/Room';
import type { GameConfig } from '../../game/types/config';
import type { PlayerSession } from '../../game/multiplayer/types';
import type { Transport, Connection } from '../../server/transports/Transport';
import type { ServerMessage } from '../../shared/multiplayer/protocol';
import * as kernelModule from '../../kernel/kernel';

// Mock transport that captures sent messages
class MockTransport implements Transport {
  public sentMessages: Map<string, ServerMessage[]> = new Map();

  send(clientId: string, message: ServerMessage): void {
    const messages = this.sentMessages.get(clientId) ?? [];
    messages.push(message);
    this.sentMessages.set(clientId, messages);
  }

  connect(clientId: string): Connection {
    return {
      send: () => {
        // No-op for test
      },
      onMessage: () => {
        // No-op for test
      },
      reply: (message: ServerMessage) => {
        // Capture messages sent via reply()
        const messages = this.sentMessages.get(clientId) ?? [];
        messages.push(message);
        this.sentMessages.set(clientId, messages);
      },
      disconnect: () => {
        // No-op for test
      }
    };
  }

  async start(): Promise<void> {
    // No-op for test
  }

  async stop(): Promise<void> {
    // No-op for test
  }

  clearMessages(clientId?: string): void {
    if (clientId) {
      this.sentMessages.delete(clientId);
    } else {
      this.sentMessages.clear();
    }
  }
}

function createTestPlayers(): PlayerSession[] {
  return [
    {
      playerId: 'human-0',
      playerIndex: 0,
      controlType: 'human',
      capabilities: [
        { type: 'act-as-player', playerIndex: 0 },
        { type: 'observe-hands', playerIndices: [0] }
      ]
    },
    {
      playerId: 'human-1',
      playerIndex: 1,
      controlType: 'human',
      capabilities: [
        { type: 'act-as-player', playerIndex: 1 },
        { type: 'observe-hands', playerIndices: [1] }
      ]
    },
    {
      playerId: 'ai-2',
      playerIndex: 2,
      controlType: 'ai',
      capabilities: [
        { type: 'act-as-player', playerIndex: 2 },
        { type: 'observe-hands', playerIndices: [2] }
      ]
    },
    {
      playerId: 'ai-3',
      playerIndex: 3,
      controlType: 'ai',
      capabilities: [
        { type: 'act-as-player', playerIndex: 3 },
        { type: 'observe-hands', playerIndices: [3] }
      ]
    }
  ];
}

describe('Room Subscriptions', () => {
  let room: Room;
  let transport: MockTransport;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let buildKernelViewSpy: any;
  const gameId = 'test-game';
  const connections: Map<string, Connection> = new Map();

  // Helper to get or create connection for a client (cached to prevent duplicates)
  const getConnection = (clientId: string): Connection => {
    if (!connections.has(clientId)) {
      connections.set(clientId, transport.connect(clientId));
    }
    return connections.get(clientId)!;
  };

  beforeEach(() => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'ai', 'ai'],
      shuffleSeed: 12345
    };
    const players = createTestPlayers();

    room = new Room(gameId, config, players);
    transport = new MockTransport();
    connections.clear(); // Clear connection cache between tests

    // Spy on buildKernelView to verify filtering calls
    buildKernelViewSpy = vi.spyOn(kernelModule, 'buildKernelView');
  });

  it('sends initial state when client subscribes', () => {
    // Subscribe client
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));

    // Should send initial STATE_UPDATE
    const messages = transport.sentMessages.get('client-1') ?? [];
    expect(messages.length).toBe(1);
    expect(messages[0]?.type).toBe('STATE_UPDATE');
    if (messages[0]?.type === 'STATE_UPDATE') {
      expect(messages[0].perspective).toBe('human-0');
    }
  });

  it('filters view once per subscriber when state changes', () => {
    // Subscribe two clients with different perspectives
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));
    room.handleMessage('client-2', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-2',
      playerId: 'human-1'
    }, getConnection('client-2'));

    // Clear initial messages and spy calls
    transport.clearMessages();
    buildKernelViewSpy.mockClear();

    // Execute an action that changes state
    const availableActions = room.getActionsForPlayer('human-0');
    if (availableActions.length > 0) {
      const action = availableActions[0]!.action;
      room.executeAction('human-0', action);

      // Verify buildKernelView was called for all subscribers (2 humans + 2 AIs = 4)
      // Note: AI clients are automatically subscribed via internal transport
      expect(buildKernelViewSpy).toHaveBeenCalledTimes(4);

      // Verify each human subscriber received exactly one message
      const client1Messages = transport.sentMessages.get('client-1') ?? [];
      const client2Messages = transport.sentMessages.get('client-2') ?? [];
      expect(client1Messages.length).toBe(1);
      expect(client2Messages.length).toBe(1);

      // Verify perspectives are correct
      if (client1Messages[0]?.type === 'STATE_UPDATE') {
        expect(client1Messages[0].perspective).toBe('human-0');
      }
      if (client2Messages[0]?.type === 'STATE_UPDATE') {
        expect(client2Messages[0].perspective).toBe('human-1');
      }
    }
  });

  it('sends different filtered views to different subscribers', () => {
    // Subscribe two clients with different perspectives
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));
    room.handleMessage('client-2', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-2',
      playerId: 'human-1'
    }, getConnection('client-2'));

    transport.clearMessages();

    // Trigger state update
    const actions = room.getActionsForPlayer('human-0');
    if (actions.length > 0) {
      room.executeAction('human-0', actions[0]!.action);

      const client1Messages = transport.sentMessages.get('client-1') ?? [];
      const client2Messages = transport.sentMessages.get('client-2') ?? [];

      expect(client1Messages.length).toBe(1);
      expect(client2Messages.length).toBe(1);

      // Type guard and extract views
      if (client1Messages[0]?.type !== 'STATE_UPDATE' || client2Messages[0]?.type !== 'STATE_UPDATE') {
        throw new Error('Expected STATE_UPDATE messages');
      }

      const view1 = client1Messages[0].view;
      const view2 = client2Messages[0].view;

      // Verify they have different visible hands
      const player0HandInView1 = view1.state.players[0]?.hand.length ?? 0;
      const player1HandInView1 = view1.state.players[1]?.hand.length ?? 0;
      const player0HandInView2 = view2.state.players[0]?.hand.length ?? 0;
      const player1HandInView2 = view2.state.players[1]?.hand.length ?? 0;

      // Player 0's view should see their hand but not player 1's
      expect(player0HandInView1).toBeGreaterThan(0);
      expect(player1HandInView1).toBe(0);

      // Player 1's view should see their hand but not player 0's
      expect(player0HandInView2).toBe(0);
      expect(player1HandInView2).toBeGreaterThan(0);
    }
  });

  it('stops sending updates after unsubscribe', () => {
    // Subscribe client
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));

    transport.clearMessages();

    // Trigger state update - should receive message
    const actions1 = room.getActionsForPlayer('human-0');
    if (actions1.length > 0) {
      room.executeAction('human-0', actions1[0]!.action);
      expect(transport.sentMessages.get('client-1')?.length).toBe(1);
    }

    transport.clearMessages();

    // Unsubscribe
    room.handleMessage('client-1', {
      type: 'UNSUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));

    // Trigger another state update - should NOT receive message
    const actions2 = room.getActionsForPlayer('human-0');
    if (actions2.length > 0) {
      room.executeAction('human-0', actions2[0]!.action);
      expect(transport.sentMessages.get('client-1')?.length ?? 0).toBe(0);
    }
  });

  it('handles spectator subscriptions (no playerId)', () => {
    // Subscribe without playerId (spectator view)
    room.handleMessage('spectator-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'spectator-1'
    }, getConnection('spectator-1'));

    // Should send initial STATE_UPDATE with unfiltered view
    const messages = transport.sentMessages.get('spectator-1') ?? [];
    expect(messages.length).toBe(1);
    expect(messages[0]?.type).toBe('STATE_UPDATE');
    if (messages[0]?.type === 'STATE_UPDATE') {
      expect(messages[0].perspective).toBeUndefined();
    }
  });

  it('removes subscription on client disconnect', () => {
    // Subscribe client
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));

    transport.clearMessages();

    // Disconnect client
    room.handleClientDisconnect('client-1');

    // Trigger state update - should NOT receive message
    const actions = room.getActionsForPlayer('human-1');
    if (actions.length > 0) {
      room.executeAction('human-1', actions[0]!.action);
      expect(transport.sentMessages.get('client-1')?.length ?? 0).toBe(0);
    }
  });

  it('multiple subscribers all receive updates on each state change', () => {
    // Subscribe three clients
    room.handleMessage('client-1', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-1',
      playerId: 'human-0'
    }, getConnection('client-1'));
    room.handleMessage('client-2', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'client-2',
      playerId: 'human-1'
    }, getConnection('client-2'));
    room.handleMessage('spectator', {
      type: 'SUBSCRIBE',
      gameId,
      clientId: 'spectator'
    }, getConnection('spectator'));

    transport.clearMessages();
    buildKernelViewSpy.mockClear();

    // Trigger state update
    const actions = room.getActionsForPlayer('human-0');
    if (actions.length > 0) {
      room.executeAction('human-0', actions[0]!.action);

      // All three human subscribers should receive updates
      expect(transport.sentMessages.get('client-1')?.length).toBe(1);
      expect(transport.sentMessages.get('client-2')?.length).toBe(1);
      expect(transport.sentMessages.get('spectator')?.length).toBe(1);

      // buildKernelView should be called for all subscribers (3 humans + 2 AIs = 5)
      // Note: AI clients are automatically subscribed via internal transport
      expect(buildKernelViewSpy).toHaveBeenCalledTimes(5);
    }
  });
});
