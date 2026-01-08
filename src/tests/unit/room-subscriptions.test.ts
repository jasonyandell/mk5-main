import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Room } from '../../server/Room';
import type { GameConfig } from '../../game/types/config';
import type { ServerMessage } from '../../multiplayer/protocol';
import * as kernelModule from '../../kernel/kernel';

describe('Room Subscriptions', () => {
  let room: Room;
  let sentMessages: Map<string, ServerMessage[]>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let buildKernelViewSpy: any;

  // Helper to create room with message capture
  const createRoom = (config: GameConfig) => {
    sentMessages = new Map();
    return new Room(config, (clientId, message) => {
      const messages = sentMessages.get(clientId) ?? [];
      messages.push(message);
      sentMessages.set(clientId, messages);
    });
  };

  beforeEach(() => {
    const config: GameConfig = {
      playerTypes: ['human', 'human', 'ai', 'ai'],
      shuffleSeed: 12345
    };

    room = createRoom(config);

    // Spy on buildKernelView to verify filtering calls
    buildKernelViewSpy = vi.spyOn(kernelModule, 'buildKernelView');
  });

  it('does not send state when client connects (must JOIN first)', () => {
    // Connect client
    room.handleConnect('client-1');

    // Should NOT send any messages - client must JOIN to receive filtered state
    const messages = sentMessages.get('client-1') ?? [];
    expect(messages.length).toBe(0);
  });

  it('sends filtered state when client joins as player', () => {
    // Connect and join as player-0
    room.handleConnect('client-1');
    room.handleMessage('client-1', {
      type: 'JOIN',
      playerIndex: 0,
      name: 'Player 1'
    });

    // Should send filtered STATE_UPDATE for player-0
    const messages = sentMessages.get('client-1') ?? [];
    // Messages: JOIN sends filtered view (no message on connect anymore)
    expect(messages.length).toBe(1);

    const lastMessage = messages[messages.length - 1];
    expect(lastMessage?.type).toBe('STATE_UPDATE');
  });

  it('filters view once per connected client when state changes', () => {
    // Connect and join two clients with different players
    room.handleConnect('client-1');
    room.handleMessage('client-1', { type: 'JOIN', playerIndex: 0, name: 'P1' });

    room.handleConnect('client-2');
    room.handleMessage('client-2', { type: 'JOIN', playerIndex: 1, name: 'P2' });

    // Clear initial messages and spy calls
    sentMessages.clear();
    buildKernelViewSpy.mockClear();

    // Execute an action that changes state
    const availableActions = room.getActionsForPlayer('player-0');
    if (availableActions.length > 0) {
      const action = availableActions[0]!.action;
      room.executeAction('player-0', action);

      // Verify buildKernelView was called for connected clients
      expect(buildKernelViewSpy).toHaveBeenCalled();

      // Verify each client received exactly one message
      const client1Messages = sentMessages.get('client-1') ?? [];
      const client2Messages = sentMessages.get('client-2') ?? [];
      expect(client1Messages.length).toBe(1);
      expect(client2Messages.length).toBe(1);
    }
  });

  it('sends different filtered views to different players', () => {
    // Connect and join two clients with different players
    room.handleConnect('client-1');
    room.handleMessage('client-1', { type: 'JOIN', playerIndex: 0, name: 'P1' });

    room.handleConnect('client-2');
    room.handleMessage('client-2', { type: 'JOIN', playerIndex: 1, name: 'P2' });

    sentMessages.clear();

    // Trigger state update
    const actions = room.getActionsForPlayer('player-0');
    if (actions.length > 0) {
      room.executeAction('player-0', actions[0]!.action);

      const client1Messages = sentMessages.get('client-1') ?? [];
      const client2Messages = sentMessages.get('client-2') ?? [];

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

  it('stops sending updates after disconnect', () => {
    // Connect and join client
    room.handleConnect('client-1');
    room.handleMessage('client-1', { type: 'JOIN', playerIndex: 0, name: 'P1' });

    sentMessages.clear();

    // Trigger state update - should receive message
    const actions1 = room.getActionsForPlayer('player-0');
    if (actions1.length > 0) {
      room.executeAction('player-0', actions1[0]!.action);
      expect(sentMessages.get('client-1')?.length).toBe(1);
    }

    sentMessages.clear();

    // Disconnect
    room.handleDisconnect('client-1');

    // Trigger another state update - should NOT receive message
    const actions2 = room.getActionsForPlayer('player-0');
    if (actions2.length > 0) {
      room.executeAction('player-0', actions2[0]!.action);
      expect(sentMessages.get('client-1')?.length ?? 0).toBe(0);
    }
  });

  it('unjoined clients receive no state (security: no unfiltered views)', () => {
    // Connect without joining
    room.handleConnect('observer-1');

    // Should NOT send any state - prevents leaking unfiltered game state
    const messages = sentMessages.get('observer-1') ?? [];
    expect(messages.length).toBe(0);
  });

  it('only joined clients receive updates on state change', () => {
    // Connect three clients
    room.handleConnect('client-1');
    room.handleMessage('client-1', { type: 'JOIN', playerIndex: 0, name: 'P1' });

    room.handleConnect('client-2');
    room.handleMessage('client-2', { type: 'JOIN', playerIndex: 1, name: 'P2' });

    room.handleConnect('observer');
    // Observer doesn't join a player - should NOT receive updates

    sentMessages.clear();
    buildKernelViewSpy.mockClear();

    // Trigger state update
    const actions = room.getActionsForPlayer('player-0');
    if (actions.length > 0) {
      room.executeAction('player-0', actions[0]!.action);

      // Only joined clients should receive updates
      expect(sentMessages.get('client-1')?.length).toBe(1);
      expect(sentMessages.get('client-2')?.length).toBe(1);
      // Observer (unjoined) should NOT receive updates
      expect(sentMessages.get('observer')?.length ?? 0).toBe(0);
    }
  });
});
