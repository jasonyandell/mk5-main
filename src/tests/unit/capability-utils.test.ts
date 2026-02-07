import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { filterActionsForSession, getVisibleStateForSession } from '../../multiplayer/capabilities';
import type { GameAction } from '../../game/types';
import type { PlayerSession } from '../../multiplayer/types';

function createPlayerSession(playerIndex: 0 | 1 | 2 | 3, extras: Partial<PlayerSession> = {}): PlayerSession {
  return {
    playerId: `player-${playerIndex}`,
    playerIndex,
    controlType: 'human',
    capabilities: [
      { type: 'act-as-player', playerIndex },
      { type: 'observe-hands', playerIndices: [playerIndex] }
    ],
    ...extras
  };
}

describe('Capability Utils', () => {
  it('hides other hands when viewer lacks observe capabilities', () => {
    const state = createInitialState({ shuffleSeed: 12345 });
    const session = createPlayerSession(0);

    const visible = getVisibleStateForSession(state, session);

    expect(visible.players[0]?.hand.length).toBe(state.players[0]?.hand.length);
    expect(visible.players[1]?.hand.length).toBe(0);
    expect(visible.players[2]?.hand.length).toBe(0);
    expect(visible.players[3]?.hand.length).toBe(0);
  });

  it('allows spectators to view all hands', () => {
    const state = createInitialState({ shuffleSeed: 54321 });
    const spectator = createPlayerSession(0, {
      capabilities: [
        { type: 'observe-hands', playerIndices: 'all' }
      ]
    });

    const visible = getVisibleStateForSession(state, spectator);
    expect(visible.players[0]?.hand.length).toBe(state.players[0]?.hand.length);
    expect(visible.players[1]?.hand.length).toBe(state.players[1]?.hand.length);
    expect(visible.players[2]?.hand.length).toBe(state.players[2]?.hand.length);
    expect(visible.players[3]?.hand.length).toBe(state.players[3]?.hand.length);
  });

  it('strips hint and aiIntent metadata (future features not yet implemented)', () => {
    const action: GameAction = {
      type: 'bid',
      player: 0,
      bid: 'points',
      value: 30,
      meta: {
        hint: 'Safe opener',
        aiIntent: 'Max trump value',
        someOtherField: 'should be preserved'
      }
    };

    const session = createPlayerSession(0);
    const filteredActions = filterActionsForSession(session, [action]);

    // Hint and aiIntent should be stripped (future features)
    expect(filteredActions[0]?.meta?.hint).toBeUndefined();
    expect(filteredActions[0]?.meta?.aiIntent).toBeUndefined();
    // Other metadata should be preserved
    expect(filteredActions[0]?.meta?.someOtherField).toBe('should be preserved');
  });

  it('removes actions requiring capabilities the viewer lacks', () => {
    const lockedAction: GameAction = {
      type: 'select-trump',
      player: 0,
      trump: { type: 'suit', suit: 0 },
      meta: {
        requiredCapabilities: [
          { type: 'act-as-player', playerIndex: 0 }
        ]
      }
    };

    const unrelatedSession = createPlayerSession(1);
    const ownerSession = createPlayerSession(0);

    const filteredForOther = filterActionsForSession(unrelatedSession, [lockedAction]);
    expect(filteredForOther.length).toBe(0);

    const filteredForOwner = filterActionsForSession(ownerSession, [lockedAction]);
    expect(filteredForOwner.length).toBe(1);
  });
});
