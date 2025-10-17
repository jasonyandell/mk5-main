import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { filterActionsForSession, getVisibleStateForSession } from '../../game/multiplayer/capabilityUtils';
import type { GameAction } from '../../game/types';
import type { PlayerSession } from '../../game/multiplayer/types';

function createPlayerSession(playerIndex: 0 | 1 | 2 | 3, extras: Partial<PlayerSession> = {}): PlayerSession {
  return {
    playerId: `player-${playerIndex}`,
    playerIndex,
    controlType: 'human',
    capabilities: [
      { type: 'act-as-player', playerIndex },
      { type: 'observe-own-hand' }
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
        { type: 'observe-all-hands' },
        { type: 'observe-full-state' }
      ]
    });

    const visible = getVisibleStateForSession(state, spectator);
    expect(visible.players[1]?.hand.length).toBe(state.players[1]?.hand.length);
    expect(visible.players[2]?.hand.length).toBe(state.players[2]?.hand.length);
  });

  it('strips metadata when session lacks required capabilities', () => {
    const action: GameAction = {
      type: 'bid',
      player: 0,
      bid: 'points',
      value: 30,
      meta: {
        hint: 'Safe opener',
        aiIntent: 'Max trump value'
      }
    };

    const baseSession = createPlayerSession(0);
    const hintSession = createPlayerSession(0, {
      capabilities: [
        { type: 'act-as-player', playerIndex: 0 },
        { type: 'observe-own-hand' },
        { type: 'see-hints' }
      ]
    });

    const aiIntentSession = createPlayerSession(0, {
      capabilities: [
        { type: 'act-as-player', playerIndex: 0 },
        { type: 'observe-own-hand' },
        { type: 'see-ai-intent' }
      ]
    });

    const baseActions = filterActionsForSession(baseSession, [action]);
    expect(baseActions[0]?.meta).toBeUndefined();

    const hintActions = filterActionsForSession(hintSession, [action]);
    expect(hintActions[0]?.meta?.hint).toBe('Safe opener');
    expect(hintActions[0]?.meta?.aiIntent).toBeUndefined();

    const intentActions = filterActionsForSession(aiIntentSession, [action]);
    expect(intentActions[0]?.meta?.hint).toBeUndefined();
    expect(intentActions[0]?.meta?.aiIntent).toBe('Max trump value');
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
