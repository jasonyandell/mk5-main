// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

/**
 * The derived view knows a nello trick is complete at three plays (#98).
 *
 * Trick completion is a rule (rules.isTrickComplete), not a count of four: the
 * nello bidder's partner sits out. The kernel's currentTrickWinner and the
 * projection's isComplete must both follow the rules while the complete trick
 * waits for consensus, before complete-trick runs.
 */

import { describe, it, expect } from 'vitest';
import { Room } from '../../server/Room';
import { createViewProjection } from '../../game/view-projection';
import { createExecutionContext } from '../../game/types/execution';
import type { GameConfig } from '../../game/types/config';
import type { GameView } from '../../multiplayer/types';

const config: GameConfig = {
  playerTypes: ['human', 'human', 'human', 'human'],
  layers: ['nello'],
  shuffleSeed: 424242
};

function projection(view: GameView) {
  return createViewProjection(
    view.state,
    view.transitions.map(t => ({ id: t.id, label: t.label, action: t.action, newState: view.state })),
    { derived: view.derived, viewingPlayerIndex: 0 }
  );
}

describe('nello trick in the derived view', () => {
  it('marks the three-play trick complete and names its winner before complete-trick', () => {
    const room = new Room(config, () => {});
    const seat = () => room.getView('player-0').state.currentPlayer;
    const act = (player: number, pick: (v: GameView) => GameView['validActions'][number] | undefined) => {
      const view = room.getView(`player-${player}`);
      const chosen = pick(view);
      expect(chosen).toBeDefined();
      const result = room.executeAction(`player-${player}`, chosen!.action);
      expect(result.success).toBe(true);
    };

    // A 1-mark bid, three passes, nello
    const bidder = seat();
    act(bidder, v => v.validActions.find(a => a.action.type === 'bid' && a.action.bid === 'marks' && a.action.value === 1));
    while (room.getView('player-0').state.phase === 'bidding') {
      act(seat(), v => v.validActions.find(a => a.action.type === 'pass'));
    }
    act(seat(), v => v.validActions.find(a => a.action.type === 'select-trump' && a.action.trump.type === 'nello'));

    // Three plays, the partner sitting out; before each the trick is incomplete
    for (let plays = 0; plays < 3; plays++) {
      const before = room.getView('player-0');
      expect(before.derived.currentTrickWinner).toBe(-1);
      expect(projection(before).trick.current.isComplete).toBe(false);
      act(seat(), v => v.validActions.find(a => a.action.type === 'play'));
    }

    const core = room.getState().coreState;
    expect(core.phase).toBe('playing');
    expect(core.currentTrick).toHaveLength(3);
    expect(core.tricks).toHaveLength(0); // complete-trick has not run: the trick awaits consensus

    const rules = createExecutionContext(config).rules;
    expect(rules.isTrickComplete(core)).toBe(true);
    const winner = rules.calculateTrickWinner(core, core.currentTrick);

    const view = room.getView('player-0');
    expect(view.derived.currentTrickWinner).toBe(winner);
    expect(projection(view).trick.current).toMatchObject({ isComplete: true, winner });
  });
});
