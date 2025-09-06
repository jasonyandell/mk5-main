# Sections and Unified Transition Pipeline (Codex Plan)

This document describes a pragmatic plan to fix timing/skip issues and to enable playing bounded sections of the game (one transition, one play, one trick, one hand, one full game) in a composable way.

## Goals

- Single, deterministic transition pipeline (one way to mutate state).
- One AI driver (no competing loops triggering actions).
- Clean gating for “sections” to allow and stop transitions precisely.
- Composable pre/post UI screens around sections (easily slot in start/end states).
- Minimal changes at first to stabilize timing; extend incrementally.

## Recommendation

Fix while building forward:
- Add a unified `TransitionDispatcher` and route all transitions through it.
- Keep `tickGame` + `ai-scheduler` as the single AI driver.
- Refactor Quickplay to adjust AI speed and start sections, not to execute actions directly.
- Layer a `SectionRunner` with stop conditions to support “play any section”.

This reduces race conditions and immersion-breaking skips immediately and unlocks sectioned play without a big-bang refactor.

## Architecture Overview

- `TransitionDispatcher`: single entry for executing transitions. Centralizes gating, queuing, and events (before/after).
- `GatePolicy`: predicate to allow/deny a transition at a moment; denied transitions are queued.
- `SectionRunner`: installs a gate and stop conditions, applies consensus policy (hold/allow/inject), and optionally adjusts AI speed profile.
- `StopConditions` DSL: composable predicates by transition type/id, counts, phase changes, trick/score deltas, etc.
- AI Timing: one source of truth in `ai-scheduler` (with a settable speed profile). Unified `skip` affects the schedule consistently.

## Problems Addressed

- Double-triggering and races: multiple call sites invoked `gameActions.executeAction` or directly assigned `newState`. Dispatcher unifies the path.
- Timing split-brain: both Quickplay and the game loop could execute transitions; we choose one AI driver.
- Accidental auto-advancement (e.g., consensus): gate blocks/disciplines these transitions during sections.
- Hard-to-stop segments: stop conditions make it trivial to stop after any well-defined boundary (first play, trick completed, phase change, etc.).

## Core Pieces

### TransitionDispatcher

Single entry-point to request transitions. Applies a gate, queues disallowed transitions, executes allowed ones via the existing `gameActions.executeAction`.

```ts
// src/game/core/dispatcher.ts
import type { GameState, StateTransition } from '../types';

type Source = 'ui' | 'ai' | 'replay' | 'system';
type Listener = (ctx: { prev: GameState; next: GameState; transition: StateTransition; source: Source }) => void;

export class TransitionDispatcher {
  private before: Listener[] = [];
  private after: Listener[] = [];
  private gate: (t: StateTransition, state: GameState) => boolean = () => true;
  private queue: Array<{ t: StateTransition; source: Source }> = [];

  constructor(
    private exec: (t: StateTransition) => void,
    private getState: () => GameState
  ) {}

  setGate(g: (t: StateTransition, state: GameState) => boolean) { this.gate = g; }
  clearGate() { this.gate = () => true; this.flush(); }

  onBefore(fn: Listener) { this.before.push(fn); return () => this.off(this.before, fn); }
  onAfter(fn: Listener) { this.after.push(fn); return () => this.off(this.after, fn); }
  private off<T>(arr: T[], fn: T) { const i = arr.indexOf(fn); if (i >= 0) arr.splice(i, 1); }

  requestTransition(t: StateTransition, source: Source) {
    const state = this.getState();
    if (!this.gate(t, state)) { this.queue.push({ t, source }); return; }
    this.execute(t, source);
  }

  flush() {
    const pending = this.queue; this.queue = [];
    for (const item of pending) {
      const { t, source } = item;
      if (this.gate(t, this.getState())) this.execute(t, source);
      else this.queue.push(item);
    }
  }

  private execute(t: StateTransition, source: Source) {
    const prev = this.getState();
    this.before.forEach(l => l({ prev, next: prev, transition: t, source }));
    this.exec(t);
    const next = this.getState();
    this.after.forEach(l => l({ prev, next, transition: t, source }));
  }
}
```

Wire-up in the store and export for app-wide use:

```ts
// src/stores/gameStore.ts (near exports)
import { TransitionDispatcher } from '../game/core/dispatcher';
export const dispatcher = new TransitionDispatcher(
  (t) => gameActions.executeAction(t),
  () => get(gameState)
);
```

Route call sites through dispatcher (illustrative):

```ts
// ActionPanel.svelte / PlayingArea.svelte
// before: gameActions.executeAction(action)
dispatcher.requestTransition(action, 'ui');

// HumanController
// before: this.executeTransition(transition)
dispatcher.requestTransition(transition, 'ui');

// Game loop (ai-scheduler)
if (result.action) dispatcher.requestTransition(result.action, 'ai');

// Quickplay
// Refactor to not execute transitions directly.
```

### Stop Conditions DSL

Composable helpers to define when a section should end.

```ts
// src/game/core/stopConditions.ts
import type { GameState, StateTransition } from '../types';

export type StopCtx = { prev: GameState; next: GameState; transition: StateTransition };
export type StopWhen = (ctx: StopCtx) => boolean;

export const whenFirstTransition = (): StopWhen => {
  let fired = false; return () => !fired ? (fired = true) : false;
};

export const whenTransitionType = (...types: StateTransition['action']['type'][]): StopWhen =>
  ({ transition }) => types.includes(transition.action.type);

export const whenTransitionId = (re: RegExp): StopWhen =>
  ({ transition }) => re.test(transition.id);

export const whenPlayCount = (n: number): StopWhen => {
  let c = 0; return ({ transition }) => transition.action.type === 'play' && ++c >= n;
};

export const whenTrickCompleted = (): StopWhen =>
  ({ prev, next }) => prev.currentTrick.length === 4 && next.currentTrick.length === 0 && next.tricks.length === prev.tricks.length + 1;

export const whenPhaseIn = (...phases: GameState['phase'][]): StopWhen =>
  ({ next }) => phases.includes(next.phase);

export const or = (...conds: StopWhen[]): StopWhen => (ctx) => conds.some(c => c(ctx));
export const and = (...conds: StopWhen[]): StopWhen => (ctx) => conds.every(c => c(ctx));
export const not = (cond: StopWhen): StopWhen => (ctx) => !cond(ctx);
```

### SectionRunner

Installs a gate, applies consensus policy, listens to after-execute events, and resolves when stop condition hits.

```ts
// src/game/core/sectionRunner.ts
import type { GameState, StateTransition } from '../types';
import { dispatcher } from '../../stores/gameStore';
import type { StopWhen } from './stopConditions';

export type Allow = (t: StateTransition, state: GameState) => boolean;
export type ConsensusPolicy = 'allow' | 'hold' | 'injectAll';

export interface SectionSpec {
  name: string;
  allow: Allow;
  stopWhen: StopWhen;
  consensus?: ConsensusPolicy;
  aiSpeed?: 'instant' | 'fast' | 'normal' | ((t: StateTransition) => number);
}

export function startSection(spec: SectionSpec) {
  const executed: StateTransition[] = [];
  let resolved = false;

  const unsubAfter = dispatcher.onAfter(({ prev, next, transition }) => {
    if (resolved) return;
    executed.push(transition);
    if (spec.stopWhen({ prev, next, transition })) {
      resolved = true;
      dispatcher.clearGate();
      unsubAfter();
      resolve({ state: next, actions: executed });
    }
  });

  // Consensus policy: modify gate
  const baseGate = (t: StateTransition, s: GameState) => spec.allow(t, s);
  if (spec.consensus === 'hold') {
    dispatcher.setGate((t, s) => !t.id.startsWith('agree-') && baseGate(t, s));
  } else {
    dispatcher.setGate(baseGate);
  }

  // TODO: if 'injectAll', at trick/score boundary, synthesize agrees via dispatcher

  let resolve!: (r: { state: GameState; actions: StateTransition[] }) => void;
  const done = new Promise<{ state: GameState; actions: StateTransition[] }>((r) => (resolve = r));

  return {
    done,
    cancel() { if (!resolved) { resolved = true; dispatcher.clearGate(); unsubAfter(); } },
    pause() { dispatcher.setGate(() => false); },
    resume() { dispatcher.setGate(baseGate); }
  };
}
```

### AI Speed Centralization

Use one place to control timing; Quickplay and Sections adjust the profile instead of executing transitions.

```ts
// src/game/core/ai-scheduler.ts
let speedProfile: 'instant' | 'fast' | 'normal' = 'normal';
export function setAISpeedProfile(p: typeof speedProfile) { speedProfile = p; }

export function getAIDelayTicks(action: StateTransition): number {
  if (speedProfile === 'instant') return 0;
  if (speedProfile === 'fast') {
    // shorter ranges
    if (action.action.type === 'select-trump') return Math.ceil((500 + Math.random() * 500) / 16.67);
    if (action.action.type === 'play') return Math.ceil((250 + Math.random() * 300) / 16.67);
    return Math.ceil((300 + Math.random() * 400) / 16.67);
  }
  // existing 'normal' calculation stays as-is
  // ...
}
```

Quickplay refactor intent:
- No `gameActions.executeAction` calls.
- Only: `setAISpeedProfile('instant'|'fast'|'normal')` and start an appropriate section.

## Preset Sections (examples)

Names only (straightforward to wire using `SectionRunner` and `StopConditions`):

- `oneTransition()`
  - allow: all
  - stopWhen: `whenFirstTransition()`
  - consensus: `hold`

- `onePlay()`
  - allow: `t.action.type === 'play'`
  - stopWhen: `whenPlayCount(1)` (aka “stop after anything has been played”)
  - consensus: `hold`

- `oneTrick()`
  - allow: `play-*`, `agree-complete-trick-*`, `complete-trick`
  - stopWhen: `whenTrickCompleted()`
  - consensus: `injectAll` (optional enhancement)

- `oneHand()`
  - allow: all
  - stopWhen: `whenPhaseIn('scoring','game_end')`
  - consensus: `injectAll`

- `fullGame()`
  - allow: all
  - stopWhen: `whenPhaseIn('game_end') && next.isComplete`

## Usage Examples

Stop after any single transition (first transition):

```ts
import { startSection } from '../game/core/sectionRunner';
import { whenFirstTransition } from '../game/core/stopConditions';

const runner = startSection({
  name: 'single',
  allow: () => true,
  stopWhen: whenFirstTransition(),
  consensus: 'hold'
});

const { state, actions } = await runner.done;
```

Play one domino (stop after anything has been played):

```ts
import { startSection } from '../game/core/sectionRunner';
import { whenPlayCount } from '../game/core/stopConditions';

const runner = startSection({
  name: 'onePlay',
  allow: (t) => t.action.type === 'play',
  stopWhen: whenPlayCount(1),
  consensus: 'hold'
});

const result = await runner.done;
```

Play one trick:

```ts
import { startSection } from '../game/core/sectionRunner';
import { whenTrickCompleted } from '../game/core/stopConditions';

const allow = (t: StateTransition) =>
  t.id.startsWith('play-') || t.id.startsWith('agree-complete-trick') || t.id === 'complete-trick';

const runner = startSection({
  name: 'oneTrick',
  allow,
  stopWhen: whenTrickCompleted(),
  consensus: 'injectAll'
});

await runner.done;
```

Play one hand:

```ts
import { startSection } from '../game/core/sectionRunner';
import { whenPhaseIn } from '../game/core/stopConditions';

const runner = startSection({
  name: 'oneHand',
  allow: () => true,
  stopWhen: whenPhaseIn('scoring', 'game_end'),
  consensus: 'injectAll'
});

await runner.done;
```

Compose pre/post UI around sections (pseudo-Svelte):

```svelte
<script lang="ts">
  import { startSection } from '../game/core/sectionRunner';
  import { whenPlayCount } from '../game/core/stopConditions';
  let banner = 'We are going to play a hand';
  async function run() {
    const runner = startSection({ name: 'onePlay', allow: t => t.action.type==='play', stopWhen: whenPlayCount(1), consensus: 'hold' });
    const { state } = await runner.done;
    banner = 'You played a hand!';
  }
</script>
```

## Rollout Plan

- Phase 1 (stabilize now):
  - Add `TransitionDispatcher`; route UI, HumanController, and game loop to it.
  - Keep `gameActions.executeAction` internals untouched (still validates, updates history/URL, notifies controllers).
  - Disable Quickplay’s direct `executeAction` calls; limit to AI speed control or starting sections.

- Phase 2 (sections):
  - Implement `SectionRunner` and `StopConditions` with presets: `oneTransition`, `onePlay`, `oneTrick`, `oneHand`.
  - Add minimal UI hooks to run sections.

- Phase 3 (consensus + speed polish):
  - Implement `injectAll` for consensus transitions at trick/scoring boundaries.
  - Expose AI speed profile in settings and Quickplay.

- Phase 4 (replay integration):
  - Provide offline replay helpers; optionally a live “replay section” mode via dispatcher.

## Expected Outcomes

- Deterministic, race-free transition flow (single mutation path).
- Unified timing and skip behavior (one AI scheduler; centralized skip).
- Reliable sectioned play with clear start/stop semantics.
- Easier debugging and telemetry via dispatcher events and queued transitions.
- Extensible path to richer tooling (dev overlay, metrics, reproducible seeds).

***

This plan fixes current timing/skip issues by unifying triggers and selecting one AI driver, while also providing the primitives to implement “play any section” in a maintainable way.

