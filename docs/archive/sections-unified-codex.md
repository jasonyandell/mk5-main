# FAILED PLAN. COST ME SO MUCH TIME. KEPT TO SHOW LLMS AN EXAMPLE OF A PLAN THAT DOES NOT WORK

This plan had a critical flaw.  The dispatcher was a total disaster.  It ended
up making new, worse race conditions.  It introduced mutation to the core game state, breaking the pure combinators at the heart of the project.  
I passed this through Opus 4.1 and GPT5-high and both were absolutely convinced that it was a great idea where nothing could possibly go wrong.

# Sections: Unified Codex Approach

This document consolidates the best overall approach to: (1) fix timing/skips reliably and (2) enable playing bounded sections (one transition, one play, one trick, one hand, full game) with clean composition.

## Guiding Principles

- Single mutation path for all transitions.
- One AI driver to avoid races (keep tick + ai-scheduler initially).
- Deterministic gating and stop conditions for sections.
- Centralized skip/delay behavior; adjustable AI speed profile.
- Incremental rollout with minimal disruption to current code.

## Overview

- TransitionDispatcher: unified entry for transition requests with gate/queue and before/after events.
- AI Driver: continue using `tickGame` + `ai-scheduler` as the sole driver; expose a global speed profile.
- SectionRunner: installs a gate and stop conditions; optional consensus policy and speed override.
- StopConditions DSL: small, composable predicates for ending sections (by type, id, count, phase, trick completion, etc.).
- Quickplay: stops executing transitions; becomes a speed and section controller.

## Components

### TransitionDispatcher (single entry)

```ts
// src/game/core/dispatcher.ts
import type { GameState, StateTransition } from '../types';

type Source = 'ui' | 'ai' | 'replay' | 'system';

type Listener = (ctx: {
  prev: GameState; next: GameState; transition: StateTransition; source: Source
}) => void;

export class TransitionDispatcher {
  private before: Listener[] = [];
  private after: Listener[] = [];
  private gate: (t: StateTransition, s: GameState) => boolean = () => true;
  private queue: Array<{ t: StateTransition; source: Source }> = [];

  constructor(private exec: (t: StateTransition) => void,
              private getState: () => GameState) {}

  onBefore(fn: Listener) { this.before.push(fn); return () => this.off(this.before, fn); }
  onAfter(fn: Listener)  { this.after.push(fn);  return () => this.off(this.after, fn); }
  setGate(g: (t: StateTransition, s: GameState) => boolean) { this.gate = g; }
  clearGate() { this.gate = () => true; this.flush(); }

  requestTransition(t: StateTransition, source: Source) {
    const s = this.getState();
    if (!this.gate(t, s)) { this.queue.push({ t, source }); return; }
    this.execute(t, source);
  }

  flush() {
    const pending = this.queue; this.queue = [];
    for (const item of pending) {
      if (this.gate(item.t, this.getState())) this.execute(item.t, item.source);
      else this.queue.push(item);
    }
  }

  private execute(t: StateTransition, source: Source) {
    const prev = this.getState();
    this.before.forEach(l => l({ prev, next: prev, transition: t, source }));
    this.exec(t); // calls gameActions.executeAction
    const next = this.getState();
    this.after.forEach(l => l({ prev, next, transition: t, source }));
  }

  private off<T>(arr: T[], fn: T) { const i = arr.indexOf(fn); if (i >= 0) arr.splice(i, 1); }
}
```

Wire in store:

```ts
// src/stores/gameStore.ts
import { TransitionDispatcher } from '../game/core/dispatcher';
export const dispatcher = new TransitionDispatcher(
  (t) => gameActions.executeAction(t),
  () => get(gameState)
);
```

Route call sites:

```ts
// UI (ActionPanel/PlayingArea)
dispatcher.requestTransition(action, 'ui');

// HumanController
// before: executeTransition(transition)
dispatcher.requestTransition(transition, 'ui');

// Game loop (ai-scheduler result)
if (result.action) dispatcher.requestTransition(result.action, 'ai');
```

### AI Driver + Speed Profile

Keep current tick/ai-scheduler and centralize speed:

```ts
// src/game/core/ai-scheduler.ts
let speedProfile: 'instant' | 'fast' | 'normal' = 'normal';
export function setAISpeedProfile(p: typeof speedProfile) { speedProfile = p; }
export function getAIDelayTicks(action: StateTransition): number {
  if (speedProfile === 'instant') return 0;
  if (speedProfile === 'fast') {
    if (action.action.type === 'select-trump') return Math.ceil((500 + Math.random()*500)/16.67);
    if (action.action.type === 'play') return Math.ceil((250 + Math.random()*300)/16.67);
    return Math.ceil((300 + Math.random()*400)/16.67);
  }
  // existing normal calc remains
  // ...
}
```

Quickplay only sets speed/sections, no direct execute:

```ts
// quickplayStore.ts (concept)
setAISpeedProfile('instant');
const runner = startSection(presets.oneGame());
await runner.done;
```

### StopConditions DSL

```ts
// src/game/core/stopConditions.ts
import type { GameState, StateTransition } from '../types';
export type StopCtx = { prev: GameState; next: GameState; transition: StateTransition };
export type StopWhen = (ctx: StopCtx) => boolean;

export const whenFirstTransition = (): StopWhen => { let f=false; return () => !f ? (f=true) : false; };
export const whenTransitionType = (...types: StateTransition['action']['type'][]): StopWhen =>
  ({ transition }) => types.includes(transition.action.type);
export const whenTransitionId = (re: RegExp): StopWhen => ({ transition }) => re.test(transition.id);
export const whenPlayCount = (n: number): StopWhen => { let c=0; return ({transition}) => transition.action.type==='play' && ++c>=n; };
export const whenTrickCompleted = (): StopWhen => ({prev,next}) => prev.currentTrick.length===4 && next.currentTrick.length===0 && next.tricks.length===prev.tricks.length+1;
export const whenPhaseIn = (...phases: GameState['phase'][]): StopWhen => ({next}) => phases.includes(next.phase);
export const or = (...conds: StopWhen[]): StopWhen => ctx => conds.some(c=>c(ctx));
export const and = (...conds: StopWhen[]): StopWhen => ctx => conds.every(c=>c(ctx));
export const not = (cond: StopWhen): StopWhen => ctx => !cond(ctx);
```

### SectionRunner (+ consensus policy)

```ts
// src/game/core/sectionRunner.ts
import type { GameState, StateTransition } from '../types';
import { dispatcher } from '../../stores/gameStore';
import type { StopWhen } from './stopConditions';

export type Allow = (t: StateTransition, s: GameState) => boolean;
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
  let resolve!: (r:{state:GameState; actions:StateTransition[]}) => void;
  const done = new Promise<{state:GameState; actions:StateTransition[]}>((r)=>resolve=r);

  const baseGate = (t:StateTransition,s:GameState)=> spec.allow(t,s);
  const gate = (t:StateTransition,s:GameState)=> spec.consensus==='hold' ? (!t.id.startsWith('agree-') && baseGate(t,s)) : baseGate(t,s);
  dispatcher.setGate(gate);

  const off = dispatcher.onAfter(({prev,next,transition})=>{
    executed.push(transition);
    if (spec.stopWhen({prev,next,transition})) {
      dispatcher.clearGate();
      off();
      resolve({ state: next, actions: executed });
    }
  });

  return {
    done,
    cancel(){ dispatcher.clearGate(); off(); },
    pause(){ dispatcher.setGate(()=>false); },
    resume(){ dispatcher.setGate(gate); }
  };
}
```

Consensus 'injectAll' can be added later by synthesizing agree-* transitions when boundaries are reached.

## Presets & Usage

- oneTransition: allow all; stop on first transition; consensus: hold.
- onePlay: allow plays only; stop after first play; consensus: hold.
- oneTrick: allow plays + agrees + complete; stop when trick completed; consensus: injectAll.
- oneHand: allow all; stop on phase scoring/game_end; consensus: injectAll.

Examples:

```ts
// One play
const runner = startSection({
  name: 'onePlay',
  allow: (t) => t.action.type === 'play',
  stopWhen: whenPlayCount(1),
  consensus: 'hold'
});
await runner.done;

// One trick
const allow = (t: StateTransition) => t.id.startsWith('play-') || t.id.startsWith('agree-complete-trick') || t.id==='complete-trick';
const trick = startSection({ name:'oneTrick', allow, stopWhen: whenTrickCompleted(), consensus:'injectAll' });
await trick.done;
```

## Rollout Plan

- Phase 1 (stabilize now)
  - Add `TransitionDispatcher`; route UI, HumanController, and game loop through it.
  - Keep `gameActions.executeAction` internals; select tick/ai-scheduler as sole AI driver.
  - Refactor Quickplay to stop executing transitions; only set `setAISpeedProfile` and start sections.

- Phase 2 (sections)
  - Implement `SectionRunner` and `StopConditions` with presets (oneTransition, onePlay, oneTrick, oneHand).
  - Add minimal UI controls for sections; optional dev overlay for gate/queue visibility.

- Phase 3 (polish)
  - Implement consensus 'injectAll'. Expose AI speed profile in settings and Quickplay.
  - Add metrics/logging via dispatcher events for debugging.

- Phase 4 (replay)
  - Provide offline replay helpers and optionally a live replay via dispatcher with appropriate gating.

## Risks & Mitigations

- Dual AI paths: mitigated by explicitly disabling Quickplay direct execution.
- Gate deadlocks: mitigated with dev overlay/telemetry and a `pause/resume/cancel` API.
- Back-compat: dispatcher preserves existing `gameActions.executeAction` behavior and URL/history updates.

## Outcome

- Immediate stabilization of timing/skips (single entry, one AI driver).
- Reliable sectioned play with clear boundaries and composable screens.
- A measured path to further centralization if desired (e.g., a future engine), without blocking near-term goals.

