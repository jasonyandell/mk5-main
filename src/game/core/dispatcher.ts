import type { GameState, StateTransition } from '../types';

type Source = 'ui' | 'ai' | 'replay' | 'system';

type Listener = (ctx: {
  prev: GameState;
  next: GameState;
  transition: StateTransition;
  source: Source;
}) => void;

interface ScheduledTransition {
  transition: StateTransition;
  executeAtTick: number;
  source: Source;
}

/**
 * Unified dispatcher for all transition execution requests.
 * Applies a gate, queues disallowed transitions, and emits before/after events.
 */
export class TransitionDispatcher {
  private before: Listener[] = [];
  private after: Listener[] = [];
  private gate: (t: StateTransition, s: GameState) => boolean = () => true;
  private queue: Array<{ t: StateTransition; source: Source }> = [];
  private customGateActive = false;
  private frozen = false; // when true, deny and drop (no queue accumulation)
  private scheduledByTick: Map<number, ScheduledTransition[]> = new Map();
  private currentTick = 0; // Internal tick counter for UI timing

  constructor(
    private exec: (t: StateTransition) => void,
    private getState: () => GameState
  ) {}

  onBefore(fn: Listener) {
    this.before.push(fn);
    return () => this.off(this.before, fn);
  }

  onAfter(fn: Listener) {
    this.after.push(fn);
    return () => this.off(this.after, fn);
  }

  setGate(g: (t: StateTransition, s: GameState) => boolean) {
    this.gate = g;
    this.customGateActive = true;
  }

  clearGate() {
    this.gate = () => true;
    this.flush();
    this.customGateActive = false;
  }

  requestTransition(t: StateTransition & { delayTicks?: number }, source: Source) {
    const s = this.getState();
    if (this.frozen || !this.gate(t, s)) {
      if (!this.frozen) {
        // Gate-denied: queue for later if not frozen
        this.queue.push({ t, source });
      }
      return;
    }
    
    // If transition has a delay, schedule for future tick
    if (t.delayTicks && t.delayTicks > 0) {
      const executeAt = this.currentTick + t.delayTicks;
      const scheduled = this.scheduledByTick.get(executeAt) || [];
      scheduled.push({ transition: t, executeAtTick: executeAt, source });
      this.scheduledByTick.set(executeAt, scheduled);
    } else {
      // Execute immediately
      this.execute(t, source);
    }
  }

  flush() {
    const pending = this.queue;
    this.queue = [];
    for (const item of pending) {
      if (this.gate(item.t, this.getState())) this.execute(item.t, item.source);
      else this.queue.push(item);
    }
  }

  private execute(t: StateTransition, source: Source) {
    const prev = this.getState();
    this.before.forEach((l) => l({ prev, next: prev, transition: t, source }));
    this.exec(t);
    const next = this.getState();
    this.after.forEach((l) => l({ prev, next, transition: t, source }));
  }

  private off<T>(arr: T[], fn: T) {
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  /**
   * Returns true if a custom gate is currently active (used by sections to pause auto-chaining).
   */
  hasCustomGate(): boolean {
    return this.customGateActive;
  }

  /** Freeze dispatcher: deny and drop new transitions, clear pending queue. */
  setFrozen(frozen: boolean): void {
    this.frozen = frozen;
    if (frozen) {
      this.queue = [];
    }
  }

  /**
   * Advance internal tick and process scheduled transitions
   */
  advanceTick(): void {
    this.currentTick++;
    const ready = this.scheduledByTick.get(this.currentTick);
    if (ready) {
      for (const item of ready) {
        this.execute(item.transition, item.source);
      }
      this.scheduledByTick.delete(this.currentTick);
    }
  }

  /**
   * Get current internal tick
   */
  getCurrentTick(): number {
    return this.currentTick;
  }

  /**
   * Execute all scheduled transitions immediately (for "hurry up")
   */
  executeAllScheduled(): void {
    const allScheduled = Array.from(this.scheduledByTick.values()).flat();
    this.scheduledByTick.clear();
    
    // Execute in tick order
    allScheduled.sort((a, b) => a.executeAtTick - b.executeAtTick);
    for (const item of allScheduled) {
      this.execute(item.transition, item.source);
    }
  }

  /**
   * Check if there are any scheduled transitions for a player
   */
  hasScheduledAction(playerId: number): boolean {
    for (const scheduled of this.scheduledByTick.values()) {
      for (const item of scheduled) {
        if ('player' in item.transition.action && 
            item.transition.action.player === playerId) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Reset internal tick counter (for test resets, etc.)
   */
  resetTick(): void {
    this.currentTick = 0;
    this.scheduledByTick.clear();
  }
}
