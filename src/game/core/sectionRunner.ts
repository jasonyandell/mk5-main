import type { GameState, StateTransition } from '../types';
import { dispatcher } from '../../stores/gameStore';
import { setCurrentScenario } from '../../stores/gameStore';
import type { StopWhen } from './stopConditions';
import { setAISpeedProfile, getAISpeedProfile } from './ai-scheduler';

export type Allow = (t: StateTransition, s: GameState) => boolean;

export interface SectionSpec {
  name: string;
  allow: Allow;
  stopWhen: StopWhen;
  autoScoreHand?: boolean; // default true
  injectDelayMs?: number; // reserved for future Option B fallback
  aiSpeed?: 'instant' | 'fast' | 'normal' | 'slow' | ((t: StateTransition) => number);
}

/**
 * Starts a section with a gate and stop condition; resolves when stop is met.
 */
export function startSection(spec: SectionSpec) {
  const executed: StateTransition[] = [];
  let resolve!: (r: { state: GameState; actions: StateTransition[] }) => void;
  const done = new Promise<{ state: GameState; actions: StateTransition[] }>((r) => (resolve = r));

  const gate = (t: StateTransition, s: GameState) => spec.allow(t, s);

  // Reflect section scenario in URL updates; keep URL updating during section
  setCurrentScenario(spec.name);
  dispatcher.setGate(gate);

  // Optional AI speed override
  let prevSpeed: ReturnType<typeof getAISpeedProfile> | null = null;
  if (spec.aiSpeed && typeof spec.aiSpeed === 'string') {
    prevSpeed = getAISpeedProfile();
    setAISpeedProfile(spec.aiSpeed);
  }

  const off = dispatcher.onAfter(({ prev, next, transition }) => {
    executed.push(transition);

    if (spec.stopWhen({ prev, next, transition })) {
      dispatcher.clearGate();
      // Clear scenario now that section is complete
      setCurrentScenario(null);
      // Restore AI speed if overridden
      if (prevSpeed) setAISpeedProfile(prevSpeed);
      off();
      resolve({ state: next, actions: executed });
    }
  });

  return {
    done,
    cancel() {
      dispatcher.clearGate();
      setCurrentScenario(null);
      if (prevSpeed) setAISpeedProfile(prevSpeed);
      off();
    },
    pause() {
      dispatcher.setGate(() => false);
    },
    resume() {
      dispatcher.setGate(gate);
    }
  };
}
