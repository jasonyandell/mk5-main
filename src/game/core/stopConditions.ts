import type { GameState, StateTransition } from '../types';

export type StopCtx = { prev: GameState; next: GameState; transition: StateTransition };
export type StopWhen = (ctx: StopCtx) => boolean;

export const whenFirstTransition = (): StopWhen => {
  let fired = false;
  return () => {
    if (!fired) {
      fired = true;
      return true;
    }
    return false;
  };
};

export const whenTransitionType = (
  ...types: Array<StateTransition['action']['type']>
): StopWhen => ({ transition }) => types.includes(transition.action.type);

export const whenTransitionId = (re: RegExp): StopWhen =>
  ({ transition }) => re.test(transition.id);

export const whenPlayCount = (n: number): StopWhen => {
  let c = 0;
  return ({ transition }) => transition.action.type === 'play' && ++c >= n;
};

export const whenTrickCompleted = (): StopWhen =>
  ({ prev, next }) =>
    prev.currentTrick.length === 4 &&
    next.currentTrick.length === 0 &&
    next.tricks.length === prev.tricks.length + 1;

export const whenPhaseIn = (...phases: GameState['phase'][]): StopWhen =>
  ({ next }) => phases.includes(next.phase);

export const or = (...conds: StopWhen[]): StopWhen => (ctx) => conds.some((c) => c(ctx));
export const and = (...conds: StopWhen[]): StopWhen => (ctx) => conds.every((c) => c(ctx));
export const not = (cond: StopWhen): StopWhen => (ctx) => !cond(ctx);

