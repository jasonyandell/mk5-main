import type { SectionSpec } from './sectionRunner';
import { whenFirstTransition, whenPhaseIn, whenPlayCount, whenTrickCompleted } from './stopConditions';

export function oneTransition(): SectionSpec {
  return {
    name: 'oneTransition',
    allow: () => true,
    stopWhen: whenFirstTransition()
  };
}

export function onePlay(): SectionSpec {
  return {
    name: 'onePlay',
    allow: (t) => t.action.type === 'play',
    stopWhen: whenPlayCount(1)
  };
}

export function oneTrick(): SectionSpec {
  return {
    name: 'oneTrick',
    allow: (t) =>
      t.id.startsWith('play-') || t.id.startsWith('agree-complete-trick') || t.id === 'complete-trick',
    stopWhen: whenTrickCompleted()
    // TODO: Handle consensus as part of game state machine
  };
}

export function oneHand(): SectionSpec {
  return {
    name: 'oneHand',
    allow: () => true,
    stopWhen: whenPhaseIn('scoring', 'game_end'),
    // TODO: Handle consensus for AI players as part of game state machine
    autoScoreHand: false,
    aiSpeed: 'slow'
  };
}

export function oneGame(): SectionSpec {
  return {
    name: 'oneGame',
    allow: () => true,
    stopWhen: whenPhaseIn('game_end')
    // TODO: Handle consensus as part of game state machine
  };
}

// Resolve a section preset by scenario slug (e.g., 'one_hand', 'one_trick')
export function fromSlug(slug: string): (() => SectionSpec) | null {
  const norm = slug
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  const map: Record<string, () => SectionSpec> = {
    one_transition: oneTransition,
    one_play: onePlay,
    one_trick: oneTrick,
    one_hand: oneHand,
    one_game: oneGame
  };
  return map[norm] || null;
}
