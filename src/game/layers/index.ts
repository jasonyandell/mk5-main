/**
 * Layers module - Pure functional game rule composition.
 *
 * Exports all layers, composition utilities, and types for the
 * threaded rules architecture.
 */

// Types
export * from './types';

// Helpers
export * from './helpers';

// Layers
export { baseLayer } from './base';
export { nelloLayer } from './nello';
export { plungeLayer } from './plunge';
export { splashLayer } from './splash';
export { sevensLayer } from './sevens';
export { tournamentLayer } from './tournament';
export { oneHandLayer } from './oneHand';
export { speedLayer } from './speed';
export { hintsLayer } from './hints';
export { consensusLayer } from './consensus';

// Composition
export { composeRules, composeGetValidActions } from './compose';
export { LAYER_REGISTRY, getLayerByName, getLayersByNames } from './registry';

// Base rule implementations (canonical source of truth for trump/suit/follow-suit)
export {
  getLedSuitBase,
  suitsWithTrumpBase,
  canFollowBase,
  rankInTrickBase,
  isTrumpBase,
  isValidPlayBase,
  getValidPlaysBase
} from './rules-base';
