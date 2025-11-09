/**
 * Standard capability builders per vision document §4.3
 *
 * These helper functions provide standard sets of capabilities for common player types.
 * They replace scattered inline capability construction with reusable, tested builders.
 */

import type { Capability } from './types';

/**
 * Standard capabilities for a human player.
 * Can act as player and observe their own hand.
 *
 * Vision spec §4.3: humanCapabilities(playerIndex)
 */
export function humanCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-own-hand' }
  ];
}

/**
 * Standard capabilities for an AI player.
 * Can act as player, observe their own hand, and can be replaced by human.
 *
 * Vision spec §4.3: aiCapabilities(playerIndex)
 */
export function aiCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-own-hand' },
    { type: 'replace-ai' }
  ];
}

/**
 * Standard capabilities for a spectator.
 * Can observe all hands and full state but cannot execute actions.
 *
 * Vision spec §4.3: spectatorCapabilities
 */
export function spectatorCapabilities(): Capability[] {
  return [
    { type: 'observe-all-hands' },
    { type: 'observe-full-state' }
  ];
}

/**
 * Standard capabilities for a coach watching a specific student.
 * Can see the student's hand and hints, but cannot execute actions.
 *
 * Vision spec §4.3: coachCapabilities(studentIndex)
 */
export function coachCapabilities(studentIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'observe-hand', playerIndex: studentIndex },
    { type: 'see-hints' }
  ];
}

/**
 * Standard capabilities for a tutorial student.
 * Can act as player, see hints, and undo actions for learning.
 *
 * Vision spec §4.3: tutorialCapabilities(playerIndex)
 */
export function tutorialCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-own-hand' },
    { type: 'see-hints' },
    { type: 'undo-actions' }
  ];
}

/**
 * Builder for custom capability sets.
 * Provides fluent API for composing capabilities.
 *
 * @example
 * const caps = buildCapabilities()
 *   .actAsPlayer(0)
 *   .observeOwnHand()
 *   .seeHints()
 *   .build();
 */
export class CapabilityBuilder {
  private capabilities: Capability[] = [];

  actAsPlayer(playerIndex: 0 | 1 | 2 | 3): this {
    this.capabilities.push({ type: 'act-as-player', playerIndex });
    return this;
  }

  observeOwnHand(): this {
    this.capabilities.push({ type: 'observe-own-hand' });
    return this;
  }

  observeHand(playerIndex: 0 | 1 | 2 | 3): this {
    this.capabilities.push({ type: 'observe-hand', playerIndex });
    return this;
  }

  observeAllHands(): this {
    this.capabilities.push({ type: 'observe-all-hands' });
    return this;
  }

  observeFullState(): this {
    this.capabilities.push({ type: 'observe-full-state' });
    return this;
  }

  seeHints(): this {
    this.capabilities.push({ type: 'see-hints' });
    return this;
  }

  seeAIIntent(): this {
    this.capabilities.push({ type: 'see-ai-intent' });
    return this;
  }

  replaceAI(): this {
    this.capabilities.push({ type: 'replace-ai' });
    return this;
  }

  configureActionTransformer(): this {
    this.capabilities.push({ type: 'configure-action-transformer' });
    return this;
  }

  undoActions(): this {
    this.capabilities.push({ type: 'undo-actions' });
    return this;
  }

  build(): Capability[] {
    return [...this.capabilities];
  }
}

/**
 * Create a new capability builder.
 */
export function buildCapabilities(): CapabilityBuilder {
  return new CapabilityBuilder();
}

/**
 * Build base capability set per control type.
 *
 * Pure helper that returns standard capabilities for human or AI players.
 * This is the canonical way to initialize capabilities when creating or updating player sessions.
 *
 * Vision spec §4.3: Uses humanCapabilities() or aiCapabilities()
 *
 * @param playerIndex - The player index (0-3)
 * @param controlType - Either 'human' or 'ai'
 * @returns Array of capabilities appropriate for the control type
 */
export function buildBaseCapabilities(playerIndex: number, controlType: 'human' | 'ai'): Capability[] {
  const idx = playerIndex as 0 | 1 | 2 | 3;
  return controlType === 'human'
    ? humanCapabilities(idx)
    : aiCapabilities(idx);
}
