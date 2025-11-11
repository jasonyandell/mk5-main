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
    { type: 'observe-hands', playerIndices: [playerIndex] }
  ];
}

/**
 * Standard capabilities for an AI player.
 * Can act as player and observe their own hand.
 *
 * Vision spec §4.3: aiCapabilities(playerIndex)
 */
export function aiCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-hands', playerIndices: [playerIndex] }
  ];
}

/**
 * Standard capabilities for a spectator.
 * Can observe all hands but cannot execute actions.
 *
 * Vision spec §4.3: spectatorCapabilities
 */
export function spectatorCapabilities(): Capability[] {
  return [
    { type: 'observe-hands', playerIndices: 'all' }
  ];
}


/**
 * Builder for custom capability sets.
 * Provides fluent API for composing capabilities.
 *
 * @example
 * const caps = buildCapabilities()
 *   .actAsPlayer(0)
 *   .observeHands([0, 1])
 *   .build();
 */
export class CapabilityBuilder {
  private capabilities: Capability[] = [];

  actAsPlayer(playerIndex: 0 | 1 | 2 | 3): this {
    this.capabilities.push({ type: 'act-as-player', playerIndex });
    return this;
  }

  observeHands(playerIndices: number[] | 'all'): this {
    this.capabilities.push({ type: 'observe-hands', playerIndices });
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
