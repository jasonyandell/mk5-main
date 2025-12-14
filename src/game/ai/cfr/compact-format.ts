/**
 * Compact Binary Format for MCCFR Strategies
 *
 * Designed for mobile game deployment - target: < 10MB for trained strategy.
 *
 * Two format modes:
 * 1. Full format (.cfr.gz) - includes regrets for continued training
 * 2. Deploy format (.cfr-deploy.gz) - strategy only, smallest size
 *
 * Format Overview:
 * ================
 * Header (16 bytes):
 *   - Magic: "CFR1" or "CFD1" (4 bytes) - CFR1=full, CFD1=deploy
 *   - Version: uint8 (1 byte)
 *   - Trump type count: uint8 (1 byte) - typically 3
 *   - Node count: uint32 (4 bytes)
 *   - Iterations: uint32 (4 bytes)
 *   - Flags: uint16 (2 bytes) - reserved
 *
 * Info Set Key Encoding (4 bytes = 32 bits):
 *   Bit layout:
 *   - bits 0-4: Count dominoes bitmap (5 bits) - which of 5 count dominoes in hand
 *   - bits 5-7: US score bucket (3 bits) - 0-7 (0,5,10,15,20,25,30,35 / 5)
 *   - bits 8-10: THEM score bucket (3 bits)
 *   - bits 11-13: POT score bucket (3 bits)
 *   - bit 14: LEAD (1 bit)
 *   - bits 15-17: TR trump count (3 bits) - 0-7
 *   - bits 18-20: TK trick number (3 bits) - 0-6
 *   - bits 21-22: POS position (2 bits) - 0-3
 *   - bits 23-25: NC non-count (3 bits) - 0-7
 *   - bits 26-27: TT trump type (2 bits) - suit=0, doubles=1, no-trump=2
 *   - bits 28-31: reserved (4 bits)
 *
 * Domino Action Encoding (5 bits):
 *   28 dominoes = 0-27, packed as 5 bits each
 *   Domino order: 0-0, 1-0, 1-1, 2-0, 2-1, 2-2, ... (sorted by id)
 *
 * Deploy Format Node Data (smallest - strategy weights only):
 *   - Info set key: uint32 (4 bytes)
 *   - Action count: uint8 (1 byte) - 1-7 typical
 *   - Actions packed: ceil(actionCount * 5 / 8) bytes
 *   - Strategy weights: uint8 per action (1 byte each) - quantized 0-255
 *
 * Full Format Node Data (includes regrets):
 *   - Info set key: uint32 (4 bytes)
 *   - Action count: uint8 (1 byte) - 1-7 typical
 *   - Actions packed: ceil(actionCount * 5 / 8) bytes
 *   - Max weight: float32 (4 bytes) - for denormalization
 *   - Regrets: int16 per action (2 bytes each) - quantized
 *   - Strategy weights: uint16 per action (2 bytes each) - quantized 0-65535
 *
 * Compression:
 *   Final output is gzipped (.cfr.gz or .cfr-deploy.gz extension)
 */

import type { SerializedStrategy, InfoSetKey, ActionKey } from './types';

// ============================================================================
// Constants
// ============================================================================

const MAGIC_FULL = 'CFR1';
const MAGIC_DEPLOY = 'CFD1';
const VERSION = 1;
const HEADER_SIZE = 16;

/** Count dominoes in order for bitmap encoding */
const COUNT_DOMINOES = ['3-2', '4-1', '5-0', '5-5', '6-4'] as const;
const COUNT_DOMINO_SET = new Set<string>(COUNT_DOMINOES);
const COUNT_DOMINO_TO_BIT: Record<string, number> = {
  '3-2': 0,
  '4-1': 1,
  '5-0': 2,
  '5-5': 3,
  '6-4': 4
};

/** Trump type encoding */
const TRUMP_TYPE_MAP: Record<string, number> = {
  'suit': 0,
  'doubles': 1,
  'no-trump': 2
};
const TRUMP_TYPE_REVERSE = ['suit', 'doubles', 'no-trump'] as const;

/** All 28 dominoes in sorted order for action encoding */
const ALL_DOMINOES: string[] = [];
for (let high = 0; high <= 6; high++) {
  for (let low = 0; low <= high; low++) {
    ALL_DOMINOES.push(`${high}-${low}`);
  }
}
const DOMINO_TO_INDEX = new Map(ALL_DOMINOES.map((d, i) => [d, i]));

// ============================================================================
// Info Set Key Encoding/Decoding
// ============================================================================

/**
 * Parse a string info set key into components.
 * Format: "C:3-2,5-5|US:10|THEM:5|POT:0|LEAD:1|TR:2|TK:3|POS:1|NC:4|TT:suit"
 */
function parseInfoSetKey(key: InfoSetKey): {
  countDominoes: string[];
  us: number;
  them: number;
  pot: number;
  lead: boolean;
  tr: number;
  tk: number;
  pos: number;
  nc: number;
  tt: string;
} {
  const parts = key.split('|');
  const result: Record<string, string> = {};

  for (const part of parts) {
    const colonIdx = part.indexOf(':');
    if (colonIdx > 0) {
      result[part.slice(0, colonIdx)] = part.slice(colonIdx + 1);
    }
  }

  const countStr = result['C'] ?? 'none';
  const countDominoes = countStr === 'none' ? [] : countStr.split(',');

  return {
    countDominoes,
    us: parseInt(result['US'] ?? '0', 10),
    them: parseInt(result['THEM'] ?? '0', 10),
    pot: parseInt(result['POT'] ?? '0', 10),
    lead: result['LEAD'] === '1',
    tr: parseInt(result['TR'] ?? '0', 10),
    tk: parseInt(result['TK'] ?? '0', 10),
    pos: parseInt(result['POS'] ?? '0', 10),
    nc: parseInt(result['NC'] ?? '0', 10),
    tt: result['TT'] ?? 'suit'
  };
}

/**
 * Encode info set key to 32-bit integer.
 */
export function encodeInfoSetKey(key: InfoSetKey): number {
  const parsed = parseInfoSetKey(key);

  let encoded = 0;

  // Bits 0-4: Count dominoes bitmap
  let countBitmap = 0;
  for (const d of parsed.countDominoes) {
    if (COUNT_DOMINO_SET.has(d)) {
      const bit = COUNT_DOMINO_TO_BIT[d];
      if (bit !== undefined) {
        countBitmap |= (1 << bit);
      }
    }
  }
  encoded |= (countBitmap & 0x1F);

  // Bits 5-7: US score bucket (divide by 5, clamp to 0-7)
  encoded |= ((Math.min(7, Math.floor(parsed.us / 5)) & 0x7) << 5);

  // Bits 8-10: THEM score bucket
  encoded |= ((Math.min(7, Math.floor(parsed.them / 5)) & 0x7) << 8);

  // Bits 11-13: POT score bucket
  encoded |= ((Math.min(7, Math.floor(parsed.pot / 5)) & 0x7) << 11);

  // Bit 14: LEAD
  encoded |= ((parsed.lead ? 1 : 0) << 14);

  // Bits 15-17: TR
  encoded |= ((Math.min(7, parsed.tr) & 0x7) << 15);

  // Bits 18-20: TK
  encoded |= ((Math.min(7, parsed.tk) & 0x7) << 18);

  // Bits 21-22: POS
  encoded |= ((parsed.pos & 0x3) << 21);

  // Bits 23-25: NC
  encoded |= ((Math.min(7, parsed.nc) & 0x7) << 23);

  // Bits 26-27: TT
  encoded |= ((TRUMP_TYPE_MAP[parsed.tt] ?? 0) << 26);

  return encoded >>> 0; // Ensure unsigned
}

/**
 * Decode 32-bit integer to info set key string.
 */
export function decodeInfoSetKey(encoded: number): InfoSetKey {
  // Bits 0-4: Count dominoes bitmap
  const countBitmap = encoded & 0x1F;
  const countDominoes: string[] = [];
  for (let i = 0; i < 5; i++) {
    if (countBitmap & (1 << i)) {
      const domino = COUNT_DOMINOES[i];
      if (domino) {
        countDominoes.push(domino);
      }
    }
  }

  const us = ((encoded >> 5) & 0x7) * 5;
  const them = ((encoded >> 8) & 0x7) * 5;
  const pot = ((encoded >> 11) & 0x7) * 5;
  const lead = ((encoded >> 14) & 0x1) === 1;
  const tr = (encoded >> 15) & 0x7;
  const tk = (encoded >> 18) & 0x7;
  const pos = (encoded >> 21) & 0x3;
  const nc = (encoded >> 23) & 0x7;
  const ttIndex = (encoded >> 26) & 0x3;
  const tt = TRUMP_TYPE_REVERSE[ttIndex] ?? 'suit';

  const countStr = countDominoes.length > 0 ? countDominoes.sort().join(',') : 'none';

  return `C:${countStr}|US:${us}|THEM:${them}|POT:${pot}|LEAD:${lead ? 1 : 0}|TR:${tr}|TK:${tk}|POS:${pos}|NC:${nc}|TT:${tt}`;
}

// ============================================================================
// Action (Domino) Encoding
// ============================================================================

/**
 * Encode a domino action key to a 5-bit index.
 */
export function encodeAction(action: ActionKey): number {
  return DOMINO_TO_INDEX.get(action) ?? 0;
}

/**
 * Decode a 5-bit index to domino action key.
 */
export function decodeAction(index: number): ActionKey {
  return ALL_DOMINOES[index] ?? '0-0';
}

/**
 * Pack multiple 5-bit action indices into bytes.
 */
export function packActions(actions: number[]): Uint8Array {
  const bitCount = actions.length * 5;
  const byteCount = Math.ceil(bitCount / 8);
  const bytes = new Uint8Array(byteCount);

  let bitOffset = 0;
  for (const action of actions) {
    // Write 5 bits starting at bitOffset
    const byteIdx = Math.floor(bitOffset / 8);
    const bitInByte = bitOffset % 8;

    // First byte - safe because byteIdx is always in range
    const currentByte = bytes[byteIdx] ?? 0;
    bytes[byteIdx] = currentByte | ((action << bitInByte) & 0xFF);

    // Overflow to second byte if needed
    if (bitInByte > 3 && byteIdx + 1 < bytes.length) {
      const nextByte = bytes[byteIdx + 1] ?? 0;
      bytes[byteIdx + 1] = nextByte | ((action >> (8 - bitInByte)) & 0xFF);
    }

    bitOffset += 5;
  }

  return bytes;
}

/**
 * Unpack 5-bit action indices from bytes.
 */
export function unpackActions(bytes: Uint8Array, count: number): number[] {
  const actions: number[] = [];

  let bitOffset = 0;
  for (let i = 0; i < count; i++) {
    const byteIdx = Math.floor(bitOffset / 8);
    const bitInByte = bitOffset % 8;

    const firstByte = bytes[byteIdx] ?? 0;

    // Read 5 bits
    let value: number;

    // If we need bits from next byte
    if (bitInByte > 3 && byteIdx + 1 < bytes.length) {
      const bitsFromFirst = 8 - bitInByte;
      const secondByte = bytes[byteIdx + 1] ?? 0;
      value = ((firstByte >> bitInByte) | (secondByte << bitsFromFirst)) & 0x1F;
    } else {
      value = (firstByte >> bitInByte) & 0x1F;
    }

    actions.push(value);
    bitOffset += 5;
  }

  return actions;
}

// ============================================================================
// Value Quantization
// ============================================================================

/**
 * Quantize a regret value to int16.
 * Regrets can be large positive or negative, so we clip to ±32767.
 */
export function quantizeRegret(regret: number): number {
  return Math.max(-32767, Math.min(32767, Math.round(regret)));
}

/**
 * Quantize a strategy sum value to uint16.
 * Strategy sums are always non-negative. We normalize by the max in the node.
 */
export function quantizeStrategyWeight(weight: number, maxWeight: number): number {
  if (maxWeight <= 0) return 0;
  const normalized = weight / maxWeight;
  return Math.round(normalized * 65535);
}

/**
 * Dequantize strategy weight back to probability-proportional value.
 */
export function dequantizeStrategyWeight(quantized: number): number {
  return quantized / 65535;
}

// ============================================================================
// Full Format Serialization (includes regrets)
// ============================================================================

export interface CompactHeader {
  version: number;
  nodeCount: number;
  iterations: number;
}

/**
 * Serialize a strategy to compact binary format (full - includes regrets).
 * Returns a Buffer that should be gzipped for final storage.
 */
export function serializeCompact(strategy: SerializedStrategy): Buffer {
  const { nodes, iterationsCompleted } = strategy;

  // Filter to nodes with data
  const validNodes = nodes.filter(n => n.regrets.length > 0);

  // Calculate buffer size
  // Header: 16 bytes
  // Per node: 4 (key) + 1 (count) + ceil(count*5/8) (actions) + 2*count (regrets) + 2*count (strat) + 4 (maxWeight)
  let totalSize = HEADER_SIZE;
  for (const node of validNodes) {
    const actionCount = node.regrets.length;
    totalSize += 4 + 1 + Math.ceil(actionCount * 5 / 8) + actionCount * 2 + actionCount * 2 + 4;
  }

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Write header
  buffer.write(MAGIC_FULL, offset, 4, 'ascii');
  offset += 4;
  buffer.writeUInt8(VERSION, offset++);
  buffer.writeUInt8(3, offset++); // Trump type count
  buffer.writeUInt32LE(validNodes.length, offset);
  offset += 4;
  buffer.writeUInt32LE(iterationsCompleted, offset);
  offset += 4;
  buffer.writeUInt16LE(0, offset); // Reserved flags
  offset += 2;

  // Write nodes
  for (const node of validNodes) {
    const actionCount = node.regrets.length;

    // Info set key (4 bytes)
    const keyEncoded = encodeInfoSetKey(node.key);
    buffer.writeUInt32LE(keyEncoded, offset);
    offset += 4;

    // Action count (1 byte)
    buffer.writeUInt8(actionCount, offset++);

    // Find max strategy weight for normalization
    let maxWeight = 0;
    for (const [, w] of node.strategySum) {
      maxWeight = Math.max(maxWeight, w);
    }

    // Store max weight as float32 (4 bytes) for denormalization
    buffer.writeFloatLE(maxWeight, offset);
    offset += 4;

    // Pack action indices
    const actionIndices = node.regrets.map(([action]) => encodeAction(action));
    const packedActions = packActions(actionIndices);
    for (let i = 0; i < packedActions.length; i++) {
      buffer.writeUInt8(packedActions[i] ?? 0, offset + i);
    }
    offset += packedActions.length;

    // Write quantized regrets (int16)
    for (const [, regret] of node.regrets) {
      buffer.writeInt16LE(quantizeRegret(regret), offset);
      offset += 2;
    }

    // Write quantized strategy weights (uint16)
    for (const [, weight] of node.strategySum) {
      buffer.writeUInt16LE(quantizeStrategyWeight(weight, maxWeight), offset);
      offset += 2;
    }
  }

  return buffer.subarray(0, offset);
}

/**
 * Deserialize compact binary format to strategy.
 */
export function deserializeCompact(buffer: Buffer): SerializedStrategy {
  let offset = 0;

  // Read header
  const magic = buffer.toString('ascii', offset, offset + 4);
  offset += 4;

  const isDeploy = magic === MAGIC_DEPLOY;
  if (magic !== MAGIC_FULL && magic !== MAGIC_DEPLOY) {
    throw new Error(`Invalid magic: expected ${MAGIC_FULL} or ${MAGIC_DEPLOY}, got ${magic}`);
  }

  const version = buffer.readUInt8(offset++);
  if (version !== VERSION) {
    throw new Error(`Unsupported version: ${version}`);
  }

  offset++; // Skip trump type count
  const nodeCount = buffer.readUInt32LE(offset);
  offset += 4;
  const iterations = buffer.readUInt32LE(offset);
  offset += 4;
  offset += 2; // Skip flags

  // Read nodes
  const nodes: SerializedStrategy['nodes'] = [];

  while (offset < buffer.length && nodes.length < nodeCount) {
    // Info set key
    const keyEncoded = buffer.readUInt32LE(offset);
    offset += 4;
    const key = decodeInfoSetKey(keyEncoded);

    // Action count
    const actionCount = buffer.readUInt8(offset++);
    if (actionCount === 0) continue;

    if (isDeploy) {
      // Deploy format: no max weight, no regrets, uint8 weights
      const packedSize = Math.ceil(actionCount * 5 / 8);
      const packedBytes = new Uint8Array(buffer.subarray(offset, offset + packedSize));
      offset += packedSize;
      const actionIndices = unpackActions(packedBytes, actionCount);

      // Read uint8 strategy weights and normalize
      const strategySum: Array<[ActionKey, number]> = [];
      let totalWeight = 0;
      const rawWeights: number[] = [];
      for (let i = 0; i < actionCount; i++) {
        const w = buffer.readUInt8(offset++);
        rawWeights.push(w);
        totalWeight += w;
      }

      for (let i = 0; i < actionCount; i++) {
        const actionIndex = actionIndices[i];
        const prob = totalWeight > 0 ? (rawWeights[i] ?? 0) / totalWeight : 1 / actionCount;
        strategySum.push([decodeAction(actionIndex ?? 0), prob]);
      }

      nodes.push({
        key,
        regrets: [], // No regrets in deploy format
        strategySum,
        visitCount: 0
      });
    } else {
      // Full format
      const maxWeight = buffer.readFloatLE(offset);
      offset += 4;

      const packedSize = Math.ceil(actionCount * 5 / 8);
      const packedBytes = new Uint8Array(buffer.subarray(offset, offset + packedSize));
      offset += packedSize;
      const actionIndices = unpackActions(packedBytes, actionCount);

      // Read regrets
      const regrets: Array<[ActionKey, number]> = [];
      for (let i = 0; i < actionCount; i++) {
        const regret = buffer.readInt16LE(offset);
        offset += 2;
        const actionIndex = actionIndices[i];
        regrets.push([decodeAction(actionIndex ?? 0), regret]);
      }

      // Read strategy weights and denormalize
      const strategySum: Array<[ActionKey, number]> = [];
      for (let i = 0; i < actionCount; i++) {
        const quantized = buffer.readUInt16LE(offset);
        offset += 2;
        const weight = dequantizeStrategyWeight(quantized) * maxWeight;
        const actionIndex = actionIndices[i];
        strategySum.push([decodeAction(actionIndex ?? 0), weight]);
      }

      nodes.push({
        key,
        regrets,
        strategySum,
        visitCount: 0
      });
    }
  }

  return {
    version: 1,
    config: {
      iterations,
      seed: 0
    },
    iterationsCompleted: iterations,
    nodes,
    trainedAt: new Date().toISOString(),
    trainingTimeMs: 0
  };
}

// ============================================================================
// Deploy Format Serialization (strategy only - smallest)
// ============================================================================

/**
 * Serialize a strategy to deploy format (strategy weights only).
 * This is the smallest format, suitable for mobile deployment.
 */
export function serializeCompactDeploy(strategy: SerializedStrategy): Buffer {
  const { nodes, iterationsCompleted } = strategy;

  // Filter to nodes with data
  const validNodes = nodes.filter(n => n.strategySum.length > 0);

  // Calculate buffer size
  // Header: 16 bytes
  // Per node: 4 (key) + 1 (count) + ceil(count*5/8) (actions) + 1*count (weights)
  let totalSize = HEADER_SIZE;
  for (const node of validNodes) {
    const actionCount = node.strategySum.length;
    totalSize += 4 + 1 + Math.ceil(actionCount * 5 / 8) + actionCount;
  }

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Write header with deploy magic
  buffer.write(MAGIC_DEPLOY, offset, 4, 'ascii');
  offset += 4;
  buffer.writeUInt8(VERSION, offset++);
  buffer.writeUInt8(3, offset++); // Trump type count
  buffer.writeUInt32LE(validNodes.length, offset);
  offset += 4;
  buffer.writeUInt32LE(iterationsCompleted, offset);
  offset += 4;
  buffer.writeUInt16LE(0, offset); // Reserved flags
  offset += 2;

  // Write nodes
  for (const node of validNodes) {
    const actionCount = node.strategySum.length;

    // Info set key (4 bytes)
    const keyEncoded = encodeInfoSetKey(node.key);
    buffer.writeUInt32LE(keyEncoded, offset);
    offset += 4;

    // Action count (1 byte)
    buffer.writeUInt8(actionCount, offset++);

    // Pack action indices
    const actionIndices = node.strategySum.map(([action]) => encodeAction(action));
    const packedActions = packActions(actionIndices);
    for (let i = 0; i < packedActions.length; i++) {
      buffer.writeUInt8(packedActions[i] ?? 0, offset + i);
    }
    offset += packedActions.length;

    // Normalize weights to probabilities, then scale to 0-255
    let totalSum = 0;
    for (const [, w] of node.strategySum) {
      totalSum += w;
    }

    for (const [, weight] of node.strategySum) {
      const prob = totalSum > 0 ? weight / totalSum : 1 / actionCount;
      const quantized = Math.min(255, Math.max(0, Math.round(prob * 255)));
      buffer.writeUInt8(quantized, offset++);
    }
  }

  return buffer.subarray(0, offset);
}

// ============================================================================
// Runtime Strategy Lookup (Optimized for mobile)
// ============================================================================

/**
 * Compact runtime strategy - optimized for fast lookups during gameplay.
 * Uses Map<number, Float32Array> for O(1) lookups with minimal memory.
 */
export class CompactStrategy {
  private strategyMap: Map<number, { actions: number[]; weights: Float32Array }>;

  constructor() {
    this.strategyMap = new Map();
  }

  /**
   * Load from serialized strategy.
   */
  static fromSerialized(strategy: SerializedStrategy): CompactStrategy {
    const compact = new CompactStrategy();

    for (const node of strategy.nodes) {
      if (node.strategySum.length === 0) continue;

      const keyEncoded = encodeInfoSetKey(node.key);
      const actions = node.strategySum.map(([action]) => encodeAction(action));

      // Compute average strategy (normalize strategy sum)
      let totalSum = 0;
      for (const [, sum] of node.strategySum) {
        totalSum += sum;
      }

      const weights = new Float32Array(node.strategySum.length);
      if (totalSum > 0) {
        for (let i = 0; i < node.strategySum.length; i++) {
          const entry = node.strategySum[i];
          if (entry) {
            weights[i] = entry[1] / totalSum;
          }
        }
      } else {
        // Uniform if no strategy accumulated
        const uniform = 1 / node.strategySum.length;
        for (let i = 0; i < weights.length; i++) {
          weights[i] = uniform;
        }
      }

      compact.strategyMap.set(keyEncoded, { actions, weights });
    }

    return compact;
  }

  /**
   * Load from compact binary buffer.
   */
  static fromCompactBuffer(buffer: Buffer): CompactStrategy {
    const strategy = deserializeCompact(buffer);
    return CompactStrategy.fromSerialized(strategy);
  }

  /**
   * Get action probabilities for an info set key.
   * Returns Map<ActionKey, probability>
   */
  getStrategy(infoSetKey: InfoSetKey): Map<ActionKey, number> {
    const keyEncoded = encodeInfoSetKey(infoSetKey);
    const entry = this.strategyMap.get(keyEncoded);

    const result = new Map<ActionKey, number>();
    if (!entry) {
      return result; // Empty - caller should use uniform over legal actions
    }

    for (let i = 0; i < entry.actions.length; i++) {
      const actionIndex = entry.actions[i];
      const weight = entry.weights[i];
      if (actionIndex !== undefined && weight !== undefined) {
        result.set(decodeAction(actionIndex), weight);
      }
    }

    return result;
  }

  /**
   * Sample an action according to strategy.
   */
  sampleAction(infoSetKey: InfoSetKey, legalActions: ActionKey[], rng: () => number): ActionKey {
    const strategy = this.getStrategy(infoSetKey);

    const fallback = legalActions[Math.floor(rng() * legalActions.length)];
    if (!fallback) {
      return '0-0'; // Should never happen with valid input
    }

    if (strategy.size === 0) {
      return fallback;
    }

    // Build probability distribution over legal actions
    const probs: number[] = [];
    let totalProb = 0;

    for (const action of legalActions) {
      const prob = strategy.get(action) ?? 0;
      probs.push(prob);
      totalProb += prob;
    }

    // Normalize if needed (handles missing actions)
    if (totalProb <= 0) {
      return fallback;
    }

    // Sample
    let r = rng() * totalProb;
    for (let i = 0; i < legalActions.length; i++) {
      const prob = probs[i] ?? 0;
      r -= prob;
      if (r <= 0) {
        return legalActions[i] ?? fallback;
      }
    }

    return legalActions[legalActions.length - 1] ?? fallback;
  }

  /**
   * Get memory usage estimate in bytes.
   */
  getMemoryUsage(): number {
    let bytes = 0;
    for (const [, entry] of this.strategyMap) {
      bytes += 4; // key (number)
      bytes += entry.actions.length * 4; // actions array
      bytes += entry.weights.length * 4; // Float32Array
      bytes += 32; // object overhead estimate
    }
    return bytes;
  }

  /**
   * Get node count.
   */
  get size(): number {
    return this.strategyMap.size;
  }
}
