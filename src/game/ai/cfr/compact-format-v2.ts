/**
 * Ultra-Compact CFD2 Format for MCCFR Strategy Deployment
 *
 * Designed for mobile game deployment - target: < 10MB for 1M+ iteration strategy.
 *
 * Optimizations over CFD1:
 * 1. Action Set Dictionary - deduplicate repeated action sets (many info sets share legal actions)
 * 2. Delta Key Encoding - sort nodes by key, store differences with varint encoding
 * 3. Strategy Pattern Encoding - special compact representations for common patterns
 * 4. Varint Encoding - protobuf-style variable-length integers throughout
 *
 * Format Overview:
 * ================
 * Header (24 bytes):
 *   - Magic: "CFD2" (4 bytes)
 *   - Version: uint8 (1 byte)
 *   - Flags: uint8 (1 byte)
 *   - Node count: varint (up to 5 bytes, stored fixed 4 bytes in header)
 *   - Iterations: uint32 (4 bytes)
 *   - Action dict size: uint16 (2 bytes)
 *   - Padding: 8 bytes (reserved)
 *
 * Action Set Dictionary:
 *   - For each unique action set: count (1 byte) + packed actions (ceil(count*5/8) bytes)
 *   - Nodes reference by index (varint, typically 1-2 bytes)
 *
 * Node Data:
 *   - Delta key: varint (typically 1-3 bytes vs 4 bytes fixed)
 *   - Action set index: varint (typically 1-2 bytes)
 *   - Strategy pattern: 2 bits in action set index
 *     - 00: Uniform distribution (0 extra bytes)
 *     - 01: Dominant action >90% (1 byte - action local index)
 *     - 10: Binary split (2 bytes - dominant index + percentage)
 *     - 11: Full weights (1 byte per action, quantized 0-255)
 *
 * Total size estimate:
 *   - CFD1: 4 + 1 + ceil(n*5/8) + n bytes per node = ~8 bytes avg for 4 actions
 *   - CFD2: 2 + 2 + pattern bytes = ~4-5 bytes avg for 4 actions
 *   - Expected 40-50% reduction
 */

import type { SerializedStrategy, InfoSetKey, ActionKey } from './types';
import { encodeInfoSetKey, decodeInfoSetKey, encodeAction, decodeAction, packActions, unpackActions } from './compact-format';

// ============================================================================
// Constants
// ============================================================================

const MAGIC_V2 = 'CFD2';
const VERSION_V2 = 1;
const HEADER_SIZE_V2 = 24;

// Strategy pattern types (2 bits)
const PATTERN_UNIFORM = 0b00;      // All actions equal probability
const PATTERN_DOMINANT = 0b01;     // One action >90%
const PATTERN_BINARY = 0b10;       // Two actions sum to >95%
const PATTERN_FULL = 0b11;         // Full weight list

// Thresholds for pattern detection
const DOMINANT_THRESHOLD = 0.90;   // Action probability for dominant pattern
const BINARY_THRESHOLD = 0.95;     // Sum of top 2 for binary pattern

// ============================================================================
// Varint Encoding (Protobuf-style)
// ============================================================================

/**
 * Encode unsigned integer to varint bytes.
 * Each byte uses 7 bits for data, high bit indicates continuation.
 */
export function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  let v = value >>> 0; // Ensure unsigned

  while (v >= 0x80) {
    bytes.push((v & 0x7F) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);

  return new Uint8Array(bytes);
}

/**
 * Decode varint from buffer at offset.
 * Returns [value, bytesRead].
 */
export function decodeVarint(buffer: Uint8Array, offset: number): [number, number] {
  let result = 0;
  let shift = 0;
  let bytesRead = 0;

  while (offset + bytesRead < buffer.length) {
    const byte = buffer[offset + bytesRead]!;
    bytesRead++;

    result |= (byte & 0x7F) << shift;

    if ((byte & 0x80) === 0) {
      return [result >>> 0, bytesRead];
    }

    shift += 7;
    if (shift >= 35) {
      throw new Error('Varint too large');
    }
  }

  throw new Error('Unexpected end of buffer reading varint');
}

/**
 * Calculate varint encoded size for a value.
 */
export function varintSize(value: number): number {
  let v = value >>> 0;
  let size = 1;
  while (v >= 0x80) {
    size++;
    v >>>= 7;
  }
  return size;
}

// ============================================================================
// Action Set Dictionary
// ============================================================================

interface ActionSetEntry {
  actions: number[];  // Encoded action indices
  packed: Uint8Array; // Packed binary representation
}

/**
 * Build action set dictionary from nodes.
 * Returns [dictionary, nodeToIndex] where nodeToIndex maps node key to dict index.
 */
function buildActionSetDictionary(
  nodes: SerializedStrategy['nodes']
): [ActionSetEntry[], Map<string, number>] {
  const actionSetMap = new Map<string, number>(); // canonical string -> index
  const dictionary: ActionSetEntry[] = [];
  const nodeToIndex = new Map<string, number>();

  for (const node of nodes) {
    if (node.strategySum.length === 0) continue;

    // Create canonical representation (sorted action indices)
    const actions = node.strategySum.map(([action]) => encodeAction(action)).sort((a, b) => a - b);
    const canonical = actions.join(',');

    let dictIndex = actionSetMap.get(canonical);
    if (dictIndex === undefined) {
      dictIndex = dictionary.length;
      actionSetMap.set(canonical, dictIndex);
      dictionary.push({
        actions,
        packed: packActions(actions)
      });
    }

    nodeToIndex.set(node.key, dictIndex);
  }

  return [dictionary, nodeToIndex];
}

// ============================================================================
// Strategy Pattern Detection
// ============================================================================

interface PatternResult {
  type: number;
  data: Uint8Array;
}

/**
 * Analyze strategy weights and determine best pattern encoding.
 */
function detectPattern(weights: number[]): PatternResult {
  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    // No data - use uniform
    return { type: PATTERN_UNIFORM, data: new Uint8Array(0) };
  }

  // Normalize to probabilities
  const probs = weights.map(w => w / sum);

  // Check for uniform (all within 5% of expected)
  const expected = 1 / weights.length;
  const isUniform = probs.every(p => Math.abs(p - expected) < 0.05);
  if (isUniform) {
    return { type: PATTERN_UNIFORM, data: new Uint8Array(0) };
  }

  // Find max and second max
  let maxIdx = 0;
  let maxProb = probs[0] ?? 0;
  let secondIdx = -1;
  let secondProb = 0;

  for (let i = 1; i < probs.length; i++) {
    const p = probs[i] ?? 0;
    if (p > maxProb) {
      secondIdx = maxIdx;
      secondProb = maxProb;
      maxIdx = i;
      maxProb = p;
    } else if (p > secondProb) {
      secondIdx = i;
      secondProb = p;
    }
  }

  // Check for dominant (>90% on one action)
  if (maxProb >= DOMINANT_THRESHOLD) {
    return {
      type: PATTERN_DOMINANT,
      data: new Uint8Array([maxIdx])
    };
  }

  // Check for binary (top 2 sum to >95%)
  if (secondIdx >= 0 && maxProb + secondProb >= BINARY_THRESHOLD) {
    // Encode: dominant index (4 bits) | second index (4 bits), percentage byte
    const percentage = Math.round(maxProb * 255 / (maxProb + secondProb));
    return {
      type: PATTERN_BINARY,
      data: new Uint8Array([
        ((maxIdx & 0x0F) << 4) | (secondIdx & 0x0F),
        percentage
      ])
    };
  }

  // Full encoding - quantize all weights to uint8
  const fullData = new Uint8Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    const p = probs[i] ?? 0;
    fullData[i] = Math.min(255, Math.max(0, Math.round(p * 255)));
  }

  return { type: PATTERN_FULL, data: fullData };
}

/**
 * Decode pattern to weight array.
 */
function decodePattern(
  type: number,
  data: Uint8Array,
  actionCount: number
): number[] {
  switch (type) {
    case PATTERN_UNIFORM: {
      const uniform = 1 / actionCount;
      return Array(actionCount).fill(uniform);
    }

    case PATTERN_DOMINANT: {
      const dominantIdx = data[0] ?? 0;
      const weights = Array(actionCount).fill(0.0001); // Small epsilon for non-dominant
      if (dominantIdx < actionCount) {
        weights[dominantIdx] = 1;
      }
      return weights;
    }

    case PATTERN_BINARY: {
      const byte0 = data[0] ?? 0;
      const dominantIdx = (byte0 >> 4) & 0x0F;
      const secondIdx = byte0 & 0x0F;
      const percentage = (data[1] ?? 128) / 255;

      const weights = Array(actionCount).fill(0.0001);
      if (dominantIdx < actionCount) {
        weights[dominantIdx] = percentage;
      }
      if (secondIdx < actionCount) {
        weights[secondIdx] = 1 - percentage;
      }
      return weights;
    }

    case PATTERN_FULL:
    default: {
      const weights: number[] = [];
      for (let i = 0; i < actionCount; i++) {
        weights.push((data[i] ?? 0) / 255);
      }
      return weights;
    }
  }
}

// ============================================================================
// CFD2 Serialization
// ============================================================================

/**
 * Serialize strategy to CFD2 format.
 * Returns uncompressed buffer (caller should gzip).
 */
export function serializeCFD2(strategy: SerializedStrategy): Buffer {
  const { nodes, iterationsCompleted } = strategy;

  // Filter to nodes with strategy data
  const validNodes = nodes.filter(n => n.strategySum.length > 0);

  // Build action set dictionary
  const [dictionary, nodeToIndex] = buildActionSetDictionary(validNodes);

  // Sort nodes by encoded key for delta encoding
  const sortedNodes = validNodes
    .map(node => ({
      node,
      encodedKey: encodeInfoSetKey(node.key),
      dictIndex: nodeToIndex.get(node.key) ?? 0
    }))
    .sort((a, b) => a.encodedKey - b.encodedKey);

  // Calculate buffer size (estimate, may need to resize)
  // Header: 24 bytes
  // Dictionary: sum of (1 + packed.length) for each entry
  // Nodes: ~5 bytes average
  let dictSize = 0;
  for (const entry of dictionary) {
    dictSize += 1 + entry.packed.length;
  }

  const estimatedSize = HEADER_SIZE_V2 + dictSize + sortedNodes.length * 8;
  let buffer = Buffer.alloc(estimatedSize);
  let offset = 0;

  // Write header
  buffer.write(MAGIC_V2, offset, 4, 'ascii');
  offset += 4;
  buffer.writeUInt8(VERSION_V2, offset++);
  buffer.writeUInt8(0, offset++); // Flags
  buffer.writeUInt32LE(sortedNodes.length, offset);
  offset += 4;
  buffer.writeUInt32LE(iterationsCompleted, offset);
  offset += 4;
  buffer.writeUInt16LE(dictionary.length, offset);
  offset += 2;
  // Padding
  offset += 8;

  // Write action set dictionary
  for (const entry of dictionary) {
    buffer.writeUInt8(entry.actions.length, offset++);
    for (let i = 0; i < entry.packed.length; i++) {
      buffer.writeUInt8(entry.packed[i]!, offset++);
    }
  }

  // Write nodes with delta encoding
  let prevKey = 0;

  for (const { node, encodedKey, dictIndex } of sortedNodes) {
    // Ensure buffer has space
    if (offset + 20 > buffer.length) {
      const newBuffer = Buffer.alloc(buffer.length * 2);
      buffer.copy(newBuffer);
      buffer = newBuffer;
    }

    // Delta key (varint)
    const delta = encodedKey - prevKey;
    const deltaBytes = encodeVarint(delta);
    for (let i = 0; i < deltaBytes.length; i++) {
      buffer.writeUInt8(deltaBytes[i]!, offset++);
    }
    prevKey = encodedKey;

    // Get the dictionary entry for action ordering
    const dictEntry = dictionary[dictIndex]!;

    // Reorder weights to match dictionary's sorted action order
    const actionToWeight = new Map(
      node.strategySum.map(([action, w]) => [encodeAction(action), w])
    );
    const sortedWeights = dictEntry.actions.map(actionIdx => actionToWeight.get(actionIdx) ?? 0);

    // Detect pattern (now indices refer to sorted order)
    const pattern = detectPattern(sortedWeights);

    // Action set index with pattern type (index << 2 | pattern)
    const combinedIndex = (dictIndex << 2) | pattern.type;
    const indexBytes = encodeVarint(combinedIndex);
    for (let i = 0; i < indexBytes.length; i++) {
      buffer.writeUInt8(indexBytes[i]!, offset++);
    }

    // Pattern data
    for (let i = 0; i < pattern.data.length; i++) {
      buffer.writeUInt8(pattern.data[i]!, offset++);
    }
  }

  return buffer.subarray(0, offset);
}

// ============================================================================
// CFD2 Deserialization
// ============================================================================

/**
 * Deserialize CFD2 format to strategy.
 */
export function deserializeCFD2(buffer: Buffer): SerializedStrategy {
  const bytes = new Uint8Array(buffer);
  let offset = 0;

  // Read header
  const magic = buffer.toString('ascii', offset, offset + 4);
  offset += 4;

  if (magic !== MAGIC_V2) {
    throw new Error(`Invalid magic: expected ${MAGIC_V2}, got ${magic}`);
  }

  const version = bytes[offset++]!;
  if (version !== VERSION_V2) {
    throw new Error(`Unsupported CFD2 version: ${version}`);
  }

  offset++; // Skip flags
  const nodeCount = buffer.readUInt32LE(offset);
  offset += 4;
  const iterations = buffer.readUInt32LE(offset);
  offset += 4;
  const dictSize = buffer.readUInt16LE(offset);
  offset += 2;
  offset += 8; // Skip padding

  // Read action set dictionary
  const dictionary: ActionSetEntry[] = [];
  for (let i = 0; i < dictSize; i++) {
    const actionCount = bytes[offset++]!;
    const packedSize = Math.ceil(actionCount * 5 / 8);
    const packed = new Uint8Array(buffer.subarray(offset, offset + packedSize));
    offset += packedSize;

    const actions = unpackActions(packed, actionCount);
    dictionary.push({ actions, packed });
  }

  // Read nodes
  const nodes: SerializedStrategy['nodes'] = [];
  let prevKey = 0;

  while (nodes.length < nodeCount && offset < bytes.length) {
    // Delta key (varint)
    const [delta, deltaBytes] = decodeVarint(bytes, offset);
    offset += deltaBytes;
    const encodedKey = prevKey + delta;
    prevKey = encodedKey;

    // Combined index (varint)
    const [combined, indexBytes] = decodeVarint(bytes, offset);
    offset += indexBytes;

    const patternType = combined & 0x03;
    const dictIndex = combined >>> 2;

    // Get action set from dictionary
    const entry = dictionary[dictIndex];
    if (!entry) {
      throw new Error(`Invalid dictionary index: ${dictIndex}`);
    }

    const actionCount = entry.actions.length;

    // Determine pattern data size
    let patternDataSize = 0;
    switch (patternType) {
      case PATTERN_UNIFORM:
        patternDataSize = 0;
        break;
      case PATTERN_DOMINANT:
        patternDataSize = 1;
        break;
      case PATTERN_BINARY:
        patternDataSize = 2;
        break;
      case PATTERN_FULL:
        patternDataSize = actionCount;
        break;
    }

    const patternData = new Uint8Array(buffer.subarray(offset, offset + patternDataSize));
    offset += patternDataSize;

    // Decode pattern to weights
    const weights = decodePattern(patternType, patternData, actionCount);

    // Build strategySum
    const strategySum: Array<[ActionKey, number]> = [];
    for (let i = 0; i < actionCount; i++) {
      const actionIndex = entry.actions[i];
      if (actionIndex !== undefined) {
        strategySum.push([decodeAction(actionIndex), weights[i] ?? 0]);
      }
    }

    // Decode info set key
    const key = decodeInfoSetKey(encodedKey);

    nodes.push({
      key,
      regrets: [], // Deploy format has no regrets
      strategySum,
      visitCount: 0
    });
  }

  return {
    version: 2,
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
// Runtime Strategy (Optimized for mobile)
// ============================================================================

/**
 * Ultra-compact runtime strategy for CFD2 format.
 * Optimized for fast lookups with minimal memory footprint.
 */
export class CompactStrategyV2 {
  private strategyMap: Map<number, { actions: number[]; weights: Float32Array }>;

  constructor() {
    this.strategyMap = new Map();
  }

  /**
   * Load from CFD2 buffer (uncompressed).
   */
  static fromBuffer(buffer: Buffer): CompactStrategyV2 {
    const strategy = deserializeCFD2(buffer);
    return CompactStrategyV2.fromSerialized(strategy);
  }

  /**
   * Load from serialized strategy.
   */
  static fromSerialized(strategy: SerializedStrategy): CompactStrategyV2 {
    const compact = new CompactStrategyV2();

    for (const node of strategy.nodes) {
      if (node.strategySum.length === 0) continue;

      const keyEncoded = encodeInfoSetKey(node.key);
      const actions = node.strategySum.map(([action]) => encodeAction(action));

      // Normalize weights to probabilities
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
   * Get action probabilities for an info set key.
   */
  getStrategy(infoSetKey: InfoSetKey): Map<ActionKey, number> {
    const keyEncoded = encodeInfoSetKey(infoSetKey);
    const entry = this.strategyMap.get(keyEncoded);

    const result = new Map<ActionKey, number>();
    if (!entry) {
      return result;
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
      return '0-0';
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
      bytes += 4; // key
      bytes += entry.actions.length * 4; // actions array
      bytes += entry.weights.length * 4; // Float32Array
      bytes += 32; // object overhead
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

// ============================================================================
// Format Detection
// ============================================================================

/**
 * Detect format version from buffer.
 */
export function detectFormat(buffer: Buffer): 'CFR1' | 'CFD1' | 'CFD2' | 'unknown' {
  if (buffer.length < 4) return 'unknown';

  const magic = buffer.toString('ascii', 0, 4);
  switch (magic) {
    case 'CFR1':
      return 'CFR1';
    case 'CFD1':
      return 'CFD1';
    case 'CFD2':
      return 'CFD2';
    default:
      return 'unknown';
  }
}
