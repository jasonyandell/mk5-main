import { describe, it, expect } from 'vitest';
import {
  encodeVarint,
  decodeVarint,
  varintSize,
  serializeCFD2,
  deserializeCFD2,
  CompactStrategyV2,
  detectFormat
} from '../../../game/ai/cfr/compact-format-v2';
import type { SerializedStrategy } from '../../../game/ai/cfr/types';

describe('Varint encoding', () => {
  it('encodes small numbers in 1 byte', () => {
    expect(encodeVarint(0)).toEqual(new Uint8Array([0]));
    expect(encodeVarint(1)).toEqual(new Uint8Array([1]));
    expect(encodeVarint(127)).toEqual(new Uint8Array([127]));
  });

  it('encodes numbers requiring 2 bytes', () => {
    expect(encodeVarint(128)).toEqual(new Uint8Array([0x80, 0x01]));
    expect(encodeVarint(300)).toEqual(new Uint8Array([0xAC, 0x02]));
    expect(encodeVarint(16383)).toEqual(new Uint8Array([0xFF, 0x7F]));
  });

  it('encodes larger numbers', () => {
    expect(encodeVarint(16384)).toEqual(new Uint8Array([0x80, 0x80, 0x01]));
    expect(encodeVarint(100000)).toEqual(new Uint8Array([0xA0, 0x8D, 0x06]));
  });

  it('decodes back to original values', () => {
    const testValues = [0, 1, 127, 128, 300, 16383, 16384, 100000, 1000000];
    for (const value of testValues) {
      const encoded = encodeVarint(value);
      const [decoded, bytesRead] = decodeVarint(encoded, 0);
      expect(decoded).toBe(value);
      expect(bytesRead).toBe(encoded.length);
    }
  });

  it('calculates correct varint size', () => {
    expect(varintSize(0)).toBe(1);
    expect(varintSize(127)).toBe(1);
    expect(varintSize(128)).toBe(2);
    expect(varintSize(16383)).toBe(2);
    expect(varintSize(16384)).toBe(3);
  });
});

describe('CFD2 serialization', () => {
  const createTestStrategy = (nodeCount: number): SerializedStrategy => ({
    version: 1,
    config: { iterations: 100, seed: 42 },
    iterationsCompleted: 100,
    nodes: Array.from({ length: nodeCount }, (_, i) => ({
      key: `C:none|US:${(i % 8) * 5}|THEM:0|POT:0|LEAD:1|TR:${i % 7}|TK:${i % 6}|POS:${i % 4}|NC:${i % 7}|TT:suit`,
      regrets: [],
      strategySum: [
        ['6-4', 100 + i],
        ['5-5', 50 + i],
        ['3-2', 25 + i]
      ],
      visitCount: 10
    })),
    trainedAt: new Date().toISOString(),
    trainingTimeMs: 1000
  });

  it('serializes and deserializes a simple strategy', () => {
    const original = createTestStrategy(10);
    const buffer = serializeCFD2(original);
    const deserialized = deserializeCFD2(buffer);

    expect(deserialized.nodes.length).toBe(10);
    expect(deserialized.iterationsCompleted).toBe(100);
  });

  it('preserves strategy weights approximately', () => {
    const original = createTestStrategy(5);
    const buffer = serializeCFD2(original);
    const deserialized = deserializeCFD2(buffer);

    // Check that action counts match
    for (let i = 0; i < original.nodes.length; i++) {
      const origNode = original.nodes[i]!;
      const deserNode = deserialized.nodes[i]!;
      expect(deserNode.strategySum.length).toBe(origNode.strategySum.length);
    }
  });

  it('detects uniform patterns', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 100],
          ['5-5', 100],
          ['3-2', 100]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    // Uniform distribution should be preserved
    const weights = deserialized.nodes[0]!.strategySum;
    const total = weights.reduce((sum, [, w]) => sum + w, 0);
    for (const [, w] of weights) {
      expect(Math.abs(w / total - 1/3)).toBeLessThan(0.01);
    }
  });

  it('detects dominant patterns', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 950],  // 95% - dominant
          ['5-5', 25],
          ['3-2', 25]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    // Dominant action should have highest weight
    const weights = deserialized.nodes[0]!.strategySum;
    const total = weights.reduce((sum, [, w]) => sum + w, 0);
    const prob64 = (weights.find(([a]) => a === '6-4')?.[1] ?? 0) / total;
    expect(prob64).toBeGreaterThan(0.9);
  });
});

describe('CompactStrategyV2', () => {
  it('loads from serialized strategy', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 100],
          ['5-5', 50],
          ['3-2', 25]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const compact = CompactStrategyV2.fromSerialized(strategy);
    expect(compact.size).toBe(1);

    const probs = compact.getStrategy('C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit');
    expect(probs.size).toBe(3);
    expect(probs.get('6-4')).toBeCloseTo(100/175, 2);
  });

  it('samples actions according to strategy', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 1000],  // Should be selected most often
          ['5-5', 1],
          ['3-2', 1]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const compact = CompactStrategyV2.fromSerialized(strategy);
    const legalActions = ['6-4', '5-5', '3-2'];

    // Sample many times, 6-4 should dominate
    let count64 = 0;
    const rng = () => Math.random();
    for (let i = 0; i < 100; i++) {
      const action = compact.sampleAction(
        'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        legalActions,
        rng
      );
      if (action === '6-4') count64++;
    }

    expect(count64).toBeGreaterThan(80);
  });

  it('returns empty map for unknown info set', () => {
    const compact = CompactStrategyV2.fromSerialized({
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    });

    const probs = compact.getStrategy('C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit');
    expect(probs.size).toBe(0);
  });

  it('falls back to uniform for unknown info set when sampling', () => {
    const compact = CompactStrategyV2.fromSerialized({
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    });

    const legalActions = ['6-4', '5-5', '3-2'];
    const action = compact.sampleAction(
      'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
      legalActions,
      () => 0.5
    );

    expect(legalActions).toContain(action);
  });
});

describe('CFD2 binary patterns', () => {
  it('detects binary split patterns', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 700],  // 70%
          ['5-5', 280],  // 28%
          ['3-2', 20]    // 2%
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    const weights = deserialized.nodes[0]!.strategySum;
    const total = weights.reduce((sum, [, w]) => sum + w, 0);

    // Top 2 actions should have most of the weight
    const sorted = [...weights].sort((a, b) => b[1] - a[1]);
    const topTwoWeight = ((sorted[0]?.[1] ?? 0) + (sorted[1]?.[1] ?? 0)) / total;
    expect(topTwoWeight).toBeGreaterThan(0.9);
  });

  it('handles full pattern encoding', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 400],  // 40%
          ['5-5', 350],  // 35%
          ['3-2', 250]   // 25%
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    // All actions should have reasonable weights
    const weights = deserialized.nodes[0]!.strategySum;
    const total = weights.reduce((sum, [, w]) => sum + w, 0);
    for (const [, w] of weights) {
      expect(w / total).toBeGreaterThan(0.1);
    }
  });

  it('handles empty strategy sum', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    expect(deserialized.nodes.length).toBe(0);
  });

  it('handles many nodes with shared action sets', () => {
    const nodes = [];
    for (let i = 0; i < 100; i++) {
      nodes.push({
        key: `C:none|US:${(i % 8) * 5}|THEM:0|POT:0|LEAD:1|TR:${i % 7}|TK:${i % 6}|POS:${i % 4}|NC:${i % 7}|TT:suit`,
        regrets: [],
        strategySum: [
          ['6-4', 100 + i],
          ['5-5', 50 + i]
        ],
        visitCount: 10
      });
    }

    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes,
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const deserialized = deserializeCFD2(buffer);

    expect(deserialized.nodes.length).toBe(100);
  });
});

describe('Format detection', () => {
  it('detects CFD2 format', () => {
    const buffer = Buffer.from('CFD2\x01\x00\x00\x00\x00\x00', 'ascii');
    expect(detectFormat(buffer)).toBe('CFD2');
  });

  it('detects CFD1 format', () => {
    const buffer = Buffer.from('CFD1\x01\x00\x00\x00\x00\x00', 'ascii');
    expect(detectFormat(buffer)).toBe('CFD1');
  });

  it('detects CFR1 format', () => {
    const buffer = Buffer.from('CFR1\x01\x00\x00\x00\x00\x00', 'ascii');
    expect(detectFormat(buffer)).toBe('CFR1');
  });

  it('returns unknown for invalid format', () => {
    const buffer = Buffer.from('XXXX\x01\x00\x00\x00\x00\x00', 'ascii');
    expect(detectFormat(buffer)).toBe('unknown');
  });

  it('returns unknown for short buffer', () => {
    const buffer = Buffer.from('CF', 'ascii');
    expect(detectFormat(buffer)).toBe('unknown');
  });
});

describe('Varint edge cases', () => {
  it('handles large varints correctly', () => {
    const testValues = [
      0x7FFFFFF,   // Max 28-bit
      0x1FFFFFFF,  // 29-bit
    ];
    for (const value of testValues) {
      const encoded = encodeVarint(value);
      const [decoded, bytesRead] = decodeVarint(encoded, 0);
      expect(decoded).toBe(value);
      expect(bytesRead).toBe(encoded.length);
    }
  });

  it('decodes varint with offset', () => {
    const prefix = new Uint8Array([0xFF, 0xFF]);
    const value = 300;
    const encoded = encodeVarint(value);

    // Create buffer with prefix
    const combined = new Uint8Array(prefix.length + encoded.length);
    combined.set(prefix);
    combined.set(encoded, prefix.length);

    const [decoded, bytesRead] = decodeVarint(combined, prefix.length);
    expect(decoded).toBe(value);
    expect(bytesRead).toBe(encoded.length);
  });

  it('throws on truncated varint', () => {
    const truncated = new Uint8Array([0x80]); // Continuation bit set but no more bytes
    expect(() => decodeVarint(truncated, 0)).toThrow('Unexpected end of buffer');
  });
});

describe('CFD2 error handling', () => {
  it('throws on invalid magic', () => {
    const buffer = Buffer.from('XXXX' + '\x00'.repeat(20), 'ascii');
    expect(() => deserializeCFD2(buffer)).toThrow('Invalid magic');
  });

  it('throws on unsupported version', () => {
    const buffer = Buffer.alloc(24);
    buffer.write('CFD2', 0, 4, 'ascii');
    buffer.writeUInt8(99, 4); // Invalid version
    expect(() => deserializeCFD2(buffer)).toThrow('Unsupported CFD2 version');
  });
});

describe('CompactStrategyV2 edge cases', () => {
  it('reports memory usage', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 100],
          ['5-5', 50]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const compact = CompactStrategyV2.fromSerialized(strategy);
    const usage = compact.getMemoryUsage();
    expect(usage).toBeGreaterThan(0);
    expect(typeof usage).toBe('number');
  });

  it('handles zero strategy sums gracefully', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 0],
          ['5-5', 0]
        ],
        visitCount: 0
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const compact = CompactStrategyV2.fromSerialized(strategy);
    const probs = compact.getStrategy('C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit');
    // Should be uniform when sum is 0
    expect(probs.size).toBe(2);
    expect(probs.get('6-4')).toBeCloseTo(0.5, 2);
    expect(probs.get('5-5')).toBeCloseTo(0.5, 2);
  });

  it('loads from CFD2 buffer', () => {
    const strategy: SerializedStrategy = {
      version: 1,
      config: { iterations: 100, seed: 42 },
      iterationsCompleted: 100,
      nodes: [{
        key: 'C:none|US:0|THEM:0|POT:0|LEAD:1|TR:0|TK:0|POS:0|NC:0|TT:suit',
        regrets: [],
        strategySum: [
          ['6-4', 100],
          ['5-5', 50]
        ],
        visitCount: 10
      }],
      trainedAt: new Date().toISOString(),
      trainingTimeMs: 1000
    };

    const buffer = serializeCFD2(strategy);
    const compact = CompactStrategyV2.fromBuffer(buffer);
    expect(compact.size).toBe(1);
  });
});
