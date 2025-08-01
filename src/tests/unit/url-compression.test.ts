import { describe, it, expect } from 'vitest';
import { 
  compressGameState, 
  expandMinimalState, 
  compressActionId, 
  decompressActionId,
  encodeURLData,
  decodeURLData
} from '../../game/core/url-compression';
import { createInitialState } from '../../game';
import { testLog } from '../helpers/testConsole';

describe('URL Compression', () => {
  it('should compress game state to minimal representation', () => {
    const state = createInitialState();
    const compressed = compressGameState(state);
    
    // Should only have seed by default
    expect(compressed).toHaveProperty('s');
    expect(compressed.s).toBe(state.shuffleSeed);
    
    // Default values should not be included
    expect(compressed.d).toBeUndefined(); // dealer is 3 by default
    expect(compressed.t).toBeUndefined(); // target is 7 by default
    expect(compressed.m).toBeUndefined(); // tournament mode is true by default
  });

  it('should include non-default values in compressed state', () => {
    const state = createInitialState();
    state.dealer = 1;
    state.gameTarget = 10;
    state.tournamentMode = false;
    
    const compressed = compressGameState(state);
    
    expect(compressed.d).toBe(1);
    expect(compressed.t).toBe(10);
    expect(compressed.m).toBe(false);
  });

  it('should expand minimal state to full game state', () => {
    const original = createInitialState();
    const compressed = compressGameState(original);
    const expanded = expandMinimalState(compressed);
    
    // Key properties should match
    expect(expanded.shuffleSeed).toBe(original.shuffleSeed);
    expect(expanded.dealer).toBe(original.dealer);
    expect(expanded.gameTarget).toBe(original.gameTarget);
    expect(expanded.tournamentMode).toBe(original.tournamentMode);
    
    // Hands should be recreated identically (same seed)
    expect(expanded.players[0].hand.map(d => d.id)).toEqual(
      original.players[0].hand.map(d => d.id)
    );
  });

  it('should compress and decompress action IDs', () => {
    // Test bidding actions
    expect(compressActionId('pass')).toBe('p');
    expect(decompressActionId('p')).toBe('pass');
    
    expect(compressActionId('bid-30')).toBe('30');
    expect(decompressActionId('30')).toBe('bid-30');
    
    expect(compressActionId('bid-1-marks')).toBe('m1');
    expect(decompressActionId('m1')).toBe('bid-1-marks');
    
    // Test trump selection
    expect(compressActionId('select-trump-0')).toBe('t0');
    expect(decompressActionId('t0')).toBe('select-trump-0');
    
    // Test domino plays
    expect(compressActionId('play-6-4')).toBe('64');
    expect(decompressActionId('64')).toBe('play-6-4');
    
    // Test special cases
    expect(compressActionId('play-3-0')).toBe('30d');
    expect(decompressActionId('30d')).toBe('play-3-0');
    
    // Unknown actions should pass through
    expect(compressActionId('unknown-action')).toBe('unknown-action');
    expect(decompressActionId('xyz')).toBe('xyz');
  });

  it('should encode and decode URL data', () => {
    const urlData = {
      v: 1 as const,
      s: { s: 12345, d: 2 },
      a: [
        { i: 'p' },
        { i: '30' },
        { i: 't5' },
        { i: '64' }
      ]
    };
    
    const encoded = encodeURLData(urlData);
    
    // Should be base64url encoded
    expect(encoded).toMatch(/^[A-Za-z0-9_-]+$/);
    
    // Should decode back to original
    const decoded = decodeURLData(encoded);
    expect(decoded).toEqual(urlData);
  });

  it('should create more compact URLs than original format', () => {
    // Create a game with some actions
    const state = createInitialState();
    const actions = [
      { id: 'pass', label: 'Pass' },
      { id: 'bid-30', label: 'Bid 30 points' },
      { id: 'pass', label: 'Pass' },
      { id: 'pass', label: 'Pass' },
      { id: 'select-trump-5', label: 'Select fives as trump' },
      { id: 'play-6-4', label: 'Play 6-4' },
      { id: 'play-5-5', label: 'Play 5-5' },
      { id: 'play-5-0', label: 'Play 5-0' },
      { id: 'play-4-1', label: 'Play 4-1' }
    ];
    
    // Old format
    const oldData = {
      initial: state,
      actions: actions
    };
    const oldParam = encodeURIComponent(JSON.stringify(oldData));
    
    // New format
    const newData = {
      v: 1 as const,
      s: compressGameState(state),
      a: actions.map(a => ({ i: compressActionId(a.id) }))
    };
    const newParam = encodeURLData(newData);
    
    testLog('Old URL length:', oldParam.length);
    testLog('New URL length:', newParam.length);
    testLog('Compression ratio:', (newParam.length / oldParam.length * 100).toFixed(1) + '%');
    
    // Should be significantly smaller
    expect(newParam.length).toBeLessThan(oldParam.length * 0.1); // Less than 10% of original
  });
});