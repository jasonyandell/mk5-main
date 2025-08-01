import { describe, it, expect } from 'vitest';
import { compressGameState, expandMinimalState } from '../../game/core/url-compression';
import { createInitialState } from '../../game/core/state';
import { analyzeSuits } from '../../game/core/suit-analysis';

describe('URL Compression Suit Analysis Fix', () => {
  it('should maintain identical suit analysis after compress/expand cycle', () => {
    // Create initial state with specific seed for deterministic test
    const originalState = createInitialState({ shuffleSeed: 12345 });
    
    // Store original suit analyses
    const originalSuitAnalyses = originalState.players.map(player => player.suitAnalysis);
    
    // Compress and expand the state
    const compressed = compressGameState(originalState);
    const expandedState = expandMinimalState(compressed);
    
    // Verify each player's suit analysis is identical
    expandedState.players.forEach((player, _i) => {
      expect(player.suitAnalysis).toBeDefined();
      expect(player.suitAnalysis).toEqual(originalSuitAnalyses[_i]);
      
      // Also verify it matches what we'd get from fresh analysis
      const freshAnalysis = analyzeSuits(player.hand);
      expect(player.suitAnalysis).toEqual(freshAnalysis);
    });
  });

  it('should handle suit analysis correctly with trump set', () => {
    // Create initial state
    const originalState = createInitialState({ shuffleSeed: 54321 });
    
    // Set trump to test trump-dependent suit analysis
    originalState.trump = 6; // sixes trump
    
    // Update suit analyses with trump
    originalState.players.forEach(player => {
      player.suitAnalysis = analyzeSuits(player.hand, originalState.trump);
    });
    
    // Compress and expand
    const compressed = compressGameState(originalState);
    const expandedState = expandMinimalState(compressed);
    
    // Verify hands are correct
    expandedState.players.forEach((player, i) => {
      expect(player.hand).toEqual(originalState.players[i].hand);
    });
    
    // Note: expandMinimalState resets trump to null and recalculates without trump
    // This is correct behavior for initial state expansion
    // The trump will be set later during action replay
    expandedState.players.forEach((player) => {
      const expectedAnalysis = analyzeSuits(player.hand, null);
      expect(player.suitAnalysis).toEqual(expectedAnalysis);
    });
  });

  it('should produce consistent results across multiple expansions', () => {
    const originalState = createInitialState({ shuffleSeed: 99999 });
    const compressed = compressGameState(originalState);
    
    // Expand multiple times
    const expanded1 = expandMinimalState(compressed);
    const expanded2 = expandMinimalState(compressed);
    const expanded3 = expandMinimalState(compressed);
    
    // All expansions should be identical
    expect(expanded1).toEqual(expanded2);
    expect(expanded2).toEqual(expanded3);
    
    // Specifically check suit analyses
    expanded1.players.forEach((player, i) => {
      expect(player.suitAnalysis).toEqual(expanded2.players[i].suitAnalysis);
      expect(player.suitAnalysis).toEqual(expanded3.players[i].suitAnalysis);
    });
  });

  it('should maintain suit analysis accuracy with different seeds', () => {
    const seeds = [1, 1337, 42424, 777777];
    
    seeds.forEach(seed => {
      const originalState = createInitialState({ shuffleSeed: seed });
      const compressed = compressGameState(originalState);
      const expandedState = expandMinimalState(compressed);
      
      // Verify each player's suit analysis is recalculated correctly
      expandedState.players.forEach((player, i) => {
        const expectedAnalysis = analyzeSuits(player.hand);
        expect(player.suitAnalysis).toEqual(expectedAnalysis);
        
        // Also verify it matches the original (since both have null trump)
        expect(player.suitAnalysis).toEqual(originalState.players[i].suitAnalysis);
      });
    });
  });

  it('should handle edge case hands correctly', () => {
    // Use seeds that might produce unusual hands
    const edgeCaseSeeds = [0, Number.MAX_SAFE_INTEGER - 1, 123456789];
    
    edgeCaseSeeds.forEach(seed => {
      const originalState = createInitialState({ shuffleSeed: seed });
      const compressed = compressGameState(originalState);
      const expandedState = expandMinimalState(compressed);
      
      // Verify hands are identical
      expandedState.players.forEach((player, i) => {
        expect(player.hand).toEqual(originalState.players[i].hand);
      });
      
      // Verify suit analysis is correct for each hand
      expandedState.players.forEach(player => {
        expect(player.suitAnalysis).toBeDefined();
        
        // Verify suit analysis totals are meaningful
        const analysisTotalPips = [0, 1, 2, 3, 4, 5, 6].reduce((sum, suit) => {
          return sum + ((player.suitAnalysis?.count[suit as keyof typeof player.suitAnalysis.count] || 0) * suit);
        }, 0);
        
        // Analysis should show meaningful suit distribution
        expect(analysisTotalPips).toBeGreaterThan(0);
      });
    });
  });
});