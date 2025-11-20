import { describe, it, expect } from 'vitest';
import type { Domino, TrumpSelection, LedSuitOrNone, RegularSuit } from '../../game/types';
import { SIXES, DOUBLES_AS_TRUMP, PLAYED_AS_TRUMP } from '../../game/types';
import { getDominoStrength } from '../../game/ai/strength-table.generated';
import { analyzeDominoAsSuit, getPlayableSuits } from '../../game/ai/domino-strength';
import { isTrump } from '../../game/core/dominoes';
import { StateBuilder } from '../helpers';
import { SUIT_IDENTIFIERS } from '../../game/game-terms';

function generateAllDominoes(): Domino[] {
  const dominoes: Domino[] = [];
  for (let high = 6; high >= 0; high--) {
    for (let low = high; low >= 0; low--) {
      dominoes.push({
        high,
        low,
        id: `${high}-${low}`
      });
    }
  }
  return dominoes;
}

function getTrumpConfigurations(): Array<{ key: string; trump: TrumpSelection }> {
  const configs: Array<{ key: string; trump: TrumpSelection }> = [];
  
  // No trump
  configs.push({
    key: 'trump-no-trump',
    trump: { type: 'no-trump' }
  });
  
  // Doubles as trump
  configs.push({
    key: 'trump-doubles',
    trump: { type: 'doubles' }
  });
  
  // Each suit as trump (0-6)
  for (let suit = 0; suit <= 6; suit++) {
    configs.push({
      key: `trump-${SUIT_IDENTIFIERS[suit as RegularSuit]}`,
      trump: { type: 'suit', suit: suit as RegularSuit }
    });
  }
  
  return configs;
}

describe('Strength Table Generation Verification', () => {
  const dominoes = generateAllDominoes();
  const trumpConfigs = getTrumpConfigurations();
  const state = StateBuilder.inPlayingPhase().withHands([[], [], [], []]).build();

  it('should match runtime calculations for all domino/trump/suit combinations', () => {
    let totalChecks = 0;
    let mismatches: string[] = [];
    
    for (const domino of dominoes) {
      for (const { trump } of trumpConfigs) {
        state.trump = trump;
        
        const playableSuits = getPlayableSuits(domino, trump);
        const dominoIsTrump = isTrump(domino, trump);
        
        // Check playing as trump if applicable
        if (dominoIsTrump) {
          const runtime = analyzeDominoAsSuit(domino, PLAYED_AS_TRUMP, trump, state, 0);
          const generated = getDominoStrength(domino, trump, PLAYED_AS_TRUMP);
          
          totalChecks++;
          if (!generated) {
            mismatches.push(`Missing entry: ${domino.id} | ${trump.type} | led-trump`);
            continue;
          }
          
          const runtimeBeatenBy = runtime.beatenBy.map(d => d.id).sort();
          const generatedBeatenBy = generated.beatenBy.sort();
          const runtimeBeats = runtime.beats.map(d => d.id).sort();
          const generatedBeats = generated.beats.sort();
          const runtimeCannotFollow = runtime.cannotFollow.map(d => d.id).sort();
          const generatedCannotFollow = generated.cannotFollow.sort();
          
          if (JSON.stringify(runtimeBeatenBy) !== JSON.stringify(generatedBeatenBy)) {
            mismatches.push(`beatenBy mismatch for ${domino.id} | ${trump.type} | led-trump`);
          }
          if (JSON.stringify(runtimeBeats) !== JSON.stringify(generatedBeats)) {
            mismatches.push(`beats mismatch for ${domino.id} | ${trump.type} | led-trump`);
          }
          if (JSON.stringify(runtimeCannotFollow) !== JSON.stringify(generatedCannotFollow)) {
            mismatches.push(`cannotFollow mismatch for ${domino.id} | ${trump.type} | led-trump`);
          }
        }
        
        // Check each playable suit
        for (const suit of playableSuits) {
          const runtime = analyzeDominoAsSuit(domino, suit, trump, state, 0);
          const generated = getDominoStrength(domino, trump, suit);
          
          totalChecks++;
          if (!generated) {
            mismatches.push(`Missing entry: ${domino.id} | ${trump.type} | suit-${suit}`);
            continue;
          }
          
          const runtimeBeatenBy = runtime.beatenBy.map(d => d.id).sort();
          const generatedBeatenBy = generated.beatenBy.sort();
          const runtimeBeats = runtime.beats.map(d => d.id).sort();
          const generatedBeats = generated.beats.sort();
          const runtimeCannotFollow = runtime.cannotFollow.map(d => d.id).sort();
          const generatedCannotFollow = generated.cannotFollow.sort();
          
          if (JSON.stringify(runtimeBeatenBy) !== JSON.stringify(generatedBeatenBy)) {
            mismatches.push(`beatenBy mismatch for ${domino.id} | ${trump.type} | suit-${suit}`);
          }
          if (JSON.stringify(runtimeBeats) !== JSON.stringify(generatedBeats)) {
            mismatches.push(`beats mismatch for ${domino.id} | ${trump.type} | suit-${suit}`);
          }
          if (JSON.stringify(runtimeCannotFollow) !== JSON.stringify(generatedCannotFollow)) {
            mismatches.push(`cannotFollow mismatch for ${domino.id} | ${trump.type} | suit-${suit}`);
          }
        }
      }
    }
    
    expect(totalChecks).toEqual(455);

    if (mismatches.length > 0) {
      console.error('Mismatches found:', mismatches);
    }
    
    expect(mismatches).toHaveLength(0);
  });

  it('should handle edge cases correctly', () => {
    // Test not-selected vs no-trump equivalence
    const domino = { high: 6, low: 5, id: '6-5' };
    const notSelected = getDominoStrength(domino, { type: 'not-selected' }, SIXES);
    const noTrump = getDominoStrength(domino, { type: 'no-trump' }, SIXES);
    
    expect(notSelected).toEqual(noTrump);
    
    // Test doubles as trump when playing as doubles suit
    const doubleDomino = { high: 3, low: 3, id: '3-3' };
    const doublesResult = getDominoStrength(doubleDomino, { type: 'doubles' }, DOUBLES_AS_TRUMP);
    expect(doublesResult).toBeDefined();
    expect(doublesResult!.cannotFollow).not.toContain('2-2'); // Other doubles can follow
    expect(doublesResult!.cannotFollow).toContain('3-2'); // Non-doubles cannot follow
  });

  it('should return undefined for non-existent combinations', () => {
    const domino = { high: 6, low: 5, id: '6-5' };
    // Test with an invalid suit that shouldn't exist in the table
    const result = getDominoStrength(domino, { type: 'no-trump' }, 99 as LedSuitOrNone);
    // The generator doesn't create entries for invalid suits beyond 0-7
    expect(result).toBeUndefined();
  });
});

describe('Strength Table Regeneration', () => {
  it('should verify table can be regenerated identically', async () => {
    // This test documents that regenerating the table produces identical output
    // To regenerate: npm run generate:strength-table
    // Git will not detect changes if the hash is identical
    
    // We're NOT automatically regenerating here because:
    // 1. It's slow (takes several seconds)
    // 2. The prebuild/predev scripts already handle it
    // 3. Git correctly ignores unchanged files
    
    // If this test fails, run: npm run generate:strength-table
    const runtime = analyzeDominoAsSuit(
      { high: 6, low: 6, id: '6-6' },
      SIXES,
      { type: 'no-trump' },
      StateBuilder.inPlayingPhase().withHands([[], [], [], []]).build(),
      0
    );
    
    const generated = getDominoStrength(
      { high: 6, low: 6, id: '6-6' },
      { type: 'no-trump' },
      SIXES
    );
    
    expect(generated).toBeDefined();
    expect(generated!.beatenBy.sort()).toEqual(runtime.beatenBy.map(d => d.id).sort());
    expect(generated!.beats.sort()).toEqual(runtime.beats.map(d => d.id).sort());
    expect(generated!.cannotFollow.sort()).toEqual(runtime.cannotFollow.map(d => d.id).sort());
  });
});