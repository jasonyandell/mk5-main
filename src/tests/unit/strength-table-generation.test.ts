import { describe, it, expect } from 'vitest';
import type { Domino, TrumpSelection, LedSuitOrNone, LedSuit, RegularSuit } from '../../game/types';
import { SIXES, DOUBLES_AS_TRUMP, PLAYED_AS_TRUMP, NO_BIDDER, NO_LEAD_SUIT } from '../../game/types';
import { getDominoStrength } from '../../game/ai/strength-table.generated';
import { analyzeDominoAsSuit, getPlayableSuits } from '../../game/ai/domino-strength';
import { isTrump } from '../../game/core/dominoes';
import type { GameState } from '../../game/types';

function createMinimalState(): GameState {
  return {
    phase: 'playing',
    players: [
      { id: 0, name: 'P0', hand: [], teamId: 0, marks: 0 },
      { id: 1, name: 'P1', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'P2', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'P3', hand: [], teamId: 1, marks: 0 },
    ],
    currentPlayer: 0,
    dealer: 0,
    bids: [],
    currentBid: { type: 'pass', player: NO_BIDDER },
    winningBidder: NO_BIDDER,
    trump: { type: 'not-selected' },
    tricks: [],
    currentTrick: [],
    currentSuit: NO_LEAD_SUIT,
    teamScores: [0, 0],
    teamMarks: [0, 0],
    gameTarget: 7,
    tournamentMode: false,
    shuffleSeed: 0,
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    consensus: {
      completeTrick: new Set(),
      scoreHand: new Set()
    },
    actionHistory: [],
    aiSchedule: {},
    currentTick: 0,
    theme: 'coffee',
    colorOverrides: {}
  };
}

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
  const suitNames = ['blanks', 'aces', 'deuces', 'tres', 'fours', 'fives', 'sixes'];
  for (let suit = 0; suit <= 6; suit++) {
    configs.push({
      key: `trump-${suitNames[suit]}`,
      trump: { type: 'suit', suit: suit as RegularSuit }
    });
  }
  
  return configs;
}

describe('Strength Table Generation Verification', () => {
  const dominoes = generateAllDominoes();
  const trumpConfigs = getTrumpConfigurations();
  const state = createMinimalState();

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
        
        // Also check "invalid" plays that runtime allows for trump dominoes
        if (dominoIsTrump) {
          const allSuits = new Set<number>();
          if (domino.high !== domino.low) {
            allSuits.add(domino.high);
            allSuits.add(domino.low);
          } else {
            allSuits.add(domino.high);
          }
          
          for (const suit of allSuits) {
            if (!playableSuits.includes(suit as LedSuit)) {
              const runtime = analyzeDominoAsSuit(domino, suit as LedSuitOrNone, trump, state, 0);
              const generated = getDominoStrength(domino, trump, suit as LedSuitOrNone);
              
              totalChecks++;
              if (!generated) {
                mismatches.push(`Missing entry for invalid play: ${domino.id} | ${trump.type} | suit-${suit}`);
                continue;
              }
              
              const runtimeBeatenBy = runtime.beatenBy.map(d => d.id).sort();
              const generatedBeatenBy = generated.beatenBy.sort();
              const runtimeBeats = runtime.beats.map(d => d.id).sort();
              const generatedBeats = generated.beats.sort();
              const runtimeCannotFollow = runtime.cannotFollow.map(d => d.id).sort();
              const generatedCannotFollow = generated.cannotFollow.sort();
              
              if (JSON.stringify(runtimeBeatenBy) !== JSON.stringify(generatedBeatenBy)) {
                mismatches.push(`beatenBy mismatch for invalid play ${domino.id} | ${trump.type} | suit-${suit}`);
              }
              if (JSON.stringify(runtimeBeats) !== JSON.stringify(generatedBeats)) {
                mismatches.push(`beats mismatch for invalid play ${domino.id} | ${trump.type} | suit-${suit}`);
              }
              if (JSON.stringify(runtimeCannotFollow) !== JSON.stringify(generatedCannotFollow)) {
                mismatches.push(`cannotFollow mismatch for invalid play ${domino.id} | ${trump.type} | suit-${suit}`);
              }
            }
          }
        }
      }
    }
    
    if (mismatches.length > 0) {
      console.error('Mismatches found:', mismatches);
    }
    
    expect(mismatches).toHaveLength(0);
    console.log(`✅ Verified ${totalChecks} entries successfully`);
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
      createMinimalState(),
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