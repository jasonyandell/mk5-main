#!/usr/bin/env tsx
/**
 * Generate precomputed strength table for all domino/trump/suit combinations
 * This eliminates runtime computation of domino comparisons
 */

import { writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Domino, TrumpSelection, GameState, LedSuitOrNone } from '../src/game/types';
import { PLAYED_AS_TRUMP, NO_BIDDER, NO_LEAD_SUIT } from '../src/game/types';
import { analyzeDominoAsSuit } from '../src/game/ai/domino-strength';
import { suitsWithTrumpBase } from '../src/game/layers/compose';
import { isTrump } from '../src/game/core/dominoes';
import { getLedSuitName, SUIT_IDENTIFIERS } from '../src/game/game-terms';

// Use centralized LED_SUIT_NAMES from game-terms.ts
function numberToSuitName(n: number): string {
  return getLedSuitName(n as LedSuitOrNone);
}

// Generate all 28 dominoes
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

// Get all trump configurations (8 unique scenarios)
function getTrumpConfigurations(): Array<{ key: string; trump: TrumpSelection }> {
  const configs: Array<{ key: string; trump: TrumpSelection }> = [];
  
  // No trump (treating not-selected and no-trump as identical)
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
      key: `trump-${SUIT_IDENTIFIERS[suit as keyof typeof SUIT_IDENTIFIERS]}`,
      trump: { type: 'suit', suit: suit as any }
    });
  }
  
  return configs;
}

// Create a minimal game state for analysis
function createMinimalState(): GameState {
  return {
    initialConfig: {
      playerTypes: ['ai', 'ai', 'ai', 'ai'],
      shuffleSeed: 0,
      theme: 'business',
      colorOverrides: {}
    },
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
    shuffleSeed: 0,
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    actionHistory: [],
    theme: 'business',
    colorOverrides: {}
  };
}

interface StrengthEntry {
  beatenBy: string[];
  beats: string[];
  cannotFollow: string[];
}

function formatTableCompact(table: Record<string, StrengthEntry>): string {
  const lines: string[] = ['{'];
  const entries = Object.entries(table);
  
  entries.forEach(([key, value], index) => {
    const beatenBy = value.beatenBy.length > 0 
      ? `[${value.beatenBy.map(id => `"${id}"`).join(', ')}]`
      : '[]';
    const beats = value.beats.length > 0
      ? `[${value.beats.map(id => `"${id}"`).join(', ')}]`
      : '[]';
    const cannotFollow = value.cannotFollow.length > 0
      ? `[${value.cannotFollow.map(id => `"${id}"`).join(', ')}]`
      : '[]';
    
    const comma = index < entries.length - 1 ? ',' : '';
    lines.push(`  "${key}": { beatenBy: ${beatenBy}, beats: ${beats}, cannotFollow: ${cannotFollow} }${comma}`);
  });
  
  lines.push('}');
  return lines.join('\n');
}

function generateTable(): void {
  const dominoes = generateAllDominoes();
  const trumpConfigs = getTrumpConfigurations();
  const state = createMinimalState();
  
  const table: Record<string, StrengthEntry> = {};
  let entryCount = 0;
  
  console.log('Generating strength table...');
  console.log(`- ${dominoes.length} dominoes`);
  console.log(`- ${trumpConfigs.length} trump configurations`);
  
  for (const domino of dominoes) {
    for (const { key: trumpKey, trump } of trumpConfigs) {
      // Update state's trump for the analysis
      state.trump = trump;
      
      // Get valid playable suits for this domino
      const playableSuits = suitsWithTrumpBase(state, domino);
      const dominoIsTrump = isTrump(domino, trump);
      
      // Generate entry for playing as trump (PLAYED_AS_TRUMP) if applicable
      if (dominoIsTrump) {
        const analysisAsTrump = analyzeDominoAsSuit(domino, PLAYED_AS_TRUMP, trump, state, 0);
        const keyAsTrump = `${domino.id}|${trumpKey}|played-as-trump`;
        table[keyAsTrump] = {
          beatenBy: analysisAsTrump.beatenBy.map(d => d.id.toString()),
          beats: analysisAsTrump.beats.map(d => d.id.toString()),
          cannotFollow: analysisAsTrump.cannotFollow.map(d => d.id.toString())
        };
        entryCount++;
      }
      
      // Generate entries for each valid playable suit
      for (const suit of playableSuits) {
        const analysis = analyzeDominoAsSuit(domino, suit, trump, state, 0);
        const key = `${domino.id}|${trumpKey}|${numberToSuitName(suit)}`;
        table[key] = {
          beatenBy: analysis.beatenBy.map(d => d.id.toString()),
          beats: analysis.beats.map(d => d.id.toString()),
          cannotFollow: analysis.cannotFollow.map(d => d.id.toString())
        };
        entryCount++;
      }      
    }
  }
  
  console.log(`Generated ${entryCount} entries`);
  
  // Generate TypeScript file
  const output = `/**
 * Auto-generated domino strength lookup table
 * DO NOT EDIT - This file is generated by scripts/generate-strength-table.ts
 *
 * Use getDominoStrength() to lookup precomputed strength analysis for any domino.
 * This module contains optimized lookups for all possible domino/trump/suit combinations.
 */

import type { Domino, TrumpSelection, LedSuitOrNone } from '../types';
import { getLedSuitName, SUIT_IDENTIFIERS } from '../game-terms';

export interface StrengthEntry {
  beatenBy: string[];      // Domino IDs that can follow AND beat this
  beats: string[];         // Domino IDs that can follow but lose to this
  cannotFollow: string[];  // Domino IDs that cannot follow suit
}

/**
 * Get precomputed strength analysis for a domino
 * @param domino The domino to analyze
 * @param trump The current trump selection
 * @param playedAsSuit The suit the domino is played as (PLAYED_AS_TRUMP for trump play)
 * @returns The strength entry, or undefined if not found
 */
export function getDominoStrength(
  domino: Domino,
  trump: TrumpSelection,
  playedAsSuit: LedSuitOrNone
): StrengthEntry | undefined {
  const trumpKey = getTrumpKey(trump);
  const suitName = getLedSuitName(playedAsSuit);

  const key = \`\${domino.id}|\${trumpKey}|\${suitName}\`;
  return STRENGTH_TABLE[key];
}

const STRENGTH_TABLE: Record<string, StrengthEntry> = ${formatTableCompact(table)};

// Helper function to get trump key
function getTrumpKey(trump: TrumpSelection): string {
  if (trump.type === 'not-selected' || trump.type === 'no-trump') {
    return 'trump-no-trump';
  } else if (trump.type === 'doubles') {
    return 'trump-doubles';
  } else if (trump.type === 'suit') {
    const suitName = SUIT_IDENTIFIERS[trump.suit as keyof typeof SUIT_IDENTIFIERS];
    return \`trump-\${suitName}\`;
  } else {
    return 'trump-no-trump';
  }
}

// Export metadata for debugging
export const TABLE_METADATA = {
  totalEntries: ${entryCount},
  dominoCount: ${dominoes.length},
  trumpConfigs: ${trumpConfigs.length}
};
`;
  
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const outputPath = join(__dirname, '..', 'src', 'game', 'ai', 'strength-table.generated.ts');
  writeFileSync(outputPath, output, 'utf-8');
  
  console.log(`✓ Written to ${outputPath}`);
  console.log(`✓ File size: ${(output.length / 1024).toFixed(2)} KB`);
}

// Run generation
generateTable();