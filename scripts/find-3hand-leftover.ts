/**
 * Find 3-Hand Perfect Partitions and Analyze Leftover Hands
 *
 * Finds all combinations of 3 non-overlapping perfect hands,
 * then analyzes the 7 leftover dominoes to determine their strength.
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { getDominoStrength } from '../src/game/ai/strength-table.generated.js';
import type { Domino, TrumpSelection, LedSuitOrNone } from '../src/game/types.js';
import { PLAYED_AS_TRUMP } from '../src/game/types.js';
import { getTrumpIdentifier } from '../src/game/game-terms.js';

// Get current directory (ES modules compatible)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Types
interface PerfectHandData {
  dominoes: string[];
  trump: string;
  type: 'gold' | 'platinum';
}

interface PerfectHand extends PerfectHandData {
  canonical: Set<string>;
  index: number;
}

interface Partition {
  hands: PerfectHand[];
  leftoverDominoes: string[];
}

interface LeftoverAnalysis {
  partition: Partition;
  bestTrump: TrumpSelection;
  bestTrumpName: string;
  externalBeaters: number;
  detailedCounts: Map<string, number>; // domino -> external beater count
}

// Canonicalize a domino to high-low format
function canonicalizeDomino(domino: string): string {
  const [a, b] = domino.split('-').map(Number);
  const high = Math.max(a!, b!);
  const low = Math.min(a!, b!);
  return `${high}-${low}`;
}

// Canonicalize an entire hand
function canonicalizeHand(dominoes: string[]): string[] {
  return dominoes.map(canonicalizeDomino);
}

// Create a Domino object from string
function createDomino(dominoStr: string): Domino {
  const [high, low] = dominoStr.split('-').map(Number);
  return { id: dominoStr, high: high!, low: low! };
}

// All 28 canonical dominoes
const ALL_DOMINOES = new Set([
  "6-6", "6-5", "6-4", "6-3", "6-2", "6-1", "6-0",
  "5-5", "5-4", "5-3", "5-2", "5-1", "5-0",
  "4-4", "4-3", "4-2", "4-1", "4-0",
  "3-3", "3-2", "3-1", "3-0",
  "2-2", "2-1", "2-0",
  "1-1", "1-0",
  "0-0"
]);

// Get play contexts for a domino when LEADING
function getPlayContexts(domino: Domino, trump: TrumpSelection): LedSuitOrNone[] {
  const contexts: LedSuitOrNone[] = [];

  // Check if it's trump
  const isTrumpDomino = isTrump(domino, trump);

  if (isTrumpDomino) {
    contexts.push(PLAYED_AS_TRUMP);
  }

  // When we LEAD (which is what we care about for analysis)
  if (domino.high === domino.low) {
    // Double leads as its suit or as doubles if doubles are trump
    if (trump.type === 'doubles') {
      contexts.push(7 as LedSuitOrNone);
    } else {
      contexts.push(domino.high as LedSuitOrNone);
    }
  } else {
    // Non-double ALWAYS leads as its HIGH pip
    contexts.push(domino.high as LedSuitOrNone);
  }

  return contexts;
}

// Check if a domino is trump
function isTrump(domino: Domino, trump: TrumpSelection): boolean {
  switch (trump.type) {
    case 'suit':
      return domino.high === trump.suit || domino.low === trump.suit;
    case 'doubles':
      return domino.high === domino.low;
    case 'no-trump':
      return false;
    default:
      return false;
  }
}

// Get all trump selections
function getAllTrumpSelections(): Array<{ trump: TrumpSelection; name: string }> {
  const trumps: TrumpSelection[] = [
    { type: 'suit', suit: 0 },
    { type: 'suit', suit: 1 },
    { type: 'suit', suit: 2 },
    { type: 'suit', suit: 3 },
    { type: 'suit', suit: 4 },
    { type: 'suit', suit: 5 },
    { type: 'suit', suit: 6 },
    { type: 'doubles' },
    { type: 'no-trump' }
  ];

  return trumps.map(trump => ({
    trump,
    name: getTrumpIdentifier(trump)
  }));
}

// Analyze a leftover hand's strength
function analyzeLeftoverHand(leftoverDominoes: string[]): {
  trump: TrumpSelection;
  name: string;
  externalBeaters: number;
  details: Map<string, number>;
} {
  let bestTrump: TrumpSelection | null = null;
  let bestTrumpName = '';
  let minExternalBeaters = Infinity;
  let bestDetails = new Map<string, number>();

  const leftoverSet = new Set(leftoverDominoes);

  // Try each trump selection
  for (const { trump, name } of getAllTrumpSelections()) {
    let totalExternalBeaters = 0;
    const details = new Map<string, number>();

    // For each domino in the leftover hand
    for (const dominoStr of leftoverDominoes) {
      const domino = createDomino(dominoStr);
      const contexts = getPlayContexts(domino, trump);

      let dominoExternalBeaters = 0;
      const externalBeaterSet = new Set<string>();

      // Check each context where we might lead this domino
      for (const context of contexts) {
        const strength = getDominoStrength(domino, trump, context);

        if (strength) {
          // Count external beaters (not in leftover hand)
          for (const beaterId of strength.beatenBy) {
            const canonicalBeater = canonicalizeDomino(beaterId);
            if (!leftoverSet.has(canonicalBeater)) {
              externalBeaterSet.add(canonicalBeater);
            }
          }
        }
      }

      dominoExternalBeaters = externalBeaterSet.size;
      details.set(dominoStr, dominoExternalBeaters);
      totalExternalBeaters += dominoExternalBeaters;
    }

    // Track the best trump selection (fewest external beaters)
    if (totalExternalBeaters < minExternalBeaters) {
      minExternalBeaters = totalExternalBeaters;
      bestTrump = trump;
      bestTrumpName = name;
      bestDetails = details;
    }
  }

  return {
    trump: bestTrump!,
    name: bestTrumpName,
    externalBeaters: minExternalBeaters,
    details: bestDetails
  };
}

// Check if two hands overlap
function handsOverlap(hand1: PerfectHand, hand2: PerfectHand): boolean {
  for (const domino of hand1.canonical) {
    if (hand2.canonical.has(domino)) {
      return true;
    }
  }
  return false;
}

// Main partition finder
class PartitionFinder {
  private hands: PerfectHand[];
  private handOverlaps: boolean[][];
  private partitions: Partition[];
  private searchCount: number;

  constructor(handsData: PerfectHandData[], silent = false) {
    if (!silent) {
      console.log(`Loading ${handsData.length} perfect hands...`);
    }

    // Convert to internal format with canonical dominoes
    this.hands = handsData.map((data, index) => ({
      ...data,
      canonical: new Set(canonicalizeHand(data.dominoes)),
      index
    }));

    // Pre-compute hand overlaps
    if (!silent) {
      console.log('Pre-computing hand overlaps...');
    }
    this.handOverlaps = Array(this.hands.length);
    for (let i = 0; i < this.hands.length; i++) {
      this.handOverlaps[i] = Array(this.hands.length);
      for (let j = 0; j < this.hands.length; j++) {
        this.handOverlaps[i]![j] = i === j || handsOverlap(this.hands[i]!, this.hands[j]!);
      }
    }

    this.partitions = [];
    this.searchCount = 0;
  }

  // Recursive backtracking search for 3-hand partitions
  private search(
    startIndex: number,
    usedHands: number[],
    usedDominoes: Set<string>
  ): void {
    this.searchCount++;

    // Progress indicator
    if (this.searchCount % 100000 === 0) {
      process.stdout.write(`\rSearched ${this.searchCount} nodes, found ${this.partitions.length} partitions...`);
    }

    // Base case: found 3 hands
    if (usedHands.length === 3) {
      // Calculate leftover dominoes
      const leftoverDominoes: string[] = [];
      for (const domino of ALL_DOMINOES) {
        if (!usedDominoes.has(domino)) {
          leftoverDominoes.push(domino);
        }
      }

      // Should be exactly 7 leftovers (28 - 21)
      if (leftoverDominoes.length === 7) {
        this.partitions.push({
          hands: usedHands.map(i => this.hands[i]!),
          leftoverDominoes
        });
      }
      return;
    }

    // Try each remaining hand
    for (let i = startIndex; i < this.hands.length; i++) {
      const hand = this.hands[i];

      // Skip if this hand overlaps with any used hand
      let overlaps = false;
      for (const usedIndex of usedHands) {
        if (this.handOverlaps[i]![usedIndex]) {
          overlaps = true;
          break;
        }
      }
      if (overlaps) continue;

      // Add this hand and recurse
      const newUsedHands = [...usedHands, i];
      const newUsedDominoes = new Set(usedDominoes);
      for (const domino of hand!.canonical) {
        newUsedDominoes.add(domino);
      }

      this.search(i + 1, newUsedHands, newUsedDominoes);
    }
  }

  // Find all 3-hand partitions
  findPartitions(silent = false): Partition[] {
    if (!silent) {
      console.log('\nSearching for 3-hand partitions...');
    }
    this.search(0, [], new Set());
    if (!silent) {
      console.log(`\n\nSearch complete! Explored ${this.searchCount} nodes`);
    }
    return this.partitions;
  }
}

// Main function
async function main() {
  try {
    // Check for JSON output mode
    const isJsonMode = process.argv.includes('--json');

    // Load the cached perfect hands
    const dataPath = join(__dirname, '..', 'data', 'perfect-hands.json');
    const jsonData = JSON.parse(readFileSync(dataPath, 'utf-8'));

    if (!isJsonMode) {
      console.log('3-Hand Perfect Partition & Leftover Analysis');
      console.log('=============================================');
      console.log(`Loaded ${jsonData.summary.total} perfect hands from cache`);
      console.log(`  Gold: ${jsonData.summary.byType.gold}`);
      console.log(`  Platinum: ${jsonData.summary.byType.platinum}`);
    }

    // Find all 3-hand partitions
    const finder = new PartitionFinder(jsonData.perfectHands, isJsonMode);
    const partitions = finder.findPartitions(isJsonMode);

    if (!isJsonMode) {
      console.log(`Found ${partitions.length} valid 3-hand partitions`);

      if (partitions.length === 0) {
        console.log('No valid partitions found');
        return;
      }
    }

    // Analyze each partition's leftover hand
    if (!isJsonMode) {
      console.log('\nAnalyzing leftover hands...');
    }
    const analyses: LeftoverAnalysis[] = [];

    for (let i = 0; i < partitions.length; i++) {
      if (!isJsonMode && i % 100 === 0) {
        process.stdout.write(`\rAnalyzing partition ${i + 1}/${partitions.length}...`);
      }

      const partition = partitions[i]!;
      const analysis = analyzeLeftoverHand(partition.leftoverDominoes);

      analyses.push({
        partition,
        bestTrump: analysis!.trump,
        bestTrumpName: analysis!.name,
        externalBeaters: analysis!.externalBeaters,
        detailedCounts: analysis!.details
      });
    }

    if (!isJsonMode) {
      console.log('\n\nAnalysis complete!');
    }

    // Sort by strength (fewest external beaters = strongest)
    analyses.sort((a, b) => a.externalBeaters - b.externalBeaters);

    if (isJsonMode) {
      // JSON output mode
      const jsonOutput = {
        partitions: analyses.map(analysis => ({
          hands: analysis.partition.hands.map(h => ({
            dominoes: canonicalizeHand(h.dominoes),
            trump: h.trump,
            type: h.type
          })),
          leftover: {
            dominoes: analysis.partition.leftoverDominoes,
            bestTrump: analysis.bestTrumpName,
            externalBeaters: analysis.externalBeaters
          }
        })),
        summary: {
          total: analyses.length,
          minBeaters: Math.min(...analyses.map(a => a.externalBeaters)),
          maxBeaters: Math.max(...analyses.map(a => a.externalBeaters)),
          avgBeaters: analyses.reduce((sum, a) => sum + a.externalBeaters, 0) / analyses.length
        }
      };

      console.log(JSON.stringify(jsonOutput, null, 2));
    } else {
      // Human-readable output mode
      // Display results
      console.log('\n' + '='.repeat(70));
      console.log('TOP 10 STRONGEST LEFTOVER HANDS (fewest external beaters)');
      console.log('='.repeat(70));

      for (let i = 0; i < Math.min(10, analyses.length); i++) {
        const analysis = analyses[i]!;
        console.log(`\n#${i + 1}: External beaters: ${analysis.externalBeaters} (trump: ${analysis.bestTrumpName})`);

        // Show the 3 perfect hands
        for (let j = 0; j < 3; j++) {
          const hand = analysis.partition.hands[j]!;
          const canonical = canonicalizeHand(hand.dominoes);
          console.log(`  Hand ${j + 1} (${hand.trump}): ${canonical.join(', ')}`);
        }

        // Show leftover hand
        console.log(`  Leftover: ${analysis.partition.leftoverDominoes.join(', ')}`);
      }

      // Summary statistics
      console.log('\n' + '='.repeat(70));
      console.log('SUMMARY STATISTICS');
      console.log('='.repeat(70));
      console.log(`Total partitions analyzed: ${analyses.length}`);

      const minBeaters = Math.min(...analyses.map(a => a.externalBeaters));
      const maxBeaters = Math.max(...analyses.map(a => a.externalBeaters));
      const avgBeaters = analyses.reduce((sum, a) => sum + a.externalBeaters, 0) / analyses.length;

      console.log(`Strongest leftover (min external beaters): ${minBeaters}`);
      console.log(`Weakest leftover (max external beaters): ${maxBeaters}`);
      console.log(`Average external beaters: ${avgBeaters.toFixed(1)}`);

      // Distribution
      const distribution = new Map<number, number>();
      for (const analysis of analyses) {
        const count = distribution.get(analysis.externalBeaters) || 0;
        distribution.set(analysis.externalBeaters, count + 1);
      }

      console.log('\nDistribution of external beater counts:');
      const sortedDist = Array.from(distribution.entries()).sort((a, b) => a[0] - b[0]);
      for (const [beaters, count] of sortedDist.slice(0, 10)) {
        console.log(`  ${beaters} beaters: ${count} partitions`);
      }
    }

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the program
main().catch(console.error);