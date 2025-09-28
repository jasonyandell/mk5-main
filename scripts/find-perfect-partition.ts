/**
 * Find Perfect Hand Partitions
 *
 * Finds all combinations of 4 perfect hands that together use all 28 dominoes exactly once.
 * Uses the cached perfect hands from data/perfect-hands.json
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

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

interface Solution {
  hands: PerfectHand[];
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
  private dominoToHands: Map<string, number[]>;
  private handOverlaps: boolean[][];
  private solutions: Solution[];
  private searchCount: number;

  constructor(handsData: PerfectHandData[]) {
    console.log(`Loading ${handsData.length} perfect hands...`);

    // Convert to internal format with canonical dominoes
    this.hands = handsData.map((data, index) => ({
      ...data,
      canonical: new Set(canonicalizeHand(data.dominoes)),
      index
    }));

    // Build inverted index
    this.dominoToHands = new Map();
    for (let i = 0; i < this.hands.length; i++) {
      for (const domino of this.hands[i]!.canonical) {
        if (!this.dominoToHands.has(domino)) {
          this.dominoToHands.set(domino, []);
        }
        this.dominoToHands.get(domino)!.push(i);
      }
    }

    // Pre-compute hand overlaps
    console.log('Pre-computing hand overlaps...');
    this.handOverlaps = Array(this.hands.length);
    for (let i = 0; i < this.hands.length; i++) {
      this.handOverlaps[i] = Array(this.hands.length);
      for (let j = 0; j < this.hands.length; j++) {
        this.handOverlaps[i]![j] = i === j || handsOverlap(this.hands[i]!, this.hands[j]!);
      }
    }

    this.solutions = [];
    this.searchCount = 0;
  }

  // Check if remaining dominoes can be covered by remaining hands
  private canCoverRemaining(usedDominoes: Set<string>, remainingSlots: number): boolean {
    const uncovered = new Set<string>();
    for (const domino of ALL_DOMINOES) {
      if (!usedDominoes.has(domino)) {
        uncovered.add(domino);
      }
    }

    // Check if each uncovered domino appears in enough remaining hands
    for (const domino of uncovered) {
      const handsWithDomino = this.dominoToHands.get(domino) || [];
      if (handsWithDomino.length < remainingSlots) {
        return false; // Not enough hands contain this domino
      }
    }

    return true;
  }

  // Recursive backtracking search
  private search(
    startIndex: number,
    usedHands: number[],
    usedDominoes: Set<string>
  ): void {
    this.searchCount++;

    // Progress indicator
    if (this.searchCount % 10000 === 0) {
      process.stdout.write(`\rSearched ${this.searchCount} nodes...`);
    }

    // Base case: found 4 hands
    if (usedHands.length === 4) {
      if (usedDominoes.size === 28) {
        // Found a valid partition!
        this.solutions.push({
          hands: usedHands.map(i => this.hands[i]!)
        });
        console.log(`\nFound solution #${this.solutions.length}!`);
      }
      return;
    }

    // Prune if remaining dominoes can't be covered
    const remainingSlots = 4 - usedHands.length;
    if (!this.canCoverRemaining(usedDominoes, remainingSlots)) {
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

      // Symmetry breaking: if this is the first hand and no solutions yet,
      // all solutions with this first hand would have been found
      if (usedHands.length === 0 && this.solutions.length === 0) {
        // Continue searching with different first hands
      }
    }
  }

  // Find all partitions
  findPartitions(): Solution[] {
    console.log('Analyzing domino frequencies...');

    // Check if any domino appears in fewer than 4 hands
    let impossible = false;
    for (const [domino, hands] of this.dominoToHands) {
      if (hands.length < 4) {
        console.log(`❌ Domino ${domino} only appears in ${hands.length} hands (need 4+)`);
        impossible = true;
      }
    }

    if (impossible) {
      console.log('No solution possible - some dominoes appear in too few hands');
      return [];
    }

    // Find rarest dominoes for optimization
    const dominoFrequencies = Array.from(this.dominoToHands.entries())
      .sort((a, b) => a[1].length - b[1].length);

    console.log('\nRarest dominoes:');
    for (const [domino, hands] of dominoFrequencies.slice(0, 5)) {
      console.log(`  ${domino}: appears in ${hands.length} hands`);
    }

    console.log('\nStarting search...');
    this.search(0, [], new Set());

    console.log(`\n\nSearch complete! Explored ${this.searchCount} nodes`);
    return this.solutions;
  }

  // Display a solution
  displaySolution(solution: Solution, index: number): void {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Solution #${index + 1}:`);
    console.log('='.repeat(60));

    const allUsedDominoes = new Set<string>();

    for (let i = 0; i < solution.hands.length; i++) {
      const hand = solution.hands[i]!;
      const canonical = canonicalizeHand(hand.dominoes);
      console.log(`Hand ${i + 1} (${hand.trump}, ${hand.type}): ${canonical.join(', ')}`);

      for (const domino of canonical) {
        allUsedDominoes.add(domino);
      }
    }

    // Verify all 28 dominoes are used
    if (allUsedDominoes.size === 28) {
      console.log('✅ All 28 dominoes used exactly once');
    } else {
      console.log(`❌ Only ${allUsedDominoes.size} dominoes used`);
      const missing = [];
      for (const domino of ALL_DOMINOES) {
        if (!allUsedDominoes.has(domino)) {
          missing.push(domino);
        }
      }
      if (missing.length > 0) {
        console.log(`Missing: ${missing.join(', ')}`);
      }
    }
  }
}

// Main function
async function main() {
  try {
    // Load the cached perfect hands
    const dataPath = join(__dirname, '..', 'data', 'perfect-hands.json');
    const jsonData = JSON.parse(readFileSync(dataPath, 'utf-8'));

    console.log('Perfect Hand Partition Finder');
    console.log('==============================');
    console.log(`Loaded ${jsonData.summary.total} perfect hands from cache`);
    console.log(`  Gold: ${jsonData.summary.byType.gold}`);
    console.log(`  Platinum: ${jsonData.summary.byType.platinum}`);
    console.log();

    // Create finder and search
    const finder = new PartitionFinder(jsonData.perfectHands);
    const solutions = finder.findPartitions();

    // Display results
    if (solutions.length > 0) {
      console.log(`\nFound ${solutions.length} partition(s)!`);
      for (let i = 0; i < solutions.length; i++) {
        finder.displaySolution(solutions[i]!, i);
      }
    } else {
      console.log('\nNo valid 4-hand partitions found.');
    }

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the program
main().catch(console.error);