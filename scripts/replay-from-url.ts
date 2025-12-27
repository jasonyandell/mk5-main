#!/usr/bin/env npx tsx
/**
 * Replay game actions from a v2 URL containing encoded state.
 *
 * URL format (v2): ?s=seed&l=layers&a=compressedActions
 * Parameters:
 *   s - Seed (base36)
 *   i - Initial hands (base64url, mutually exclusive with s)
 *   p - Player types (e.g., "haaa" = human + 3 AI)
 *   d - Dealer (0-3)
 *   l - Layers (e.g., "n,t" = nello, tournament)
 *   t - Theme
 *   v - Color overrides
 *   a - Compressed actions
 *
 * Usage:
 *   npx tsx scripts/replay-from-url.ts <url> [options]
 *
 * Options:
 *   --stop-at N         Stop replay at action N
 *   --verbose           Show each action as it's replayed
 *   --generate-test     Generate a test file in scratch/
 *   --action-range S E  Show details only for actions S to E
 *   --show-tricks       Display trick-by-trick breakdown
 *   --hand N            Focus on just hand N
 *   --compact           One-line-per-action format
 *
 * Examples:
 *   npx tsx scripts/replay-from-url.ts "http://localhost:60101/?s=abc&a=CAAS"
 *   npx tsx scripts/replay-from-url.ts "?s=1a2b&a=CAASkb" --show-tricks
 *   npx tsx scripts/replay-from-url.ts "<url>" --generate-test
 */

import { replayFromUrl, type ReplayOptions } from '../src/game/utils/urlReplay';
import { decodeGameUrl, compressEvents, decompressEvents } from '../src/game/core/url-compression';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Parse command line arguments
const args = process.argv.slice(2);
const url = args[0];

if (!url) {
  console.error('Usage: npx tsx scripts/replay-from-url.ts <url> [options]');
  console.error('\nOptions:');
  console.error('  --stop-at N         Stop replay at action N');
  console.error('  --verbose           Show each action as it\'s replayed');
  console.error('  --generate-test     Generate a test file in scratch/');
  console.error('  --action-range S E  Show details only for actions S to E');
  console.error('  --show-tricks       Display trick-by-trick breakdown');
  console.error('  --hand N            Focus on just hand N');
  console.error('  --compact           One-line-per-action format');
  console.error('\nExamples:');
  console.error('  npx tsx scripts/replay-from-url.ts "http://localhost:60101/?s=abc&a=CAAS"');
  console.error('  npx tsx scripts/replay-from-url.ts "?s=1a2b&a=CAASkb3" --show-tricks');
  console.error('  npx tsx scripts/replay-from-url.ts "<url>" --generate-test');
  process.exit(1);
}

// Parse options
let stopAt: number | undefined;
let verbose = false;
let generateTest = false;
let actionRangeStart: number | undefined;
let actionRangeEnd: number | undefined;
let showTricks = false;
let focusHand: number | undefined;
let compact = false;

for (let i = 1; i < args.length; i++) {
  if (args[i] === '--stop-at' && args[i + 1]) {
    stopAt = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i] === '--verbose') {
    verbose = true;
  } else if (args[i] === '--generate-test') {
    generateTest = true;
  } else if (args[i] === '--action-range' && args[i + 1] && args[i + 2]) {
    actionRangeStart = parseInt(args[i + 1], 10);
    actionRangeEnd = parseInt(args[i + 2], 10);
    i += 2;
  } else if (args[i] === '--show-tricks') {
    showTricks = true;
  } else if (args[i] === '--hand' && args[i + 1]) {
    focusHand = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i] === '--compact') {
    compact = true;
  }
}

// Handle test generation
if (generateTest) {
  // Decode URL to extract components
  const queryString = url.includes('?') ? url.split('?')[1] : url;
  const decoded = decodeGameUrl(queryString || '');

  // Ensure scratch directory exists
  const scratchDir = path.join(__dirname, '..', 'scratch');
  if (!fs.existsSync(scratchDir)) {
    fs.mkdirSync(scratchDir, { recursive: true });
  }

  // Generate test file with timestamp
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const testFile = path.join(scratchDir, `test-${timestamp}.test.ts`);

  // Compress actions back to get the URL string
  const compressedActions = compressEvents(decoded.actions);

  // Build query string for embedding in test
  const queryParams: string[] = [];
  if (decoded.initialHands) {
    // Use initialHands - would need to re-encode, but for now we'll use seed approach
    console.error('Note: initialHands-based URLs not yet fully supported in test generation. Using actions array directly.');
  } else {
    queryParams.push(`s=${decoded.seed.toString(36)}`);
  }
  if (decoded.playerTypes.join('') !== 'haaa') {
    queryParams.push(`p=${decoded.playerTypes.map(t => t[0]).join('')}`);
  }
  if (decoded.dealer !== 3) {
    queryParams.push(`d=${decoded.dealer}`);
  }
  if (decoded.layers && decoded.layers.length > 0) {
    // Map layer names to short codes
    const layerCodes: Record<string, string> = {
      'nello': 'n', 'plunge': 'p', 'splash': 'S', 'sevens': 'v',
      'tournament': 't', 'oneHand': 'o', 'speed': 's', 'hints': 'h'
    };
    const codes = decoded.layers.map(l => layerCodes[l] || l).join(',');
    queryParams.push(`l=${codes}`);
  }
  queryParams.push(`a=${compressedActions}`);
  const reconstructedQuery = '?' + queryParams.join('&');

  const testContent = `/**
 * Test generated from URL replay
 * Generated: ${new Date().toISOString()}
 *
 * Original URL: ${url}
 * Reconstructed query: ${reconstructedQuery}
 *
 * Run with: npx vitest --config vitest.scratch.config.ts run scratch/test-${timestamp}.test.ts
 * Or move to src/tests/ to run with production suite
 */

import { describe, it, expect } from 'vitest';
import { replayFromUrl } from '../src/game/utils/urlReplay';
import { decodeGameUrl } from '../src/game/core/url-compression';

describe('URL replay test', () => {
  const testUrl = '${reconstructedQuery.replace(/'/g, "\\'")}';

  it('should replay all actions without errors', () => {
    const result = replayFromUrl(testUrl);

    expect(result.errors ?? []).toEqual([]);
    expect(result.actionCount).toBe(${decoded.actions.length});
  });

  it('should reach expected game state', () => {
    const result = replayFromUrl(testUrl);
    const state = result.state;

    // Snapshot current state - update these assertions as needed
    console.log('Phase:', state.phase);
    console.log('Team Scores:', state.teamScores);
    console.log('Team Marks:', state.teamMarks);
    console.log('Hands Played:', state.teamMarks[0] + state.teamMarks[1], 'marks total');

    // TODO: Add your specific assertions here
    // Example assertions:
    // expect(state.phase).toBe('playing');
    // expect(state.teamScores).toEqual([17, 22]);
  });

  it('should decode URL correctly', () => {
    const decoded = decodeGameUrl(testUrl);

    expect(decoded.seed).toBe(${decoded.seed});
    expect(decoded.actions.length).toBe(${decoded.actions.length});
    expect(decoded.playerTypes).toEqual(${JSON.stringify(decoded.playerTypes)});
    ${decoded.layers ? `expect(decoded.layers).toEqual(${JSON.stringify(decoded.layers)});` : '// No layers specified'}
  });

  // Utility: replay to a specific action for debugging
  it.skip('debug: replay to specific action', () => {
    const stopAt = 50; // Change this to the action you want to stop at

    const result = replayFromUrl(testUrl, { stopAt });
    const state = result.state;

    console.log('=== State at action', stopAt, '===');
    console.log('Phase:', state.phase);
    console.log('Team Scores:', state.teamScores);
    console.log('Current Trick:', state.currentTrick);
    console.log('Tricks Played:', state.tricks.length);

    if (state.phase === 'playing' || state.phase === 'bidding' || state.phase === 'trump-selection') {
      console.log('\\n=== Player Hands ===');
      state.players.forEach((p, i) => {
        console.log(\`P\${i}: \${p.hand.map(d => d.id).join(', ')}\`);
      });
    }
  });
});
`;

  fs.writeFileSync(testFile, testContent);
  console.log(`Generated test file: ${testFile}`);
  console.log('\nTo run the test:');
  console.log(`  npx vitest --config vitest.scratch.config.ts run ${testFile}`);
  console.log('\nNext steps:');
  console.log('1. Add assertions for the specific bug you\'re testing');
  console.log('2. Run the test to verify it catches the issue');
  console.log('3. Fix the bug in the game logic');
  console.log('4. Re-run the test to confirm the fix');
  process.exit(0);
}

// Normal replay mode
const options: ReplayOptions = {
  stopAt,
  verbose,
  actionRangeStart,
  actionRangeEnd,
  showTricks,
  focusHand,
  compact
};

try {
  console.log('Replaying game from URL...\n');

  const result = replayFromUrl(url, options);

  if (result.errors && result.errors.length > 0) {
    console.error('\n Replay encountered errors:');
    result.errors.forEach(err => console.error(`  ${err}`));
  }

  console.log(`\n Replayed ${result.actionCount} actions successfully`);
  console.log('\n=== Final Game State ===');
  console.log('Phase:', result.state.phase);
  console.log('Team Scores:', result.state.teamScores);
  console.log('Team Marks:', result.state.teamMarks);
  console.log('Hands Played:', result.state.teamMarks[0] + result.state.teamMarks[1], 'marks total');

  if (result.state.phase === 'playing' || result.state.phase === 'scoring') {
    console.log('\n=== Current Hand ===');
    console.log('Dealer: Player', result.state.dealer);
    console.log('Winning Bidder: Player', result.state.winningBidder);
    console.log('Bid:', result.state.currentBid?.value ?? 'N/A');
    console.log('Trump:', result.state.trump.type === 'suit'
      ? `Suit ${result.state.trump.suit}`
      : result.state.trump.type);
    console.log('Tricks Played:', result.state.tricks.length);

    if (result.state.tricks.length > 0) {
      let team0Points = 0;
      let team1Points = 0;

      result.state.tricks.forEach((trick) => {
        const winnerTeam = trick.winner !== undefined && trick.winner % 2 === 0 ? 0 : 1;
        if (winnerTeam === 0) {
          team0Points += trick.points;
        } else {
          team1Points += trick.points;
        }
      });

      console.log('Points Won So Far: Team 0:', team0Points, ', Team 1:', team1Points);
    }
  }

  if (verbose || stopAt !== undefined) {
    console.log('\n=== Full State (JSON) ===');
    console.log(JSON.stringify(result.state, null, 2));
  }

} catch (error) {
  console.error(' Error:', error instanceof Error ? error.message : error);
  if (verbose) {
    console.error(error);
  }
  process.exit(1);
}
