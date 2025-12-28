#!/usr/bin/env npx tsx
/**
 * Encode game parameters to a v2 URL.
 *
 * Creates shareable URLs using the current URL compression format.
 *
 * Usage:
 *   npx tsx scripts/encode-url.ts <seed> [action1] [action2] ...
 *
 * Examples:
 *   npx tsx scripts/encode-url.ts 12345
 *   npx tsx scripts/encode-url.ts 12345 pass pass pass bid-30
 *   npx tsx scripts/encode-url.ts 12345 pass pass pass bid-30 trump-blanks play-6-6
 *
 * Action IDs can be:
 *   - Bidding: pass, redeal, bid-30, bid-31, ..., bid-42, bid-1-marks, bid-2-marks, bid-3-marks
 *   - Trump: trump-blanks, trump-aces, ..., trump-sixes, trump-doubles, trump-no-trump
 *   - Dominoes: play-0-0, play-1-0, play-1-1, ..., play-6-6
 *   - Consensus: complete-trick, score-hand
 */

import { encodeGameUrl, compressEvents } from '../src/game/core/url-compression';

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('Usage: npx tsx scripts/encode-url.ts <seed> [action1] [action2] ...');
  console.error('\nExamples:');
  console.error('  npx tsx scripts/encode-url.ts 12345');
  console.error('  npx tsx scripts/encode-url.ts 12345 pass pass pass bid-30');
  console.error('  npx tsx scripts/encode-url.ts 12345 pass pass pass bid-30 trump-blanks play-6-6');
  console.error('\nAction IDs:');
  console.error('  Bidding: pass, redeal, bid-30, bid-31, ..., bid-42, bid-1-marks, bid-2-marks, bid-3-marks');
  console.error('  Trump: trump-blanks, trump-aces, ..., trump-sixes, trump-doubles, trump-no-trump');
  console.error('  Dominoes: play-0-0, play-1-0, play-1-1, ..., play-6-6');
  console.error('  Consensus: complete-trick, score-hand');
  process.exit(1);
}

const seedArg = args[0];
if (!seedArg) {
  console.error('Error: seed is required');
  process.exit(1);
}

const seed = parseInt(seedArg, 10);
if (isNaN(seed)) {
  console.error('Error: First argument must be a numeric seed');
  process.exit(1);
}

const actions = args.slice(1);

// Validate actions are known
const knownActions = new Set([
  'pass', 'redeal',
  'bid-30', 'bid-31', 'bid-32', 'bid-33', 'bid-34', 'bid-35', 'bid-36', 'bid-37',
  'bid-38', 'bid-39', 'bid-40', 'bid-41', 'bid-42',
  'bid-1-marks', 'bid-2-marks', 'bid-3-marks',
  'bid-plunge', 'bid-splash', 'bid-84-honors', 'bid-sevens',
  'trump-blanks', 'trump-aces', 'trump-deuces', 'trump-tres',
  'trump-fours', 'trump-fives', 'trump-sixes',
  'trump-doubles', 'trump-no-trump',
  'play-0-0',
  'play-1-0', 'play-1-1',
  'play-2-0', 'play-2-1', 'play-2-2',
  'play-3-0', 'play-3-1', 'play-3-2', 'play-3-3',
  'play-4-0', 'play-4-1', 'play-4-2', 'play-4-3', 'play-4-4',
  'play-5-0', 'play-5-1', 'play-5-2', 'play-5-3', 'play-5-4', 'play-5-5',
  'play-6-0', 'play-6-1', 'play-6-2', 'play-6-3', 'play-6-4', 'play-6-5', 'play-6-6',
  'complete-trick', 'score-hand',
  'agree-trick-p0', 'agree-trick-p1', 'agree-trick-p2', 'agree-trick-p3',
  'agree-score-p0', 'agree-score-p1', 'agree-score-p2', 'agree-score-p3',
  'timeout-p0', 'timeout-p1', 'timeout-p2', 'timeout-p3',
  'retry-one-hand', 'new-one-hand'
]);

for (const action of actions) {
  if (!knownActions.has(action)) {
    console.warn(`Warning: Unknown action "${action}" - may not encode correctly`);
  }
}

// Generate URL
const queryString = encodeGameUrl(seed, actions);
console.log(`http://localhost:60101/${queryString}`);

// Also show the compressed format
if (actions.length > 0) {
  console.log(`\nCompressed actions: ${compressEvents(actions)}`);
}
