#!/usr/bin/env npx tsx
/**
 * Decode a v2 URL to show its components.
 *
 * Usage:
 *   npx tsx scripts/decode-url.ts <url-or-query-string>
 *
 * Examples:
 *   npx tsx scripts/decode-url.ts "http://localhost:60101/?s=abc&a=CAAS"
 *   npx tsx scripts/decode-url.ts "?s=9ix&l=n,t&a=CAASkb3"
 *   npx tsx scripts/decode-url.ts "s=abc&a=CAAS"
 */

import { decodeGameUrl } from '../src/game/core/url-compression';

const input = process.argv[2];

if (!input) {
  console.error('Usage: npx tsx scripts/decode-url.ts <url-or-query-string>');
  console.error('\nExamples:');
  console.error('  npx tsx scripts/decode-url.ts "http://localhost:60101/?s=abc&a=CAAS"');
  console.error('  npx tsx scripts/decode-url.ts "?s=9ix&l=n,t&a=CAASkb3"');
  console.error('  npx tsx scripts/decode-url.ts "s=abc&a=CAAS"');
  process.exit(1);
}

try {
  const decoded = decodeGameUrl(input);

  console.log('=== URL DECODED ===');
  console.log('Seed:', decoded.seed, `(base36: ${decoded.seed.toString(36)})`);
  console.log('Player Types:', decoded.playerTypes.join(', '));
  console.log('Dealer:', decoded.dealer);
  console.log('Theme:', decoded.theme);

  if (decoded.layers && decoded.layers.length > 0) {
    console.log('Layers:', decoded.layers.join(', '));
  }

  if (decoded.initialHands) {
    console.log('\n=== INITIAL HANDS ===');
    decoded.initialHands.forEach((hand, i) => {
      console.log(`Player ${i}: ${hand.map(d => d.id).join(', ')}`);
    });
  }

  if (Object.keys(decoded.colorOverrides).length > 0) {
    console.log('\n=== COLOR OVERRIDES ===');
    Object.entries(decoded.colorOverrides).forEach(([varName, value]) => {
      console.log(`  ${varName}: ${value}`);
    });
  }

  console.log('\n=== ACTIONS ===');
  console.log(`Total: ${decoded.actions.length} actions`);

  if (decoded.actions.length > 0) {
    decoded.actions.forEach((action, index) => {
      console.log(`  ${index + 1}. ${action}`);
    });
  }

  console.log('\n=== RAW DATA ===');
  console.log(JSON.stringify(decoded, null, 2));

} catch (error) {
  console.error('Failed to decode:', error instanceof Error ? error.message : error);
  process.exit(1);
}
