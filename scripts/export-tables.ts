/**
 * Export domino tables as JSON for Python test comparison.
 *
 * Usage: npx tsx scripts/export-tables.ts
 */

import {
  DOMINO_PIPS,
  EFFECTIVE_SUIT,
  SUIT_MASK,
  HAS_POWER,
  RANK,
} from '../src/game/core/domino-tables';

const tables = {
  DOMINO_PIPS: Array.from(DOMINO_PIPS),
  EFFECTIVE_SUIT: Array.from(EFFECTIVE_SUIT),
  SUIT_MASK: Array.from(SUIT_MASK),
  HAS_POWER: Array.from(HAS_POWER),
  RANK: Array.from(RANK),
};

console.log(JSON.stringify(tables, null, 2));
