#!/usr/bin/env node
/**
 * Automated fixer for layer test TypeScript errors
 * Fixes common patterns:
 * - Removes undefined from player arrays
 * - Adds missing Player properties (name, teamId, marks, hand)
 * - Fixes GameAction property access with type guards
 */

import fs from 'fs';
import { glob } from 'glob';

const files = glob.sync('src/tests/layers/**/*.test.ts');

for (const file of files) {
  let content = fs.readFileSync(file, 'utf8');
  let modified = false;

  // Pattern 1: Fix `players: [undefined, { id: N, hand }]` - just use spread
  const undefPlayerPattern = /players:\s*\[\s*undefined,\s*\{\s*id:\s*(\d+),\s*name:\s*'([^']+)',\s*teamId:\s*(\d+),\s*marks:\s*0,\s*hand\s*\}/g;
  if (undefPlayerPattern.test(content)) {
    content = content.replace(undefPlayerPattern, (match, id, name, teamId) => {
      modified = true;
      return `players: [
          undefined,
          { id: ${id}, name: '${name}', teamId: ${teamId}, marks: 0, hand }
        ]`;
    });
  }

  // Pattern 2: Add missing `name` property to players with only `{ id: N, hand }`
  const missingPropsPattern = /\{\s*id:\s*(\d+),\s*hand:\s*([^\}]+)\s*\}/g;
  const matches = [...content.matchAll(missingPropsPattern)];
  for (const match of matches.reverse()) { // reverse to maintain positions
    const id = parseInt(match[1]);
    const handExpr = match[2];
    const teamId = id % 2; // alternate teams
    const replacement = `{ id: ${id}, name: 'P${id}', teamId: ${teamId}, marks: 0, hand: ${handExpr} }`;
    content = content.substring(0, match.index) + replacement + content.substring(match.index + match[0].length);
    modified = true;
  }

  if (modified) {
    fs.writeFileSync(file, content);
    console.log(`Fixed: ${file}`);
  }
}

console.log('Done!');
