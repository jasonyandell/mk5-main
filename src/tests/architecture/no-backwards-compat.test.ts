/**
 * Architecture tests for greenfield invariant.
 *
 * This is a greenfield project with no external users - backwards compatibility
 * code is never needed. This test detects and prevents patterns like:
 * - @deprecated annotations
 * - Legacy/backward compatibility comments
 * - _legacy, _old, _deprecated suffixes
 */

import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '../../..');

/**
 * Run grep and return matches, handling exit code 1 (no matches) gracefully
 */
function grepForPattern(pattern: string, grepFlags = ''): string[] {
  let result: string;
  try {
    result = execSync(
      `grep -r ${grepFlags} "${pattern}" src/ --include="*.ts" --exclude-dir=node_modules`,
      { encoding: 'utf-8', cwd: projectRoot }
    );
  } catch (error: unknown) {
    const err = error as { status?: number };
    if (err.status === 1) {
      // No matches found - this is good
      return [];
    }
    throw error;
  }

  return result.split('\n').filter(line => line.trim());
}

/**
 * Filter grep results to exclude allowed patterns
 */
function filterAllowed(lines: string[], allowedPatterns: string[]): string[] {
  return lines.filter(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) return false;

    const filePath = line.substring(0, colonIndex);
    return !allowedPatterns.some(pattern => filePath.includes(pattern));
  });
}

describe('Architecture: No Backwards Compatibility', () => {
  // Allowlist for legitimate patterns (should be empty in a clean codebase)
  const allowedPatterns: string[] = [
    // Test files may reference patterns for documentation purposes
    'no-backwards-compat.test.ts'
  ];

  it('no @deprecated annotations', () => {
    const lines = grepForPattern('@deprecated');
    const violations = filterAllowed(lines, allowedPatterns);

    if (violations.length > 0) {
      console.error('Found @deprecated annotations (this is a greenfield project):');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Found @deprecated annotations. This is a greenfield project - ' +
      'delete deprecated code instead of marking it deprecated.'
    ).toHaveLength(0);
  });

  it('no legacy compatibility comments', () => {
    // Match "legacy" followed by "compat" (case insensitive)
    const legacyCompat = grepForPattern('legacy.*compat', '-i');
    // Match "backward" followed by "compat" (case insensitive)
    const backwardCompat = grepForPattern('backward.*compat', '-i');

    const allLines = [...legacyCompat, ...backwardCompat];
    const violations = filterAllowed(allLines, allowedPatterns);

    if (violations.length > 0) {
      console.error('Found backwards compatibility comments:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Found backwards compatibility comments. This is a greenfield project - ' +
      'no backwards compatibility needed.'
    ).toHaveLength(0);
  });

  it('no _legacy, _old, or _deprecated suffixes', () => {
    // Match identifiers ending with these suffixes
    const lines = grepForPattern('_legacy\\|_old\\|_deprecated', '-E');
    const violations = filterAllowed(lines, allowedPatterns);

    if (violations.length > 0) {
      console.error('Found deprecated suffix patterns:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Found _legacy, _old, or _deprecated suffixed identifiers. ' +
      'Delete deprecated code instead of keeping it with suffixes.'
    ).toHaveLength(0);
  });

  it('no DEPRECATED markers in comments', () => {
    // Match uppercase DEPRECATED (common informal deprecation marker)
    const lines = grepForPattern('DEPRECATED');
    const violations = filterAllowed(lines, allowedPatterns);

    if (violations.length > 0) {
      console.error('Found DEPRECATED markers:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Found DEPRECATED markers. This is a greenfield project - ' +
      'delete deprecated code instead of marking it.'
    ).toHaveLength(0);
  });
});
