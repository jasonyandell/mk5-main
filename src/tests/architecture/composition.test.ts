/**
 * Architecture tests for composition invariant.
 *
 * Verifies that ExecutionContext composition happens only at designated points:
 * 1. Room.ts - Production composition
 * 2. HeadlessRoom.ts - Tool/simulation composition
 * 3. Test helpers - Test utilities
 * 4. Test files - Direct composition testing only
 */

import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Architecture: Single Composition Point', () => {
  it('only allowed files import createExecutionContext', () => {
    // Use grep to find all imports of createExecutionContext
    let result: string;
    try {
      result = execSync(
        'grep -r "createExecutionContext" src/ --include="*.ts" --exclude-dir=node_modules',
        { encoding: 'utf-8', cwd: path.join(__dirname, '../../..') }
      );
    } catch (error: unknown) {
      // grep exits with code 1 if no matches found
      const err = error as { status?: number; stdout?: string };
      if (err.status === 1) {
        // No matches is actually OK for this test (means no violations)
        result = '';
      } else {
        throw error;
      }
    }

    // Filter to only lines that have import statements
    const lines = result
      .split('\n')
      .filter(line => line.trim())
      .filter(line => {
        // Match import statements (both named and type imports)
        return line.includes('import') && line.includes('createExecutionContext');
      });

    // Allowed files/patterns (relative to project root)
    const allowedPatterns = [
      'src/server/Room.ts',
      'src/server/HeadlessRoom.ts',
      'src/tests/helpers/',
      'src/game/types/execution.ts', // Definition file
      'src/game/ai/strategies/intermediate.ts', // Monte Carlo AI needs direct composition for fast simulations
      '.test.ts',
      '.spec.ts'
    ];

    // Check each line for violations
    const violations = lines.filter(line => {
      // Extract file path from grep output (format: "filepath:line content")
      const colonIndex = line.indexOf(':');
      if (colonIndex === -1) return false;

      const filePath = line.substring(0, colonIndex);

      // Check if file matches any allowed pattern
      const isAllowed = allowedPatterns.some(pattern => filePath.includes(pattern));

      return !isAllowed;
    });

    if (violations.length > 0) {
      console.error('Found violations of single composition point invariant:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Found createExecutionContext imports outside allowed files. ' +
      'Only Room.ts, HeadlessRoom.ts, test helpers, and test files may import createExecutionContext.'
    ).toHaveLength(0);
  });

  it('client code does not import engine helpers', () => {
    // Directories that should NOT import engine execution helpers
    const clientDirs = [
      'src/stores/'
    ];

    // Imports that are banned in client code
    const bannedImports = [
      'getNextStates',
      'executeAction',
      'createExecutionContext'
    ];

    // No exceptions needed - multiplayer authorization now lives in src/multiplayer/
    // which is not a client directory
    const allowedFiles: string[] = [];

    const violations: string[] = [];

    clientDirs.forEach(dir => {
      bannedImports.forEach(importName => {
        let result: string;
        try {
          result = execSync(
            `grep -r "import.*${importName}" ${dir} --include="*.ts" || true`,
            { encoding: 'utf-8', cwd: path.join(__dirname, '../../..') }
          );
        } catch {
          result = '';
        }

        if (result.trim()) {
          // Check if it's actually an import (not just a comment or string)
          const lines = result.split('\n').filter(line => {
            return line.includes('import') && line.includes(importName);
          });

          lines.forEach(line => {
            // Extract file path from grep output
            const colonIndex = line.indexOf(':');
            if (colonIndex === -1) return;

            const filePath = line.substring(0, colonIndex);

            // Check if this file is allowed
            const isAllowed = allowedFiles.some(allowed => filePath.includes(allowed));
            if (!isAllowed) {
              violations.push(`${dir}: ${line.trim()}`);
            }
          });
        }
      });
    });

    if (violations.length > 0) {
      console.error('Found banned imports in client code:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Client code (stores, multiplayer) must not import engine execution helpers. ' +
      'Use Room/HeadlessRoom for execution, trust server for state.'
    ).toHaveLength(0);
  });

  it('Room.ts is the primary composition point', () => {
    const roomPath = path.join(__dirname, '../../../src/server/Room.ts');

    // Verify Room.ts exists
    expect(fs.existsSync(roomPath), 'Room.ts must exist as primary composition point').toBe(true);

    // Verify Room.ts imports createExecutionContext
    const roomContent = fs.readFileSync(roomPath, 'utf-8');
    expect(
      roomContent.includes('createExecutionContext'),
      'Room.ts must import createExecutionContext for composition'
    ).toBe(true);

    // Verify Room creates ExecutionContext in constructor
    expect(
      roomContent.includes('this.ctx') || roomContent.includes('ExecutionContext'),
      'Room.ts must create ExecutionContext'
    ).toBe(true);
  });

  it('test helpers centralize test composition', () => {
    const helperPath = path.join(__dirname, '../helpers/executionContext.ts');

    // Verify test helper exists
    expect(fs.existsSync(helperPath), 'executionContext.ts test helper must exist').toBe(true);

    // Verify helper provides factory functions
    const helperContent = fs.readFileSync(helperPath, 'utf-8');
    expect(
      helperContent.includes('createTestContext'),
      'Test helper must provide createTestContext factory'
    ).toBe(true);

    expect(
      helperContent.includes('createExecutionContext'),
      'Test helper must use createExecutionContext'
    ).toBe(true);
  });

  it('ExecutionContext type is properly defined', () => {
    const typesPath = path.join(__dirname, '../../../src/game/types/execution.ts');

    // Verify types file exists
    expect(fs.existsSync(typesPath), 'execution.ts types file must exist').toBe(true);

    // Verify it exports ExecutionContext and createExecutionContext
    const typesContent = fs.readFileSync(typesPath, 'utf-8');

    expect(
      typesContent.includes('export interface ExecutionContext') ||
      typesContent.includes('export type ExecutionContext'),
      'Must export ExecutionContext type'
    ).toBe(true);

    expect(
      typesContent.includes('export function createExecutionContext') ||
      typesContent.includes('export const createExecutionContext'),
      'Must export createExecutionContext function'
    ).toBe(true);
  });

  it('no direct composition in utils or ai modules', () => {
    // Files that MUST NOT import createExecutionContext or getNextStates
    // These should use Room/HeadlessRoom for composition
    const restrictedDirs = [
      'src/game/utils/',
      'src/game/ai/'
    ];

    // Allowlist for files that legitimately need direct composition
    // Monte Carlo AI needs createExecutionContext for fast simulations
    const allowedFiles = [
      'src/game/ai/strategies/intermediate.ts'
    ];

    const violations: string[] = [];

    restrictedDirs.forEach(dir => {
      const bannedImports = ['createExecutionContext', 'getNextStates'];

      bannedImports.forEach(importName => {
        let result: string;
        try {
          result = execSync(
            `grep -r "import.*${importName}" ${dir} --include="*.ts" || true`,
            { encoding: 'utf-8', cwd: path.join(__dirname, '../../..') }
          );
        } catch {
          result = '';
        }

        if (result.trim()) {
          // Check if it's actually an import (not just a comment or string)
          const lines = result.split('\n').filter(line => {
            return line.includes('import') && line.includes(importName);
          });

          lines.forEach(line => {
            const colonIndex = line.indexOf(':');
            if (colonIndex !== -1) {
              const file = line.substring(0, colonIndex);
              // Skip allowed files
              if (!allowedFiles.some(allowed => file.endsWith(allowed))) {
                violations.push(file);
              }
            }
          });
        }
      });
    });

    if (violations.length > 0) {
      console.error('Found composition violations in utils/ai modules:');
      violations.forEach(v => console.error(`  ${v}`));
    }

    expect(violations,
      'Utils and AI modules must use Room/HeadlessRoom, not direct composition. ' +
      'These files should have been migrated as part of the URL replay refactor.'
    ).toHaveLength(0);
  });
});
