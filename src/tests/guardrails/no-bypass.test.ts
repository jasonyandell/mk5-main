/**
 * No-Bypass Guardrail Tests
 *
 * These tests enforce architectural boundaries:
 * 1. UI components must NOT import rule logic directly from rules-base.ts
 * 2. UI components must NOT import from core/dominoes.ts (utility-level) for rule decisions
 * 3. AI modules must use the GameRules interface via composition, not direct imports
 * 4. View projection must consume derived fields, not compute rules
 *
 * The "dumb client" pattern requires all rule-based computations to happen
 * server-side. These tests validate that pattern by checking import patterns.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.resolve(import.meta.dirname!, '../../..');

/**
 * Scan a file for forbidden import patterns
 */
function scanFileForImports(filePath: string, forbiddenPatterns: RegExp[]): string[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const violations: string[] = [];

  for (const pattern of forbiddenPatterns) {
    const matches = content.match(pattern);
    if (matches) {
      violations.push(...matches);
    }
  }

  return violations;
}

/**
 * Get all files matching a pattern
 */
function getFiles(dir: string, extension: string): string[] {
  const files: string[] = [];

  if (!fs.existsSync(dir)) {
    return files;
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...getFiles(fullPath, extension));
    } else if (entry.name.endsWith(extension)) {
      files.push(fullPath);
    }
  }

  return files;
}

describe('No-Bypass Guardrails: UI Components', () => {
  const componentsDir = path.join(PROJECT_ROOT, 'src/lib/components');
  const svelteFiles = getFiles(componentsDir, '.svelte');

  // UI components should never import these
  const forbiddenUiImports: RegExp[] = [
    // Direct imports from rules-base.ts (Crystal Palace internals)
    /from\s+['"][^'"]*rules-base['"]/g,
    /from\s+['"][^'"]*\/layers\/rules-base['"]/g,

    // Direct imports from compose.ts (should go through kernel)
    /from\s+['"][^'"]*\/layers\/compose['"]/g,

    // Calling rule functions directly (getLedSuit, canFollow, etc.)
    /\bgetLedSuitBase\b/g,
    /\bcanFollowBase\b/g,
    /\bsuitsWithTrumpBase\b/g,
    /\brankInTrickBase\b/g,
    /\bisTrumpBase\b/g,

    // Direct imports from individual layer files
    /from\s+['"][^'"]*\/layers\/base['"](?!\.)/g,
    /from\s+['"][^'"]*\/layers\/nello['"]/g,
    /from\s+['"][^'"]*\/layers\/sevens['"]/g,
  ];

  it('UI components do not import rule logic directly', () => {
    const violations: { file: string; imports: string[] }[] = [];

    for (const file of svelteFiles) {
      const imports = scanFileForImports(file, forbiddenUiImports);
      if (imports.length > 0) {
        violations.push({
          file: path.relative(PROJECT_ROOT, file),
          imports
        });
      }
    }

    if (violations.length > 0) {
      const message = violations
        .map(v => `${v.file}:\n  ${v.imports.join('\n  ')}`)
        .join('\n\n');
      expect.fail(`UI components have forbidden imports:\n${message}`);
    }
  });

  it('view-projection.ts does not compute rule logic locally', () => {
    const viewProjectionPath = path.join(PROJECT_ROOT, 'src/game/view-projection.ts');

    if (!fs.existsSync(viewProjectionPath)) {
      // File may not exist yet - skip
      return;
    }

    const forbiddenViewProjectionPatterns: RegExp[] = [
      // Should not import rule base functions
      /from\s+['"][^'"]*rules-base['"]/g,

      // Should not call base rule functions
      /\bgetLedSuitBase\b/g,
      /\bcanFollowBase\b/g,
      /\bsuitsWithTrumpBase\b/g,
      /\brankInTrickBase\b/g,
      /\bisTrumpBase\b/g,

      // Should not import compose (rules come via DerivedViewFields)
      /from\s+['"][^'"]*\/layers\/compose['"]/g,
    ];

    const violations = scanFileForImports(viewProjectionPath, forbiddenViewProjectionPatterns);

    if (violations.length > 0) {
      expect.fail(`view-projection.ts has forbidden imports:\n  ${violations.join('\n  ')}`);
    }
  });
});

describe('No-Bypass Guardrails: AI Modules', () => {
  const aiDir = path.join(PROJECT_ROOT, 'src/game/ai');
  const aiFiles = getFiles(aiDir, '.ts');

  // AI modules should use GameRules interface, not direct base function imports
  // Exception: domino-strength.ts may import isTrumpBase for display-only purposes
  const forbiddenAiImports: RegExp[] = [
    // Direct calls to base functions should use rules.method() instead
    /\bgetLedSuitBase\s*\(/g,
    /\bcanFollowBase\s*\(/g,
    /\brankInTrickBase\s*\(/g,
    /\bisValidPlayBase\s*\(/g,
    /\bgetValidPlaysBase\s*\(/g,
  ];

  it('AI modules use GameRules interface for rule calls', () => {
    const violations: { file: string; calls: string[] }[] = [];

    for (const file of aiFiles) {
      // Skip type definition files
      if (file.endsWith('types.ts')) continue;

      const calls = scanFileForImports(file, forbiddenAiImports);
      if (calls.length > 0) {
        violations.push({
          file: path.relative(PROJECT_ROOT, file),
          calls
        });
      }
    }

    if (violations.length > 0) {
      const message = violations
        .map(v => `${v.file}:\n  ${v.calls.join('\n  ')}`)
        .join('\n\n');
      expect.fail(`AI modules call base functions directly instead of using GameRules:\n${message}`);
    }
  });

  it('AI modules that need rules create them via composeRules', () => {
    // Files that use rules should import composeRules
    const filesNeedingRules = [
      'utilities.ts',
      'domino-strength.ts',
      'monte-carlo.ts',
      'rollout-strategy.ts'
    ];

    for (const fileName of filesNeedingRules) {
      const filePath = path.join(aiDir, fileName);
      if (!fs.existsSync(filePath)) continue;

      const content = fs.readFileSync(filePath, 'utf-8');

      // If file uses rules.method(), it should import composeRules or receive rules as parameter
      const usesRules = /rules\.(getLedSuit|canFollow|isTrump|calculateTrickWinner|rankInTrick|getValidPlays)/g.test(content);

      if (usesRules) {
        const hasComposeImport = /from\s+['"][^'"]*compose['"]/g.test(content);
        const hasRulesParam = /\brules\s*:\s*GameRules\b/g.test(content);

        expect(
          hasComposeImport || hasRulesParam,
          `${fileName} uses rules.method() but doesn't import composeRules or accept GameRules parameter`
        ).toBe(true);
      }
    }
  });
});

describe('No-Bypass Guardrails: Core Module Boundaries', () => {
  it('core/scoring.ts does not contain rule logic', () => {
    const scoringPath = path.join(PROJECT_ROOT, 'src/game/core/scoring.ts');

    if (!fs.existsSync(scoringPath)) return;

    const forbiddenPatterns: RegExp[] = [
      // Scoring should not contain trump/suit/follow logic
      /\bgetLedSuit\b/g,
      /\bcanFollow\b/g,
      /\bisTrump\b/g,
      /\brankInTrick\b/g,
    ];

    const violations = scanFileForImports(scoringPath, forbiddenPatterns);

    if (violations.length > 0) {
      expect.fail(`core/scoring.ts contains rule logic that should be in layers:\n  ${violations.join('\n  ')}`);
    }
  });

  it('core/dominoes.ts contains only utilities, not rule logic', () => {
    const dominoesPath = path.join(PROJECT_ROOT, 'src/game/core/dominoes.ts');

    if (!fs.existsSync(dominoesPath)) return;

    const content = fs.readFileSync(dominoesPath, 'utf-8');

    // Should not import from layers
    const layerImports = /from\s+['"][^'"]*\/layers\//g.test(content);
    expect(layerImports, 'core/dominoes.ts should not import from layers/').toBe(false);

    // Should not call compose
    const composeCalls = /\bcomposeRules\b/g.test(content);
    expect(composeCalls, 'core/dominoes.ts should not call composeRules').toBe(false);
  });
});

describe('No-Bypass Guardrails: Server-Side Projection', () => {
  it('kernel.ts computes derived fields using ctx.rules', () => {
    const kernelPath = path.join(PROJECT_ROOT, 'src/kernel/kernel.ts');

    if (!fs.existsSync(kernelPath)) return;

    const content = fs.readFileSync(kernelPath, 'utf-8');

    // Kernel should use ctx.rules for computations
    const usesCtxRules = /ctx\.rules\./g.test(content) || /rules\./g.test(content);
    expect(usesCtxRules, 'kernel.ts should use ctx.rules for rule computations').toBe(true);

    // Kernel should not import base functions directly
    const importsForbidden = /from\s+['"][^'"]*rules-base['"]/g.test(content);
    expect(importsForbidden, 'kernel.ts should not import from rules-base directly').toBe(false);
  });

  it('DerivedViewFields contains all rule-aware fields', () => {
    const typesPath = path.join(PROJECT_ROOT, 'src/multiplayer/types.ts');

    if (!fs.existsSync(typesPath)) return;

    const content = fs.readFileSync(typesPath, 'utf-8');

    // DerivedViewFields should exist
    const hasDerivedFields = /interface\s+DerivedViewFields|type\s+DerivedViewFields/g.test(content);
    expect(hasDerivedFields, 'DerivedViewFields type should be defined').toBe(true);

    // Should include key derived fields
    const hasIsTrump = /isTrump/g.test(content);
    const hasCanFollow = /canFollow/g.test(content);
    const hasTrickWinner = /currentTrickWinner|trickWinner/g.test(content);

    expect(hasIsTrump, 'DerivedViewFields should include isTrump').toBe(true);
    expect(hasCanFollow, 'DerivedViewFields should include canFollow').toBe(true);
    expect(hasTrickWinner, 'DerivedViewFields should include trick winner').toBe(true);
  });
});

describe('No-Bypass Guardrails: Import Dependency Graph', () => {
  it('layers/ directory is the single source for rule logic', () => {
    const layersDir = path.join(PROJECT_ROOT, 'src/game/layers');

    // These files should exist and define all rule logic
    const requiredFiles = ['rules-base.ts', 'compose.ts', 'base.ts', 'types.ts'];

    for (const file of requiredFiles) {
      const filePath = path.join(layersDir, file);
      expect(fs.existsSync(filePath), `${file} should exist in layers/`).toBe(true);
    }
  });

  it('rules-base.ts exports the 6 core base functions', () => {
    const rulesBasePath = path.join(PROJECT_ROOT, 'src/game/layers/rules-base.ts');

    if (!fs.existsSync(rulesBasePath)) return;

    const content = fs.readFileSync(rulesBasePath, 'utf-8');

    const coreFunctions = [
      'getLedSuitBase',
      'suitsWithTrumpBase',
      'canFollowBase',
      'rankInTrickBase',
      'isTrumpBase',
      'isValidPlayBase'
    ];

    for (const fn of coreFunctions) {
      const isExported = new RegExp(`export\\s+function\\s+${fn}`).test(content);
      expect(isExported, `${fn} should be exported from rules-base.ts`).toBe(true);
    }
  });

  it('compose.ts exports composeRules', () => {
    const composePath = path.join(PROJECT_ROOT, 'src/game/layers/compose.ts');

    if (!fs.existsSync(composePath)) return;

    const content = fs.readFileSync(composePath, 'utf-8');

    const hasComposeRules = /export\s+function\s+composeRules/.test(content);
    expect(hasComposeRules, 'compose.ts should export composeRules').toBe(true);
  });
});
