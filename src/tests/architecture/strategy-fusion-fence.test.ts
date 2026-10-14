// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

/**
 * STRATEGY-FUSION FENCE: the guard that keeps the fence self-enforcing.
 *
 * A clairvoyant (double-dummy, perfect-information) value used per world and
 * then averaged as a seat's value is strategy fusion: "strong but wrong"
 * (Jason, 2026-10-13). The repo keeps its clairvoyant solvers as fenced
 * references. What a player uses instead is an information-state solve: one
 * choice per information set, never one per world (Silas's (certain-make),
 * Zeb's plan). The ruling, the inventory and the sanctioned uses:
 * wiki/topics/strategy-fusion-fence.md.
 *
 * The rule enforced here: every code file git sees (tracked or new) that defines, imports or
 * calls a clairvoyant API carries the token `STRATEGY-FUSION FENCE`. A new file
 * that reaches for one without the token fails this test. Adding the token is
 * a claim that the use is sanctioned (a bound, a baseline or a measurement of
 * the fusion itself); record it on the wiki page.
 */

import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const projectRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const TOKEN = 'STRATEGY-FUSION FENCE';
const RULING = 'wiki/topics/strategy-fusion-fence.md';

/** The clairvoyant APIs, by home. A use of any of them needs the token. */
const CLAIRVOYANT_APIS: Record<string, RegExp> = {
  // silas/src/clairvoyant.rs and the predicate in silas/src/outcome.rs
  silas: /\b(ClairvoyantSolver|clairvoyant_grade_line|OpenPosition)\b|clairvoyant-dd-make/,
  // src/game/ai/clairvoyant-minimax.ts and scripts/clairvoyant-minimax-eval.ts
  engine: /\b(clairvoyantMinimaxEvaluate|clairvoyantMinimaxFullGame)\b|clairvoyant-minimax/,
  // the oracle era: forge's solver, Stage1Oracle, E[Q], the E[Q] players, otis's clairvoyant leg
  oracle: new RegExp(
    '\\b(solve_one_seed|solve_gpu|Stage1Oracle|query_model|compute_eq_pdf|compute_eq_with_counts' +
      '|compute_eq_weighted_mean|compute_posterior_weighted_eq|generate_eq_games_gpu' +
      '|generate_eq_from_snapshots|ClairvoyantPolicy|load_clairvoyant_policy|LensPlay' +
      '|evaluate_eq_vs_random|eq_player)\\b'
  )
};

/** The modules that define a clairvoyant API: the token sits at the top. */
const DEFINING_MODULES = [
  'silas/src/clairvoyant.rs',
  'silas/src/outcome.rs',
  'src/game/ai/clairvoyant-minimax.ts',
  'src/game/ai/monte-carlo.ts',
  'scripts/clairvoyant-minimax-eval.ts',
  'forge/oracle/solve.py',
  'forge/oracle/__init__.py',
  'forge/eq/oracle.py',
  'forge/eq/generate/eq_compute.py',
  'forge/eq/generate/pipeline.py',
  'forge/eq/generate/posterior.py',
  'forge/eq/generate/model.py',
  'forge/zeb/eq_player.py',
  'arena/lens_play.py',
  'otis/tiedroll.py'
];

const CODE = /\.(py|rs|ts|tsx|js|mjs|cjs|svelte|sh|toml|silas)$/;

/** Every code file git sees: tracked, and new ones not yet added (ignored ones excepted). */
function trackedCode(): string[] {
  const listing = execSync('git ls-files --cached --others --exclude-standard', {
    cwd: projectRoot,
    encoding: 'utf-8',
    maxBuffer: 64 * 1024 * 1024
  });
  return listing.split('\n').filter(f => CODE.test(f) && existsSync(path.join(projectRoot, f)));
}

function read(file: string): string {
  return readFileSync(path.join(projectRoot, file), 'utf-8');
}

describe('Architecture: the strategy-fusion fence', () => {
  const files = trackedCode().map(file => ({ file, text: read(file) }));

  it('every file that uses a clairvoyant API carries the fence token', () => {
    const unfenced = files
      .filter(({ text }) => !text.includes(TOKEN))
      .flatMap(({ file, text }) =>
        Object.entries(CLAIRVOYANT_APIS).flatMap(([home, api]) => {
          const hit = text.match(api);
          return hit ? [`${file} (${home}: ${hit[0]})`] : [];
        })
      );
    expect(
      unfenced,
      `These files use a clairvoyant API without the token "${TOKEN}". A per-world ` +
        `perfect-information value averaged as a seat's value is strategy fusion. Use an ` +
        `information-state solve instead, or, for a sanctioned use (a bound, a baseline, a ` +
        `measurement of fusion), add the token with its reason and record it in ${RULING}.`
    ).toEqual([]);
  });

  it('every defining module carries the token at its top', () => {
    const late = DEFINING_MODULES.filter(file => !read(file).split('\n').slice(0, 20).join('\n').includes(TOKEN));
    expect(late, `the token belongs in the first lines of each defining module`).toEqual([]);
  });

  it('the guard still sees the APIs it guards', () => {
    // A rename that leaves a pattern matching nothing would silently retire the guard.
    for (const [home, api] of Object.entries(CLAIRVOYANT_APIS)) {
      expect(files.some(({ text }) => api.test(text)), `no file matches the ${home} pattern`).toBe(true);
    }
  });

  it('the old, unwarning names are gone', () => {
    const stale = files
      .filter(({ file }) => !file.endsWith('strategy-fusion-fence.test.ts'))
      .filter(({ text }) => /(?<![\w-])dd-make\b|(?<![A-Za-z])minimaxEvaluate\b|\bfn grade_line\b/.test(text))
      .map(({ file }) => file);
    expect(stale).toEqual([]);
  });

  it('Silas builds without the clairvoyant reference by default', () => {
    const cargo = read('silas/Cargo.toml');
    const defaults = cargo.match(/^default\s*=\s*\[(.*)\]/m)?.[1] ?? '';
    expect(defaults).not.toContain('clairvoyant');
    expect(cargo).toMatch(/^clairvoyant\s*=\s*\[\]/m);
    expect(read('silas/src/lib.rs')).toMatch(/#\[cfg\(feature = "clairvoyant"\)\]\npub use clairvoyant::/);
  });

  it('the ruling is where the token points', () => {
    const page = read(RULING);
    expect(page).toContain(TOKEN);
    const claude = read('CLAUDE.md');
    expect(claude).toContain(TOKEN);
    expect(claude).toContain(RULING);
  });
});
