/**
 * Vitest config for running scratch/ tests.
 *
 * Usage:
 *   npx vitest --config vitest.scratch.config.ts run scratch/test-*.test.ts
 *   npx vitest --config vitest.scratch.config.ts run  # all scratch tests
 */
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 60_000,
    reporters: ['verbose'],
    include: ['scratch/**/*.test.ts'],
  },
});
