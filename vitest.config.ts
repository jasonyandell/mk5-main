import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 60_000,
    reporters: ['dot'],
    include: ['src/tests/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/tests/',
        'dist/',
        'scratch/',
        'scripts/',
        '**/*.d.ts',
        '**/*.config.js',
        '**/*.config.ts',
        '**/*.svelte',
        'src/lib/**',
        'src/stores/**',
        'src/routes/**',
        'src/app.html',
        'src/app.css',
        'src/main.ts',
      ],
      thresholds: {
        statements: 59,
        branches: 81.5,
        functions: 71,
        lines: 59,
      },
    },
  },
});
