import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['src/tests/**/*.test.ts'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/tests/',
        'dist/',
        '**/*.d.ts',
      ],
    },
  },
});