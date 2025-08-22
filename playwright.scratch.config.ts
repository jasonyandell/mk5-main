import baseConfig from './playwright.config';
import { defineConfig } from '@playwright/test';

// Config for running scratch tests
export default defineConfig({
  ...baseConfig,
  testMatch: ['scratch/**/*.test.ts'],
});