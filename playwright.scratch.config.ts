// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

import baseConfig from './playwright.config';
import { defineConfig } from '@playwright/test';

// Config for running scratch tests
export default defineConfig({
  ...baseConfig,
  testMatch: ['scratch/**/*.test.ts'],
});