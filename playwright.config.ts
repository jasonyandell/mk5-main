import { defineConfig, devices } from '@playwright/test';

const PORT = parseInt(process.env.PORT || '60101');

export default defineConfig({
  testDir: '.',
  testMatch: ['src/tests/e2e/**/*.spec.ts'],
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : 4,
  reporter: [['html', { open: 'never' }]],
  timeout: 10000,
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    actionTimeout: 2000,
    navigationTimeout: 5000,
    // Disable animations for faster, more reliable tests
    launchOptions: {
      args: ['--force-prefers-reduced-motion']
    },
    // Emulate reduced motion preference
    reducedMotion: 'reduce',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    command: 'npm run dev',
    port: PORT,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});