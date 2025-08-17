import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './src/tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [['html', { open: 'never' }]],
  timeout: 10000,
  use: {
    baseURL: 'http://localhost:60101',
    trace: 'on-first-retry',
    actionTimeout: 4000,
    navigationTimeout: 6000,
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
    port: 60101,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});