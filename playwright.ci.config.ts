import { defineConfig, devices } from '@playwright/test';

// Import base configuration
import baseConfig from './playwright.config';

const PORT = parseInt(process.env.PORT || '60101');

export default defineConfig({
  ...baseConfig,
  
  // Override for CI optimization
  testDir: '.',
  testMatch: ['src/tests/e2e/**/*.spec.ts'],
  
  // CI-specific settings
  workers: 1, // GitHub Actions has 2 CPUs, but 1 worker is more stable
  retries: 2, // More retries in CI
  
  // Mobile-first testing for faster execution
  projects: [
    {
      name: 'mobile-chrome',
      use: { 
        ...devices['Pixel 5'],
        // Force landscape orientation for game
        viewport: { width: 812, height: 375 }
      },
    },
  ],
  
  // Use the same reporter config
  reporter: [['html', { open: 'never' }]],
  
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    actionTimeout: 2000,
    navigationTimeout: 5000,
    // Headless for CI
    headless: true,
    // Disable animations for faster, more reliable tests
    launchOptions: {
      args: ['--force-prefers-reduced-motion']
    },
    // Emulate reduced motion preference
    reducedMotion: 'reduce',
  },

  webServer: {
    command: 'npm run dev',
    port: PORT,
    reuseExistingServer: false, // Always start fresh in CI
    timeout: 30000,
  },
});