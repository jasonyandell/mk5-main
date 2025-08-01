/**
 * Test console helper that respects VERBOSE_TESTS environment variable
 */

// Check if verbose output is enabled via environment variable
export const VERBOSE_TESTS = process.env.VERBOSE_TESTS === 'true' || process.env.VERBOSE_TESTS === '1';

/**
 * Conditional console log that only outputs when VERBOSE_TESTS is enabled
 */
export const testLog = (...args: any[]) => {
  if (VERBOSE_TESTS) {
    console.log(...args);
  }
};

/**
 * Conditional console info that only outputs when VERBOSE_TESTS is enabled
 */
export const testInfo = (...args: any[]) => {
  if (VERBOSE_TESTS) {
    console.info(...args);
  }
};

/**
 * Conditional console warn that only outputs when VERBOSE_TESTS is enabled
 */
export const testWarn = (...args: any[]) => {
  if (VERBOSE_TESTS) {
    console.warn(...args);
  }
};

/**
 * Conditional console error - always outputs (errors should always be visible)
 */
export const testError = (...args: any[]) => {
  console.error(...args);
};

/**
 * Conditional console debug that only outputs when VERBOSE_TESTS is enabled
 */
export const testDebug = (...args: any[]) => {
  if (VERBOSE_TESTS) {
    console.debug(...args);
  }
};

/**
 * Conditional console table that only outputs when VERBOSE_TESTS is enabled
 */
export const testTable = (data: any, columns?: string[]) => {
  if (VERBOSE_TESTS) {
    console.table(data, columns);
  }
};