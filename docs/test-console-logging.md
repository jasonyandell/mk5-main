# Test Console Logging

## Overview

Tests are configured to be silent by default to reduce noise during test runs. Console output can be enabled when needed for debugging.

## How It Works

1. **Default Behavior**: Tests run silently, no console output is displayed
2. **Verbose Mode**: Set `VERBOSE_TESTS=true` to enable console output
3. **Test Helper**: Use `testLog()` instead of `console.log()` in tests

## Usage

### Running Tests Silently (Default)
```bash
npm test                  # Run tests once, silently
npm run test:watch       # Watch mode, silent
```

### Running Tests with Console Output
```bash
npm run test:verbose          # Run tests once with console output
npm run test:watch:verbose    # Watch mode with console output

# Or manually:
VERBOSE_TESTS=true npm test
```

### In Test Files

```typescript
import { testLog, testInfo, testWarn, testError } from '../helpers/testConsole';

describe('My Test', () => {
  it('should do something', () => {
    testLog('This will only print when VERBOSE_TESTS=true');
    testInfo('Info message');
    testWarn('Warning message');
    testError('Error message - always visible');
    
    // These are conditionally displayed:
    testLog('Debug info:', someObject);
    testTable(data); // Displays a table
  });
});
```

## Migration from console.log

Replace direct console calls with test helpers:

```typescript
// Before:
console.log('Debug info');

// After:
testLog('Debug info');
```

## Configuration

The test console helper (`src/tests/helpers/testConsole.ts`) checks the `VERBOSE_TESTS` environment variable to determine whether to output console messages.

When `VERBOSE_TESTS` is not set or is false, all `testLog()`, `testInfo()`, `testWarn()`, and `testDebug()` calls are suppressed. Only `testError()` calls will always be displayed.