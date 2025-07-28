# Automated Test Fixing Script

## Overview
`fix_tests.sh` is an automated script that runs tests in a loop and uses Claude Code to fix failures until all tests pass.

## Usage

### Basic Usage
```bash
./fix_tests.sh
```

### Advanced Usage
```bash
# Set custom max iterations
MAX_ITERATIONS=20 ./fix_tests.sh

# Use different test command
TEST_COMMAND="npm run test:unit" ./fix_tests.sh
```

## How It Works

1. **Run Tests**: Executes `npm test` and captures output
2. **Check Results**: 
   - ✅ If tests pass → Print success message and exit
   - ❌ If tests fail → Continue to step 3
3. **Call Claude**: Runs `CLAUDE_AUTO_EXIT=true claude --dangerously-skip-permissions` with test output
4. **Repeat**: Goes back to step 1 until tests pass or max iterations reached

## Configuration

### Environment Variables
- `MAX_ITERATIONS` - Maximum loops before giving up (default: 10)
- `TEST_COMMAND` - Command to run tests (default: "npm test")

### Claude Auto-Exit
The script automatically sets `CLAUDE_AUTO_EXIT=true` so Claude will exit when done, allowing the loop to continue.

## Features

- 🎨 **Colorized output** for easy reading
- 📊 **Progress tracking** with iteration counters
- 🧪 **Test summary** extraction from output
- 🤖 **Intelligent prompting** with recent test failures
- 🛡️ **Error handling** and cleanup
- ⚙️ **Configurable** via environment variables

## Example Output

```bash
🔧 Starting automated test fixing loop
📊 Max iterations: 10
🧪 Test command: npm test

=== Iteration 1/10 ===
🧪 Running tests...
❌ Tests failed
Test summary:
 Test Files  13 failed | 8 passed (21)
      Tests  47 failed | 265 passed (312)

🤖 Calling Claude to fix tests...
✅ Claude completed fixes

=== Iteration 2/10 ===
🧪 Running tests...
✅ ALL TESTS PASSED!
🎉 Test fixing complete after 2 iteration(s)
```

## Safety Features

- **Max iterations** prevent infinite loops
- **Test output capture** for debugging
- **Graceful error handling** if Claude fails
- **Cleanup** of temporary files

## Troubleshooting

### Script won't run
```bash
# Make sure it's executable
chmod +x ./fix_tests.sh
```

### Claude not found
Make sure Claude Code is installed and accessible as `claude` command.

### Tests keep failing
- Check if the test failures require manual intervention
- Increase `MAX_ITERATIONS` for complex fixes
- Review the test output logs for patterns

### Claude doesn't exit
Ensure the Claude Code auto-exit hook is properly configured (see main setup instructions).