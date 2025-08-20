#!/bin/bash

# fix_tests.sh - Automated test fixing loop with Claude Code
# This script runs tests in a loop and uses Claude to fix failures

set -e  # Exit on any error except test failures
set -o pipefail  # Ensure pipeline failures are detected

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_ITERATIONS=${MAX_ITERATIONS:-10}
TEST_COMMAND="npm run test:all"
CLAUDE_COMMAND="claude --dangerously-skip-permissions"

echo -e "${BLUE}🔧 Starting automated test fixing loop${NC}"
echo -e "${BLUE}📊 Max iterations: $MAX_ITERATIONS${NC}"
echo -e "${BLUE}🧪 Test command: $TEST_COMMAND${NC}"
echo ""

iteration=1

while [ $iteration -le $MAX_ITERATIONS ]; do
    echo -e "${YELLOW}=== Iteration $iteration/$MAX_ITERATIONS ===${NC}"
    echo -e "${BLUE}🧪 Running tests...${NC}"
    
    # Run tests and capture output
    if $TEST_COMMAND 2>&1 | tee test_output.log && false; then
        # Tests passed!
        echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
        echo -e "${GREEN}🎉 Test fixing complete after $iteration iteration(s)${NC}"
        
        # Show final test summary
        echo ""
        echo -e "${GREEN}Final test results:${NC}"
        tail -10 test_output.log
        
        # Clean up
        rm -f test_output.log
        exit 0
    else
        # Tests failed
        echo -e "${RED}❌ Tests failed${NC}"
        
        # Show test summary or error type
        echo -e "${YELLOW}Test/Build summary:${NC}"
        if grep -E "(Test Files|Tests|failed|passed|PASS|FAIL)" test_output.log | tail -5; then
            :  # Found test summary
        elif grep -E "(error TS|TypeError|SyntaxError)" test_output.log | tail -5; then
            echo "TypeScript errors detected"
        elif grep -E "(ESLint|Prettier|warning|error)" test_output.log | tail -5; then
            echo "Linting errors detected"
        else
            echo "Build or test failure (see details below)"
        fi
        
        echo ""
        echo -e "${BLUE}🤖 Calling Claude to fix tests...${NC}"
        
        # Set auto-exit and call Claude
        export CLAUDE_AUTO_EXIT=true
        
        # Extract only failure and warning details for Claude
        echo -e "${BLUE}📝 Extracting failure details for Claude...${NC}"
        
        # Create filtered output with only failures and warnings (limited to prevent context flooding)
        {
            echo "=== FAILURE AND WARNING DETAILS ==="
            echo ""
            
            # Extract TypeScript errors first (highest priority)
            if grep -q "error TS" test_output.log; then
                echo "=== TYPESCRIPT ERRORS ==="
                # Get full file paths with errors
                grep -E "\.ts.*error TS" test_output.log | head -150 || true
                echo ""
            fi
            
            # Extract lint errors
            if grep -q -E "(ESLint|Prettier|⚠|✖)" test_output.log; then
                echo "=== LINT ERRORS ==="
                grep -A 2 -B 1 -E "(ESLint|Prettier|⚠|✖|error|warning)" test_output.log | head -100 || true
                echo ""
            fi
            
            # Extract Playwright E2E test failures - SIMPLE approach
            # Just grab any lines that look like Playwright failures without complex parsing
            if grep -q "\[chromium\].*› " test_output.log; then
                echo "=== PLAYWRIGHT E2E TEST FAILURES ==="
                
                # Get all test failure lines and their error messages (up to 50 lines of context each)
                grep -A 50 "^.*) \[chromium\].*›" test_output.log | head -2000 || true
                
                echo ""
            fi
            
            # Extract Vitest unit test failures (avoid listing all passing tests)
            if grep -q "Test Files.*failed" test_output.log; then
                echo "=== VITEST UNIT TEST FAILURES ==="
                # Skip the long list of passing tests, get failures only
                sed -n '/FAIL.*\.test\.ts/,/Test Files/p' test_output.log | head -150 || true
                echo ""
            fi
            
            # Add truncation notice if there are many failures
            total_failures=$(grep -c -E "(error TS|FAIL|ERROR|WARN|✕|❌|Failed|Error:|Warning:|ESLint|✖)" test_output.log 2>/dev/null || echo "0")
            if [ "$total_failures" -gt 30 ]; then
                echo ""
                echo "... (Output truncated - $total_failures total issues found)"
                echo "Focus on fixing the first few issues above, which should resolve many others."
            fi
            
            echo ""
            echo "=== SUMMARY ==="
            # Extract test summaries from each tool
            echo "TypeScript: $(grep -c "error TS" test_output.log 2>/dev/null || echo "0") errors"
            echo "ESLint: $(grep -c "✖" test_output.log 2>/dev/null || echo "0") issues"
            
            # Get Vitest summary if present
            grep "Test Files" test_output.log | tail -1 2>/dev/null || true
            
            # Get Playwright summary - look for the pattern "  X failed" where X is a number
            grep -E "^  [0-9]+ failed" test_output.log | tail -1 2>/dev/null || true
            
            # Count individual Playwright test failures (simple pattern)
            playwright_failures=$(grep -c ") \[chromium\].*›" test_output.log 2>/dev/null || echo "0")
            if [ "$playwright_failures" -gt 0 ]; then
                echo "Playwright E2E: $playwright_failures test failures detected"
            fi
            
            # Get overall test command result
            tail -5 test_output.log | grep -E "(failed|passed|error|Error)" || echo "See details above for errors"
            
        } > claude_failures.txt
        
        # Create a prompt for Claude with only failure details
        cat > claude_prompt.txt << EOF
We have changed the frontend and now the e2e tests are failing and there may be other code quality issues. 
Please analyze the output and fix all issues including:
- Test failures
- Linting errors
- TypeScript errors
- Build errors
- Slow tests (timeouts > 2000)
- Non-deterministic e2e tests. Make a best effort to make them deterministic.
- Any other code quality issues

REQUIREMENT: The core game in src/game must be correct by construction.  Do not perform quick-fix solutions to the core game, do analyze how we can improve the code to be more correct.
REQUIREMENT: Do not change the UI (svelte) just because a test is failing.  The test is more likely (but not guaranteed) to be incorrect.
REQUIREMENT: Put locators into playwrightHelper.ts and not in the tests.  The tests should be pure and readable.
HINT: If you find a race condition, it is likely not a race condition at all, but that the helper is doing the wrong thing or the test is expecting the wrong behavior.  Look for comments that indicate places we know the helper is likely incorrect.  Find the correct behavior that does not involve switching tabs and update the helper.
CRITICAL: this is all local and there's no network delay.  when addressing timeouts, try to make things deterministic, including within the browser, if possible    

Warnings are errors. This is a greenfield project.
Make all tests (unit and e2e) thorough and maintainable.
Ensure code follows project conventions and passes all quality checks.

Once the tests are passing, you have ONE PRIMARY GOAL: IMPROVE E2E TEST PERFORMANCE.  Deterministic. LOW TIMEOUTS.

Failure and warning details:
\`\`\`
$(cat claude_failures.txt)
\`\`\`
EOF
        
        # Call Claude with the prompt
        if $CLAUDE_COMMAND "$(cat claude_prompt.txt)"; then
            echo -e "${GREEN}✅ Claude completed fixes${NC}"
        else
            echo -e "${RED}❌ Claude encountered an error${NC}"
            echo -e "${YELLOW}Continuing to next iteration anyway...${NC}"
        fi
        
        # Clean up prompt files
        rm -f claude_prompt.txt claude_failures.txt
        
        echo ""
        echo -e "${YELLOW}Waiting 2 seconds before next iteration...${NC}"
        sleep 2
    fi
    
    iteration=$((iteration + 1))
done

# Max iterations reached
echo -e "${RED}❌ Maximum iterations ($MAX_ITERATIONS) reached${NC}"
echo -e "${YELLOW}Final test output:${NC}"
tail -20 test_output.log

# Clean up
rm -f test_output.log

echo ""
echo -e "${YELLOW}💡 You can increase MAX_ITERATIONS or run the script again${NC}"
echo -e "${YELLOW}   Example: MAX_ITERATIONS=20 ./fix_tests.sh${NC}"

exit 1