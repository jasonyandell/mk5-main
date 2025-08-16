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
MAX_ITERATIONS=${MAX_ITERATIONS:-100}
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
    if $TEST_COMMAND 2>&1 | tee test_output.log; then
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
                grep -A 2 -B 1 "error TS" test_output.log | head -150 || true
                echo ""
            fi
            
            # Extract lint errors
            if grep -q -E "(ESLint|Prettier|⚠|✖)" test_output.log; then
                echo "=== LINT ERRORS ==="
                grep -A 2 -B 1 -E "(ESLint|Prettier|⚠|✖|error|warning)" test_output.log | head -100 || true
                echo ""
            fi
            
            # Extract test failures, errors, warnings, and related context
            if grep -q -E "(FAIL|✕|❌|Failed|Error:|Warning:|expect\(|received:|AssertionError)" test_output.log; then
                echo "=== TEST FAILURES ==="
                grep -A 3 -B 1 -E "(FAIL|ERROR|WARN|✕|❌|Failed|Error:|Warning:|expect\(|received:|AssertionError)" test_output.log | head -100 || true
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
            # Try to extract any summary information
            tail -20 test_output.log | grep -E "(Test Files|Tests|failed|passed|Suites|✓|✕|error TS|ESLint|files? checked|problems?)" || echo "See details above for errors"
            
        } > claude_failures.txt
        
        # Create a prompt for Claude with only failure details
        cat > claude_prompt.txt << EOF
The tests are failing and there may be other code quality issues. 
Please analyze the output and fix all issues including:
- Test failures
- Linting errors
- TypeScript errors
- Build errors
- Any other code quality issues

REQUIREMENT: The core game in src/game must be correct by construction.  Do not perform quick-fix solutions to the core game, do analyze how we can improve the code to be more correct.

Warnings are errors. This is a greenfield project.
Make all tests (unit and e2e) thorough and maintainable.
Ensure code follows project conventions and passes all quality checks.

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