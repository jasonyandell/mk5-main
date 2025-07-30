#!/bin/bash

# fix_tests.sh - Automated test fixing loop with Claude Code
# This script runs tests in a loop and uses Claude to fix failures

set -e  # Exit on any error except test failures

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
    set -o pipefail
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
        
        # Show test summary
        echo -e "${YELLOW}Test summary:${NC}"
        if grep -E "(Test Files|Tests|failed|passed)" test_output.log | tail -5; then
            :  # Command succeeded
        else
            echo "Could not extract test summary"
        fi
        
        echo ""
        echo -e "${BLUE}🤖 Calling Claude to fix tests...${NC}"
        
        # Set auto-exit and call Claude
        export CLAUDE_AUTO_EXIT=true
        
        # Create a prompt for Claude with test output
        cat > claude_prompt.txt << EOF
The tests are failing. Please analyze the test output and fix the issues.  Warnings are errors.

Recent test output:
\`\`\`
$(tail -100 test_output.log)
\`\`\`

This is an exercise in creating tests to a rules spec
Referencing @rules.md and @docs/rules-gherkin-plan.md we need to make new tests for a theoretical implementation for strictly tournament rules
The tests should be written assuming the implementation does the right thing
The complete and final code definitions and state are found in src/game/
CRITICAL: never reference or implement anything in src/game that does not already have a definition in src/game.  
Do not run these tests you are about to create.  Do not implement any game logic.  
Find the first unmarked scenario create the tests and test-only implementation as described for ONLY that block
Check the block off when this task is complete.

After making changes, I'll run the tests again automatically.
EOF
        
        # Call Claude with the prompt
        if $CLAUDE_COMMAND "$(cat claude_prompt.txt)"; then
            echo -e "${GREEN}✅ Claude completed fixes${NC}"
        else
            echo -e "${RED}❌ Claude encountered an error${NC}"
            echo -e "${YELLOW}Continuing to next iteration anyway...${NC}"
        fi
        
        # Clean up prompt file
        rm -f claude_prompt.txt
        
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