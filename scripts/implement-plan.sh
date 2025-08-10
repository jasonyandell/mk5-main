#!/bin/bash

# implement-plan.sh - Automated plan implementation loop with Claude Code
# This script checks for uncompleted tasks and uses Claude to implement them

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_ITERATIONS=${MAX_ITERATIONS:-100}
PLAN_FILE="docs/rules-gherkin-plan.md"
CLAUDE_COMMAND="claude --dangerously-skip-permissions"

echo -e "${BLUE}🔧 Starting automated plan implementation loop${NC}"
echo -e "${BLUE}📊 Max iterations: $MAX_ITERATIONS${NC}"
echo -e "${BLUE}📋 Plan file: $PLAN_FILE${NC}"
echo ""

iteration=1

while [ $iteration -le $MAX_ITERATIONS ]; do
    echo -e "${YELLOW}=== Iteration $iteration/$MAX_ITERATIONS ===${NC}"
    echo -e "${BLUE}📋 Checking for uncompleted tasks...${NC}"
    
    # Check if there are any uncompleted tasks ([ ])
    if ! grep -q '\[ \]' "$PLAN_FILE"; then
        # No uncompleted tasks found!
        echo -e "${GREEN}✅ ALL TASKS COMPLETED!${NC}"
        echo -e "${GREEN}🎉 Plan implementation complete after $iteration iteration(s)${NC}"
        
        # Show summary of completed tasks
        echo ""
        echo -e "${GREEN}Completed tasks summary:${NC}"
        grep -c '\[x\]' "$PLAN_FILE" | xargs echo -e "${GREEN}Total completed tasks:${NC}"
        
        exit 0
    else
        # Uncompleted tasks found
        task_count=$(grep -c '\[ \]' "$PLAN_FILE")
        echo -e "${YELLOW}📝 Found $task_count uncompleted task(s)${NC}"
        
        # Show next uncompleted task
        echo -e "${YELLOW}Next task:${NC}"
        grep -m 1 -B 2 -A 2 '\[ \]' "$PLAN_FILE" | head -10
        
        echo ""
        echo -e "${BLUE}🤖 Calling Claude to implement next task...${NC}"
        
        # Set auto-exit and call Claude
        export CLAUDE_AUTO_EXIT=true
        
        # Create a prompt for Claude
        cat > claude_prompt.txt << EOF
Please implement the next uncompleted task from @docs/rules-gherkin-plan.md.

This is an exercise in creating tests to a rules spec.
Referencing @docs/rules.md and @docs/rules-gherkin-plan.md we need to make new tests for a theoretical implementation for strictly tournament rules.
The tests should be written assuming the implementation does the right thing.
The complete and final code definitions and state are found in src/game/.
CRITICAL: never reference or implement anything in src/game that does not already have a definition in src/game.  
Do not run these tests you are about to create. Do not implement any game logic.
Find the first unmarked scenario create the tests and test-only implementation as described for ONLY that block.
Check the block off when this task is complete and then STOP.
EOF
        
        # Call Claude with the prompt
        if $CLAUDE_COMMAND "$(cat claude_prompt.txt)"; then
            echo -e "${GREEN}✅ Claude completed implementation${NC}"
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
echo -e "${YELLOW}Remaining uncompleted tasks:${NC}"
grep '\[ \]' "$PLAN_FILE" | head -10

echo ""
echo -e "${YELLOW}💡 You can increase MAX_ITERATIONS or run the script again${NC}"
echo -e "${YELLOW}   Example: MAX_ITERATIONS=20 ./implement-plan.sh${NC}"

exit 1