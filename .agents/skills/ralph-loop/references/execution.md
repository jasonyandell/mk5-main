# Running Ralph Loops

## Using Official Plugin

```bash
# Install plugin (one-time)
/install-github-plugin anthropics/claude-code plugins/ralph-wiggum

# Start loop
/ralph-loop "Read PROMPT.md and execute" \
  --max-iterations 50 \
  --completion-promise "DONE"

# Cancel active loop
/cancel-ralph
```

## Using Bash Loop

```bash
# Basic loop (runs forever until completion signal)
while :; do cat PROMPT.md | claude -p ; done

# With iteration limit
MAX=50
i=0
while [ $i -lt $MAX ]; do
  cat PROMPT.md | claude -p
  ((i++))
done

# With git push after each iteration
while :; do
  cat PROMPT.md | claude -p --dangerously-skip-permissions
  git push origin "$(git branch --show-current)"
done
```

## Two-Phase Loop (Plan/Build)

```bash
#!/bin/bash
# loop.sh - Usage: ./loop.sh [plan|N]

MODE="${1:-build}"
if [ "$MODE" = "plan" ]; then
  PROMPT_FILE="PROMPT_plan.md"
  MAX_ITERATIONS="${2:-5}"
else
  PROMPT_FILE="PROMPT_build.md"
  MAX_ITERATIONS="${1:-20}"
fi

i=0
while [ $i -lt $MAX_ITERATIONS ]; do
  cat "$PROMPT_FILE" | claude -p
  git push origin "$(git branch --show-current)"
  ((i++))
done
```

## Overnight Batch

```bash
#!/bin/bash
# overnight.sh - Run multiple projects

PROJECTS=(
  "/path/to/project1"
  "/path/to/project2"
  "/path/to/project3"
)

for proj in "${PROJECTS[@]}"; do
  cd "$proj"
  echo "=== Starting $proj ==="
  /ralph-loop "Read PROMPT.md and execute" --max-iterations 30
done
```

## Monitoring

```bash
# Watch loop progress (in separate terminal)
watch -n 5 'git log --oneline -10'

# Tail logs
tail -f logs/ralph-*.log

# Check cost (if using API directly)
# Monitor your API dashboard
```

## Safety Mechanisms

### Iteration Limits
```bash
# ALWAYS set max iterations
--max-iterations 50  # Plugin
MAX=50               # Bash variable
```

### Completion Signals
In PROMPT.md:
```markdown
When complete, output: DONE
If stuck, output: STUCK
If blocked, output: BLOCKED
```

### Cost Control
- 50 iterations on medium codebase: ~$50-100
- Set limits based on budget
- Smaller, focused loops are cheaper

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Loop runs forever | Add completion signal, set max iterations |
| Same error repeating | Add guardrail for that error pattern |
| No commits happening | Add explicit commit step to PROMPT.md |
| Context filling up | Use subagents for file reads |
| Wrong files edited | Add "only modify X" guardrail |

## Exit Codes

If using bash loop, detect completion:
```bash
while :; do
  OUTPUT=$(cat PROMPT.md | claude -p)
  echo "$OUTPUT"
  if echo "$OUTPUT" | grep -q "DONE\|COMPLETE"; then
    echo "Loop completed successfully"
    break
  fi
  if echo "$OUTPUT" | grep -q "STUCK\|BLOCKED"; then
    echo "Loop needs intervention"
    break
  fi
done
```
