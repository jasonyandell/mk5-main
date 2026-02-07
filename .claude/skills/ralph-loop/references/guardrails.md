# Common Guardrail Patterns

Add these to PROMPT.md when you observe specific failure modes.

## File Operations

```markdown
- ALWAYS read files before editing them
- NEVER assume a file exists - check first
- Search codebase before implementing (don't duplicate existing code)
```

**When to add**: Ralph edits files without reading, duplicates existing functions.

## Testing

```markdown
- NEVER skip failing tests
- NEVER commit with failing tests
- If tests fail, fix before continuing
- If tests fail 3x on same issue, output: STUCK
- Run tests after every change: `npm test`
```

**When to add**: Ralph commits broken code, ignores test failures.

## Scope Control

```markdown
- Don't refactor unrelated code
- Keep changes focused on current task
- Only modify files directly related to the task
- Complete one task fully before starting another
```

**When to add**: Ralph goes on tangents, refactors everything it sees.

## Quality Gates

```markdown
Before marking any task complete:
- All tests pass
- No TypeScript errors: `npm run typecheck`
- Lint passes: `npm run lint`
- Changes are committed
```

**When to add**: Ralph marks tasks done without verification.

## Escape Hatches

```markdown
- If blocked for 3 iterations, output: BLOCKED
- If tests fail 3x on same issue, output: STUCK
- If unable to find required files, output: MISSING_FILES
- After 15 iterations with no progress, output: STALLED
```

**When to add**: Ralph loops forever on unsolvable problems.

## Context Preservation

```markdown
- Update TODO.md after completing each task
- Commit after each completed task (not batched)
- Log learnings in progress.txt for future iterations
- Keep AGENTS.md current with build/test commands
```

**When to add**: Ralph loses track of what it's done, repeats work.

## Defensive Patterns

```markdown
- Confirm before deleting any files
- Create backups before major refactors
- Test in isolation before integrating
- Don't assume functionality is missing - search first
```

**When to add**: Ralph deletes important files, assumes things don't exist.

## API/External Services

```markdown
- Never hardcode API keys
- Use environment variables for secrets
- Mock external services in tests
- Handle rate limits gracefully
```

**When to add**: Working with external APIs.

## Database Operations

```markdown
- Never run migrations without backup
- Test migrations on copy of data first
- Use transactions for multi-step operations
- **HARD STOP** before any destructive database operation
```

**When to add**: Working with databases.

## Progressive Guardrails

Start minimal, add guardrails reactively:

1. **First attempt**: Minimal PROMPT.md
2. **First failure**: Add specific guardrail for that failure
3. **Pattern emerges**: Generalize guardrail
4. **Works reliably**: Document for future loops

This is the Ralph way: "Failures are data. Use them to tune prompts."
