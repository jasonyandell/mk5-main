# PROMPT.md Template

```markdown
# PROMPT.md

## Project
[One-line description of what you're building/doing]

## Requirements
[Reference specs if complex, or list 3-5 key requirements here]

## Each Iteration

1. Read TODO.md for current tasks
2. Pick the highest priority incomplete task
3. Search codebase first (don't assume not implemented)
4. Read files before editing them
5. Implement the task completely
6. Run tests/validation: `npm test` (or appropriate command)
7. If tests fail, fix before continuing
8. Mark task complete in TODO.md with [x]
9. Commit changes: `git add -A && git commit -m "descriptive message"`
10. Continue to next task

## Guardrails

- ALWAYS read files before editing
- NEVER skip failing tests
- NEVER commit with failing tests
- If tests fail 3x on same issue, output: STUCK
- Don't refactor unrelated code
- Keep changes focused on current task
- Search before implementing (don't duplicate existing code)

## Completion

When ALL tasks in TODO.md are marked [x] AND tests pass, output:

DONE
```

## Variations

### Minimal (for simple tasks)
```markdown
# PROMPT.md

Implement [feature] per TODO.md.

Each iteration: pick highest priority task → implement → test → commit.

If tests fail, fix before continuing. Never commit broken code.

When all tasks complete and tests pass, output: DONE
```

### With Subagents (for large codebases)
```markdown
# PROMPT.md

## Project
[Description]

## Each Iteration

0a. Study specs/* with up to 250 parallel Sonnet subagents
0b. Study TODO.md and existing implementation

1. Pick most important incomplete item from TODO.md
2. Use 500 Sonnet subagents for file reads/searches
3. Use only 1 Sonnet subagent for build/tests (backpressure)
4. After implementing, run: `npm test`
5. If tests pass, update TODO.md, commit, push
6. Continue to next task

## Guardrails
[Same as above]

## Completion
When all tasks done and tests pass, output: DONE
```
