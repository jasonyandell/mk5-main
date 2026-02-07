# Effective Language Patterns

Specific phrasings that work well in Ralph prompts (from Geoffrey Huntley's methodology).

## Cognitive Framing

| Use | Instead of | Why |
|-----|------------|-----|
| "Study the codebase" | "Read the files" | Implies deeper analysis |
| "Search before implementing" | "Check if it exists" | Action-oriented |
| "Don't assume not implemented" | "Make sure it doesn't exist" | Double-negative forces attention |
| "Capture the why" | "Document it" | Emphasizes reasoning over description |

## Subagent Delegation

```markdown
# Distribute context load
"Use up to 500 parallel Sonnet subagents for file reads/searches"

# Control backpressure
"Use only 1 Sonnet subagent for build/tests"

# Complex reasoning
"Use an Opus 4.5 subagent with 'ultrathink' for architectural decisions"
```

**Why**: Main context acts as scheduler; subagents extend memory without filling smart zone.

## Completion Signals

Strong completion language:
```markdown
When ALL of the following are true, output: DONE
- All tasks in TODO.md are marked [x]
- All tests pass
- No TypeScript errors
- No lint errors
```

Escape hatches:
```markdown
If unable to complete after 5 attempts, output: STUCK
If missing required files, output: MISSING_FILES
If blocked by external dependency, output: BLOCKED
```

## Iteration Structure

Clear step-by-step (numbered, imperative):
```markdown
Each iteration:
1. Read TODO.md
2. Pick highest priority incomplete task
3. Search codebase first
4. Implement the task
5. Run tests
6. If tests fail, fix before continuing
7. Mark complete in TODO.md
8. Commit changes
9. Continue to next task
```

## Constraint Phrasing

Strong constraints:
```markdown
NEVER commit with failing tests
ALWAYS read files before editing
NEVER assume functionality is missing
```

Conditional constraints:
```markdown
If tests fail 3x on same issue, output: STUCK
After implementing, run tests
Before marking complete, verify quality gates
```

## Task Sizing

Good task phrasing:
```markdown
- Add JWT verification middleware to /api/* routes
- Fix login.test.ts - mock the database connection
- Add memoization to calculateScore() - cache by gameId
```

Bad task phrasing:
```markdown
- Add authentication (too vague)
- Fix the tests (which ones?)
- Make it faster (not measurable)
```

## The "One Sentence Without And" Test

Each topic/task should pass this test:

- "The auth system handles login and logout and sessions" - 3 separate topics
- "The auth system verifies JWT tokens" - single focused topic

Apply to specs, tasks, and iteration goals.

## Declarative > Imperative

```markdown
# Declarative (better for specs)
"Users can create accounts with email and password.
Duplicate emails are rejected with a 409 error."

# Imperative (better for iteration steps)
"1. Pick the highest priority task
 2. Implement it completely
 3. Run tests"
```

Use declarative for describing desired state, imperative for process steps.
