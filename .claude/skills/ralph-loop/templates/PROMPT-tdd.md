# TDD-Focused PROMPT.md

```markdown
# PROMPT.md

## Project
[One-line description]

## TDD Workflow

Each iteration, follow strict TDD:

1. Read TODO.md for next requirement
2. Write a **failing test** that defines the requirement
3. Run tests: `npm test` - confirm new test FAILS (red)
4. Implement **minimum code** to make test pass
5. Run tests: `npm test` - confirm ALL tests PASS (green)
6. Refactor if needed (tests must stay green)
7. Mark task complete in TODO.md with [x]
8. Commit: `git add -A && git commit -m "feat: [description]"`
9. Continue to next requirement

## Guardrails

- NEVER write implementation before the failing test
- NEVER commit with failing tests
- If stuck on same test 3x, output: STUCK
- Each commit should be: red → green → refactor
- Keep tests focused (one assertion per behavior)
- Don't over-engineer - minimum code to pass

## Quality Gates

Before marking any task complete:
- All tests pass
- No TypeScript errors: `npm run typecheck`
- Lint passes: `npm run lint`

## Completion

When all TODO items are [x] AND all quality gates pass, output:

DONE
```

## When to Use TDD Prompt

- Greenfield features where you define the behavior
- Bug fixes (write failing test that reproduces bug first)
- Refactoring with safety net
- When test coverage is a goal

## TDD TODO.md Pattern

```markdown
# TODO

## Requirements to Test
- [ ] User can create account with email/password
- [ ] User cannot create account with duplicate email
- [ ] User can login with valid credentials
- [ ] User gets 401 with invalid credentials
- [ ] **HARD STOP** - Core auth flow complete

## Edge Cases
- [ ] Handle empty email
- [ ] Handle empty password
- [ ] Handle malformed email format
```
