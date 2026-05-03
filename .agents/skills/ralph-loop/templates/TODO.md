# TODO.md Template

```markdown
# TODO

## Critical (MVP - Must Complete)
- [ ] Task 1: [Specific, measurable outcome]
- [ ] Task 2: [Another specific task with clear done criteria]
- [ ] **HARD STOP** - Verify core flow works end-to-end before continuing

## High Priority
- [ ] Task 3: [Description with success criteria]
- [ ] Task 4: [Reference specific files when known]

## Medium Priority
- [ ] Task 5: [Description]
- [ ] **HARD STOP** - Review progress before continuing to low priority

## Low Priority / Nice-to-Have
- [ ] Task 6: [Optional enhancement]
- [ ] Task 7: [Polish item]

---
## Completed
(Tasks move here when done)
- [x] Example completed task
```

## Task Writing Rules

1. **Be specific**: "Add JWT auth middleware to /api routes" not "Add authentication"
2. **Be measurable**: Include success criteria when possible
3. **One thing per task**: No "and" in task descriptions
4. **Include context**: Reference specific files when known
5. **Order by dependency**: Tasks that enable others come first

## Bad vs Good Examples

| Bad | Good |
|-----|------|
| Add authentication | Add JWT verification middleware to /api/* routes |
| Fix the tests | Fix failing login.test.ts - mock the database connection |
| Improve performance | Add memoization to calculateScore() - cache by gameId |
| Update the docs | Add API docs for /users endpoint in README.md |
| Handle errors | Add try/catch to fetchUser() with 500 response on failure |

## HARD STOP Usage

Insert `**HARD STOP**` markers to force review:
- After MVP tasks (ensure core works before extras)
- Before low-priority work (confirm high-priority done)
- At risky transitions (e.g., before database changes)

Ralph will pause at these markers, giving you a chance to review and adjust.
