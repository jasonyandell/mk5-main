# Real-World Ralph Loop Examples

Actual prompts and use cases from the community.

## Example 1: Jest to Vitest Migration

**PROMPT.md:**
```markdown
# PROMPT.md

## Project
Migrate all tests from Jest to Vitest.

## Each Iteration
1. Read TODO.md for next file to migrate
2. Update imports: jest → vitest
3. Update matchers and mocks to Vitest syntax
4. Run: `npm test -- {filename}`
5. If passes, mark complete and commit
6. Continue to next file

## Guardrails
- One file per iteration
- Don't modify non-test files
- If syntax unclear, check Vitest docs first

## Completion
When all TODO items marked [x] and `npm test` passes, output: MIGRATED
```

**TODO.md:**
```markdown
# TODO
- [ ] src/tests/auth.test.ts
- [ ] src/tests/api.test.ts
- [ ] src/tests/utils.test.ts
- [ ] **HARD STOP** - Run full suite
- [ ] Update package.json scripts
- [ ] Remove Jest dependencies
```

---

## Example 2: Test Coverage Loop

**PROMPT.md:**
```markdown
# PROMPT.md

## Project
Add tests for all uncovered functions in src/

## Each Iteration
1. Run: `npm test -- --coverage`
2. Find lowest-coverage file
3. Pick one uncovered function
4. Write tests for that function
5. Run tests, ensure they pass
6. Commit: "test: add tests for {function}"
7. Repeat

## Guardrails
- Test behavior, not implementation
- One function per iteration
- If function is trivial (< 3 lines), skip it

## Completion
When coverage > 80% for all files, output: COVERAGE_COMPLETE
```

---

## Example 3: Codebase Refactor to Standards

From HumanLayer blog - ran for 6 hours autonomously.

**Setup:**
1. Spent 30 minutes creating `REACT_CODING_STANDARDS.md`
2. Created simple prompt

**PROMPT.md:**
```markdown
# PROMPT.md

Make sure the codebase matches the standards in REACT_CODING_STANDARDS.md.

Each iteration:
1. Read REACT_CODING_STANDARDS.md
2. Find ONE file that violates standards
3. Fix it to match standards
4. Run tests
5. Commit
6. Continue

When all files match standards and tests pass, output: DONE
```

**Result:** Generated `REACT_REFACTOR_PLAN.md` and completed entire refactor.

---

## Example 4: Programming Language (3 months)

Geoffrey Huntley's famous example.

**PROMPT.md:**
```markdown
Make me a programming language like Golang but with Gen Z slang keywords.
```

**Result:** [Cursed](https://github.com/AustralianDisability/cursed) - functional compiler, LLVM compilation, standard library, editor support.

**Key insight:** Even extremely ambitious goals work if the loop can make incremental progress.

---

## Example 5: PRD-Driven Development

From snarktank/ralph repository.

**Structure:**
```
project/
├── ralph.sh      # Loop script
├── prompt.md     # Instructions
├── prd.json      # User stories with passes status
└── progress.txt  # Learnings between iterations
```

**prompt.md:**
```markdown
1. Read prd.json and progress log
2. Verify correct branch checkout
3. Select highest-priority incomplete user story
4. Implement that single story
5. Run quality checks (typecheck, lint, test)
6. Commit: `feat: [Story ID] - [Story Title]`
7. Update PRD and progress documentation

Reply "COMPLETE" when all stories have passes: true
```

---

## Example 6: Multi-Repo Overnight

Y Combinator hackathon - shipped 6 repos overnight for $297.

**Batch script:**
```bash
#!/bin/bash
# overnight-work.sh

cd /path/to/project1
/ralph-loop "Read PROMPT.md and execute" --max-iterations 50

cd /path/to/project2
/ralph-loop "Read PROMPT.md and execute" --max-iterations 50

# ... more projects
```

---

## Pattern: What Works

| Task Type | Why It Works |
|-----------|--------------|
| Migrations | Clear start/end, mechanical |
| Test coverage | Measurable progress |
| Refactoring to standards | Document defines "done" |
| Greenfield with specs | Clear requirements |

## Pattern: What Doesn't Work

| Task Type | Why It Fails |
|-----------|--------------|
| "Make it better" | No measurable criteria |
| Design decisions | Requires human judgment |
| Debugging without repro | Can't verify fix |
| Unclear requirements | Loops forever |
