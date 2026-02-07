---
name: ralph-loop
description: Ralph loop creation and refinement. Transforms vague prompts into well-structured autonomous development loops with PROMPT.md, TODO.md, guardrails, and completion criteria. Use when user wants to set up overnight/autonomous coding runs.
---

# Ralph Loop Builder

Transform ideas into autonomous development loops that run until completion.

## Activation Examples

- "Set up a ralph loop to migrate from Jest to Vitest"
- "Create an overnight loop to add test coverage"
- "Make a ralph prompt for refactoring the auth system"
- "Help me write a ralph loop for this feature"

## Core Philosophy

Ralph loops work by:
1. Running the same prompt repeatedly in fresh context windows
2. Persisting state via git commits and text files (not conversation history)
3. Using clear completion criteria to know when to stop
4. Adding guardrails when failures occur

## Workflow

### Step 1: Understand the Task

Ask clarifying questions:
- What's the desired end state? (Tests passing? Coverage threshold? Migration complete?)
- What verification exists? (Tests, lints, type checks?)
- What files/directories are involved?
- Any existing patterns to follow?

### Step 2: Generate Loop Files

Create the following structure in the target directory:

```
project/
├── PROMPT.md           # Main loop instructions
├── TODO.md             # Prioritized task checklist
├── AGENTS.md           # Build/test commands (optional)
└── specs/              # Requirements (if complex)
    └── {topic}.md
```

### Step 3: Write PROMPT.md

Use the template in `templates/PROMPT.md`. Key elements:
- Clear one-line project description
- Reference to TODO.md for current tasks
- Explicit iteration steps (pick task → implement → test → commit)
- Guardrails section for known failure modes
- Completion signal (e.g., `Output: DONE`)

### Step 4: Write TODO.md

Use the template in `templates/TODO.md`. Key elements:
- Priority sections (Critical/High/Medium/Low)
- Each task is specific and measurable
- **HARD STOP** markers for review points
- Completed section at bottom

### Step 5: Explain Execution

Tell user how to run:

```bash
# Using official plugin
/ralph-loop "Read PROMPT.md and execute" \
  --max-iterations 50 \
  --completion-promise "DONE"

# Using bash loop (if no plugin)
while :; do cat PROMPT.md | claude -p ; done
```

## Refinement Workflow

When user reports failures:

1. **Identify the pattern** - What went wrong repeatedly?
2. **Add guardrail** - Specific instruction to prevent it
3. **Update PROMPT.md** - Add to Guardrails section

Example evolution:
```
v1: "Build the feature. Output DONE when complete."
    → Fails: skips tests

v2: "Build the feature. Run tests. Output DONE."
    → Fails: commits with failing tests

v3: "Build the feature. After each change: run tests.
     If tests fail, fix before continuing.
     Never commit with failing tests. Output DONE."
    → Works
```

## Reference Files

- `templates/PROMPT.md` - Standard prompt structure
- `templates/TODO.md` - Task list format
- `templates/PROMPT-tdd.md` - TDD-specific prompt
- `references/guardrails.md` - Common guardrail patterns
- `references/language-patterns.md` - Effective phrasing

## Anti-Patterns

| DON'T | DO |
|-------|-----|
| Vague completion criteria | Explicit, testable conditions |
| 1500 words | 100 words (minimal is best) |
| "Make it good" | "Tests pass, coverage > 80%" |
| Assume functionality missing | "Search codebase first" |
| Let it run forever | Set `--max-iterations` |
| One massive TODO | Break into prioritized items |

## Cost Awareness

- 50-iteration loop on medium codebase: $50-100+ in API
- Always set `--max-iterations` as cost control
- Prefer small, focused loops over massive ones
