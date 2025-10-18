---
name: Creating handoffs
description: Creates comprehensive handoff documents for context continuity between Claude sessions. Use when the user asks to create a handoff, prepare for the next session, document current work, or transition context to another Claude instance. Captures session accomplishments, technical decisions, work in progress, and actionable next steps.
---

# Creating Session Handoffs

## Structure

1. **Session Summary** - Accomplishments (✅/⚠️ status), git branch, uncommitted changes
2. **Task Context** - Original request, why it matters, scope decisions
3. **Technical Details** - Files modified (with line numbers), architecture decisions, dependencies
4. **Work In Progress** - Next steps (ordered), open questions, blockers, tests needed
5. **Knowledge Transfer** - What worked, what didn't, gotchas, useful commands
6. **Quick Start** - Verification commands, immediate action, success criteria
7. **Reference Links** - Files (clickable), docs consulted, git context

## Execution

Review conversation for completed work, decisions, and lessons learned.

## Output

Save to `scratch/handoff-{timestamp}.md` (format: `YYYY-MM-DD-HHMM`)

Use VSCode links: `[file.ts:42](path/to/file.ts#L42)`

## Project Context

See [CLAUDE.md](../../CLAUDE.md) and [docs/GAME_ONBOARDING.md](../../docs/GAME_ONBOARDING.md) for architecture.

## Example Tone

```markdown
## Session Summary
- ✅ Implemented bidding validation
- ⚠️  E2E test failing (timing issue)

## Work In Progress
**Next steps**:
1. Fix E2E test - replace `setTimeout()` with `waitFor()`
2. Run full test suite

**Open questions**:
- Validate bid values here or in executeAction()?

## Quick Start
**Verification**: `git status && npm run typecheck`
**Immediate action**: Fix E2E test timeout
**Success criteria**: All tests pass
```
