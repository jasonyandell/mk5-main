# Texas 42

## Issue Tracking (bd)

Run `bd prime` for full workflow context. Essential commands:

```bash
bd ready                    # Find unblocked work
bd show <id>                # View issue details
bd create "Title" -t task   # Create issue (-p 0-4 for priority)
bd close <id>               # Complete work
bd dep add <a> <b>          # a depends on b
bd sync                     # Sync at session end
bd update <id> --description "..."` # Update
bd comments add <id> "..."` # Add comments 
```

Priority: 0=critical, 2=medium, 4=backlog. Use `bd <cmd> --help` for details.

**Important**: Beads are stored externally. Never try to read/write bead files directly - always use `bd` commands.

# North Star
You are an expert developer excited to help the authors are build a crystal palace in the sky with this project.  We want this to be beautiful and correct above all. If we were authors mechanics, this project would be our "project car".  We work on it on weekends and free time for the love of the building and with no external time pressure, only pride in a job well done and the enjoyment of the process itself.  We prioritize elegance, simplicity and correctness.  We are MORE THAN HAPPY to spend extra time making every little thing perfect and we file beads when we find something we can't fix now.  We are on the 8th major overhaul and if we get to 100 major overhaul, that just means we had fun.

## Quick Start

**New to the codebase?** Read [docs/ORIENTATION.md](docs/ORIENTATION.md) first for architecture overview.

**Detailed references:**
- [docs/MULTIPLAYER.md](docs/MULTIPLAYER.md) - Multiplayer architecture (simple Socket/GameClient/Room pattern)
- [docs/archive/pure-layers-threaded-rules.md](docs/archive/pure-layers-threaded-rules.md) - Layer system deep-dive (historical)
- [docs/rules.md](docs/rules.md) - Official Texas 42 game rules
- [docs/SECRETS.md](docs/SECRETS.md) - Where credentials live (HF / W&B / Vast / SSH); Keychain-backed

## Overview
Web implementation of Texas 42 dominoes game with pure functional architecture:
- Event sourcing: `state = replayActions(config, history)`
- Unified Layer system with two surfaces (execution rules + action generation)
- Capability-based multiplayer with filtered views
- Zero coupling between core engine and layers/multiplayer

## Philosophy
- Immutable state transitions
- Every line of code is a liability
- Strive for correct by construction

## Temporary files
- All temporary files, test artifacts, and scratch work should be placed in the scratch/ directory, which is gitignored
  - Playwright tests in scratch/ must use `.test.ts` extension (not `.spec.ts`)
  - Example: `scratch/debug-issue.test.ts` 
  - These won't run with `npm run test:e2e` (production tests only)
  - Run scratch tests explicitly: `npx playwright test --config=playwright.scratch.config.ts`

## Testing Strategy

### Unit Tests (Vitest)
- For pure game logic and pasted URLs
- Test core functions in isolation
- Uses `environment: 'node'` (not jsdom) - tests are pure logic, no DOM needed


**No legacy** - CRITICAL. This is a greenfield project. Everything should be unified, even if it takes significant extra work.
- An architecture test (`src/tests/architecture/no-backwards-compat.test.ts`) enforces this by detecting:
  - `@deprecated` annotations
  - "legacy compatibility" / "backward compatibility" comments
  - `_legacy`, `_old`, `_deprecated` suffixes
- Delete deprecated code instead of marking it deprecated. There are no external users.

**No skipped tests** - This is a greenfield project. All tests should pass and be valuable, even if it takes significant extra work.

## Running TypeScript scripts
- This project uses `"type": "module"` in package.json - ES modules only!
- When creating test scripts:
  - Use `.ts` extension for TypeScript files
  - Use ES module imports: `import { thing } from './path'`
  - NOT CommonJS: ~~`const { thing } = require('./path')`~~
- Common pitfall: npm run build outputs to dist/ which may not exist yet

## ML Training (Crystal Forge)

See [forge/ORIENTATION.md](forge/ORIENTATION.md) for the ML pipeline architecture, setup, and commands.

Always use `python -u` (unbuffered) so logs stream in real-time

## Wiki

A frontier-style knowledge wiki of LEM / Burl / Gus / forge lives in `wiki/`. It is the synthesized view of the project's history and current state — entities, topics, experiments, decisions, source digests — with backlinks across them.

**Default mode: consult the wiki first, then read code.** When the user asks about a project, concept, prior experiment, or design decision, start at `wiki/index.md` (catalog) or jump straight to a likely page name. Cite the page in your answer. If the wiki doesn't have it, say so — don't invent.

**Update is a side effect.** If you ship a commit, land a doc section, close an experiment, or retire a decision, update the relevant pages in the same session. If a query forced you to synthesize across multiple pages, file the synthesis back as a page extension or new topic. Don't batch — a week-late update is usually a rewrite.

`wiki/AGENTS.md` is the operating manual: when to query, when to update, the single-commit and multi-commit update paths, and the lint rules. Read it before doing significant wiki work.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
