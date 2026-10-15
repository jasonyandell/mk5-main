# AI Agent Guidelines

**Read [CLAUDE.md](CLAUDE.md)** — it is the single entrypoint for any agent working in this
repo: north star, buddy-mode collaboration stance, wiki-first knowledge base, issue
tracking (GitHub issues; beads is retired), hard rules (no legacy, no skipped tests,
scratch/ hygiene, ES modules, GPU-only forge, the strategy-fusion fence, the book is the authority, the license line), quality gates, and the session-completion
protocol.

The project's knowledge base is the wiki at `wiki/` — consult it before reading code, and
update it as a side effect of any work that changes what's true. `wiki/AGENTS.md` is the
wiki's own operating manual.

## Sharing the machine and `main` with the other sessions

Several sessions (Claude desks, Codex worktrees) work this repo at once; the rules are in
[wiki/playbooks/standing-desks.md](wiki/playbooks/standing-desks.md). In short:

- **Every heavy run goes through `scripts/cpu-slot -- <command>`** (3 shared slots, `nice` 10,
  300 s cap). Runs outside it starve the others and skew anything timed on a clock. One
  owner (worktree, or `--owner TAG`) holds at most 2 slots while another waits; a wall-clock
  run that needs a quiet machine says `--exclusive "why"` (#169).
- **Merge your own work**: rebase on `origin/main`, `scripts/check` GREEN, fast-forward push,
  never force. Add a line to `wiki/log.md`; that log and `git log` are how the others hear.
- **Coordinated changes get a heads-up first**: Silas core pins or the standard registry,
  `silas/src/physics.rs` and `frame.rs`, `scenario/`, `core/src/`, `CLAUDE.md`, `LICENSE`,
  `scripts/check`, `scripts/cpu-slot`, releases. Open a draft PR (or an issue) titled
  `[gated] …` so the Mayor session sees it before the push.
