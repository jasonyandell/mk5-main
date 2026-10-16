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

- **Every heavy run goes through `scripts/cpu-slot [--threads T] -- <command>`** (starts when
  the busy cores plus T fit in the machine, `nice` 10, 300 s cap; #173). Runs outside it starve
  the others. While another owner (worktree, or `--owner TAG`) that holds less waits, a new run
  leaves room for it.
- **Don't wait for a quiet machine; time with the right watch.** FAIR CLOCKS (CLAUDE.md): every
  arena and tournament player thinks on a deterministic budget, so results don't depend on load.
  Compare cost by **instructions retired** (cpu-slot prints them; `arena/cost.py`), and report CPU
  and wall time as descriptive only. `--exclusive "why"` is a rare last resort, not the default
  for timing studies.
- **A slot is busy only while its file is `flock`ed.** The text inside `/tmp/mk5-cpu-slots/slot-N`
  is the last holder's note and stays after release; never read it as "in use".
- **Merge your own work**: rebase on `origin/main`, `scripts/check` GREEN, fast-forward push,
  never force. Add your log entry as a new file in `wiki/log.d/`; never edit `wiki/log.md`. That log and `git log` are how the others hear.
- **Coordinated changes get a heads-up first**: Silas core pins or the standard registry,
  `silas/src/physics.rs` and `frame.rs`, `scenario/`, `core/src/`, `CLAUDE.md`, `LICENSE`,
  `scripts/check`, `scripts/cpu-slot`, releases. Open a draft PR (or an issue) titled
  `[gated] …` so the Mayor session sees it before the push.
