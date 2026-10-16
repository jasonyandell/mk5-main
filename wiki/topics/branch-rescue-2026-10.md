---
title: Branch rescue 2026-10 — every unmerged branch, sorted
kind: topic
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

Jason, 2026-10-16: "probably a ton of stuff left on the ground." The Sweep desk (Wren) inventoried every
branch not on `main` and sorted it: **merge**, **tombstone** (tag `attic/<branch>` pushed first, then the
branch deleted, nothing lost), **categorize** (waits on Jason), or **live** (another session is using it;
untouched). Snapshot taken at main `082dcf33`. Live card: The 42 Feed, post `branch-rescue`.

## The July stack (one stack, ten open PRs, Jason's call)

`worktree-walt-perf` ⊂ `worktree-perf-day` ⊂ `fused-iterate` ⊂ `worktree-equiv-census` ⊂ {`worktree-metal-hoyt`,
`codex/full-metal-hoyt-problem`, `worktree-roles` ⊂ `worktree-atlas` ⊂ `worktree-atlas-92`}. Each is the base of the
next PR (#78 → #81 → #83 → #84 → #89/#86/#91 → #92 → #95; #78 targets main). Together: the July Walt/Hoyt campaign
(`walt/kernel`, `hoyt/`, `metal_hoyt/`, `atlas/`, `roles/`, 40+ wiki pages), 428 commits behind main, 12–14
merge conflicts each (all in the wiki router files and logs). Issues citing it, all open: #72, #74, #77, #82, #87, #88,
#90, #92, #93, #94. The Rust `scenario/` Walt of 2026-10-09 is a different, later line; `hoyt/` and `atlas/` are not on main.
Wiki pages on main already name the stack as "not on main" ([[endgame-equivalence-census]]).
**Question for Jason:** merge the stack in order (conflicts resolved, `scripts/check` GREEN), or tag it `attic/` and keep only
the measurements pages?

## Table

| Branch | Last commit | Ahead / behind | Cited by | Verdict |
|---|---|---|---|---|
| `beads-sync` | 2026-04-19 | 214 / 1328 | none | TOMBSTONE — bot commits of the retired `bd` |
| `gascity` (local) | 2026-04-29 | 1 | none | TOMBSTONE — a `.beads` snapshot |
| `mk9` | 2025-12-27 | 23 / 1794 | PR #16 (CONFLICTING, base mk8) | TOMBSTONE — the 2025 codebase-review line |
| `claude/one-hand-per-page-…` | 2025-11-01 | 1 / 1970 | none (PR #13 is a different branch) | TOMBSTONE — `PerfectsApp.svelte` is gone from main |
| `worktree-walt` | 2026-07-17 | 10 / 429 | PR #76 merged, #71 | TOMBSTONE — landed as `4850dfeb` |
| `worktree-fix-jud` | 2026-07-18 | 3 / 428 | PR #70 merged, #69 | TOMBSTONE — landed as `b94cfc15` |
| `worktree-parlor` | 2026-07-12 | 1 / 1104 | PR #40 closed | TOMBSTONE — superseded by #79 |
| `worktree-parlor-promote` | 2026-07-18 | 1 / 427 | PR #79 merged | TOMBSTONE |
| `worktree-wiki-overhaul` (local) | 2026-07-06 | 1 | PR #34 merged | TOMBSTONE — WIP snapshot |
| `switch-to-github-issues` (local) | 2026-06-06 | 1 | none | TOMBSTONE — one 10-line `__init__` |
| `perf/batch`, `perf/cheap` | 2026-04-27 | 5 + 10 / ~900 | main links `burl-perf-phase1` (page only here) | RESCUE the three Burl-perf wiki pages, TOMBSTONE the rest |
| `research/recursive-generalization-2026-10-15` | 2026-10-15 | 6 / 14 | PR #174 | MERGE if GREEN (wiki only) |
| `worktree-table42` | 2026-07-18 | 9 / 427 | PR #68; #65 #66 #67 #69 #80 | see Other verdicts |
| The July stack (ten branches above) | 2026-07-18…22 | 3–56 / ~428 | PRs #78–#95; 10 issues | CATEGORIZE — Jason |
| `ccr-07735f91-fouydw` | 2026-10-16 | 6 / 14 | #175 | LIVE — a cloud session's branch |
| `codex/silas-world-constructor` | 2026-10-14 | 3 / 79 | PR #166 | LIVE — Codex worktree |
| `codex/fair-42-table-player` | 2026-05-26 | 9 (local) | none | LIVE — Codex worktree `cf79` |
| `lem/stage1`, `silas/cross-game-slice4`, `worktree-agent-ab92…` | 2026-10-16 | 2 / 4 / 6 | desks, #173 | LIVE |

## Tombstones

Each tag is `attic/<branch>`, pushed to origin before the branch was deleted. `git log attic/<branch>` reads it back.

| Branch (tag) | Tip | What it tried | Why it stopped |
|---|---|---|---|
| `beads-sync` | `a17f44b9` | 214 bot commits syncing the `bd` issue tracker | `bd` retired 2026-06; issues live on GitHub, archive at `.beads/issues.jsonl` |
| `gascity` | local only | a `.beads` snapshot for a Gas City experiment | same; nothing else in it |
| `mk9` | `33340f6d` | the 2025 codebase redundancy/unification review (PR #16, base `mk8`) | 1,794 commits behind; the engine was rebuilt since (the 8th overhaul). Closing #16 with it |
| `claude/one-hand-per-page-…` | `ff3def30` | `PerfectsApp.svelte` shows one hand per page (Nov 2025) | `src/PerfectsApp.svelte` no longer exists on main; 1,970 commits behind |
| `worktree-walt` | `8d4a7e9f` | the exact endgame info-set solver, +3.00 marks/game over jud | merged as PR #76 (`4850dfeb`); this tip is the PR head |
| `worktree-fix-jud` | `2759c8e5` | the jud #69 root cause and retrain probe | merged as PR #70 (`b94cfc15`); tip is the PR head |
| `worktree-parlor` | `7f768fa7` | session 1 of the salon, with the raw Sol session log | PR #40 closed; superseded by `worktree-parlor-promote` |
| `worktree-parlor-promote` | `3abde763` | the salon promoted to a wiki citizen | merged as PR #79; tip is the PR head |
| `worktree-wiki-overhaul` | local only | a WIP snapshot of the wiki overhaul (459 files) | merged as PR #34 from a later branch; snapshot says "worktree based on stale main" |

## Other verdicts

(Added as each branch is acted on.)

## Links

[[standing-desks]] · [[endgame-equivalence-census]] · [[scenario-player]]
