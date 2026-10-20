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

## The July stack (one stack, nine open PRs; shelved 2026-10-20 on Jason's word)

`worktree-walt-perf` ⊂ `worktree-perf-day` ⊂ `fused-iterate` ⊂ `worktree-equiv-census` ⊂ {`worktree-metal-hoyt`,
`codex/full-metal-hoyt-problem`, `worktree-roles` ⊂ `worktree-atlas` ⊂ `worktree-atlas-92`}. Each is the base of the
next PR (#78 → #81 → #83 → #84 → #89/#86/#91 → #92 → #95; #78 targets main). Together: the July Walt/Hoyt campaign
(`walt/kernel`, `hoyt/`, `metal_hoyt/`, `atlas/`, `roles/`, 40+ wiki pages), 428 commits behind main, 12–14
merge conflicts each (all in the wiki router files and logs). Issues citing it, all open: #72, #74, #77, #82, #87, #88,
#90, #92, #93, #94. The Rust `scenario/` Walt of 2026-10-09 is a different, later line; `hoyt/` and `atlas/` are not on main.
Wiki pages on main already name the stack as "not on main" ([[endgame-equivalence-census]]).
**Jason's answer (2026-10-20, feed ask `july-stack`, answered 2026-10-17T16:56:51-05:00):** "Keep the wiki pages, shelve the code in attic/ tags." Done: see § The July stack, shelved.

## The July stack, shelved

**9 branches were tagged `attic/<name>` and deleted, and their 9 stacked PRs closed** (`git ls-remote --tags origin 'attic/*'` lists them; the stack was nine branches, not the ten I first wrote). The code (`hoyt/`, `metal_hoyt/`, `atlas/`, `roles/`, `walt/kernel`) is not on main. `git log attic/worktree-atlas-92` reads the whole main chain (it contains `worktree-roles`, `worktree-atlas`, `worktree-equiv-census`, `fused-iterate`, `worktree-perf-day` and `worktree-walt-perf`); `attic/worktree-metal-hoyt` and `attic/codex/full-metal-hoyt-problem` are the two side branches.

Kept on main, eleven pages that exist only on the stack, each with a shelved-code note at the top: [[atlas]], [[hoyt]], [[metal-hoyt]], [[cfr-primer]], [[forty-two-native-object]], [[full-metal-hoyt]], [[fundamental-factorization]], [[hoyt-perf-primer]], [[perf-log]], [[perf]], [[role-threat-tensor]]. The stack's edits to pages main already has (`walt`, `walt-spec`, `count-fate-ledger`, `play-phase-algebra`, `suit-algebra-spec`, `perf-on-the-table`, `questions/open`) stay in the tags: main's newer versions win. [[endgame-equivalence-census]] was already on main. Issues #74, #82, #87, #88, #90, #92, #93, #94 stay open; their work resumes from the tags.

| Branch (tag) | Tip | PR | What it was |
|---|---|---|---|
| `worktree-walt-perf` | `2370f1a0` | #78 | the wavefront engine, world-cap, the net-free kernel and CFR+ instrument |
| `worktree-perf-day` | `2473d115` | #81 | columnar profile, cap-ledger cascade, the 200-root reference line |
| `fused-iterate` | `38106b26` | #83 | the fused iterate kernel (#82), anchor 88.6 s to 15.9 s |
| `worktree-equiv-census` | `c3e1374c` | #84 | the endgame equivalence census |
| `worktree-metal-hoyt` | `3bd3f947` | #89 | the GPU searcher under the fp64 referee |
| `codex/full-metal-hoyt-problem` | `1056ffd2` | #86 | the bounded full-Metal H5/H6 lane and playable table |
| `worktree-roles` | `2cdcaa0f` | #91 | the role threat tensor |
| `worktree-atlas` | `1d8d8c21` | #92 | atlas, the game as an addressed space |
| `worktree-atlas-92` | `f4764a2f` | #95 | the fundamental factorization |

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
| `perf/batch`, `perf/cheap` | 2026-04-27 | 5 + 10 / ~900 | main links `burl-perf-phase1` (page only here) | DONE — three pages rescued, branches tombstoned |
| `research/recursive-generalization-2026-10-15` | 2026-10-15 | 6 / 14 | PR #174 | MERGED to main (rebased, GREEN); PR #174 closed; tombstoned |
| `worktree-table42` | 2026-07-18 | 9 / 427 | PR #68; #65 #66 #67 #69 #80 | LANDED on main (`f529d358`) after the fence ruling; tombstoned |
| The July stack (nine branches above) | 2026-07-18…22 | 14–56 / ~428 | PRs #78–#95; 10 issues | TOMBSTONED 2026-10-20 — code in `attic/` tags, 11 wiki pages kept |
| `ccr-07735f91-fouydw` | 2026-10-16 | 6 / 14 | #175 | LIVE — a cloud session's branch |
| `codex/silas-world-constructor` | 2026-10-14 | 3 / 79 | PR #166 | LIVE — Codex worktree |
| `codex/fair-42-table-player` | 2026-05-26 | 9 (local) | none | LIVE — Codex worktree `cf79` |
| `lem/stage1`, `worktree-agent-ab92…` | 2026-10-16 | 2 / 6 | desks, #173 | LIVE |
| `silas/cross-game-slice4`, `book/count-lapse` | 2026-10-16 | 4 / — | Book desk | MERGED by Book (`aba0c958`, `3aba8426`), branches and worktrees removed |

## Tombstones

**13 branches were tagged `attic/<name>` and deleted** (`git ls-remote --tags origin 'attic/*'` lists 13): the nine in the first table, `perf/batch` and `perf/cheap`, `research/recursive-generalization-2026-10-15` (merged first) and `worktree-table42` (landed first). Two of the 13 (`gascity`, `worktree-wiki-overhaul`) were local-only, so their tags are the only copy of them on origin.

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

Set aside by the same pass (rescued pages only, April 2026, ~900 commits behind):

| Branch (tag) | Tip | What it tried | Why it stopped |
|---|---|---|---|
| `perf/batch` | `7d80bab7` | the Burl phase-2 clean re-validation (retracts magnitude claims), the cohort abstraction spec, mlx-lm internals | wiki pages rescued to main ([[harvest-cohort-abstraction]]); its bench JSON and edits to pages main has since moved on stay in the tag. Burl's perf program ended with the champion era |
| `perf/cheap` | `8928bfe7` | Burl phase-1 cheap wins: turn-aware budgets, temp-0 clean baselines, lever-1 pre/post codepath benches | [[burl-perf-phase1]] and [[mlx-cohort-bench-discipline]] rescued to main; bench JSON stays in the tag |

## Other verdicts

- **`research/recursive-generalization-2026-10-15`**: merged 2026-10-16 (six commits rebased, its log entry moved to a `wiki/log.d` fragment, `scripts/check --fast` GREEN). Tag `attic/research/recursive-generalization-2026-10-15`, branch deleted, PR #174 closed.
- **`worktree-table42`** (PR #68; #65 #66 #67 #69 #80): squashed onto current main with SPDX headers and landed as `f529d358`; the Mayor's fence ruling `9cebd296` makes the jud seat's E[Q] a sanctioned opponent (token plus an inventory row in [[strategy-fusion-fence]]). #65 and #66 closed with receipts; #67, #69, #80 stay open. Tag `attic/worktree-table42`, branch deleted, PR #68 closed.
- **The July stack**: shelved on Jason's answer; see § The July stack, shelved.
- **Live, untouched** `ccr-07735f91-fouydw` (a cloud session wrote it today, #175), `codex/silas-world-constructor` (PR #166) and `codex/fair-42-table-player` (Codex worktrees), `lem/stage1`, `worktree-agent-ab9223d8997972645` (locked; `lsof` shows a live `gtimeout`/`node`/`python` there, #173). Book's `book/count-lapse` and `silas/cross-game-slice4` were live earlier tonight; Book merged both (`3aba8426`, `aba0c958`) and removed the branches and worktrees.
- **Waiting on Jason, one line each**: `switch-to-github-issues` is checked out in `~/intent/workspaces/book-translation/mk5-main`, a live workspace, and holds one 10-line `strategies/__init__.py` ("T5: Analysis, wiki experiment page, manifest"): is that book-translation's work to land?
- **Worktrees on merged branches** (nothing unmerged, so not this pass's job): `zeb-package`, `perf/aggressive`, `lead-notes`, `rob-vs-eq`, `agent-a4a0…`. Whoever owns them can `git worktree remove` them.


## Links

[[standing-desks]] · [[endgame-equivalence-census]] · [[scenario-player]]
