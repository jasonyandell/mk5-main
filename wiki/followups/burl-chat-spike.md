Reviewed against code on 2026-07-07 — no issues found.

Verified: `burl/chat/server/` layout (app.py, inference.py, improvised_tools.py, tools_runner.py), `tools_library/` contains exactly the four improvised tools named (board_snapshot, legal_plays, play_brief, state_brief), `autoFillFromHarvest` in `burl/chat/web/src/App.svelte` returns `null` on args mismatch as described, `AUTO_SERVE_BASE` includes explore_game/probe_* as claimed, default model repo is `mlx-community/gemma-4-e2b-it-bf16`, and all [[wiki links]] resolve to existing pages.

Unverifiable (gitignored/local artifacts): the harvest directories under `scratch/belief_trajectory_rollout/` and the `burl/adapters/e1-rank16` adapter weights — session-level numbers (regret 12.12, +11.7/-5.8 Q, decision #1 rerun details) taken on the page's word.

- A cheap next probe (still open from "What's next"): the bucket-sampling pass (3–4 decisions per bucket) was never run before the spike was superseded by burl-lab; if burl-lab revives, that's the first thing to sample.
- The `play_brief` "adoption asymmetry" fix (naming the tool in protocol text) was diagnosed but not shown fixed here; worth confirming burl-lab's primer actually names improvised tools.
