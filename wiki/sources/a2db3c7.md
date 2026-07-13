---
title: "Source digest: a2db3c7 — burl/lab/ workbench platform"
kind: source
first_seen: a2db3c7
last_updated: a2db3c7
status: active
---

## Commit

- **SHA:** a2db3c709bb3c861e101c8d5438921551bcb2f31
- **Date:** 2026-05-02
- **Author:** Jason Yandell

> Add burl/lab/ workbench platform
>
> Deterministic experimentation platform replacing burl/chat/. Phase state
> machine (pre_game, in_run, post_turn stub), pure async Engine protocol,
> first-class ToolSpec values rendered into the system prompt, HATEOAS
> next_tools surfaced as user-selectable, events.jsonl as the only source
> of truth, and Stamp-shaped first-class timings/tokens. MLX-LM engine
> ports the single-thread executor + Gemma chat-template fixes from
> burl/chat/ and adds a sys.modules rebind for the upstream
> mlx_lm.generate.generation_stream cross-thread bug. Three base
> ToolSpecs (belief_trajectory, explore_game, commit_play) and a
> FastAPI server (port 8002) with one polymorphic /api/move endpoint.
> 15 fast tests + 1 slow MLX smoke, all green.

## What this commit establishes

The platform spine for [[burl-lab]]. The architecture is no longer paper — every property described in `SPEC.md` has a corresponding module, and 15 fast tests + 1 slow MLX smoke prove they compose end-to-end. Lands as 31 files / 4619 insertions in one shot, side-by-side with `burl/chat/` (which remains the reference predecessor).

Seven structural decisions reach disk:

1. **Journal-canonical state.** `events.jsonl` is the only on-disk source. `core/transcript.py` defines the Move union (`PhaseEnter`, `PhaseExit`, `UserText`, `UserChoice`, `EngineStart`, `EngineToken`, `EngineToolCall`, `ToolResult`, `EngineCommit`, `EngineDone`, plus the config Moves `SystemSet`, `AdvertisedSet`, `ToolAdded`, `ToolRemoved`); `fold(replay(session_dir), Registry())` reproduces the live `State`. No `state.json` companion at any point in this commit's history — the philosophy point lands ready-made.
2. **Phase as pure state machine.** `core/phase.py` is the Protocol; `phases/{pre_game,in_run,post_turn}.py` are the implementations. `pre_game` boots a session and selects a hand; `in_run` plays through; `post_turn` renders the committed-session segments and offers `start_new_session`. Each phase exports `render(state) -> Frame`, `options(state) -> [Option]`, `async handle(state, move) -> (state', next_phase)`.
3. **Engine as pure async generator.** `core/engine.py` is `step(messages, tools, *, state, budget) -> AsyncIterator[Move]` with strict yield order: `EngineStart` → 0+ `EngineToken` → 0+ `EngineToolCall` → optional `EngineCommit` → terminal `EngineDone`. Knows nothing about phases, sessions, harvests. Replaceable across real / fake / recorded.
4. **First-class ToolSpec.** `core/tool.py` defines `ToolSpec(name, description, params, example, protocol_role, protocol_phrase, impl)`. The system prompt's Decision protocol section is **rendered** from active ToolSpecs by `core/render.py` — never hand-edited. This closes the [[improvised-tools]] adoption-asymmetry hole by construction.
5. **HATEOAS tool advertisement.** Every `ToolResult` may include `next_tools: tuple[ToolSpec, ...]`. The runtime registers them and adds their names to `state.advertised`, surfacing them as user-selectable options on subsequent steps. `tool.ToolResult` (call-time, full specs) and `transcript.ToolResult` (journal Move, names only) are intentionally distinct types.
6. **First-class timings/tokens.** Every Move carries a `Stamp(t_wall_ms, t_mono_ns, tok_in, tok_out, tok_cum_in, tok_cum_out, ms_ttft, ms_decode, tok_per_s)`. Performance is observed by reading the journal, not by sprinkling print statements.
7. **MLX-LM engine fixes.** `core/engine.py:_load()` rebinds `mlx_lm.generate.generation_stream` via `sys.modules["mlx_lm.generate"]` after `load()` runs inside the executor thread, defeating the upstream module-scope-stream bug that the [[burl-chat]] import path leaves latent. Single-thread executor + Gemma chat-template fix carry over from `burl/chat/server/inference.py`. See [[mlx-lm]] "Upstream bug: module-level generation_stream" for the full diagnosis.

Drive is engine-shaped (no `PhaseExit`/`PhaseEnter`); server is transition-shaped (owns those Moves whenever a phase's `handle()` returns a non-`None` `next_phase`). Server runs on port `8002` (test runs use `18002`); web on `5174` (chat is `8001`/`5173`).

## Files changed

| File | Lines | Purpose |
|---|---|---|
| `burl/lab/SPEC.md` | new, +355 | Canonical contract — philosophy, module layout, Move kinds, Wire format, build order |
| `burl/lab/README.md` | new, +77 | Run instructions, port table, side-by-side with burl/chat/ |
| `burl/lab/__init__.py` | new, +4 | Package marker |
| `burl/lab/core/transcript.py` | new, +544 | Stamp, Move union, Frame, Option, State, append/replay/fold; the spine |
| `burl/lab/core/tool.py` | new, +77 | `ToolSpec`, `ToolResult`, `Registry`; first-class tools |
| `burl/lab/core/phase.py` | new, +42 | `Phase` Protocol + module-level `PHASES` registry |
| `burl/lab/core/engine.py` | new, +559 | Real MLX-LM engine + fake-engine path; `step()` async generator + `mlx_lm` rebind fix |
| `burl/lab/core/render.py` | new, +140 | Pure system-prompt + chat-message renderer; rendered protocol text from active ToolSpecs |
| `burl/lab/core/drive.py` | new, +259 | Phase-aware drive loop; engine-shaped only (no phase Moves) |
| `burl/lab/core/hf_sink.py` | new, +60 | `push_session(session_dir)`; gated by `BURL_HARNESS_HF_PUSH=1` (default off) |
| `burl/lab/phases/__init__.py` | new, +21 | Imports + registers each phase via `phase.register()` |
| `burl/lab/phases/pre_game.py` | new, +275 | Session bootstrap, pick a hand |
| `burl/lab/phases/in_run.py` | new, +146 | Play through a hand |
| `burl/lab/phases/post_turn.py` | new, +65 | Renders committed-session segments; offers `start_new_session` → `pre_game` |
| `burl/lab/server/__init__.py` | new, +12 | Package marker |
| `burl/lab/server/app.py` | new, +373 | FastAPI on `:8002`; `/api/health`, `/api/sessions`, polymorphic `/api/move`; `_load_state` is `fold(replay)` |
| `burl/lab/server/stream.py` | new, +75 | SSE rendering of Move sequence |
| `burl/lab/tests/test_transcript_roundtrip.py` | new, +333 | Move serialization + `fold(replay)` round-trip; `test_state_journal_only_no_state_json_needed` |
| `burl/lab/tests/test_engine_smoke.py` | new, +66 | Fake-engine yield-order contract |
| `burl/lab/tests/test_render.py` | new, +125 | System-prompt rendering from active ToolSpec set |
| `burl/lab/tests/test_drive_with_fake_engine.py` | new, +353 | Drive loop against a scripted fake engine, 5 cases |
| `burl/lab/tests/test_server_smoke.py` | new, +301 | FastAPI lifecycle, polymorphic move; `test_journal_is_canonical_no_state_json` |
| `burl/lab/tests/test_tools_base.py` | new, +152 | All three base ToolSpecs round-trip + `protocol_phrase` discipline |
| `burl/lab/tools/__init__.py` | new, +19 | Top-level tools export |
| `burl/lab/tools/base/__init__.py` | new, +13 | Re-exports `BELIEF_TRAJECTORY`, `EXPLORE_GAME`, `COMMIT_PLAY` |
| `burl/lab/tools/base/belief_trajectory.py` | new, +48 | First-read ToolSpec; wraps `burl.wax_museum` belief surface |
| `burl/lab/tools/base/explore_game.py` | new, +55 | Candidate-eval ToolSpec; reuses `WaxContext.get_or_build()` cache |
| `burl/lab/tools/base/commit_play.py` | new, +62 | Commit-role ToolSpec; result returns `next_phase="post_turn"` |
| `.beads/issues.jsonl` | +14 | Bead bookkeeping for the spike |

15 fast tests pass (`test_{transcript_roundtrip,engine_smoke,tools_base,render,drive_with_fake_engine,server_smoke}.py`) plus one slow MLX smoke (gated; runs against a real Gemma 4 E2B load). All green at commit time.

## Naming history

The package was first called `burl/harness/` before the team noticed the collision with the existing agent tool-loop runner package (~20 importers across `wax_museum/`, `haiku_spike/`, `candlewax_spike/`, `eval/run_move4_*`, `burl/chat/server/tools_runner.py`). Renamed to `burl/lab/`; misrouted files were consolidated via `git mv` during the spike. The existing `burl/harness/` package is untouched.

## Related pages

[[burl-lab]] · [[burl-chat]] · [[mlx-lm]] · [[improvised-tools]] · [[burl-tool-wishlist]] · [[chat-mode-primer]] · [[play-adapter-lock-in]] · [[post-commit-q-and-a]] · [[burl-chat-spike]]
