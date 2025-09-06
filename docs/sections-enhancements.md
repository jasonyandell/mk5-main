# Sections Enhancements (Backlog)

This file captures optional improvements and follow-ups beyond the initial dispatcher, AI speed profile, core sections (runner + stop conditions), presets, and basic UI hooks.

## URL/History Batching During Sections
- Problem: URL/history updates on every action can create noisy history and slow navigation.
- Idea: Add a batching flag during a section. While active, suppress per-action URL updates and flush once on section end.
- Sketch:
  - Add `beginUrlBatch()` / `endUrlBatch()` in the store; track a counter.
  - `gameActions.executeAction` checks the counter and skips `pushState/replaceState` if nonzero.
  - `SectionRunner` toggles the counter for the section lifetime.

## Dev Overlay for Dispatcher / Gate
- Visualize current allowed vs queued transitions, gate policy, and sources (ui/ai/system/replay).
- Small panel showing:
  - Gate active? (yes/no)
  - Queue size
  - Last N transitions (id, type, source)
  - Current AI speed profile
- Add a debug-only Svelte component that subscribes to dispatcher events.

Additional overlay ideas:
- Recent transitions list (id, type, source, elapsed time).
- Visual mark when a custom gate is active and which predicates are in effect.
- Toggle controls to pause/resume/cancel current section.

## Consensus ‘injectAll’ Polish
- Current plan injects missing `agree-*` transitions at trick/scoring boundaries.
- Future: parameterize which players to auto-agree (e.g., only AI, never humans), and support timeouts before injection.
- Add per-section option to inject after a short delay to preserve “feel” while avoiding stalls.

## Determinism & Metrics
- Deterministic mode: seed the AI delay jitter (Math.random) for reproducible runs; log seeds.
- Metrics: emit timing and selection data via dispatcher events for profiling.
- Export a structured audit log for bug reports.
- Attach correlation IDs to sections (start/stop, actions executed) for traceability.

## Replay Integration
- Live replay via dispatcher with a special gate to step through recorded actions.
- Offline replay remains pure and separate; provide adapters to create sections from replay slices.
- UI controls to scrub through a replay, or run a replay as a section with stop conditions (e.g., stop at trump selection).

## Section Presets (Extensions)
- Additional presets beyond onePlay/oneTrick/oneHand:
  - oneBidRound: stop after four bids or redeal.
  - toTrumpSelection: stop when phase === trump_selection.
  - toScoring: stop when entering scoring; optionally auto-complete scoring agrees.
  - fullGame: stop when phase === game_end (preset exists; add UI hook when ready).

## Gate Safeguards
- Add a watchdog to avoid deadlocks (e.g., no allowed transitions for X ms): auto-cancel or relax gate.
- Expose `cancel/pause/resume` in a dev hotkey.
- Safety net to auto-clear queued transitions on `resetGame`, `loadState`, `loadFromURL`, `undo`.

## Public Dispatcher API
- Consider exposing lightweight dispatcher APIs for tooling (e.g., CLI scripts or test harnesses) with safe guards.
- Provide a minimal, documented event contract for before/after listeners (for external tools).

## Section UX Hooks
- Pre/post banners and custom toasts integrated with sections.
- Section progress indicators (e.g., 1/4 plays, trick 3/7).
- Optional toasts on section completion with brief summary (e.g., “Trick completed by P2”).

## What’s Left To Do (Roadmap)
- More presets/UI hooks: add buttons for One Hand (implemented), Full Game, and others above.
- Dev overlay and basic telemetry: visualize gate/queue and last transitions; log timings.
- Replay section mode with step-through controls.
- Consensus parameterization and delays.
- Deterministic seeds for delay jitter and audit logs.
