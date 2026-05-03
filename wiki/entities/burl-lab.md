---
title: burl-lab — Deterministic experimentation platform for Burl
kind: entity
first_seen: a2db3c7
last_updated: local-2026-05-03
status: active
---

## What it is

`burl/lab/` is a greenfield experimentation workbench for the [[burl]] LLM agent. It replaces [[burl-chat]] as the active surface for talking with Burl about decisions, but raises the architectural bar: every interaction is event-sourced, the model's view is rendered from first-class tool specs, and tool results carry their own next-move advertisements. Retraining is **not** a goal — this is a deterministic experimentation platform, not a corpus generator.

Lives entirely under `burl/lab/`. Side-by-side with `burl/chat/` until parity: server on port `8002` (test runs use `18002`), web on `5174` (chat is on `8001`/`5173`).

## Status

Server runs end-to-end against a fake engine. `/api/health`, `/api/sessions`, and `/api/move` are wired; the three base ToolSpecs (`belief_trajectory`, `explore_game`, `commit_play`) register correctly, and **`events.jsonl` is the only on-disk source of truth**. Phase handlers are now [[logged-arrows]]: `handle(state, move)` returns a `Trace[str]` carrying journalable Moves (`SystemSet`, `AdvertisedSet`, `ToolAdded`, `ToolRemoved`, `UserText`, etc.) plus an optional next phase. The server interprets that trace by appending its Moves, re-folding from `events.jsonl`, and materializing any phase transition. Verified by `tests/test_phase_arrow.py` and `tests/test_server_smoke.py::test_journal_is_canonical_no_state_json`.

15/15 fast tests pass across `tests/test_{transcript_roundtrip,engine_smoke,tools_base,render,drive_with_fake_engine,server_smoke}.py`, plus one slow MLX smoke (gated, runs against a real Gemma 4 E2B load). Real MLX engine integration is wired in `core/engine.py` ([burl/lab/core/engine.py @ a2db3c7](../sources/a2db3c7.md)), including a fix for an upstream `mlx_lm` module-level `generation_stream` bug — rebound via `sys.modules["mlx_lm.generate"]` after `load()` inside the executor thread. See [[mlx-lm]] "Upstream bug: module-level generation_stream" for the full diagnosis and the submodule-shadowing trap.

**Phase ownership of transitions.** Drive is engine-shaped — it does not mint `PhaseExit`/`PhaseEnter` Moves. The server is transition-shaped — it journals exit/enter when a phase trace's `output` is a next phase. After a commit-role tool dispatches in `in_run`, drive synthesizes `EngineCommit`, the server calls `in_run.handle(state, EngineCommit)` which returns `Trace(output="post_turn")`, and the server appends `PhaseExit("in_run") + PhaseEnter("post_turn")`. `post_turn` renders the committed-session segments and offers `start_new_session` → `pre_game`.

**Harvested chat prompt import.** `pre_game.load_decision` now mines the original [[burl-chat]] `prompt_system` as well as `prompt_user`. It keeps Burl's identity, trimmed Texas 42 rules primer, current-decision 42 framing, and user-facing state prompt, but strips the legacy `# Decision protocol (wax_museum)` tail and raw `<|tool>declaration:` blobs before journaling `SystemSet`. The next engine step then re-renders the Decision Protocol from active `ToolSpec.protocol_phrase` values. This preserves the useful chat-era grounding while keeping the tool/protocol surface algebraic and non-stale.

**Mined chat tools.** The four [[improvised-tools]] that earned their keep in [[burl-chat-spike]] now have first-class `ToolSpec` wrappers under `burl/lab/tools/chat_mined.py`: `state_brief`, `board_snapshot`, `legal_plays`, and `play_brief`. Their implementations still live in `burl/chat/server/tools_library/`; burl-lab wraps them only to attach JSON schemas, examples, `protocol_role`, and `protocol_phrase`. The server registry now boots with seven tools total: the three base decision tools plus the four mined chat tools.

**Prompt-builder UX.** Pre-game is now an explicit prompt builder rather than a hidden bootstrap step. The user picks an advertised tool set from registry checkboxes, clicks `Generate system prompt` to seed the default Burl base prompt, and sees the composed `rendered_system` immediately: base prompt plus the selected tools' `ToolSpec.protocol_phrase` values. `Load harvested decision into builder` imports harvested `prompt_system` and `prompt_user` but does not start the run; `Start run` and `Ask Gemma now` are separate phase choices. The web surface now wraps this in a guided wizard: new session → build prompt or chat → select tools → confirm and ship to Gemma. The raw HATEOAS/option controls remain available as an advanced console, but the happy path no longer requires knowing internal move names like `SystemSet`, `AdvertisedSet`, or `load_decision`. The wizard and backend now refuse `Start run` until a user/decision prompt exists, and `ToolSpec.requires_context` keeps game-state tools hidden from the engine until ctx exists.

**Seeded decision lane.** The happy path no longer requires a separate "load then ship" move. `send_seeded_decision` takes `(harvest, seed)`, loads the same harvested user prompt that [[burl-chat]] uses, preserves the currently built system prompt/tool protocol, enters `in_run`, and gives the server enough journal data to build the game ctx for state tools. The wizard surfaces this as "Send play": optional chat, generate system prompt, select/apply tools, then send a seed.

## Architecture

Five load-bearing properties, each codified in `burl/lab/SPEC.md` ([burl/lab/SPEC.md @ a2db3c7](../sources/a2db3c7.md)):

### 1. State is a fold over events

A session is `events.jsonl` on disk. State is recovered by `fold(replay(session_dir), registry)`. There is no in-memory master; the journal is the source of truth. Every Move is one JSON line: `kind` discriminator + `Stamp` + flat payload. See `core/transcript.py`.

### 2. Phase is a logged arrow

Five phases planned (`empty`, `pre_game`, `in_run`, `post_turn`, `post_session`); current cut ships `pre_game` + `in_run` + `post_turn`. Each phase implements:

- `render(state) -> Frame`
- `options(state) -> [Option]`
- `async handle(state, move) -> Trace[str]`

`Trace` is the algebra the arrows thread through the harness: `events: tuple[Move, ...]` plus `output: str | None`. Composition concatenates events; absent output stops the next arrow while keeping the emitted logs. Phases are **harness-private** and side-effect-free with respect to disk: they do not append or re-fold. The model never sees phase identifiers or transitions — only the messages a phase produces. See `core/arrow.py` and `core/phase.py` for the Protocol; `phases/{pre_game,in_run,post_turn}.py` for implementations. Drive is engine-shaped only; the server owns `PhaseExit`/`PhaseEnter` Moves whenever a phase trace outputs a next phase.

### 3. Engine is a pure async generator

```python
async def step(messages, tools, *, state, budget=1024) -> AsyncIterator[Move]
```

The engine knows nothing about phases, sessions, harvests, or UI. Yield order is strict per call: `EngineStart` → 0+ `EngineToken` → 0+ `EngineToolCall` → optional `EngineCommit` → terminal `EngineDone`. When a tool call is yielded, the engine pauses; the runtime executes the tool, appends a `ToolResult`, and may invoke `step()` again with updated messages.

This shape makes the engine replaceable: a real MLX-LM engine for live runs, a recorded engine for replay, a fake engine for tests. Ports the single-thread executor + Gemma chat-template fix from [[burl-chat]]'s `inference.py`, and adds the `mlx_lm.generate.generation_stream` rebind that [[burl-chat]]'s import path leaves latent — see [[mlx-lm]] for the diagnosis.

### 4. Tools are first-class values

A `ToolSpec` carries:

- `name`, `description`, `params` (JSON-schema draft 7), `example` (literal call string)
- `protocol_role: Literal["first_read", "candidate_eval", "diagnostic", "commit"]`
- `protocol_phrase: str` — the literal sentence the system prompt should include for this tool
- `impl: Callable[[Ctx, dict], ToolResult]`

The system prompt's **Decision protocol** section is **rendered from the active ToolSpec set**, never hand-edited. This solves the adoption asymmetry documented in [[improvised-tools]]: in the [[burl-chat-spike]] second wave, `play_brief` sat unused because the protocol text named only `explore_game(play=X)` literally. With rendered protocol text, the protocol section names whatever tools are active — the gap closes by construction.

First three base tools live as ToolSpec values: `tools/base/{belief_trajectory,explore_game,commit_play}.py`.

**No `system_text` on State.** `State` has no stored system prompt field. Every step, `core/render.py` builds the system prompt fresh from the current active ToolSpec set — concatenating each spec's `protocol_phrase` into the Decision protocol section, plus the description / params / example block per tool. Add a tool to the active set, the next render names it; remove one, the next render forgets it. This is the structural property that makes adoption first-class: there is no stale primer text to drift out of sync with the registry.

### 5. HATEOAS tool advertisement

Every `ToolResult` may include `next_tools: tuple[ToolSpec, ...]`. The runtime registers them and adds them to `state.advertised`, surfacing them as user-selectable options for subsequent steps. Tool results literally advertise next moves; the user is the gate (selection is a `UserChoice` move, not auto-execution).

**Two distinct `ToolResult` types, by design.** `tool.ToolResult` is the impl return value — carries `evidence: dict`, `next_tools: tuple[ToolSpec, ...]` (full specs), and an optional `next_phase`. `transcript.ToolResult` is the Move kind written to `events.jsonl` — carries `evidence: dict`, `name`, `call_id`, and `next_tools: list[str]` (names only; full specs are resolved through the Registry during `fold`). Names overlap, semantics differ: one is the call-time payload, the other is the journal record.

This is the structural answer to "Burl plans a tool he doesn't have" from the [[burl-tool-wishlist]] sessions: the prior tool's result can offer the next tool inline, instead of relying on system-prompt prescience.

### 6. First-class timings

Every Move carries a `Stamp`:

```python
@dataclass(frozen=True)
class Stamp:
    t_wall_ms: int
    t_mono_ns: int
    tok_in: int = 0
    tok_out: int = 0
    tok_cum_in: int = 0
    tok_cum_out: int = 0
    ms_ttft: int | None = None
    ms_decode: int | None = None
    tok_per_s: float | None = None
```

Performance is observed by reading the journal, not by sprinkling print statements. UI surfaces TTFT, decode latency, and tok/s inline; cumulative tokens come from `EngineStart`/`EngineToken` stamps.

### 7. HuggingFace sink

`core/hf_sink.py` exports `push_session(session_dir)`. Default: no-op when `BURL_HARNESS_HF_PUSH != "1"`. When enabled, bundles `events.jsonl` + metadata into a row in the `jasonyandell/burl-harness-sessions` dataset. Sessions are first-class ML artifacts; an experiment session is one row.

## Module layout

```
burl/lab/
  SPEC.md                       # canonical contract
  core/
    transcript.py               # Stamp, Move union, Frame, Option, State, append/replay/fold
    arrow.py                    # Trace algebra: journalable events + optional output
    tool.py                     # ToolSpec, ToolResult, Registry
    phase.py                    # Phase Protocol, PHASES registry
    engine.py                   # async step(messages, tools) generator
    render.py                   # pure system-prompt + chat renderer
    drive.py                    # phase-aware drive loop
    hf_sink.py                  # session -> HF dataset push
  phases/
    pre_game.py                 # prompt builder, tool picking, explicit run start
    in_run.py                   # play through a hand
    post_turn.py                # render committed-session, offer start_new_session
  tools/base/
    belief_trajectory.py        # ToolSpec
    explore_game.py             # ToolSpec
    commit_play.py              # ToolSpec
  tools/chat_mined.py           # ToolSpecs wrapping burl-chat improvised tools
  server/
    app.py                      # FastAPI on :8002
    stream.py                   # SSE streaming
  tests/
    test_transcript_roundtrip.py
    test_engine_smoke.py
    test_render.py
    test_drive_with_fake_engine.py
    test_server_smoke.py
    test_tools_base.py
```

## Move kinds

The Move union is the wire format. One JSON line per move:

| Kind | Payload | Emitted by |
|---|---|---|
| `PhaseEnter` / `PhaseExit` | `phase: str` | runtime (drive loop) |
| `UserText` | `text: str` | server (user input) |
| `UserChoice` | `option_name: str`, `args: dict` | server (option click) |
| `EngineStart` | `messages_hash: str`, `n_messages: int`, `n_tools: int` | engine |
| `EngineToken` | `text: str` | engine (per chunk) |
| `EngineToolCall` | `name: str`, `args: dict`, `call_id: str` | engine |
| `ToolResult` | `name`, `call_id`, `evidence: dict`, `next_tools: list[str]` | runtime (after `impl()`) |
| `EngineCommit` | `final: Any` | engine (phase-defined commit shape) |
| `EngineDone` | `reason: Literal["done","budget","aborted","tool_dispatch"]` | engine (terminal) |

`ToolResult.next_tools` carries tool **names** on the wire; full ToolSpecs live in the Registry, resolved during `fold`.

## Why this exists

[[burl-chat]] proved the loop "harvested decision → conversation prefix → ask Burl about it" produces useful output, and the [[burl-chat-spike]] surfaced four structural findings that drove the redesign:

1. **Adoption asymmetry.** Tools advertised in declarations but not named in the protocol text sit unused (`play_brief`). Rendered protocol text from ToolSpec is the structural fix.
2. **Hand-edited primer drift.** burl-chat's `chat_mode_primer` is a magic string spliced post-`commit_play`. burl-lab's primer is whatever a Phase emits — composable, not patched.
3. **Performance was a second-pass derivation.** `Stamp` makes timings part of the journal; no second source of truth.
4. **Reflection-deafness ([[burl-reflection-deafness]]) needs phase-scoped retries.** A `post_turn` phase can offer reflection options the model is otherwise locked out of by [[play-adapter-lock-in]].

The wishlist → Tool → Adoption pipeline documented in [[burl-tool-wishlist]] survives the redesign — improvised tools become hot-registered ToolSpecs that ride the same first-class path as base tools, including HATEOAS advertisement back to the user.

## Naming history

The package was first called `burl/harness/` before the team noticed the collision: `burl/harness/` was already an agent tool-loop runner package with ~20 importers across `wax_museum/`, `haiku_spike/`, `candlewax_spike/`, `eval/run_move4_*`, and `burl/chat/server/tools_runner.py`. The new platform was renamed to `burl/lab/` to avoid the collision. Misrouted files were consolidated via `git mv` during the spike. The existing `burl/harness/` package is untouched.

## Boundaries (per SPEC.md)

- Do not touch `burl/chat/` — it is the reference implementation until burl/lab/ reaches parity.
- Do not touch `burl/harness/` — that is the existing agent tool-loop runner.
- No legacy / no backwards-compat shims. Greenfield.

## Related

- [[burl-chat]] — the reference predecessor; the workbench burl-lab replaces.
- [[burl-chat-spike]] — the four findings that motivated the architectural choices here.
- [[improvised-tools]] — hot-register tool layer; ports forward as ToolSpec hot-registration on top of HATEOAS.
- [[burl-tool-wishlist]] — Burl's meta-asks; the corpus signal the workbench is built to harvest.
- [[chat-mode-primer]] — recency-weighted primer; in burl-lab, primer text is what a Phase emits.
- [[play-adapter-lock-in]] — the structural finding that says post-commit Q&A needs co-training, not stacking; the experimentation platform is the surface for figuring out what that co-training corpus should look like.
- [[post-commit-q-and-a]] — research direction the workbench is feeding.
- [[mlx-lm]] — the engine the live `core/engine.py` wraps.

## First wire run (session d80349b45ad6)

**Date:** 2026-05-02. **Session:** `d80349b45ad6` (web → SSE → server :8002 → real `MlxEngine`). **Decision:** `harvest_batched_20260425_072910` decision_idx=1 — `BURL_BREAKS_CONSENSUS`, follow-suit, lead `14 (4-4)`. Journal: `scratch/burl-lab-runlogs/sessions/d80349b45ad6/events.jsonl` (99 events).

**Wire commit.** Tail is the canonical commit-role envelope:

- `EngineToolCall(name=commit_play, args={"domino_id": 21})`
- `EngineDone(reason=tool_dispatch)`
- `ToolResult("COMMIT: domino_id=21 recorded")`
- `EngineCommit(final={"domino_id": 21})`
- `PhaseExit("in_run")` → `PhaseEnter("post_turn")`

`domino_id=21` is `6-0`. The canonical in-process smoke session `scratch/burl-lab-sessions/3dc73cb67bc1/events.jsonl` committed the same `domino_id=21` for the same harvest+idx — wire and in-process agree on the answer end-to-end.

**Rendered Decision Protocol — first live confirmation.** Captured at `scratch/burl-lab-runlogs/rendered-system.txt`. The "Decision Protocol" section (lines 7–21) is composed verbatim from each active `ToolSpec.protocol_phrase`, grouped by `protocol_role` (`_ROLE_ORDER = ("first_read", "candidate_eval", "diagnostic", "commit")` in `core/render.py`):

```
## Decision Protocol

You have access to the following tools. Each tool plays a specific role in the decision loop; use the protocol phrase to decide when to call it.

### first_read

- **belief_trajectory** — Call `belief_trajectory()` once per turn to read the belief state before considering candidate plays.

### candidate_eval

- **explore_game** — Call `explore_game(play=X)` for each candidate play to sample its outcome distribution.

### commit

- **commit_play** — When you've decided, call `commit_play(domino_id=X)` once to play it. Do not call other tools after committing.
```

Only three of the four canonical roles appear. There is no `diagnostic`-role tool in the base set, so `core/render.py` skips that subsection — that is the rendered behavior, not a bug. Add a `diagnostic` ToolSpec to the active set and the next render will name it; remove a tool and the section shrinks. This is the first live confirmation that the Decision Protocol is rendered, not hand-edited.

**Chat-vs-lab structural diff.** No live A/B was run for this decision — `burl/chat` was not booted (out of scope: heavy second model copy). The platforms differ by construction:

- **burl/chat:** the protocol-instruction surface is a static template string spliced into the system prompt during prompt assembly. Tools added via the improvised registry don't change the protocol section unless the template is also edited. This is the [[burl-chat]] "necessary but insufficient" framing — the executor pattern is in place but the registry and the prompt are not coupled, so adoption is a separate manual step.
- **burl/lab:** the protocol section is `core/render.py` walking the active `ToolSpec` set every step, grouped by `protocol_role`. Tools selected and registered via `ToolResult.next_tools` (HATEOAS) appear in the next render's protocol section automatically. The Decision Protocol section is whatever the active tools say it is.

This is the structural answer to the [[burl-chat-spike]] adoption asymmetry — `play_brief` sat unused in burl-chat because the protocol text named only `explore_game(play=X)` literally. With rendered protocol text, the adoption gap closes by construction: there is no static template to drift out of sync with the registry.

**Sampling-stochastic note.** A pre-existing failed session for the same harvest+idx, `scratch/burl-lab-runlogs/sessions/8ff6e41fd87e/`, reasoned to `domino_id=21` but emitted it in plaintext, missing the `<|tool_call>` envelope — the engine never dispatched. Same answer (correct domino), missed envelope (no commit). Two-of-two on the answer; one-of-two on syntactic compliance for un-adapted base Gemma 4 E2B BF16. Exactly the kind of empirical observation this platform is built to make. Tracked under follow-up bead for envelope-adoption rate measurement.
