# burl/lab — Deterministic Experimentation Platform

> Greenfield workbench for the Burl LLM agent on Texas 42. Replaces `burl/chat/`
> (which stays alongside as a reference until parity).
>
> **Side-by-side ports:** server `:8002`, web `:5174` (chat uses `:8001` / `:5173`).
> **Do not touch `burl/chat/`.**

---

## Philosophy

1. **State is a fold over events.** A session = `events.jsonl` on disk. State is
   recovered by `fold(replay(session_dir), registry)`. There is no in-memory
   master; the journal is the source of truth.

2. **Phase is a logged arrow.** Five phases planned (first pass: `pre_game`,
   `in_run`). Each phase implements `render(state) -> Frame`,
   `options(state) -> [Option]`, `async handle(state, move) -> Trace[str]`.
   The trace carries journalable `Move`s plus an optional next phase. Phases do
   not append or re-fold; the runtime interprets the trace. The model never sees
   the phase machinery — it only sees the messages each phase produces.

3. **Engine is a pure async generator.** `step(messages, tools) -> AsyncIterator[Event]`.
   It knows nothing about phases, sessions, harvests, or UI. Yields a strict
   sequence per call: `EngineStart` → 0+ `EngineToken` → 0+ `EngineToolCall` →
   optional `EngineCommit` → terminal `EngineDone`.

4. **Tools are first-class values.** A `ToolSpec` carries name, description,
   JSON-schema params, an example call, a `protocol_role`, a literal
   `protocol_phrase` for the system prompt, and an `impl`. The system prompt is
   **rendered** from the active ToolSpec set — never hand-edited.

5. **HATEOAS tool advertisement.** Every `ToolResult` from `impl()` may include
   `next_tools: tuple[ToolSpec, ...]`. The runtime registers them and adds them
   to `state.advertised`, surfacing them as user-selectable options for
   subsequent steps.

6. **First-class timings.** Every Move carries a `Stamp` with wall-time,
   monotonic-time, prompt/completion tokens, cumulative token counts, TTFT,
   decode latency, and tok/s. Performance is observed by reading the journal,
   not by sprinkling print statements.

7. **HuggingFace sink.** Sessions can be pushed to the dataset
   `jasonyandell/burl-harness-sessions` (gated by env var
   `BURL_HARNESS_HF_PUSH=1`; default off during dev).

8. **Phases are harness-private.** Phase identifiers and transitions are not
   surfaced to the model. Retraining is **not** a concern here — this is a
   deterministic experimentation platform, not a corpus generator.

---

## Module layout

```
burl/lab/
  __init__.py
  SPEC.md                       <-- this file
  core/
    __init__.py
    arrow.py                    # Trace algebra: journalable events + optional output
    transcript.py               # Stamp, Move union, Frame, Option, State, append/replay/fold
    tool.py                     # ToolSpec, ToolResult, Registry
    phase.py                    # Phase Protocol, PHASES registry
    engine.py                   # async step(messages, tools) generator       (engine agent)
    render.py                   # pure system-prompt + chat renderer          (runtime agent)
    drive.py                    # phase-aware drive loop                      (runtime agent)
    hf_sink.py                  # session -> HF dataset push                  (runtime agent)
  phases/
    __init__.py                 # imports & registers each phase
    pre_game.py                 # session bootstrap, pick a hand              (runtime agent)
    in_run.py                   # play through a hand                         (runtime agent)
  tools/
    __init__.py
    base/
      __init__.py
      belief_trajectory.py      # ToolSpec                                    (tools agent)
      explore_game.py           # ToolSpec                                    (tools agent)
      commit_play.py            # ToolSpec                                    (tools agent)
  server/
    __init__.py
    app.py                      # FastAPI on :8002                            (runtime agent)
    stream.py                   # SSE streaming                               (runtime agent)
  tests/
    __init__.py
    test_transcript_roundtrip.py    (types agent — this turn)
    test_engine_smoke.py            (engine agent)
    test_render.py                  (runtime agent)
    test_drive_with_fake_engine.py  (runtime agent)
    test_server_smoke.py            (runtime agent)
    test_tools_base.py              (tools agent)
```

---

## Build order (dependency-respecting)

1. **types** (this doc + `core/{transcript,tool,phase}.py` + transcript test) — **landing now.**
2. **engine** (`core/engine.py` + smoke test) — needs transcript types.
3. **tools** (`tools/base/*.py` + tools test) — needs ToolSpec/ToolResult.
4. **runtime** (`core/{render,drive,hf_sink}.py`, `phases/*.py`, `server/*.py`,
   tests) — needs everything above.

Engine and tools can start in parallel against this spec; runtime joins once
the engine is in.

---

## Canonical types

### `core/arrow.py`

`Trace` is the small algebraic result every phase handler returns:

```python
@dataclass(frozen=True)
class Trace(Generic[O]):
    events: tuple[Move, ...] = ()
    output: O | None = None

    def then(self, fn: Callable[[O], Trace[P]]) -> Trace[P]: ...
```

Read it as `input -> (journalable Moves, optional output)`. Composition
concatenates `events`; when `output is None`, the next arrow is skipped but the
events already produced remain durable. The runtime owns interpretation:
append the events, re-fold state from `events.jsonl`, then route on the optional
output.

### `core/transcript.py`

```python
@dataclass(frozen=True)
class Stamp:
    t_wall_ms: int                # wall ms since session start
    t_mono_ns: int                # monotonic ns; for deltas
    tok_in: int = 0               # prompt tokens this step
    tok_out: int = 0              # completion tokens this step
    tok_cum_in: int = 0
    tok_cum_out: int = 0
    ms_ttft: int | None = None
    ms_decode: int | None = None
    tok_per_s: float | None = None
```

**Move kinds** (each frozen dataclass with `kind: str` discriminator + `stamp` +
payload):

| Kind | Payload | Emitted by |
|---|---|---|
| `PhaseEnter` | `phase: str` | runtime (drive loop) |
| `PhaseExit` | `phase: str` | runtime |
| `UserText` | `text: str` | server (user input) |
| `UserChoice` | `option_name: str`, `args: dict` | server (option click) |
| `EngineStart` | `messages_hash: str`, `n_messages: int`, `n_tools: int` | engine (start of step) |
| `EngineToken` | `text: str` | engine (per chunk) |
| `EngineToolCall` | `name: str`, `args: dict`, `call_id: str` | engine |
| `ToolResult` | `name: str`, `call_id: str`, `evidence: dict`, `next_tools: list[str]` | runtime (after `impl()`); `next_tools` is tool **names** on the wire (full specs live in registry) |
| `EngineCommit` | `final: Any` | **harness** (drive loop / phase post-processor — NOT the engine) |
| `EngineError` | `message: str`, `traceback: str \| None`, `during: str \| None` | engine (mid-step failure; immediately precedes `EngineDone(reason="aborted")`); `during` is the call_id of an in-flight tool call if relevant |
| `EngineDone` | `reason: Literal["done","budget","aborted","tool_dispatch"]` | engine (terminal) |
| `SystemSet` | `text: str` | phase (typically `pre_game`) — sets the verbatim system message at `messages[0]`; idempotent |
| `AdvertisedSet` | `names: list[str]` | phase / server — replaces `state.advertised` outright; validation that names ⊆ active is the phase's responsibility |
| `ToolAdded` | `name: str` | phase — adds a tool name to `state.active_tools`; idempotent |
| `ToolRemoved` | `name: str` | phase — removes a tool name from `state.active_tools` AND `state.advertised`; idempotent on absence |

```python
Move = Union[PhaseEnter, PhaseExit, UserText, UserChoice,
             EngineStart, EngineToken, EngineToolCall, ToolResult,
             EngineCommit, EngineError, EngineDone,
             SystemSet, AdvertisedSet, ToolAdded, ToolRemoved]
```

**Config Moves: `SystemSet` / `AdvertisedSet` / `ToolAdded` / `ToolRemoved`.**
They exist so that `state = fold(replay(events))` is *total* — the journal is
the only source of truth for session configuration. Do NOT write a sidecar
`state.json`; if a configuration change can't be expressed as a Move, propose a
new Move kind, don't bypass the journal.

**Token-counter semantics on `Stamp`:**
- `tok_in` / `tok_out` are *deltas attributed to this event*, not the running total. On `EngineStart` set `tok_in = prompt_token_count` and `tok_out = 0`. On `EngineToken` set `tok_in = 0` and `tok_out = len(chunk_tokens)`. On `EngineToolCall` / `EngineDone` / `EngineError` both deltas are typically 0.
- `tok_cum_in` / `tok_cum_out` are the running totals **as of this event**. They are monotonically non-decreasing across the journal. `now_stamp(state, tok_in=..., tok_out=...)` derives them automatically from `state.cum_tok_*`.

### `core/transcript.py` — view types

```python
@dataclass(frozen=True)
class Option:
    name: str
    label: str
    args_schema: dict             # JSON schema describing what the user supplies

@dataclass(frozen=True)
class Frame:
    phase: str
    segments: list[dict]          # typed segments for UI rendering
    active_tools: list[str]       # tool names; full ToolSpecs via registry
    advertised: list[str]         # subset of active_tools advertised to model
    timing: dict                  # {wall_ms, tok_cum_in, tok_cum_out, tok_per_s}
```

### `core/transcript.py` — State

```python
@dataclass(frozen=True)
class State:
    session_dir: Path
    phase: str
    messages: tuple[dict, ...]            # render-ready chat messages
    active_tools: tuple[str, ...]         # names; full specs via registry
    advertised: tuple[str, ...]           # names; subset of active_tools
    segments: tuple[dict, ...]
    cum_tok_in: int
    cum_tok_out: int
    started_mono_ns: int
    started_wall_ns: int
```

### `core/transcript.py` — I/O

```python
def append(session_dir: Path, move: Move) -> None: ...
def replay(session_dir: Path) -> Iterator[Move]: ...
def fold(moves: Iterable[Move], session_dir: Path, registry: Registry) -> State: ...
def now_stamp(state: State, **kw) -> Stamp: ...
```

- `append`: one JSON line per move to `session_dir / "events.jsonl"`. Creates
  the parent dir if missing. Wire format:
  `{"kind": "...", "stamp": {...}, ...payload}`.
- `replay`: line-by-line; reconstructs the dataclass via `kind` dispatch.
- `fold`: rebuilds `State` from a sequence of moves. `PhaseEnter`/`PhaseExit`
  set `state.phase`. `UserText` appends to `messages`; `UserChoice` is recorded
  as a segment only. `EngineToken` accumulates an assistant segment.
  `EngineToolCall` opens a `tool_call` segment; `ToolResult` closes it
  (evidence inline) and adds `next_tools` names to both `active_tools` and
  `advertised` (HATEOAS). `EngineCommit` / `EngineError` / `EngineDone`
  produce their own segments. **Config Moves**: `SystemSet` writes/replaces
  the system message at `messages[0]`; `AdvertisedSet` replaces `advertised`
  outright; `ToolAdded` appends to `active_tools` (idempotent); `ToolRemoved`
  drops from both `active_tools` and `advertised`. Cumulative tokens come
  from `EngineStart` / `EngineToken` stamps.

### `core/tool.py`

```python
ProtocolRole = Literal["first_read", "candidate_eval", "diagnostic", "commit"]

@dataclass(frozen=True)
class ToolResult:
    evidence: dict                          # {"prose": str, "structured": Any}
    next_tools: tuple["ToolSpec", ...] = ()  # HATEOAS — full specs
    next_phase: str | None = None

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict                  # JSON schema (draft 7)
    example: str                  # literal call string, e.g. 'explore_game(play=14)'
    protocol_role: ProtocolRole
    protocol_phrase: str          # literal sentence the system prompt should include
    impl: Callable[[Any, dict], ToolResult]

class Registry:
    def add(self, spec: ToolSpec) -> None: ...
    def remove(self, name: str) -> None: ...
    def find(self, name: str) -> ToolSpec | None: ...
    def active(self) -> list[ToolSpec]: ...
```

`Ctx` (the first arg to `impl`) is whatever runtime context tools need.
Currently aliased to `Any` to avoid hard-importing `burl.wax_museum.tools.WaxContext`.
Migrate to a proper alias once runtime+tools are in.

### `core/phase.py`

```python
class Phase(Protocol):
    name: str
    def render(self, state: State) -> Frame: ...
    def options(self, state: State) -> list[Option]: ...
    async def handle(self, state: State, move: Move) -> Trace[str]: ...

PHASES: dict[str, Phase] = {}

def register(p: Phase) -> None: PHASES[p.name] = p
```

`Trace.output` is the optional next phase name. `Trace.events` are the only
state changes a phase requests. A phase handler must not call `append()` or
`fold()`; that makes the server/smoke runner the only journal interpreter.

### `core/engine.py` — engine agent's contract

```python
async def step(
    messages: list[dict],
    tools: list[ToolSpec],
    *,
    state: State,
    budget: int = 1024,
) -> AsyncIterator[Move]: ...
```

Yield order from `step()`, strict:
1. `EngineStart(messages_hash, n_messages, n_tools)`
2. zero or more `EngineToken(text)`
3. zero or more `EngineToolCall(name, args, call_id)` — when one is yielded, the
   engine **pauses**; the runtime executes the tool, appends a `ToolResult`,
   and may invoke `step()` again with updated `messages`
4. optionally `EngineError(message, traceback, during)` if the step failed
   mid-decode; pair it with `EngineDone(reason="aborted")` immediately after
5. terminal `EngineDone(reason)` — `reason="tool_dispatch"` if step ended on a
   tool call awaiting result; `"done"` for clean finish; `"budget"` on cap;
   `"aborted"` on error/cancel (the preceding `EngineError` carries the message)

**`step()` does NOT yield `EngineCommit`.** Commits are emitted by the *harness*
(the drive loop or a phase post-processor) after the engine's terminal
`EngineDone`, when the phase decides the trailing region constitutes a commit
(e.g., parsing the final assistant text into a structured `{"bid": 31}` payload).
The engine's job is to produce raw token+tool-call output; the phase decides
what is a commit.

---

## Wire format (events.jsonl)

One move per line. `kind` discriminator + `stamp` + payload, all flat:

```json
{"kind":"PhaseEnter","stamp":{"t_wall_ms":0,"t_mono_ns":42424242,"tok_in":0,"tok_out":0,"tok_cum_in":0,"tok_cum_out":0,"ms_ttft":null,"ms_decode":null,"tok_per_s":null},"phase":"pre_game"}
{"kind":"UserChoice","stamp":{...},"option_name":"start_run","args":{"hand":"66 55 44 33 22 11 00"}}
{"kind":"EngineStart","stamp":{...},"messages_hash":"abc...","n_messages":3,"n_tools":2}
{"kind":"EngineToken","stamp":{...},"text":"Looking at"}
{"kind":"EngineToolCall","stamp":{...},"name":"belief_trajectory","args":{"play":14},"call_id":"01HXY..."}
{"kind":"EngineDone","stamp":{...},"reason":"tool_dispatch"}
{"kind":"ToolResult","stamp":{...},"name":"belief_trajectory","call_id":"01HXY...","evidence":{"prose":"...","structured":{...}},"next_tools":["explore_game"]}
```

---

## HF sink (runtime agent)

`core/hf_sink.py` exports `push_session(session_dir: Path) -> None`.

Default: no-op when `BURL_HARNESS_HF_PUSH != "1"`. When enabled: bundle
`events.jsonl` + metadata into a row in `jasonyandell/burl-harness-sessions`.

---

## Server (runtime agent)

- `server/app.py` — FastAPI on port **8002** (chat workbench is on 8001).
- `server/stream.py` — SSE rendering of Move sequence to the web UI.
- Web UI on port **5174** (chat workbench is on 5173). Web UI is *not* in scope
  for this team; assume an existing/parallel team supplies it.

---

## Boundaries

- **Do not touch `burl/chat/`.** It is the reference implementation.
- **Do not touch `burl/harness/`.** Despite the name overlap, that package is
  the existing agent tool-loop runner used by `wax_museum`, `haiku_spike`,
  `candlewax_spike`, `eval/run_move4_*`, and the chat tools_library. We chose
  `burl/lab/` to avoid the collision.
- **No legacy / no backwards-compat shims.** Greenfield.
- **No `@deprecated` markers, `_legacy` suffixes, "for compatibility" comments.**
  The architecture test in `src/tests/architecture/no-backwards-compat.test.ts`
  enforces this.

---

## Coordination

When this SPEC.md and `core/{transcript,tool,phase}.py` land on disk, types
notifies `engine`, `runtime`, and `tools` peers. Each peer drafts against this
contract; corrections flow back through `SendMessage` → SPEC.md update.
