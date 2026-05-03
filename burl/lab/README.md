# burl/lab

Deterministic experimentation platform for Burl. Greenfield rewrite that runs
side-by-side with `burl/chat/` (the reference impl) on different ports:

|              | chat | **lab** |
|--------------|------|---------|
| server port  | 8001 | **8002** |
| web port     | 5173 | **5174** |

## Run the server

```bash
# from repo root
uvicorn burl.lab.server.app:app --port 8002 --reload
```

Smoke check:

```bash
curl -s http://localhost:8002/api/health | jq .
```

Create a session and post a no-op move:

```bash
SID=$(curl -s -X POST http://localhost:8002/api/sessions | jq -r .session_id)
curl -N -X POST http://localhost:8002/api/move \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"move\":{\"kind\":\"UserChoice\",\"option_name\":\"set_system\",\"args\":{\"text\":\"hello burl\"}}}"
curl -s http://localhost:8002/api/sessions/$SID/frame | jq .
```

## Environment

| var | default | meaning |
|-----|---------|---------|
| `BURL_HARNESS_SESSION_ROOT` | `~/.cache/burl-harness` | where session dirs live |
| `BURL_HARNESS_HARVEST_ROOT` | `scratch/belief_trajectory_rollout` | harvest root for `load_decision`; this fills the prompt builder, it does not start a run |
| `BURL_HARNESS_MODEL_REPO` | `mlx-community/gemma-4-e2b-it-bf16` | MLX model for `MlxEngine` |
| `BURL_HARNESS_ADAPTER_PATH` | unset | optional LoRA adapter path |
| `BURL_HARNESS_HF_PUSH` | unset | set to `1` to enable HF Hub session push (default off) |

## Layout

```
burl/lab/
  SPEC.md                 canonical contract — read first
  core/
    arrow.py              Trace algebra: journalable Moves + optional output
    transcript.py         Stamp, Move union, Frame, Option, State, append/replay/fold
    tool.py               ToolSpec, ToolResult, Registry
    phase.py              Phase Protocol + PHASES registry
    engine.py             MlxEngine — async step() generator
    render.py             pure system-prompt + chat renderer
    drive.py              phase-aware drive loop
    hf_sink.py            optional HF Hub push (gated)
  phases/
    pre_game.py           prompt builder + advertised tool set + explicit run start
    in_run.py             live model run + interject/abort/select_tool
    post_turn.py          committed-turn review surface
  server/
    app.py                FastAPI on :8002
    stream.py             SSE emitter (CRLF-aware)
  tools/
    base/                 belief_trajectory, explore_game, commit_play
  tests/
```

## Boundaries

- **No** backwards-compat shims with `burl/chat/`. Greenfield.
- **Do not touch** `burl/harness/` — that's the existing tool-loop runner used
  by `wax_museum`, `haiku_spike`, etc., despite the name overlap.
- `core/render.py` is the **single source of truth** for prompt assembly. If
  the model sees something, this file produced it.
- HATEOAS surface: `ToolResult.next_tools` *registers* surfaced tools and
  promotes them into `state.active_tools`, but the user must call
  `select_tool` in `in_run` to advertise them to the model.
