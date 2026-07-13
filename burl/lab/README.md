# burl/lab

This project's knowledge lives in the wiki — see `wiki/entities/burl-lab.md`.

Runs side-by-side with `burl/chat/` on different ports:

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

The Svelte app on port `5174` opens with a guided wizard: start a session,
build or load the prompt, select the advertised tools, then confirm and ship
the run to Gemma.

## Environment

| var | default | meaning |
|-----|---------|---------|
| `BURL_HARNESS_SESSION_ROOT` | `~/.cache/burl-harness` | where session dirs live |
| `BURL_HARNESS_HARVEST_ROOT` | `scratch/belief_trajectory_rollout` | harvest root for `load_decision`; this fills the prompt builder, it does not start a run |
| `BURL_HARNESS_MODEL_REPO` | `mlx-community/gemma-4-e2b-it-bf16` | MLX model for `MlxEngine` |
| `BURL_HARNESS_ADAPTER_PATH` | unset | optional LoRA adapter path |
| `BURL_HARNESS_HF_PUSH` | unset | set to `1` to enable HF Hub session push (default off) |
