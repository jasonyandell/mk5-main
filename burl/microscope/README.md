# Burl microscope

A small human-in-the-loop experiment loop for Burl prompt/tool work.

The microscope is not a production agent and not a training corpus generator. It
loads one harvested decision, one editable recipe, and one Gemma conversation.
Then you step the model turn-by-turn, inspect tool calls/results, edit the
recipe, and rerun the same case.

## Run from Pi

This repo includes a project-local Pi extension at `.pi/extensions/burl-microscope.ts`.
After `/reload` in Pi:

```text
/burl open 1                # load global_idx=1 from the default harvest
/burl step                  # one Gemma turn, dispatch at most one tool
/burl auto                  # run until commit or the model pauses
/burl say why 6-0?          # chat with Burl, then one Gemma turn
/burl prompt                # show rendered system/user prompt
/burl tools                 # list available tools
/burl tools legal_plays play_brief commit_play
/burl mode off              # stop routing normal typed input to Burl
```

`/burl open` starts `python -m burl.microscope.server` automatically if the
server is not already listening on `127.0.0.1:8765`.

## Run directly

```bash
python -m burl.microscope.server
curl -s http://127.0.0.1:8765/api/health | jq .
SID=$(curl -s -X POST http://127.0.0.1:8765/api/sessions \
  -H 'Content-Type: application/json' \
  -d '{"idx":1,"recipe":"baseline"}' | jq -r .session_id)
curl -s -X POST http://127.0.0.1:8765/api/sessions/$SID/step \
  -H 'Content-Type: application/json' -d '{"text":""}' | jq .
```

## Recipes

Recipes live under `burl/microscope/recipes/<name>/`:

| File | Purpose |
|---|---|
| `system.md` | Base system prompt. The tool protocol is appended from active ToolSpecs. |
| `play.md` | User/play prompt template. Supports simple `{{name}}` substitutions. |
| `tools.json` | Active tools, either names or objects with description/protocol overrides. |
| `params.json` | Model repo, adapter path, max tokens. |
| `tool_responses/<tool>.md` | Optional tool-response renderer for that tool. |

`tools.json` can be a list of names:

```json
["belief_trajectory", "play_brief", "commit_play"]
```

or override how a tool is described in the rendered protocol:

```json
[
  {
    "name": "play_brief",
    "protocol_role": "candidate_eval",
    "protocol_phrase": "Call `play_brief(play=X)` for each plausible legal play before committing."
  }
]
```

A tool response template can use:

```text
{{prose}}
{{structured_json}}
{{structured_json_compact}}
```

Edit recipe files, then `/burl open` the same case again to compare.

Useful starting recipes:

| Recipe | Shape |
|---|---|
| `baseline` | Minimal user prompt: pre-rendered `board_snapshot()` output plus decide text; broad tool set. |
| `legal-brief` | Forces `legal_plays()` then `play_brief()` over legal candidates. |
| `snapshot-first` | Minimal user prompt: pre-rendered `board_snapshot()` output plus decide text; no oracle/original-Burl references. |
| `snapshot-hypothesis` | `snapshot-first` plus Burl's requested `simulate_hand_impact` tool for targeted hidden-hand hypotheses. Try with `/burl open 1 --recipe snapshot-hypothesis`. |
