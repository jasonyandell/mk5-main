# wax_museum — hard-gated HATEOAS tool surface

This project's knowledge lives in the wiki — see `wiki/entities/wax-museum.md`.

## Run

```bash
# Plumbing check — stub model, no Modal.
source .venv/bin/activate
python -u -m burl.wax_museum.run_pilot --model stub --n 2

# Base Gemma 4 E2B, local MLX-LM, hits 5/5 bot-match on held-out trick-6 decisions.
python -u -m burl.wax_museum.run_pilot --model local --n 5 --run-id my_gemma_run --no-live-stdout

# Base Qwen3.6-35B-A3B-4bit, local MLX-LM. Slow (big <think> blocks) but coherent.
python -u -m burl.wax_museum.run_pilot --model qwen --n 5 --run-id my_qwen_run --no-live-stdout

# Modal L4 (needs deploy first).
modal deploy burl/wax_museum/modal_serve.py    # first time only
python -u -m burl.wax_museum.run_pilot --model modal --n 5
```

## Tail the run

```bash
# Human-readable event stream, one line per event.
tail -f burl/wax_museum/logs/<run_id>/tail.log

# Full completion for a single turn (the actual reasoning).
cat burl/wax_museum/logs/<run_id>/thoughts/d0_t1.md

# Machine-readable for later analysis.
jq . burl/wax_museum/logs/<run_id>/events.jsonl | less
```

`summary.json` + `decisions.jsonl` hold the per-run rollup.

## Auditing new backends

Any new model requires a three-step audit (the bug that ate 145+ decisions would
have been caught in 60s of this):

1. **Render a fixture.** `tokenizer.apply_chat_template([system, user, assistant with tool_call, tool_response])` with `tokenize=False`. Grep the output for expected content.
2. **Check the tool-call syntax** the base model emits zero-shot. Different models want different shapes (`<|tool_call>…<tool_call|>`, `<tool_call><function=…>`, `[{"name":…}]`).
3. **Verify `role` handling.** Some templates drop `role="tool"` entirely; some require it. Test both `role="tool"` and `assistant.tool_responses` and see which one appears in the rendered prompt.

If the content isn't in the serialized prompt, the experiment isn't measuring
what you think it's measuring.
