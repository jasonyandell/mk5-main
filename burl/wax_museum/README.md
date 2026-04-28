# wax_museum — hard-gated HATEOAS tool surface

A minimal pilot to answer one question: if we **force** a model to call `explore_game` → probe → commit (via a turn-indexed tool menu), does it start reasoning about outcome distributions instead of pip-counting its way to a commit?

**Supports both Gemma 4 E2B (local MLX-LM) and Qwen3.6-35B-A3B-4bit (local MLX-LM).** The harness is backend-pluggable: each model supplies its own completion parser and tool-response shape (see `--model` + `harness.py::run_decision_waxed`'s `parse_completion` / `tool_response_style` kwargs).

## What the pilot found

1. **The `conditional_outcome = 0/145` observation was a harness bug, not a model ceiling.** The shared Burl tool harness was packing tool responses into `role="tool"` messages. Gemma 4's chat template silently **drops** those — every probe response was `{value:None}` in the actual prompt Gemma saw. Every Burl adapter was trained on traces where tool outputs were invisible to the model. See [`../PRACTICALITIES.md`](../PRACTICALITIES.md) §9.
2. **With the fix (tool responses live on `assistant.tool_responses=[{name, response}]` for Gemma), base Gemma 4 E2B hits 5/5 bot-match and quotes tool values verbatim.**
3. **Qwen3.6 does something qualitatively different**: derives 42 rules from scratch inside a multi-thousand-token `<think>` block, arrives at the answer before calling the tool, and treats the tool as confirmation. See `logs/n5_qwen/live.log` for a preserved turn-1 trace. See [`../PRACTICALITIES.md`](../PRACTICALITIES.md) §10.

## The bet

On turn 1 only `explore_game` is in the tool menu. After it fires, `probe_best_case` / `probe_worst_case` / `ask_rule` appear. Only after ≥1 probe does `commit_play` appear. HATEOAS-style — each tool response includes a `next_actions` block announcing what's now reachable.

## Files

- [`HYPOTHESIS.md`](HYPOTHESIS.md) — the explicit prior before the pilot ran. Read this first, then compare to the actual traces.
- `schemas.py` — tool JSON schemas + `GateState` state machine (INITIAL → AFTER_EXPLORE → AFTER_PROBE).
- `tools.py` — HATEOAS tool implementations. Thin wrappers over `eq_outcome_distribution` + `conditional_outcome` + `what_would_change_my_mind`. Each tool returns `{prose, structured, next_actions}`. Prose renders as an ASCII histogram + if/then scenarios + pivot-synthesis line; the model quotes it.
- `harness.py` — per-decision loop. Enforces the gate, detects turn-1 silent bail, handles commit legality + retry. **Backend-pluggable** via `parse_completion` + `tool_response_style` kwargs so Gemma and Qwen can share it.
- `qwen_parser.py` — Qwen3.6-specific completion parser (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` + Hermes JSON fallback).
- `modal_serve.py` — Modal L4 endpoint, 32k context, base Gemma 4 E2B (no LoRA). Distinct app name from `burl.modal.gemma_serve_native` so it can coexist.
- `run_pilot.py` — N=5 orchestration with tail-able logs. `--model` selects backend (`stub` / `local` = Gemma / `qwen` / `modal`).

## Run

```bash
# Plumbing check — stub model, no Modal.
source .venv/bin/activate
python -u -m burl.wax_museum.run_pilot --model stub --n 2

# Base Gemma 4 E2B, local MLX-LM, hits 5/5 bot-match on held-out trick-6 decisions.
python -u -m burl.wax_museum.run_pilot --model local --n 5 --run-id my_gemma_run --no-live-stdout

# Base Qwen3.6-35B-A3B-4bit, local MLX-LM. Slow (big <think> blocks) but coherent.
python -u -m burl.wax_museum.run_pilot --model qwen --n 5 --run-id my_qwen_run --no-live-stdout

# Eagerly precompute the bounded per-decision tool lattice.
python -u -m burl.wax_museum.run_pilot --model local --n 5 --eager-tool-cache

# Modal L4 (needs deploy first).
modal deploy burl/wax_museum/modal_serve.py    # first time only
python -u -m burl.wax_museum.run_pilot --model modal --n 5
```

`--eager-tool-cache` is only a latency/cache layer. `menu_for` and `advance`
still decide which tools the model is allowed to call on each turn.

## Tail the run

```bash
# Human-readable event stream, one line per event.
tail -f burl/wax_museum/logs/<run_id>/tail.log

# Full completion for a single turn (the actual reasoning).
cat burl/wax_museum/logs/<run_id>/thoughts/d0_t1.md

# Machine-readable for later analysis.
jq . burl/wax_museum/logs/<run_id>/events.jsonl | less
```

## What the logs show

- `tail.log` — tail-friendly. Per decision, per turn: menu composition, tool calls, menu transitions, commit attempts.
- `events.jsonl` — structured events (`turn_start`, `completion`, `tool_call`, `menu_change`, `gate_reject`, `commit_attempt`, `bail`, `decision_end`).
- `thoughts/d<N>_t<T>.md` — full raw model completion for each turn. This is where you actually read Gemma's reasoning and see whether 32k bought us legibility.
- `summary.json` + `decisions.jsonl` — the per-run rollup.

## Bail behavior

- **Per-decision**: if turn 1 emits no tool call AND the thought is <200 chars, the decision is marked bailed and we skip to the next one.
- **Run-level**: if two consecutive decisions bail, the run stops. The hypothesis behind the hard gate is dead in that case — Gemma isn't engaging with the gate at all.

## What to look for

Per [`HYPOTHESIS.md`](HYPOTHESIS.md), the original success signals were:

1. ≥1 probe call on ≥3/5 decisions (moves the 0/145 number off zero).
2. Turn-1 thoughts that reference spike_drivers vocabulary ("disaster branch", "if partner holds the 5-5").
3. At least one decision where the committed play is different from the unconditional argmax because of what a probe revealed.

Failure signals:

- Two consecutive silent-turn-1 bails (hard gate is alien).
- Model explores but never probes — commits immediately after `explore_game`.
- All probes are `probe_best_case` (cargo-culting the menu).

## Auditing new backends

Any new model requires a three-step audit (the bug that ate 145+ decisions would have been caught in 60s of this):

1. **Render a fixture.** `tokenizer.apply_chat_template([system, user, assistant with tool_call, tool_response])` with `tokenize=False`. Grep the output for expected content.
2. **Check the tool-call syntax** the base model emits zero-shot. Different models want different shapes (`<|tool_call>…<tool_call|>`, `<tool_call><function=…>`, `[{"name":…}]`).
3. **Verify `role` handling.** Some templates drop `role="tool"` entirely; some require it. Test both `role="tool"` and `assistant.tool_responses` and see which one appears in the rendered prompt.

If the content isn't in the serialized prompt, the experiment isn't measuring what you think it's measuring.
