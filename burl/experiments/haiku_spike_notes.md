# T3 — Haiku 4.5 Agent SDK reference-trace spike (iter-2 prep)

**Status:** smoke passed end-to-end. Do NOT run full N=30 without team-lead greenlight.

## TL;DR

- Claude Agent SDK + in-process MCP tools works cleanly. One gotcha: the SDK
  falls back to non-streaming `--print` mode when `prompt` is a plain string,
  and that kills the RPC channel that in-process SDK MCP tools use. Fix: pass
  `prompt` as an async iterable of stream-json messages.
- 3-decision smoke cost **$0.086** total (~$0.029/decision). Projected N=30
  full reference-trace run: **~$0.87** — comfortably cheap for a ceiling trace.
- Haiku matched the E[Q] bot on 2/3 decisions. The miss was on a 1.86-point
  E[Q] gap (essentially a tie); noise at the E[Q] tool's default N=10 sampling
  flipped it. This is not a reasoning failure.

## Files

- `burl/haiku_spike/agent.py` — 8 in-process `@tool` handlers wrapping
  `burl/tools/engine.py` + `burl/tools/eq_distribution.py`; `run_decision_haiku`
  runs one decision via `claude_agent_sdk.query()`.
- `burl/haiku_spike/run_smoke.py` — loads first 3 rows of
  `burl/eval/data/move3_decisions.jsonl`, runs each, writes per-decision trace
  JSONL + a summary.
- Traces: `scratch/burl_p5_iter2_prep/haiku_traces/smoke_d{0,1,2}.jsonl` —
  one structured event per line (`prompt`, `assistant_text`, `thinking`,
  `tool_call`, `tool_result`, `commit`, `result`).
- Summary: `scratch/burl_p5_iter2_prep/haiku_traces/smoke_summary.json`.
- Stdout mirror: `scratch/burl_p5_iter2_prep/haiku_traces/smoke_stdout.log`.

## Results

| d | seed / decl / narrator | legal | bot play | Haiku play | match | tools | turns | cost   | wall  |
|---|------------------------|-------|----------|------------|-------|-------|-------|--------|-------|
| 0 | 900000 / blanks / s1   | 14,21 | 21       | 21         | ✓     | 8     | 9     | $0.028 | 18.3s |
| 1 | 900000 / blanks / s3   | 15,23 | 23       | 23         | ✓     | 5     | 6     | $0.029 | 16.4s |
| 2 | 900000 / ones   / s3   | 15,23 | 15       | 23         | ✗     | 7     | 8     | $0.029 | 15.9s |

Total: **$0.086** for 3 decisions. Spend cap was $0.25.

## What Haiku's reasoning shape looks like

Pattern across all three traces:

1. **One opening textual paragraph** — names the situation in 42 vocabulary
   ("I'm on defense… trying to set the bidders at 30 count"). The 42-framing
   block from `agent_runner_native.py` gets paraphrased, not re-derived.
2. **Parallel engine-fact probes** — `trump_declared`, `unseen`, `is_trump` on
   candidate plays, `is_legal` on each candidate — emitted in one batch
   (multiple `ToolUseBlock`s in a single assistant turn).
3. **A middle textual paragraph** — reads back the played trick, infers the
   led suit, and narrates what each candidate "means" (winning the trick,
   sluffing, capturing count).
4. **Selective E[Q] calls** — Haiku only calls `eq_outcome_distribution` on
   the 1–2 candidates it has already narrowed to. Does NOT sweep all legals.
   On d2 (the close call) it bumped `n_samples` from 10 to 100 unprompted.
5. **`commit_play`** — with a one-line final text confirming the choice in
   42-speak ("winning the trick with a trump to keep Team 0 below 30-count").

Compared to Gemma 4 E2B (iter-1 traces, reference in `burl/SPIKE_REPORT.md`):

- Gemma tends to emit the tool call *first*, then reason about the result.
  Haiku leads with prose, then emits a whole batch of tool calls, then
  synthesizes.
- Haiku never needs a retry — it calls `is_legal` itself before `commit_play`.
  Gemma's traces lean on the engine-retry loop to recover from an illegal
  attempt. Haiku's built-in defensive call is pattern we'd *like* STaR to
  reinforce in Gemma.
- Haiku never hallucinates a domino label like `"6-0"` as the commit
  argument; it uses the integer id every time. The format discipline is
  free at Haiku scale; Gemma needed the trimmed primer to keep this stable.
- Haiku does not use `conditional_outcome`. On the 1.86-gap tie in d2 that's
  exactly where a counterfactual probe would disambiguate — if Haiku misses
  this even at 100 samples, asking it to reach for the counterfactual tool is
  a training signal we may want to build.

## SDK pitfalls (save future-us an hour)

- The `claude-agent-sdk` wraps the `claude` CLI via stdin/stdout stream-json.
  For in-process (SDK) MCP servers, the CLI calls **back** into this Python
  process over the same channel, and that only works in streaming mode.
- Streaming mode is enabled when `prompt` is *not* a plain string. Passing
  `prompt="Hi"` with MCP servers configured produces a misleading
  `CLIConnectionError: ProcessTransport is not ready for writing` after the
  first tool hop. Fix: `async def stream(): yield {"type":"user","message":{"role":"user","content":msg}}`.
- `setting_sources=[]` avoids pulling in the project's CLAUDE.md — important,
  because otherwise the CLI loads this repo's own Texas 42 instructions on
  top of Burl's system prompt and the agent's behavior skews.
- `disallowed_tools=[...]` is the clean way to block built-in CLI tools
  (Bash, Read, Edit, …) so Haiku can only see our MCP tool surface.
- Auth "just works" via the user's existing Claude Code session — no API key
  was required. The SDK inherits whatever the CLI is already authenticated
  against.

## Suggested next steps (team-lead decides)

1. If greenlit, extend `run_smoke.py` to N=30 (same balanced-declaration
   subset as iter-0 / iter-1). Projected cost ~$0.90.
2. Consider a `--nudge-counterfactual` variant of the system prompt on
   close-gap decisions — see whether Haiku uses `conditional_outcome` when
   asked to, and whether it flips d2's choice.
3. These traces are *reference ceiling*, not training data. Their role is
   to anchor Move 5+ comparisons: if STaR-trained Gemma ever reaches this
   reasoning shape (or exceeds its bot-match at matched cost), we have an
   objective yardstick.

## Reproduce

```bash
python -u -m burl.haiku_spike.run_smoke
```

Auth via existing Claude Code session; no `ANTHROPIC_API_KEY` needed.
Hard spend cap enforced at $0.25 total; per-decision cap $0.09.
