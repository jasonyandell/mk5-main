---
title: "Source digest: 4b3ba3d — Move 3 shipped — base Gemma 4 E2B 70% K1"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** 4b3ba3dbb5559e467a15bc868e0729d82d20e87c
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): Move 3 shipped — base Gemma 4 E2B clears 70% K1 on held-out decisions
>
> Result: 100% legal, 0 retries, 60% bot-match, 70% P(E[Q] ≥ bot), $0.09 run.
> Base model, zero fine-tuning, 10 held-out decisions. Premise survives.
>
> Landed via a four-teammate team (burl-move3):
>
> modal/ — gemma_serve.py on L4 vLLM, threading.Lock for ZMQ race.
> harness/ — agent_runner.py, retry.py, tool_loop.py with RetryExhausted carrying .trace.
> eval/ — decision_dataset.py (50 held-out), run_move3.py (grades all metrics).
>
> GEMMA_4_ERGONOMICS.md — canonical Gemma 4 tool-use format, probe evidence,
> ranked R1-R5 fix options. Produced by a sibling research agent.
>
> OVERVIEW.md — full post-pivot refresh. Zeb parked. E[Q] N=10 distribution
> is Burl's belief primitive. Architecture + tool surface + moves updated.
>
> Beads: t42-56gu, t42-oyq9, t42-jljb, t42-2ap0, t42-ozju, t42-ng3f,
> t42-8grw, t42-snam, t42-zry5, t42-qaxg — all closed.
> t42-rr6q (Move 4) created for next session.

## Files introduced / modified

| Path | Change |
|---|---|
| `burl/modal/gemma_serve.py` | New, 169 LOC. vLLM `@app.cls` on L4; `threading.Lock` around sync `generate()` for ZMQ race; cold 107s, warm 48–56s, 9–11 tok/s |
| `burl/harness/agent_runner.py` | New, 388 LOC. Stitches remote model + 7 tools + retry into `run_decision() → BurlTrace`. Model-agnostic |
| `burl/eval/decision_dataset.py` | New, 428 LOC. 50 held-out decisions: balanced across 10 declarations, trick-6, `|legal|>=2`, `eq_gap>=1.0`, seeds ≥900000 |
| `burl/eval/run_move3.py` | New, 792 LOC. Orchestrates rollouts; grades legal/bot-match/K1/retry/tool-use/tokens; writes `report.md` + `summary.json` + `traces.jsonl` |
| `burl/GEMMA_4_ERGONOMICS.md` | New, 185 LOC. Canonical Gemma 4 tool-use format reference; probe evidence; ranked R1-R5 fix options |
| `burl/OVERVIEW.md` | Heavily revised: post-pivot refresh, Zeb parked, E[Q] distribution promoted, architecture current |
| `burl/harness/retry.py` | Modified: `RetryExhausted` now carries `.trace` |
| `burl/harness/tool_loop.py` | Minor fixes |

## Move 3 findings

- Base model emits XML zero-shot when prompt is unambiguous.
- Only calls `is_legal`. Never reaches for distribution tools.
- Hallucinates `play` tool 80% of time — harmless but signals format mismatch.
- `enable_thinking` is a no-op on Gemma 4's Jinja template.
- Four-teammate agent team (burl-move3) used for parallel landing.

## Results

100% legal, 60% bot-match, 70% K1, $0.09. See [[experiments/burl-move3-base]] for full analysis.

## Related pages

[[experiments/burl-move3-base]] · [[burl]] · [[gemma-4-e2b]] · [[modal]] · [[tool-orchestration]] · [[sources/d9baf3b]] · [[sources/3781dce]]
