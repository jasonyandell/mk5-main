---
title: Burl Move 3 — Base Gemma 4 E2B Baseline
kind: experiment
first_seen: 4b3ba3d
last_updated: 4b3ba3d
status: active
---

## Summary

Base [[gemma-4-e2b]] (no fine-tuning) on Burl's XML tool-call harness, evaluated on 10 held-out trick-6 decisions. The premise survives: a 2B model plays legally at 70% K1 via tool-use, zero training required. But tool-use breadth (the key Burl differentiator) does not emerge spontaneously on XML.

([burl/eval/run_move3.py @ 4b3ba3d](../sources/4b3ba3d.md))

## Setup

- **Model:** base [[gemma-4-e2b]], zero fine-tuning
- **Harness:** XML tool-call format (`<tool>`, `<commit>`) via `burl/harness/agent_runner.py`
- **Eval set:** 10 of the 50 held-out decisions from `burl/eval/decision_dataset.py` — balanced across 10 declarations, trick-6, `|legal|>=2`, `eq_gap>=1.0`, seeds ≥900000
- **Infra:** [[modal]] L4, `burl/modal/gemma_serve.py` with `threading.Lock` around vLLM V1's sync `generate()` (ZMQ race under Modal's threaded dispatch)
- **Cost:** $0.09

## Results

| Metric | Value |
|---|---|
| Legal rate | 100% |
| Bot-match rate | 60% |
| K1 (P(E[Q] >= bot)) | 70% |
| Illegal plays | 0 |
| Retries | 0 |

## Tool-use pattern

Only `is_legal` called. Never reaches for `eq_outcome_distribution`, `conditional_outcome`, `unseen`, `void_audit`, or `trump_declared` — the distribution tools that are Burl's whole point.

Fake `play` tool hallucinated in 80% of trials. Harmless (commit still lands via actual XML commit path), but signals the harness format is not Gemma's trained grammar.

## Infrastructure notes

- Cold start: 107s; warm: 48–56s; throughput: 9–11 tok/s
- `enable_thinking` is a no-op on Gemma 4's Jinja template — `max_tokens` budget is what actually controls thinking-channel consumption
- `RetryExhausted` raises loudly (carries `.trace`) rather than silently falling back — preserves diagnostic signal

## Significance

Premise survives: base 2B model plays legally at 70% K1 with no fine-tuning. But the tool-use BREADTH that is Burl's whole point — querying `eq_outcome_distribution`, incorporating beliefs about hidden state — does not emerge spontaneously on XML format. This motivates [[experiments/burl-move4-native-spike]]: the harness format, not the model, is the bottleneck.

## Related pages

[[burl]] · [[gemma-4-e2b]] · [[tool-orchestration]] · [[k1-grading]] · [[modal]] · [[decisions/native-tool-use-format]] · [[experiments/burl-move4-native-spike]] · [[sources/4b3ba3d]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; 1 correction applied in place and independently re-verified.

- The move3 results directory (`burl/eval/results/move3_<timestamp>/`) is not in the repo; metrics trace only to the 4b3ba3d commit message and [[4b3ba3d]]. Committing or archiving summary.json would make the numbers independently checkable.
- `burl/eval/data/move3_decisions.jsonl` (the dataset path in `run_move3.py`'s usage string) is absent from the worktree; regenerating it or noting it as generated-on-demand would help reproduction.
