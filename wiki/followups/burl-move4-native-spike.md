Reviewed against code on 2026-07-07 — no issues found.

- Results table matches commit 3781dce message verbatim; `burl/eval/run_move4_spike.py`, `burl/harness/agent_runner_native.py`, and `burl/harness/tool_loop_native.py` all exist and match described behavior.
- "Move 4 production default: 7" retries confirmed (`run_move4_star_rollout.py:773`, `run_move4_star_rollout_batched.py:662`); spike default is 3 as implied.
- Per-decision spike result JSONs are not in `burl/eval/results/` (only later perf benches) — the numbers rest on the commit message and SPIKE_REPORT.md at 3781dce; a cheap follow-up would be to check the archived SPIKE_REPORT if raw traces are ever needed.
