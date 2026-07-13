---
title: Perf Sprint — Playbook
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

The "let's friggin rock this perf problem" entry point.

**Dormant since sprint 2 (2026-04-28).** No perf sprint has run in the ~2 months since;
frontier work has moved to jud v1 / Zeb (see [[champion]], `wiki/log.md` 2026-07-06
entries). The contract and playbook below are still the right starting point if a perf
sprint resumes — nothing here is stale, it's just unclaimed.

## Kickoff

Paste into a fresh session at the project root:

> Read [[perf-sprint]]. Run a perf sprint targeting `<GOAL>`. e.g. *"drive Burl per-decision latency from current baseline toward ~2s on this M5 Max."*

The orchestrator's first job is the kickoff sequence in "How to work" below — load deferred tools, create the team, register the `/loop` heartbeat, spawn the first worker. Skipping any step breaks the supervision loop.

## The contract

**Metric (one number, lower is better):** `wall_s_per_decision` on `perf_subset_5`, with a paired baseline run immediately before each variant. End-to-end — token generation, tool calls, harness overhead, all of it.

**Equivalence gate (binary, must pass):** `K1_match >= 60%` AND `regret_delta` within ±10% on that same paired run. If the gate fails, the row is `discard` regardless of wall.

**Wall budget (hard cap):** a single paired bench (baseline + variant) MUST complete in **≤ 10 minutes total wall time**. No exceptions. Long benches do not earn their keep — codified 2026-04-28 after sprint 2 burned hours on subset_560[:50] runs that silent-died or didn't change the lever ladder. If a config can't show a signal in 10 min, the config is too big: shrink N, not the budget. Cross-hand confirmation lives at subset_560[:10–20]; primary discovery lives at subset_5.

**Logging discipline (hard rule):** no process started by a perf-sprint worker may go **more than 60 seconds without writing a log line**. If it does, that is a high-priority bug — fix it before the next iter runs. Long silent benches set a precedent the next worker copies; we lose the ability to distinguish "dead" from "working slowly." Required for any bench/worker process: a tail-able log (`run.log` / `live.log` / `events.jsonl`) with a heartbeat or progress line at least every 60 seconds. The existing `[bench] subproc-cohort alive (...)` pings at 30s satisfy this; any new harness must do the same. If you must spawn a process that doesn't log natively, wrap it in a watchdog that does. Codified 2026-04-28.

**Cycle:** modify, run paired baseline + variant, log one row to the ledger, `keep` or `discard` (advance branch or `git reset`). See [[perf-sprint-loop]].

**Architecture:** orchestrator + one backgrounded worker at a time, in a `TeamCreate`'d team for addressability. The worker drives iterations; the orchestrator never blocks. `/loop` is the supervision heartbeat (status pings, stuck-worker recovery), not the driver. See "Architecture" below.

**Don't give up.** Crashes are work, not a stop sign. Stop only when the metric hits the target or the user types stop.

## Editable surface (likely-candidate set, not strict)

The agent is free to modify anything that helps. These are where wins typically live:

- `burl/eval/bench_decision_latency.py` (lives on the `perf/bench` worktree; bring it forward when needed) — the bench harness, batch flag, paired-baseline plumbing.
- `burl/eval/bench_batch_throughput.py` — the raw-inference reference bench.
- `burl/eval/run_move4_star_rollout_batched.py` — the batched eval harness; sync-wave loop + tool dispatch live here.
- `burl/wax_museum/harness.py`, `schemas.py` — gate state machine + tool dispatch.
- `burl/requirements-mlx.txt` — mlx / mlx-lm version pin.
- Bench flags: `--batch`, `--model-repo`, `--max-tokens`, `prefill_batch_size`.

Read source freely. Add files to this list as the territory teaches you where the wins are.

## The ledger

`scratch/<sprint>/results.tsv`, tab-separated. One row per iteration:

```
commit  wall_s  k1_match  regret_delta  peak_gb  status  description
```

- `commit` — short SHA after the variant edit.
- `wall_s` — the contract metric. Use `0.0` for crashes.
- `k1_match`, `regret_delta` — the equivalence gate values, logged so you can see why a row was discarded.
- `peak_gb` — advisory. Useful because memory headroom unlocks larger batches. Use `0.0` for crashes.
- `status` ∈ {`keep`, `discard`, `crash`}. `keep` requires `wall_s` improved AND gate passed.
- `description` — short freelance text. Hypothesis tried, what was learned, idea for next.

The TSV *is* the digest. The user wakes up, reads the keep rows, picks winners.

## How to work

**Kickoff (run once at sprint start):**

1. Load deferred tools via `ToolSearch`: `TeamCreate`, `TeamDelete`, `SendMessage`, `TaskStop`. The orchestrator can't supervise workers or clean up without these.
2. Write `scratch/<sprint>/PERF_GOAL.md` from [[perf-sprint-goal]] — sprint goal + equivalence gate values.
3. Initialize `scratch/<sprint>/results.tsv` with the header row.
4. `TeamCreate(name="<sprint>-team")` — the team is the addressability scope for workers.
5. Register the heartbeat: paste the [[perf-sprint-loop]] message (`/loop 10m` + body) verbatim.
6. Spawn the first worker into the team using the spawn template below (`run_in_background=true`).

**Steady state (driven by worker returns, not by /loop):**

- **On worker return** (synchronous, immediate): append the returned row to `results.tsv`, **`TaskStop` the returned worker by name** (sessions don't auto-release on return — over a multi-hour sprint, completed workers leak), pick the next variant from [[perf-sprint-levers]] / TSV tail / web search, spawn the next worker. Don't wait for `/loop`.
- **On `/loop` fire while worker is in flight:** `SendMessage` the worker for a one-line status, slack-update with what it says, re-anchor on `PERF_GOAL.md` and TSV tail.
- **If worker silent for 2+ `/loop` fires:** `SendMessage` once more; if no response, `TaskStop` and respawn fresh worker with the same variant. Same variant wedges twice → log a `crash` row, pick a different variant.
- **End-of-sprint cleanup:** when the sprint ends (target hit or user said "stop"), `TaskStop` the active worker if any, then `TeamDelete` the team.

**End:** sprint ends when `wall_s_per_decision` hits the target OR the user types "stop". Append a post-mortem to [[perf-sprint-history]].

## Architecture

Single-threaded loop with a supervised, backgrounded worker. The orchestrator runs for hours; it has to stay under context-degradation thresholds. It owns the TSV but **never reads bench output, source dumps, or tracebacks directly** — that work happens in worker contexts that die on return. Per-iteration cost to the orchestrator's context is ~2k tokens (TSV row + 2-sentence note) regardless of iteration weight.

`/loop` is supervisory because the worker is non-blocking — without `run_in_background`, `/loop` can't fire while the orchestrator waits for an `Agent`, and a hung worker hangs the whole sprint. With background workers, `/loop` fires regardless and pings via `SendMessage`.

Workers don't auto-release on return — backgrounded sessions persist until explicitly stopped. The orchestrator owns cleanup: `TaskStop` each returned worker by name before spawning the next, and `TeamDelete` at end-of-sprint. Without this, completed workers accumulate over a multi-hour sprint and leak sessions.

The wiki is the cross-iteration learning channel — durable findings (new levers, new traps) are written to [[perf-sprint-levers]] or [[perf-sprint-traps]] inside the worker's context before it returns, so the next worker inherits them without the orchestrator having to relay.

### Iteration agent contract

- **One coherent variant per spawn.** Coherent = one hypothesis interpretable on its own. Co-required changes (change A is meaningless without change B) ship together. Obvious blocker fixes (typos, hardcoded constants in the way, wrong import paths) ship silently as part of the variant — they don't earn their own iteration.
- **No side quests.** Interesting findings become text in the `description` column as "hypothesis for next iteration," not work this iteration completes.
- **Wiki updates are the one sanctioned side effect.** If the iteration revealed a durable lever or trap, append to the relevant page before returning.
- **Web search is sanctioned and encouraged.** Gemma 4 E2B is weeks old as of this playbook; mlx-lm ships frequently; spec-decode and continuous-batching state-of-the-art moves weekly. Reach for `WebSearch` / `WebFetch` when: a trap recipe doesn't match what you're seeing (upstream may have shipped a fix), a lever is marked "untested" or "could be already-applied" (check upstream docs/examples first), a model variant or quant set is in play (HuggingFace cards drift), or the technique is recent. Primary sources beat priors when "common knowledge" is weeks old. `WebSearch` / `WebFetch` are deferred tools — load schemas via `ToolSearch` first.
- **Owns the git decision.** The iteration does the commit, the bench run, and the `git reset` on `discard` — all in its own context.
- **Return shape is rigid.** One TSV row + 2 sentences (what was tried, hypothesis for next). Nothing else. Knowing the return shape is small forces summarization during the iteration, not after.

### Spawn template

The orchestrator reuses this verbatim, filling in `<sprint>`, `<N>`, and the variant description:

~~~
Agent({
  description: "perf iter <N>: <one-line variant>",
  team_name: "<sprint>-team",
  name: "worker-<N>",
  run_in_background: true,
  prompt: "Read wiki/playbooks/perf-sprint.md for the contract — metric,
          equivalence gate, ledger format, iteration agent contract.
          Sprint dir: scratch/<sprint>/.

          Variant to try: <description>.

          Do: modify the editable surface, git commit, run paired baseline
          + variant on perf_subset_5, parse wall_s + k1_match + regret_delta
          + peak_gb. Decide keep (advance branch) or discard (git reset).
          Append one row to scratch/<sprint>/results.tsv.

          Coherent variant — co-required changes and obvious blocker fixes
          ship together as part of this variant. No side quests. If the
          iteration revealed a durable lever or trap, update
          wiki/playbooks/perf-sprint-levers.md or -traps.md before returning.

          Read wiki/playbooks/perf-sprint-traps.md if the bench crashes —
          known recipes are there.

          If SendMessage'd mid-flight, reply with a one-line status
          (current step, % done if estimable). Don't paste output.

          Web search is sanctioned and encouraged. Gemma 4 E2B and mlx-lm
          are moving weekly; primary sources beat priors. Load WebSearch /
          WebFetch via ToolSearch and use them when an upstream changelog,
          GitHub issue, model card, or recent paper would resolve a question
          faster than re-deriving.

          Reply with that one TSV row plus exactly 2 sentences (what you
          tried, hypothesis for next). Do NOT paste bench output, source
          code, or tracebacks."
})
~~~

## Scope

Calibrated for Burl-style mlx-lm inference on Apple Silicon (Gemma 4 E2B, M5 Max). The lever ladder, trap recipes, and `decode_tok_s` thresholds assume that workload. Other perf sprints can borrow the contract and the loop discipline; fork the levers and traps.

## Links

[[perf-sprint-goal]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
