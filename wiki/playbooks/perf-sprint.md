---
title: Perf Sprint — Playbook
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

The "read this and let's friggin rock this perf problem" entry point. Distilled from the [[burl-perf-phase0]] → [[burl-perf-phase3]] sprint and the wrap that should not have happened.

## Kickoff (one line, fresh session)

To start a sprint, paste this into Claude in the project root:

> Read [[perf-sprint]]. Run a perf sprint targeting `<GOAL>`. e.g. *"drive Burl per-decision latency from current baseline toward ~2s on this M5 Max."*

Or, once the slash command exists: `/perf-sprint <GOAL>`.

## What the orchestrator does on kickoff

1. Read this playbook and every doc it links to.
2. Establish the goal anchor from [[perf-sprint-goal]] — write `scratch/PERF_GOAL.md` with the sprint's target, anchor metric, and quality bar.
3. Spawn scribe-D first ([[perf-sprint-scribes|Phase 0 — measurement]]). Wait for the baseline ledger row + the bench harness.
4. Spawn scribes B + A in parallel (Phases 1 + 2). Then C (Phase 3) when A completes.
5. Register the loop from [[perf-sprint-loop]] verbatim.
6. Run Phase 4 (full-N attribution) once the stack is locked.
7. NEVER wrap before one of the three wrap conditions below.

## The wrap conditions (non-negotiable)

Wrap is permitted **only** when one of these is true:

1. Goal achieved on full-N attribution **and** the user approves wrap, OR
2. Every lever in [[perf-sprint-levers]] has been tried with a ledger row recorded for each, OR
3. The user has explicitly typed `stop`.

Crashes, contention, ambiguous results, "the bench is unreliable" — none of these are wrap conditions. They are work to do. See [[perf-sprint-traps]] for recipes.

The first sprint failed at the wrap step because the loop message only said "check in with slack-style updates." When the bench crashed twice, "wrap and write the digest" felt like discipline rather than what it actually was: giving up. These three conditions exist to close that rationalization door.

## Mandatory rules

- **One GPU = one benchmarker.** Scribes touch GPU only when team-lead has confirmed exclusivity. `bench.lock` file convention (write before run, delete after; refuse to start if it exists). Detection signal for missed contention: `decode_tok_s` < 100. See [[mlx-cohort-bench-discipline]].
- **Paired protocol.** Every variant runs with a fresh baseline IMMEDIATELY before it. Never compare against "latest baseline in ledger" — that's the footgun that produced two false 1.66× / 2.1× claims in the first sprint.
- **K1 + regret are primary; wall is suggestive on small subsets.** Sub-2× wall claims on a 5-row bench are noise. Use full-N for absolute attribution.
- **Pre-flight smoke test.** Every scribe runs one paired baseline+variant before touching their lever. If the bench crashes there, fix the bench first. The first sprint hit Phase 4 crash modes 80% through a long run that could have been caught at minute 5.
- **Honest reversals are bonus points.** If a claim doesn't survive scrutiny, retract it loudly and in-session. See [[burl-perf-phase1]] (1.66× → 0.3%) and [[burl-perf-phase2]] (2.1× → tie) for canonical form.
- **Wiki as you go.** Every meaningful finding becomes a wiki page or extension in the same session. The wiki growing is a first-class output.

## Team shape

Four scribes, one orchestrator, one worktree per scribe. Worktree convention: `.claude/worktrees/<sprint>-{bench,cheap,batch,aggressive}` off the working branch.

| Scribe | Phase | Role | Blocks |
|---|---|---|---|
| scribe-D | 0 | Measurement: bench, frozen subset, baseline | Everyone |
| scribe-B | 1 | Cheap wins: budgets, dispatch, no model changes | — |
| scribe-A | 2 | Big lever: continuous batching, KV-cache | — |
| scribe-C | 3 | Aggressive: quantization, spec-decode | A |

Spawn each in research-only mode first (no GPU). Convert to active mode only when team-lead grants GPU exclusivity. C in particular benefits from doing all upfront research while A holds the GPU — turns idle time into structural-finding time.

## When the sprint ends

Append a short post-mortem to [[perf-sprint-history]]: target, outcome, what worked, what didn't, what changed in the playbook. The playbook is meant to grow; each sprint feeds the next.

## Links

[[perf-sprint-loop]] [[perf-sprint-goal]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[mlx-cohort-bench-discipline]] [[harvest-cohort-abstraction]] [[continuous-batching-dispatcher-design]] [[perf-on-the-table]]
