---
title: Zeb fleet-ops — the self-healing Vast.ai spot-GPU cluster
kind: entity
first_seen: 2026-02-07
last_updated: 2026-02-15
status: complete
---

## What it is

`forge/zeb/vast/` is a self-healing Vast.ai spot-GPU fleet-management stack, built to keep
[[zeb]]'s self-play training running at a scale one dev machine couldn't sustain — the
training curve needed roughly 1M+ self-play games (`a7d6b5b`, 2026-02-06) and later 1.7M+
(`forge/zeb/models/large-belief-recap.md`, W&B run `waxffg2j`). Jason's framing of the
spend, which he would not make for paid work: *"I'd never do this for work. ever. but for
the hobby project? let's gooo"* (2026-02-08T19:41,
[[era4-zeb-era|conversation digest]]).

## Components (all present on disk today)

- `vast_monitor.sh` (1,051 lines) — the autonomous fleet manager, first commit `bcc549f`
  (2026-02-07 23:22).
- `vast_up.sh` / `vast_down.sh` / `vast_status.sh` / `vast_replenish.sh` — lifecycle
  scripts.
- `fleet.py` (181 lines), `reputation.py` (196 lines), `market_scanner.py`,
  `find_offer.py` — machine selection and reputation logic.
- A `Dockerfile` for fast worker boot, and `RUNBOOK.md`.
- `docs/zeb-worker-saga.md` (added `751c290`, 2026-02-15) — a live incident-report doc for
  standing up experiment `lb-v-eq-3740`.

## HF Hub as the worker/learner exchange bus

N independent self-play workers and one learner process needed to stay in sync on model
weights and training examples without a shared filesystem or a message queue. The answer:
route everything through Hugging Face Hub as the single source of truth (`5e3b80c`,
2026-02-07). This immediately created a rate-limit problem against HF's 128-commits/hour
ceiling: naive per-batch uploads from 4 workers projected ~960 commits/hour. `25f62e4`
(2026-02-07 00:25) fixed this by folder-batching example uploads into one commit per ~60s
interval, cutting the rate to ~4/hr per worker; `7117809`/`f73f3b5` walked the interval up
further (180s → 240s) as fleet size grew.

## Machine reputation scoring

`7d0746e` (2026-02-10) adds `reputation.py`, persisting per-machine observed-gps, boot
reliability, boot times, stalls, and errors. The formula, stated in the commit itself:
**"Effective gps = observed_avg × reliability (min 3 samples to trust)."** `db55993`/
`b100e47` (2026-02-09) add preferred-machine selection on top of the scores, plus a Docker
image for faster worker boot.

## CQRS monitor rewrite

`f8e5e28` (2026-02-08 22:41) — "CQRS monitor architecture, parallel launches, smart
downscale": background pollers write local status files; the main loop reads those files
instead of making serial `vastai logs` calls per instance. Commit claims the check cycle
drops from **60-100s to ~2s regardless of fleet size** (commit-message-sourced; no separate
benchmark artifact was located). `84cd4f8` (2026-02-09) extracts offer-selection into its
own module (`find_offer.py`). Additional same-window hardening: `70ed0ae` detects broken
Vast launch scripts, `cee336e` adds stall detection for silently-dead workers, `18df653`
switches fleet-value decisions to observed (not advertised) GPS, `1a28d9a` adds a
projection-based monitor with startup seed + warmup.

## The worker saga (2026-02-15)

`docs/zeb-worker-saga.md` records a live incident report standing up `lb-v-eq-3740` (1
learner + 2 self-play workers + 1 eval-aux worker, bootstrapped from `large-belief.pt` at
HF step 3740): Vast hosts stuck in `loading` requiring repeated culling/replacement; an
eval-aux worker crash (`No module named forge.zeb.worker.run_eval_aux`) traced to a launch
path that only tried module-mode execution on some cloned runtime images, fixed same day
with a script-path fallback in both `vast_monitor.sh` and `vast_up.sh`. Eval-aux is named
directly as *"the fragile leg right now."* `286aa97` (same evening) fixes a duplicate-
monitor bug via a PID file guard.

## Terminal status

BUILT and dormant-but-intact. The entire stack exists on disk today; `forge/zeb/` last
received a commit 2026-07-06 (`838f48d`). Fleet-ops never got its own closeout commit the
way [[full-teacher-eq-experiment]] did — its fate is tied to Zeb's: parked as [[burl]]'s
belief primitive 2026-04-18, superseded by [[gus]]/[[belief-trajectory]] 2026-04-23 (see
[[zeb]]), with the fleet-ops code itself untouched by either event.

Fleet-ops is infrastructure, not a finding about [[candlewax|the wall]] — it made the
[[full-teacher-eq-experiment]] affordable to run repeatedly, but ruled nothing in or out
about consuming E[Q] itself.

See [[zeb]] · [[eval-matrix-bradley-terry]] · [[era4-zeb-era|conversation digest]].
