---
title: "Source: a8bc35a"
kind: source
commit: a8bc35a
date: 2026-04-21
author: Jason Yandell
---

## Commit message

> docs(gus): GEN_FLEET.md — distributed diverse-seed corpus gen plan
>
> Plan for scaling joint-world corpus generation across a Vast.ai fleet
> with HuggingFace as durable storage. Pattern mirrors forge/zeb/vast/.
>
> Architecture: stateless workers run on interruptible 3090s/4090s, each
> covers a static seed range, pushes chunks directly to a public HF
> dataset repo (jasonyandell/gus-42-worlds). Local M5 Max pulls chunks
> and trains. No learner loop, no coordination service — embarrassingly
> parallel at the seed-range level.
>
> Prerequisite: --n-decl-per-seed flag in the generator (oracle trained on
> 10 decls/seed; Gus currently does 1/seed → 100× fewer states per unit
> compute).
>
> Cost envelope: ~$15 for an 8-worker 7.5-hour run covering 10k diverse-
> seed games. Cuts local wall time from ~100 hours to hours.
>
> Phasing:
>   0. Generator harden (--n-decl-per-seed flag, chunk naming)
>   1. Fleet-of-1 end-to-end (1 worker, small seed range, validate)
>   2. Scale fleet (4-16 workers via vast_monitor.sh)
>   3. Local consumer (periodic HF pull + retrain trigger)
>
> Not a learner loop. Not a coordinator service. Not a training fleet.

## Links

[[entities/gen-fleet]] · [[gus]]
