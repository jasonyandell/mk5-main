---
title: "Source digest: ceca203 — iter-5 E1 + E2 writeups"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** ceca20332dd10a2e218edcb836de1df338407967
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> docs(burl): iter-5 E1 + E2 writeups — truncation reframe, candlewax null
>
> E1: rank-16 70%, rank-64 55.6%, rank-128 0% (catastrophic). Truncation
> was the iter-4 bug, not capacity. Next lever: N=100+ at rank-16.
> E2: candlewax bimodal hints land cleanly; 0 conditional_outcome calls
> in E3 rollout (N=500). Blocker is model policy, not tool surface.
> Writeups include "What I would NOT conclude" sections.

Documentation commit landing two experiment writeups. Full analysis: [[experiments/iter5-e1-rank-sweep]] and [[experiments/iter5-e2-candlewax-null]].

## Related pages

[[experiments/iter5-e1-rank-sweep]] · [[experiments/iter5-e2-candlewax-null]] · [[preserve-thoughts]] · [[candlewax]] · [[burl]]
