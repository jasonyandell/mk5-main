---
title: Commit Discipline (primer as behavioral scaffold)
kind: decision
first_seen: 2026-04-19
last_updated: 2026-04-19
status: superseded
---

## Observation

The full LEM rules primer was acting as a load-bearing behavioral scaffold for `commit_play` emission — not just a rules reference. This was revealed by two ablations in [[experiments/burl-iter1-mixed]]:

1. **Drop primer entirely:** 0% wins, 50% retry-exhausted on 6 decisions. Killed immediately.
2. **Trim primer to ~500 words:** commit discipline partially recovers in rollout (0 retry-exhausted during N=30 corpus generation), but the SFT adapter inherits the depth-without-commit pattern and goes 5/10 retry-exhausted on held-out eval.

The primer was teaching Gemma to "structure your turn and commit" as much as it was teaching rules. Removing it removes that structure along with the rules content.

## Implications for iter-2

Three candidate mitigations:

1. **Keep full primer** — accept eq-shy pathology (iter-0 lesson) but maintain commit discipline.
2. **Trim primer + add explicit protocol line** — keep ~500-word primer, add "you MUST emit `commit_play` to end your turn" to the system prompt. Separates behavioral scaffold from rules content.
3. **Drop primer, rely on tool schema + protocol** — re-harvest from spike v2 prompt shape (no framing, no primer). Would need explicit `max_turns` limit.

**Resolved the following week**: [[iter3-rules-adapter]] (dbadb5f) shipped option 3's
spirit — rules-as-tools, no primer — at 90% bot-match / 0 retry-exhausted. See
[[rules-as-tools]].

## Generalizable principle

Prompt components can carry behavioral load that is hard to isolate from their apparent purpose. Removing a component reveals which behaviors depended on it. When ablating a prompt component, check behavioral metrics (retry-exhausted rate, completion rate) separately from quality metrics (bot-match on completed).

## Related pages

[[burl]] · [[decisions/primer-tradeoff]] · [[experiments/burl-iter1-mixed]] · [[experiments/burl-iter0-eval]] · [[burl-iter1-adapter]]
