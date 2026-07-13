---
title: logged-arrows
kind: topic
first_seen: 2026-05-02
last_updated: 2026-05-02
status: superseded
---

## What It Is

[[burl-lab]] uses logged arrows for harness steps that need both durable traceability and composition. A logged arrow has the shape:

```python
Trace[O] = (events: tuple[Move, ...], output: O | None)
```

Read it as `input -> optional output + logs`. Composition concatenates `events`; when `output is None`, the next arrow is skipped but the events already produced remain durable.

## In Burl Lab

`burl/lab/core/arrow.py` defines `Trace`. Phase handlers implement `async handle(state, move) -> Trace[str]`, where `Trace.events` are journalable Moves and `Trace.output` is an optional next phase name.

The server is the interpreter. It appends each returned Move to `events.jsonl`, re-folds state from the journal, streams those same Moves over SSE, and only then routes on the optional phase output. Phase handlers do not call `append()` or `fold()`.

## Why It Matters

The first `burl/lab` implementation preserved event sourcing and rendered `ToolSpec` prompts but left orchestration procedural: phases appended directly, server code owned some semantic reconstruction, and the algebra was implicit. Logged arrows restore the intended composition rule without changing the journal format.

## Status

Downstream of [[burl-lab]], which went dormant 2026-05-07 and was superseded by
[[jud]]. The design remains correct as documented; it is not carried
forward into the current architecture.
