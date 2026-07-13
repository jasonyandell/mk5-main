---
title: Beads Retired — GitHub Issues Are the Tracker
kind: decision
first_seen: 2026-06-14
last_updated: 2026-07-13
status: active
---

## Decision

Beads (`bd`) is retired. Durable work items are tracked as **GitHub issues via
`gh`** on `jasonyandell/mk5-main`, milestone **"Champion"** for the current push.
The old beads archive stays grep-able at `.beads/issues.jsonl` as read-only
history, not a live queue. Scratch to-do lists (TodoWrite/TaskCreate style) are
not durable work tracking — file an issue instead.

## Why

`bd` was uninstalled in 2026-06 and the champion action ladder moved to GitHub
issues at the champion-rung ingest: *"beads is retired (2026-06)… the champion
action ladder lives in GitHub issues (milestone 'Champion')… The wiki carries
information; GitHub issues carry action"* (log-archive.md, 2026-06-12). The
docs→wiki consolidation (`b89ff635`, 2026-07-11) then rewrote CLAUDE.md,
AGENTS.md, and README.md wiki-first, formalizing "beads → GitHub issues."

## Links

[[champion-ladder]] · [[the-wall]]
