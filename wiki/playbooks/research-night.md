---
title: Research Night — Playbook
kind: playbook
first_seen: 2026-07-13
last_updated: 2026-07-13
status: active
---

The "take a registered research frontier end-to-end overnight" entry point. One
autonomous session picks up an already-selected set of research lanes, runs each
to its first gate, and files graded results to the wiki by morning. Worked
example: [[research-night-2026-07-13]].

## The contract

- **Input is a lane-selection decision, not a blank slate.** The night executes
  experiments and their order from a decision that already chose the lanes and
  their first gates ([[research-lane-selection]]); it does not pick the
  architecture.
- **Every run is pre-registered.** Predictions, uncertainty bands, and falsifiers
  are written down *before* the run, per [[partnership-research-gates]]
  ("registered before the final run"). Missed predictions are kept and reported,
  not quietly dropped — the worked example logged 5/6 round-1 and 3/4 round-2
  predictions missed, all registered first.
- **A null is a result.** Lane B's "comprehensive negative" is a graded outcome,
  not a failed night.
- **Same-night close.** Every result lands as a wiki page in the same session;
  the night does not end with un-filed runs.

## The loop

1. **Close Stage 0 first.** Measurement prerequisites — sampler correctness, the
   historical-exposure scan, two-block baseline reproduction — clear before any
   lane result is graded ([[stage-0-closure]]). Expected lane deltas are small
   enough that a silent world-distribution bias could manufacture or erase them.
2. **Register predictions.** For each lane, write the gate, the predicted band,
   and the falsifier into the experiment-page stub before launching.
3. **Run with live-tailable logs.** Each rollout writes an `events.jsonl` +
   `tail.log` layout (plus `live.log` and a `thoughts/` dir) so a check-in reads
   progress without stopping the run. Disjoint lanes run in parallel.
4. **Grade each lane against its own first gate.** Held-out ranking / marks /
   calibration as the lane's page specifies; record which registered predictions
   hit and which missed.
5. **Promote only load-bearing receipts to git.** A curated `summary.json`
   (plus pooled `.txt` receipts and plots) lands under the area's `evidence/`
   dir; game-level rows a paired-CI claim depends on go to the HF evidence
   dataset, never git; raw runs and corpora stay ephemeral
   ([[run-artifacts-policy]], [[huggingface-assets]]).
6. **File the night digest.** One `sources/` digest enumerates what the night
   established and links every experiment page written; corrections found along
   the way are carried onto the pages that held the stale claim.

## Traps

- **Post-hoc predictions don't count.** If the band wasn't written before the
  run, the run is exploratory, not a graded gate.
- **Grading before Stage 0 closes.** A lane delta measured on an unrepaired
  sampler is uninterpretable — the measurement question reopens and lane grading
  suspends.
- **Silent runs.** A rollout gone quiet is indistinguishable from a dead one;
  keep the tail-able log heartbeat live.

## Links

[[research-night-2026-07-13]] · [[research-lane-selection]] ·
[[partnership-research-gates]] · [[run-artifacts-policy]] · [[stage-0-closure]] ·
[[the-wall]]
