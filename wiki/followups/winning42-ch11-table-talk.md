# Follow-ups: winning42-ch11-table-talk

Reviewed against code on 2026-07-07.

## Corrections

- Page said the harvest registered "14 measurable hypotheses"; the concept table on the page itself has 13 rows (evidence: wiki/experiments/winning42-ch11-table-talk.md, concept table).

## Verified

- Book slice `scratch/winning42/winning42.with_figures.md` lines 4332-4484 exists in the main repo and does contain Chapter 11 "Talking Across the Board" (start at page 87, ends before Chapter 12 at page 91).
- Bead `t42-ni1l.11` exists in `.beads/issues.jsonl` with matching title and the same line range.
- The "64-row ledger" reference matches [[w42-phase4-final-claim-audit]] (64 ledger rows / 64 claim ids).
- Grep confirms no `table_talk_leakage` or `trace_public_evidence_faithfulness` detectors exist in the codebase — the "phantom plan" verdict stands.

## Follow-ups

- `renege_immediate_detection` remains the cheapest live probe: it's deterministic from engine legality and needs only injected illegal-play fixtures.
- If Burl STaR traces are ever re-filtered, `trace_public_evidence_faithfulness` is a one-afternoon lint (regex for partner-hand claims + tool-citation check) with direct training-data value.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Concept table has exactly 13 rows (page lines 61-73), so 14→13 was correct; book slice, bead `t42-ni1l.11`, 64-row ledger, and absent-detectors claims all re-verified against `scratch/winning42/winning42.with_figures.md`, `.beads/issues.jsonl`, and `wiki/experiments/w42-phase4-final-claim-audit.md`.
