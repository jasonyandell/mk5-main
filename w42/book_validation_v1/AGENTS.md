---
title: W42 Book Validation v1 — Agent Rules of Engagement
status: active
parent_bead: t42-4zi6
wave_0_bead: t42-snwe
---

## Mission

Take the 64-row W42 claim ledger from "evidence-on-slice" to paired
counterfactuals on real auctions or injected late states. Book validation is
the priority. Model improvements are downstream.

## Baseline (frozen 2026-05-03)

`baseline_ledger.csv` — 64 rows from
`w42/phase4_claim_completion_board/completion_board.csv`.

`baseline_audit_summary.json` — frozen audit verdict from
`w42/phase4_final_claim_audit/`.

`baseline_unresolved_or_blocked.csv` — 39 bounded follow-up rows.

Status counts at baseline:

- supported: 23
- underpowered: 21
- context-limited: 12
- not-yet-tested: 6
- contradicted: 2

Eight overclaim risks are listed in `baseline_audit_summary.json`
under `suspect_overclaims`. Do not promote any of them broadly.

## Agent contract

Every agent output (CSV/JSON/wiki page) must record:

1. **question** — what claim or sub-claim is under test.
2. **slice** — exact data subset (corpus, declarations, seeds, seat, role,
   bid value, regime).
3. **N** — decisions, actions, paired contrasts.
4. **paired or unpaired** — same-decision contrasts or pooled.
5. **metric** — regret/Q-delta/make-rate/threshold-mass; with CI where N permits.
6. **claim-ledger impact** — one of:
   - `not-yet-tested` (default; detector exists, no run)
   - `supported` (evidence on the named slice)
   - `contradicted` (evidence against on the named slice)
   - `context-limited` (true only in restricted regime)
   - `underpowered` (sample/proxy too weak for verdict)
7. **caveats** — what the evidence does *not* cover.
8. **artifacts** — paths to CSV/JSON/.pt and exact reproducibility command.

Agent must NOT:

- promote a claim to `supported` based on detector creation alone.
- generalize beyond its slice in the wiki page body.
- merge new evidence into `phase4_claim_completion_board/` or
  `phase4_final_claim_audit/`. Reconciliation is the foreground orchestrator's
  job, not the agent's.
- silently overwrite existing artifacts. Write under
  `w42/book_validation_v1/<wave>/<task_id>/`.

## Output layout per agent

```
w42/book_validation_v1/<wave>/<bd_task_id>_<short_slug>/
├── README.md            # question, slice, N, metric, status, caveats, command
├── manifest.json        # provenance + input artifact SHAs + exact command
├── summary.json         # headline numbers, machine-readable
├── <result>.csv         # row-level data
└── (any other figures or tables)
```

The corresponding wiki page lives at
`wiki/experiments/w42-bookval-v1-<wave>-<short_slug>.md` and follows the
existing `w42-phase4-*` template.

## Reconciliation gate

When all agents in a wave finish, the orchestrator:

1. Re-runs validation scripts on every artifact.
2. Diffs the agents' status proposals against `baseline_ledger.csv`.
3. Updates `phase4_claim_completion_board/` and the synthesis page in place.
4. Closes wave beads, appends `wiki/log.md`, pushes to remote.

No agent edits the central ledger or the synthesis page directly.

## Promotion guard (added 2026-05-03 after Wave 2.E.2 demotion)

A claim may move toward `supported` only on **paired same-decision
contrast evidence on the relevant action shape**. Aggregate proxies do
not qualify. Examples:

- Paired (qualifies): same `(seed, decl_id, decision_idx)` evaluated
  under two action choices; same snapshot run through `from_snapshot`
  with action A vs action B; same hand under bid=X vs bid=Y.
- Aggregate (does not qualify): per-team mean Q across all chosen
  actions in games at bid X; per-declaration mean regret across all
  decisions; cross-corpus aggregate flip rate.

The Wave 2.B.2 promotion of `ch12-setter-pounce-high-bid-off` to
`context-limited` based on aggregate Q-delta was reversed by Wave
2.E.2's snapshot-level evidence (-10.42 EV CI [-11.25, -9.59], all 4
bids contradict). The agent had flagged the proxy as insufficient; the
orchestrator promoted anyway. This guard prevents repeating that
mistake.
