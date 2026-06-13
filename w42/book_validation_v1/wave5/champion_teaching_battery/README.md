# Champion Teaching Battery — Wave 5

## Question

Does the champion (GusBidder + GusPointsEvaluator auction, lens:ev play) obey the
Winning 42, 2nd ed. tactical claims in its own play trajectories? This is the
teaching-half question: not "is the book right in general?" but "does the champion
play the way the book says?"

## Slice

- **Seeds:** 128 deal seeds (0–127), both arena halves (champion as team 0 and team 1)
- **Games:** 256 total (128 seeds × 2 halves)
- **Decisions:** 7,168 decisions × 7 candidate actions = 19,264 action rows
- **n_worlds:** 10 sampled worlds per decision (fast-path; variance is higher than
  production runs at n=50 or n=1000, but sufficient for paired contrasts)
- **Claims tested:** 6 ch04/ch05 tactical play claims (GUS_TACTICAL_SPECS replication)
- **Model:** `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`

## N

| Object | Count |
|--------|-------|
| Games | 256 |
| Decisions | 7,168 |
| Action rows | 19,264 |
| Labeled action rows | 1,484 |

## Metric

Paired same-decision contrast: within each decision, best-by-mean labeled action vs.
best-by-mean unlabeled action. Bootstrap CI (1,000 iterations, percentile). A claim
is `supported-on-slice` when the CI is strictly above zero (labeled action better);
`contradicted` when strictly below zero.

Obey rate = fraction of decisions where champion's actual action is the best labeled
candidate (i.e., the champion independently arrives at the book recommendation).

## Verdict per claim

| claim_id | obey_rate | N_paired | delta (pts) | 95% CI | verdict |
|----------|-----------|----------|-------------|--------|---------|
| ch05-pounce-count | 0.593 | 117 | +4.93 | [+3.26, +6.73] | **supported-on-slice** |
| ch05-pounce-count-before-certainty | 0.604 | 87 | +3.95 | [+2.24, +5.72] | **supported-on-slice** |
| ch05-extra-count-to-set | 0.654 | 23 | +10.00 | [+4.65, +15.79] | **supported-on-slice** |
| ch05-reckless-count-to-bidder | 0.317 | 550 | -7.66 | [-8.62, -6.77] | **contradicted** (negative control: reckless is worse) |
| ch04-safe-partner-count-donation | 0.571 | 150 | +0.05 | [-1.12, +1.20] | within-CI |
| ch04-unsafe-partner-count-donation | 0.320 | 201 | -9.21 | [-10.96, -7.69] | **contradicted** (negative control: unsafe is worse) |

**Headline:** 3 of 6 checkable claims supported; 0 underpowered; 2 contradicted as
expected (negative controls — reckless/unsafe count plays are worse, confirming the
champion avoids these). 1 within CI (safe partner donation, ~0 delta, claim is
context-limited).

## Caveats

1. **Self-selection bias (critical):** The champion plays its own lens:ev trajectories.
   Tactical scenarios arise only when the game reaches those states under champion play.
   The battery does NOT measure whether the book's advice is correct for players who
   arrive at those states via different play — only whether the champion agrees.

2. **n_worlds=10:** Low world count inflates per-decision variance. Paired contrasts
   aggregate across 117–550 decisions, which provides sufficient power for the
   observed effect sizes (delta 4–10 pts), but per-game estimates are noisy.

3. **Promotion guard:** Paired contrasts are present for all claims. However, the
   central ledger (`phase4_claim_completion_board/completion_board.csv`) must NOT be
   updated by this probe. Reconciliation is the orchestrator's job.

4. **ch04-safe-partner-count-donation (within-CI):** The near-zero delta suggests the
   book's preference for safe count donation is partially correct on this slice but
   the champion's deviation has roughly equal value — context-limited finding.

5. **ch05-extra-count-to-set N=23:** Small paired N; CI is wide. Supported but
   borderline on N.

## Command

```bash
python -u w42/book_validation_v1/wave5/probe_champion_teaching_battery.py \
    --n-seeds 128 --n-samples 10 --device mps \
    --out-dir w42/book_validation_v1/wave5/champion_teaching_battery \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt
```

## Artifacts

| File | Description |
|------|-------------|
| `action_rows.csv` | 19,264 per-action rows with Q stats and labels |
| `receipts.csv` | Per-claim obey rate, delta, CI, verdict |
| `summary.json` | Machine-readable headline + all receipts |
| `manifest.json` | Provenance: repo SHA, exact command, input row count |
| `harness_output/` | Standard harness artifacts (label_metrics, paired_contrasts, etc.) |
| `../probe_champion_teaching_battery.py` | Probe script |

## Utility coverage (AGENTS.md wave3 requirement)

Per-action rows include all 5 utility lenses:
- `ev_setter` / `ev_bidder` — scalar EV from q_per_world mean (split by team)
- `p_make_setter` / `p_make_bidder` — P(mark won)
- `mark_ev_setter` / `mark_ev_bidder` — E[mark utility]
- `cvar_10_setter` / `cvar_10_bidder` — 10th-percentile tail
- `robust_q25_setter` / `robust_q25_bidder` — 25th-percentile robust quantile
