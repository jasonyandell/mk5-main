Reviewed against code on 2026-07-07 — no issues found.

Verified: `gus/train/train_pi_opp.py` exists and matches the description (frozen v3 trunk, 3-way seat embedding, `oracle_softmax_per_seat` targets, 3 pairs per batch item, AdamW + cosine schedule, NaN fix at the `masked_fill(~legal_mask, 0.0)` line); commits 93859a0/b4e8ecd/8106f01/dcd9365 exist with matching messages; 68.6% confirmed in the 8106f01 commit message and `gus/MORNING4_STATUS.md`.

- MORNING4_STATUS notes that even at 68.6% oracle match, lamir1-piopp rollouts were *worse* than direct π_me — the page's "useful for LAMIR-1 rollout quality" claim is aspirational; a one-line pointer to that null could strengthen the page.
- Cheap next probe: report per-seat accuracy (L-opp vs partner vs R-opp) — partner prediction likely differs and matters most for signaling.
