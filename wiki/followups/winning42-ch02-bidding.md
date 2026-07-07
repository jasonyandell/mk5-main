# winning42-ch02-bidding — audit 2026-07-07

## Corrections

- Page said Wave 2.G mark_ev deltas were "+0.07 to +0.15"; the source table shows +0.048 to +0.146 — the 35↔36 step (+0.048) falls below the claimed floor (evidence: wiki/experiments/w42-bookval-v1-wave2-ch02-multistep.md per-step table).
- Page said "Cohen d grows monotonically 0.16 → 0.47"; d is not monotone in step order (0.186 at 30↔32, dips to 0.158 at 35↔36, then 0.472 at 39↔42). What is monotone/uniform is the book-direction sign — no reversals. Reworded both the ledger row and the Wave 2.G update paragraph (evidence: same table + Monotonicity section).

## Verified clean

- Phase-4 ledger numbers (+0.037471, +0.074942, +0.292901, 0.336857, 16.774838%) match wiki/experiments/w42-phase4-bidding-count-exposure-tests.md.
- Wave 2.B.2 numbers (259,618 rows, N=14,000, paired deltas −0.076 mark_ev / −0.038 p_make) match wiki/experiments/w42-bookval-v1-wave2-bid-aware-atlas.md.
- Multistep N range 8,168–10,052, 85/85 slice cells, 0.81% transitive mismatch all match the multistep page.
- scratch/winning42/winning42.with_figures.md exists in the main checkout (absent in this worktree, as the page itself notes).

## Follow-ups

- Bead IDs (t42-ni1l.2, t42-br7n.7, t42-ey88) are unverifiable here (bd retired 2026-06); a cheap pass could rewrite them as plain text references or link to the archived .beads/issues.jsonl.
- The 35↔36 step's small delta (+0.048, count-point-only threshold shift) is a good cheap probe for whether the overbid penalty is driven by threshold mass rather than bid magnitude.
