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

- Bead IDs (t42-ni1l.2, t42-br7n.7, t42-ey88) can no longer be resolved via `bd` (retired 2026-06) but all three are present in the archived .beads/issues.jsonl; a cheap pass could rewrite them as plain text references or link to the archive.
- The 35↔36 step's small delta (+0.048, count-point-only threshold shift) is a good cheap probe for whether the overbid penalty is driven by threshold mass rather than bid magnitude.

## Review (second pass, 2026-07-07)

- Both first-pass corrections stand: per-step table in wiki/experiments/w42-bookval-v1-wave2-ch02-multistep.md confirms deltas +0.048…+0.146 (original "+0.07 to +0.15" excluded the 35↔36 step) and Cohen d 0.186/0.299/0.158/0.337/0.472 — non-monotone in step order, so the reworded "no reversals, d 0.16–0.47" is right.
- Amended the ledger row's pre-existing "(decl × seat × phase × step)" parenthetical to "(5 step pairs × [10 decls + 4 seat roles + 3 phases])" — the old wording implied a 600-cell cross-product; the source defines 85 marginal slices (evidence: w42-bookval-v1-wave2-ch02-multistep.md, Slice Analysis).
- Amended this followup's bead-ID line: t42-ni1l.2, t42-br7n.7, t42-ey88 all exist in the archived .beads/issues.jsonl, so they are archive-verifiable, not unverifiable.
- Verified-clean items re-checked: phase-4 ledger numbers match w42-phase4-bidding-count-exposure-tests.md (lines 55–59); Wave 2.B.2 numbers (259,618 rows, N=14,000, −0.076 mark_ev / −0.038 p_make, threshold_mass borderline) match w42-bookval-v1-wave2-bid-aware-atlas.md (lines 190–262); OCR source absent in this worktree, present at /Users/jason/code/mk5-main/scratch/winning42/winning42.with_figures.md.
- Note: the source page w42-bookval-v1-wave2-ch02-multistep.md itself says Cohen d "grows" in its Monotonicity section while its own table shows the 35↔36 dip; that page was out of scope here but could use the same rewording.
