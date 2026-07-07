# Follow-ups: winning42-ch05-setter-defense

Reviewed against code and result artifacts on 2026-07-07.

## Corrections

- Page said the high-bid pounce scope (`ch12-setter-pounce-high-bid-off`) "waits for Wave 2.B.2's bid-aware corpus (bead t42-8kbh)"; the probe has since run as Wave 2.E.2 and returned **contradicted** — decline-better is amplified, not reversed, at bids 35/36/39/42 (evidence: wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md). Also, t42-8kbh is the probe bead, not the corpus bead (corpus is Wave 2.B/t42-6j3k, 2.B.2/t42-7eop).

## Verified

- Book-source line citations (2314-2341 etc.) match `scratch/winning42/winning42.with_figures.md` in the main working tree (scratch/ is gitignored, so absent in this worktree — expected).
- Phase 4 numbers (+3.028 Q / 1866 pairs, +2.681 Q, 75,079 rows) match w42-phase4-sequence-handshape-tests.md.
- Wave 2.E pounce numbers (n=52 of 500, 59.6%, 65.4%, +15.68 CI [+1.60, +29.76], 80%) match w42-bookval-v1-wave2-pounce-window-bid30.md.
- Void-creation lead (n=276, contradicted, CIs) and follow (n=500, +0.77 CI [+0.12, +1.42], 52.6%) match their pages.
- Wave 1 findings (83/100 divisive setter seats, 45 left/38 right, ~30%/~37% asymmetry, regret 9.18 over 2,300 fires, 38-43 worst-case regret, beads t42-v0m5/t42-btpg) match w42-book-claim-synthesis-and-ai-directions.md, the wave-1 pages, and .beads/issues.jsonl.

## Follow-ups

- Consider adding the Wave 2.E.2 high-bid contradiction to the Claim Ledger table itself (the pounce row still reads only "context-limited" via Phase 4).
- The `t42-br7n.1` ledger citation could link to the phase4 page directly for traceability.
