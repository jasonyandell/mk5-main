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

## Review (second pass, 2026-07-07)

- The core correction stands: Wave 2.E.2 (wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md, bead `t42-8kbh`, frontmatter status `contradicted`) did run; decline-better is amplified, not reversed, at bids 35/36/39/42; every per-bid EV CI excludes zero (results table, t-stats -11.7 to -12.8). The bead reattribution is also right: `t42-8kbh` is the probe bead; the atlas page frontmatter carries `bead: t42-6j3k` (2.B) and `wave2b2_bead: t42-7eop`.
- Amended one imprecision in the auditor's replacement text: it said the probe ran "on the Wave 2.B bid-aware atlas", but the probe's data source (seeds 9000-9049 × 10 decls × 4 bids, n_samples=200/decision — see the probe page's Data Source section) is the **Wave 2.B.2 full-sweep** corpus (bead `t42-7eop`); the 2.B build was the 5-seed + seed-9430 smoke. The original page's "Wave 2.B.2's bid-aware corpus" was correct on that sub-point. Page now reads "Wave 2.B.2 full-sweep bid-aware atlas ([[w42-bookval-v1-wave2-bid-aware-atlas]], bead `t42-7eop`)". Evidence: wiki/experiments/w42-bookval-v1-wave2-bid-aware-atlas.md (Wave 2.B.2 Full Sweep section).
- Spot-checked all "Verified" bullets above: Phase 4 (+3.028 Q/1866 pairs, +2.681 Q/368, 75,079 rows), Wave 2.E bid=30 (52/500, 59.6%, 34.6%→65.4% decline-better, +15.68 CI [+1.60, +29.76], 80%), void-creation lead (n=276, CIs [-0.028,-0.003]/[-3.42,-1.84]) and follow (n=500, +0.77 CI [+0.12,+1.42], 52.6%), Wave 1 (83/100, 45 left/38 right, ~30%/~37%, 9.18 over 2,300, 38-43 worst, beads `t42-v0m5`/`t42-btpg` in .beads/issues.jsonl), and book line ranges in the main tree's scratch/winning42/winning42.with_figures.md — all confirm.
- Both Follow-ups suggestions kept: the Claim Ledger pounce rows still cite only Phase 4 (no Wave 2.E/2.E.2), and the `t42-br7n.1` row cites the bead without a page link.
- One nuance the page's "contradicted" bolding elides (acceptable, since the sentence scopes it to the high-bid slice): the probe's own status proposal is `contradicted` per-bid but `context-limited` overall, pending a narrower late-game probe and the setter-team-led subsample (39.6% pounce-better, N=280).
