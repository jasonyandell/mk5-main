# Followups: winning42-ch12-advanced-bidding-playing

Reviewed against code/artifacts on 2026-07-07.

## Corrections

- page said the high-bid pounce extension "waits for the bid-aware corpus from Wave 2.B.2"; it has since run and contradicted the claim at all four high bids, which the page's own Claim Ledger already recorded (evidence: wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md, wiki/log.md ~line 2433). Updated the Wave 2 Findings paragraph to close the internal inconsistency. Non-substantive — the ledger row was already correct.

## Verified

- Pounce-high-bid numbers (n=1,140; EV delta -10.42 CI [-11.25, -9.59]; p_set delta -0.047; pounce-better 21.6-26.6% per bid) match wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md.
- Bid=30 numbers (52 paired contrasts from 500 candidates; oracle pounced 59.6%; decline better under EV 65.4%; 10-pt subgroup n=5, +15.68 CI [+1.60, +29.76], oracle pounced 80%) match wiki/experiments/w42-bookval-v1-wave2-pounce-window-bid30.md.
- 66 / 384 natural-bucket rows (t42-br7n.7) matches wiki/experiments/w42-phase4-bidding-count-exposure-tests.md.
- Source slice file scratch/winning42/winning42.with_figures.md exists in the main working tree (gitignored, absent from this worktree's scratch — expected).
- All [[wiki links]] resolve to existing pages.

## Follow-ups

- The page's own suggestion at the end of Wave 2 Findings — re-classify each `context-limited` row by which objective it survives under — is partially done by w42-bookval-v2-utility-lens-synthesis; a one-line pointer from each context-limited ledger row to its utility-lens verdict would be a cheap consistency pass.
- Line-number anchors into scratch/winning42/winning42.with_figures.md are unverifiable from the repo (gitignored source); consider quoting a short anchor phrase per row so anchors survive re-OCR.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. The single edit (Wave 2 Findings closing paragraph) is confirmed: wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md is bead `t42-8kbh`, Wave 2.E.2, bids 35/36/39/42, status `contradicted` (n=1140; EV delta -10.42 CI [-11.25, -9.59]; p_set delta -0.047; pounce-better 21.6-26.6% per bid), matching the page's Claim Ledger row and wiki/log.md line 2433. The edit's addition of bid 39 and removal of the stale "Wave 2.B.2" attribution are both correct (2.B.2 was the aggregate-proxy pass whose promotion the paired contrast reversed). Bid=30 numbers re-checked against wiki/experiments/w42-bookval-v1-wave2-pounce-window-bid30.md; 66/384 re-checked against wiki/experiments/w42-phase4-bidding-count-exposure-tests.md. All [[links]] resolve; both Follow-ups suggestions kept (w42-bookval-v2-utility-lens-synthesis exists, so "partially done" is accurate).
