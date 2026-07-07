Reviewed against code on 2026-07-07 — no issues found.

Verified: enumeration counts (8,288,280 evaluations), duplicate-exposure 86.354%/97.964%, strong-trump-trap 2.927%/20.121%, four-five-off 17.857%/25.000%, best-candidate <=10 points 66.971% (8.059+23.557+35.355), side-protection 17.751%, bid-only-enough follow-up (1224 counterfactual rows, 1104 contrasts, -0.111555 / -0.223109 deltas, 1040/64/0 split, sha 18a5b94), and auction-pressure run (32 deals, 384 contracts, 608 contexts, 16800 rows, 12872/2104/0 split, partner overcall 77/192 = 40.104%) all match summary.json/CSV artifacts in w42/.

- A cheap next probe: the page notes 30/31/35/36 never appear in the static ceiling proxy; a small trick-loss-aware ceiling (rank coverage per off) could test whether natural buckets emerge without full auction logs.
- The bid_margin_by_context_summary.csv and opponent_pressure_summary.csv artifacts exist but are not cited in the Findings table — a one-row opponent-pressure finding would round out the phase-3 slice.
