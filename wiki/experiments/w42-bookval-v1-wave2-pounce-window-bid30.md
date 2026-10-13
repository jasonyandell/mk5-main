---
title: W42 Book Validation v1 — Wave 2.E Pounce Window (bid=30)
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-10-12
status: complete
bead: t42-ntbe
parent_epic: t42-4zi6
---

## Summary

Paired contrast probe for the `ch12-setter-pounce` claim at bid=30.
Tests whether the oracle consistently prefers setters to "pounce" (capture trick with
count tiles exposed by the bidder) versus "decline" (play a non-winning tile).

**Verdict: withdrawn pending a sign check (#122).** The probe reads E[Q] as
Team 0's view and reinterprets it for the setter; forge documents E[Q] from the
acting player's view (`forge/eq/generate/actions.py:113,160`), and the setter is
the acting player.

N=52 paired contrasts. EV delta (pounce − decline): mean +3.09, CI95
[−0.57, +6.75], p=0.104: null whichever way the sign falls. The "p_make vs
E[Q] split" (oracle pounced in 59.6% of positions while "pounce better by
E[Q]" was 34.6%) is what an inverted sign produces: 34.6% is the complement
of 65.4%. It is not a finding. Subgroups have N ≤ 27, and the 10-count slice
N=5.

See [[winning42-ch05-setter-defense]] and [[winning42-ch12-advanced-bidding-playing]].

---

## Question

When a bidder-team tile worth ≥5 count points appeared in the most recently completed
trick, and the setter is now following in the current trick with both winning and
non-winning options available, does pouncing (winning the trick) improve the setter's
outcome vs. declining (letting the trick go)?

---

## Slice

| Field | Value |
|-------|-------|
| Source | 500 oracle-greedy snapshots from legacy corpus |
| Declarations | All 10 |
| Role | Setter (player 1 or 3), following position |
| Bid value | 30 (corpus default; no bid-aware context) |
| Filter | Bidder team exposed ≥5 count pts in last completed trick |

---

## N Breakdown

| Category | N |
|----------|---|
| Total snapshots | 500 |
| Single-legal (forced play) | 217 (43.4%) |
| Setter leading next trick | 106 (21.2%) |
| Setter following, ≥2 legal | 208 (41.6%) |
| Pounce-only (no decline option) | 14 |
| Decline-only (cannot win trick) | 142 |
| **Paired (can pounce AND decline)** | **52** |

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| N paired contrasts | 52 |
| EV delta (pounce − decline) | +3.09 |
| EV delta CI95 | [−0.57, +6.75] |
| p-value (one-sample t vs 0) | 0.104 |
| p_set delta (proxy) | 0.00 [−0.12, +0.12] |
| Pounce better by E[Q] | 34.6% (18/52) |
| Oracle chose pounce | 59.6% (31/52) |

Sign convention as recorded: delta = pounce_e_q − decline_e_q, read as Team 0
perspective, negative = pounce better for setter. That reading is unverified
(#122); if E[Q] is acting-player perspective, positive = pounce better.

---

## Slice Results

| Slice | N | Mean EV delta | CI95 | Pounce better | Oracle pounce |
|-------|---|--------------|------|---------------|---------------|
| Overall | 52 | +3.09 | [−0.57, +6.75] | 34.6% | 59.6% |
| Count=0pts | 38 | +2.35 | [−1.87, +6.57] | 39.5% | 60.5% |
| Count=5pts | 9 | −0.77 | [−7.05, +5.51] | 33.3% | 44.4% |
| Count=10pts | 5 | **+15.68** | [+1.60, +29.76] | **0.0%** | 80.0% |
| Phase early | 20 | **+7.96** | [+2.67, +13.26] | 20.0% | 75.0% |
| Phase mid | 27 | +0.68 | [−4.10, +5.46] | 48.1% | 44.4% |
| Phase late | 5 | −3.37 | [−20.11, +13.38] | 20.0% | 80.0% |
| Bidder team led | 36 | **+4.48** | [+0.67, +8.29] | 30.6% | 63.9% |
| Setter team led | 16 | −0.03 | [−8.29, +8.22] | 43.8% | 50.0% |

Bold = CI excludes zero.

---

## Reading

Every slice's direction depends on the unverified sign (#122), and the overall
contrast is null. The subgroup CIs that exclude zero (10-count N=5, early N=20,
bidder-led N=36) are post-hoc slices of 52 pairs with no multiplicity
correction. Labels come from the learned Q network (qMAE 0.94 points), in
points, at bid 30, on the April corpus later found 27–67% malformed
([[otis-phase-r]]).

---

## Claim-Ledger Impact

None. The page's original `context-limited` reading and its "p_make/EV split"
are withdrawn pending #122.

---

## Caveats

1. This probe uses stored oracle E[Q] from corpus decisions, not fresh inference.
   The oracle that played the corpus maximized p_make, not EV.
2. N=52 paired contrasts; subgroup slices have N≤9 — interpret cautiously.
3. Bid=30 only. No bid-aware context in legacy corpus. High-bid behavior deferred.
4. Filter is broad: 43.4% of positions had only one legal move (no contrast possible).

---

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-ntbe_pounce_window_bid30/
├── README.md
├── manifest.json
├── summary.json
└── paired_contrasts.csv  (500 rows)
```

Input corpus: `w42/book_validation_v1/wave2/snapshots/pounce_window/snapshots.jsonl`
(SHA256: `a45352211e2a7ed7e78f9b03ae4515c6ef672a0f64ea31202eb57477774cd4b0`)

---

## Related Pages

- [[winning42-ch05-setter-defense]] — reckless count overfire context
- [[winning42-ch12-advanced-bidding-playing]] — pounce instruction source
- [[w42-phase4-claim-completion-board]] — baseline ledger
- [[w42-claim-analysis-synthesis-report]] — Wave 1 synthesis

---
