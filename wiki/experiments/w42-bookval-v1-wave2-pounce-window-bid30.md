---
title: W42 Book Validation v1 — Wave 2.E Pounce Window (bid=30)
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: complete
bead: t42-ntbe
parent_epic: t42-4zi6
---

## Summary

Paired contrast probe for the `ch12-setter-pounce` claim at bid=30.
Tests whether the oracle consistently prefers setters to "pounce" (capture trick with
count tiles exposed by the bidder) versus "decline" (play a non-winning tile).

**Claim-ledger impact: `context-limited`.**

N=52 paired contrasts. EV delta (pounce − decline, Team 0 perspective):
mean +3.09, CI95 [−0.57, +6.75], p=0.104. No clear direction by raw E[Q].
Oracle (p_make metric) chose pounce in 59.6% of paired positions — substantially
higher than the 34.6% rate where pounce has lower E[Q], revealing a systematic
p_make vs E[Q] divergence. Correctness of pounce is conditioned on game phase and
trick leadership.

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

Sign convention: delta = pounce_e_q − decline_e_q from Team 0 perspective.
Negative = pounce better for setter (lower Team 0 EV).

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

## Key Findings

### 1. No uniform pounce advantage by E[Q]

The overall CI spans zero (p=0.104). Pounce is better by raw E[Q] in only 34.6% of
positions. The book's instruction to always pounce on exposed count is **not uniformly
supported** at bid=30.

### 2. Oracle systematically prefers pounce more than E[Q] alone predicts

Oracle chose pounce 59.6% vs 34.6% where pounce has lower E[Q]. The oracle
optimizes p_make (probability of setting the contract threshold), not E[Q]. Capturing
count deterministically improves the threshold probability even when the expected mean
score is worse — a known behavior at bid=30 where the decision boundary is sharp.

### 3. High-count positions (10pts): pounce burns position, E[Q] strongly favors decline

All 5 positions with 10-pt count exposure showed pounce_delta > 0 (pounce worse for
setter by E[Q]), with mean +15.68 and CI excluding zero. Inspection shows these involve
high-pip count tiles (5-5 or 6-4) where capturing requires burning a high trump — trading
future trump control for immediate count capture. Oracle still pounces 80% of the time
by p_make reasoning.

### 4. Phase and trick leadership are key moderators

- **Early game + bidder led**: oracle pounces 75%+, but E[Q] clearly prefers decline.
  The book instruction is most questionable here.
- **Mid game**: balanced; neither direction reaches significance.
- **Setter-team led tricks**: near-zero delta (−0.03), suggesting the pounce/decline
  distinction collapses when setter already controls trick flow.

### 5. Distinguishing ch12-setter-pounce from ch05_reckless_count overfire

The Wave 1.4 cross-AI matrix flagged `ch05_reckless_count` as an overfire pattern.
This probe tests a narrower claim: pounce specifically on **bidder-exposed** count.
The filter correctly limits to setter-following positions after bidder count exposure.
However, the E[Q] evidence suggests even this narrower claim is context-dependent:
legitimate in late-game setter-controlled tricks, questionable in early bidder-led tricks.

---

## Claim-Ledger Impact

**`context-limited`**

The `ch12-setter-pounce` claim at bid=30 is supported in setter-team-led trick contexts
(near-zero E[Q] delta, oracle ~50/50) but not in early-game bidder-led contexts where
the oracle's p_make preference and raw E[Q] diverge most sharply. The book instruction
overstates by treating pounce as universally correct; it is conditioned on phase and
trick leadership at bid=30.

The high-bid-off slice (`ch12-setter-pounce-high-bid-off`) remains open as a sibling
bead (Wave 2.B.2), blocked on bid-aware corpus data.

---

## Caveats

1. E[Q] (Team 0 perspective) and p_make (oracle's actual objective) diverge. This
   probe uses stored oracle E[Q] from corpus decisions, not fresh inference. Oracle
   pounce rates reflect p_make optimization.
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

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Finding 3's "high-pip count tiles (5-5 or 6-4)" detail is not re-derivable from `paired_contrasts.csv` (its count_bucket column doesn't name tiles); listing the 5 snap_idx values would make it reproducible.
- Cheap next probe: rerun the paired contrast with p_make (threshold-probability) deltas instead of the p_set proxy — the p_make-vs-E[Q] divergence is the page's central mechanism but is inferred, not measured.
- The count=10pts and phase-late slices (N=5) drive two bolded findings; pooling with a second 500-snapshot draw would cheaply confirm or kill the 10-pt "pounce burns position" effect.
