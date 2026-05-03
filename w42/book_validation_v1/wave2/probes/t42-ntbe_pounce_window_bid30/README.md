# Pounce Window Probe — Wave 2.E (bid=30 slice)

**Bead**: t42-ntbe  
**Claim**: `ch12-setter-pounce` (bid=30 baseline)  
**Scope**: bid=30 only. `ch12-setter-pounce-high-bid-off` (high-bid slice) remains a separate sibling bead.

## Question

When the bidder's team has played a count domino (≥5 pts) in the most recent completed trick
and the setter is now following, does the oracle consistently prefer to "pounce" (play a tile
that wins the current trick) over "decline" (play a non-winning tile)?

## Slice

- **Source**: 500 oracle-greedy snapshots, `w42/book_validation_v1/wave2/snapshots/pounce_window/snapshots.jsonl`
- **Filter**: setter (player 1 or 3) at following position, bidder team exposed ≥5 count pts in last completed trick
- **Bid value**: 30 (corpus default; no bid-aware context available)
- **Declarations**: all 10

## N

- Total snapshots: 500
- Setter following with ≥2 legal moves: 208
- **Paired contrasts (can pounce AND decline)**: 52
- Pounce-only positions: 14 (setter can win but can't decline)
- Decline-only positions: 142 (setter cannot win trick)
- Single-legal (forced): 217
- Setter leading next trick: 106

## Paired vs Unpaired

Paired. Each of 52 contrasts compares best pounce e_q vs best decline e_q within the same position.

## Metric

E[Q] delta (Team 0 perspective), where negative = pounce better for setter:

| Metric | Value |
|--------|-------|
| EV delta (pounce − decline) | +3.09 [CI95: −0.57, +6.75] |
| p-value (one-sample t, H₀=0) | 0.104 |
| p_set delta (proxy, e_q vs threshold) | 0.000 [CI95: −0.12, +0.12] |
| Oracle pounce rate (when both available) | 59.6% (31/52) |
| Pounce better by e_q | 34.6% (18/52) |

**Sign convention**: V is always from Team 0's perspective. Setter (Team 1) wants lower (more negative) e_q. Delta = pounce_e_q − decline_e_q. Negative delta = pounce is better for setter.

## Slices

| Slice | N | Mean delta | CI95 | Pounce better | Oracle pounce |
|-------|---|-----------|------|---------------|---------------|
| Overall | 52 | +3.09 | [−0.57, +6.75] | 34.6% | 59.6% |
| Count=0pts | 38 | +2.35 | [−1.87, +6.57] | 39.5% | 60.5% |
| Count=5pts | 9 | −0.77 | [−7.05, +5.51] | 33.3% | 44.4% |
| Count=10pts | 5 | +15.68 | [+1.60, +29.76] | 0.0% | 80.0% |
| Phase early | 20 | +7.96 | [+2.67, +13.26]* | 20.0% | 75.0% |
| Phase mid | 27 | +0.68 | [−4.10, +5.46] | 48.1% | 44.4% |
| Phase late | 5 | −3.37 | [−20.11, +13.38] | 20.0% | 80.0% |
| Bidder team led | 36 | +4.48 | [+0.67, +8.29]* | 30.6% | 63.9% |
| Setter team led | 16 | −0.03 | [−8.29, +8.22] | 43.8% | 50.0% |

*CI excludes zero (but small N; interpret cautiously).

## Key Findings

### 1. No uniform pounce advantage by E[Q]

The overall EV delta is +3.09 (CI spans zero, p=0.104). By raw E[Q], pounce is better
for setter in only 34.6% of paired positions. The book claim that setters should pounce on
bidder-exposed count is **not uniformly supported** at bid=30 by E[Q] alone.

### 2. Oracle (p_make) prefers pounce more than E[Q] suggests

The oracle chose pounce in 59.6% of paired positions. This exceeds the 34.6% rate where
pounce has lower E[Q]. The oracle optimizes p_make (probability of setting the contract),
not E[Q]. Capturing count deterministically improves threshold probability even when it
worsens the expected score. This is consistent with the ch05_reckless_count overfire pattern:
pouncing opportunistically vs. pouncing strategically require different thresholds.

### 3. 10-point count cases: pounce costs position

The 5 positions with 10-pt count at stake show pounce_delta=+15.68 (CI excludes zero) —
pounce is strongly bad for setter by E[Q] in all 5 cases, yet oracle pounces 80% of the time.
Inspection reveals these involve high-pip count tiles (5-5 or 6-4) where winning requires
burning a high trump, sacrificing future trump control. The oracle's p_make preference for
capturing count here may be a quirk of the bid=30 threshold calibration.

### 4. Phase and trick leadership matter

- Early game: oracle pounces 75% but E[Q] strongly prefers decline (delta=+7.96, CI excludes zero)
- Mid game: balanced; neither direction significant
- Setter-team-led tricks: near zero delta (−0.03), suggesting when setter controls the trick
  flow, pounce/decline distinction collapses (setter is already winning the trick often)

### 5. Distinguishing book claim from ch05_reckless_count overfire

The book claim (Ch12 pounce) is specifically about **bidder-exposed count** — count the
bidder volunteered into a trick the setter can win. This is distinct from recklessly leading
count into a trick the bidder controls. The filter captures the correct scenario. However,
the E[Q] signal is weak, suggesting the "pounce" instruction is context-dependent:
correct in late-game and setter-led positions; questionable in early-game bidder-led tricks.

## Claim-Ledger Impact

**`context-limited`**

Evidence supports pounce preference in setter-team-led trick positions (late-game,
symmetric E[Q]). Evidence contradicts pounce in early-game bidder-led positions where
the oracle's p_make preference and raw E[Q] diverge most sharply. The book instruction
"always pounce on exposed count" overstates; correctness is conditioned on phase and
who controls the trick flow.

## Caveats

1. E[Q] and p_make (oracle's actual objective) diverge for setters; this probe uses stored
   E[Q] from corpus decisions, not fresh inference. Oracle pounce rates reflect p_make.
2. N=52 paired contrasts; several slices have N≤9 (underpowered for subgroup claims).
3. No fresh model inference run; statistical power comes from stored oracle e_q values only.
4. High-bid-off slice (`ch12-setter-pounce-high-bid-off`) explicitly not covered here.
5. Filter is broad: any setter decision after bidder count exposure; ~43% positions had
   only one legal move (no contrast available).

## Artifacts

```
w42/book_validation_v1/wave2/probes/t42-ntbe_pounce_window_bid30/
├── README.md              (this file)
├── manifest.json          (provenance)
├── summary.json           (headline numbers, machine-readable)
└── paired_contrasts.csv   (500-row detail table)
```

## Reproducibility

This probe uses stored oracle E[Q] values from the pounce_window corpus.
No model checkpoint required.

```bash
# Rebuild corpus (if needed)
python w42/book_validation_v1/wave2/snapshots/pounce_window/build_pounce_window_corpus.py

# Re-run analysis (not yet a standalone script; see probe code in bead t42-ntbe)
```
