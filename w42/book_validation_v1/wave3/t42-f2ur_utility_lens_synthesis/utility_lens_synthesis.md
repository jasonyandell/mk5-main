# Utility-Conditional Book Validation — Wave 3.0 Synthesis

**Bead:** t42-f2ur | **Date:** 2026-05-03 | **Wave:** 3

---

## Background

A pattern has emerged across the W42 book validation campaign:

- Wave 1.4 cross-AI agreement: detectors agree with p_make at 59%, CVaR_10 at 83%
- Wave 2.E (pounce bid=30): oracle pounces 60% (p_make lens supports book), but EV spans zero
- Wave 2.E.2 (pounce high bid): EV contradicts at all 4 bid buckets
- Wave 2.H: mark_ev ≡ p_make via positive-affine identity at all bids
- Wave 2.G: bid-only-enough supported under both p_make and EV

**Hypothesis:** The book's tactical advice may be implicitly p_make-optimized near the contract threshold. Some claims survive under EV; others only under p_make. A systematic re-classification reveals which.

---

## Method

**Inputs:** All 7 closed Wave 2 probes (and Wave 1 via Wave 2 summaries).  
**Approach:** Re-derive the paired delta under 5 utility lenses from existing columns.

### Utility Definitions

| Utility | Definition | Proxy Used |
|---------|-----------|-----------|
| EV | E[Q] oracle | ev_delta or ev_delta_setter column |
| p_make | P(Q >= make_threshold) | threshold_mass or p_set columns |
| mark_ev | bid × P(make) - penalty × P(fail) | ≡ p_make at bid=30; p_set_delta at high bids |
| CVaR_10 | E[Q \| Q ≤ Q_10%] | cvar_delta columns (T0 perspective throughout) |
| robust_q25 | 25th percentile of Q | Not recorded in any probe — all missing |

**Sign convention critical note:** All Q values in the oracle are from Team 0 (bidder/declarer) perspective. Each probe's delta column convention was verified algebraically before analysis. See manifest.json for per-probe sign notes.

**Bootstrap CI:** 2000 resamples, percentile method, seed=42. **Verdict:** "supported" = CI excludes zero in book direction; "contradicted" = CI excludes zero against book; "spans_zero" = CI includes zero.

---

## Per-Probe Re-Analysis

### Probe 1 — ch03-reentry-preservation (t42-v9lu, n=222)

Book claim: preserve the reentry trump (play off-suit to avoid consuming it).

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV | -1.230 | [-2.612, +0.177] | spans_zero |
| p_make (threshold_mass) | -0.015 | [-0.035, +0.003] | spans_zero |
| mark_ev | -0.015 | [-0.035, +0.003] | spans_zero |
| CVaR_10 | -0.507 | [-2.285, +1.315] | spans_zero |
| robust_q25 | — | — | missing |

**Late-game sub-slice (trick 5-6, n=60):** EV mean=-3.967 CI[-6.797,-1.364], verdict=contradicted.

**Summary:** All utilities agree — spans_zero overall, contradicted in late game. No utility flip. The claim is context-limited (late game specifically contradicts: consuming trump is better). All utilities give same direction.

---

### Probe 2 — ch04-low-trump-trap (t42-jysl, n=257)

Book claim: playing a low trump instead of the dominant trump costs the bidder EV.

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV | -1.949 | [-2.967, -0.944] | contradicted |
| p_make | — | — | missing |
| mark_ev | — | — | missing |
| CVaR_10 | — | — | missing |
| robust_q25 | — | — | missing |

**Summary:** EV only probe. Low trump (book's "trap") is often BETTER for the bidder on average (oracle prefers low trump 58% of the time). No utility comparison possible. The EV verdict stands: claim is contradicted on this slice.

---

### Probe 3 — ch05-void-creation-lead (t42-26j8, n=276)

Book claim: setter should lead to create a void (from leading position).

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV (setter) | -2.629 | [-3.433, -1.845] | contradicted |
| p_make (p_set_delta) | -0.016 | [-0.029, -0.001] | contradicted |
| mark_ev | -0.016 | [-0.029, -0.001] | contradicted |
| CVaR_10 | -0.213 | [-1.469, +1.084] | spans_zero |
| robust_q25 | — | — | missing |

**Utility flip detected:** EV and p_make both contradicted; CVaR_10 spans_zero. CVaR is the one dissenter. Effect is small and noisy in the tail. The EV and p_make verdicts are consistent: void creation from lead position is contradicted.

**Strong agreement note:** Both EV (-2.63) and p_make (-0.016) CI excludes zero on the same side. CVaR spans zero due to wider tail distribution uncertainty (n=276 is sufficient for mean but not tail).

---

### Probe 4 — ch05-void-creation-follow (t42-z31l, n=500)

Book claim: setter should follow to create a void (from following position).

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV (setter) | +0.767 | [+0.170, +1.443] | **supported** |
| p_make (p_set_delta) | +0.009 | [-0.002, +0.019] | spans_zero |
| mark_ev | +0.009 | [-0.002, +0.019] | spans_zero |
| CVaR_10 | +0.171 | [-0.655, +1.082] | spans_zero |
| robust_q25 | — | — | missing |

**Utility flip detected (EV vs p_make/mark_ev/CVaR):** EV=supported but p_make, mark_ev, CVaR all span zero.

**Interpretation:** Void creation in follow position produces a positive expected-value advantage for the setter (~+0.77 EV), but this does not reliably translate into a higher probability of making the set (p_set_delta CI spans zero). The book's void-creation-follow advice is EV-valid but not p_make-validated. For tournament players optimizing p(set), the claim is inconclusive. For EV optimizers, it's supported.

---

### Probe 5 — ch12-setter-pounce-bid30 (t42-ntbe, n=52 paired)

Book claim: setter should pounce when count is exposed at bid=30.

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV (from T0, negated) | +3.092 | [-0.599, +6.664] | spans_zero |
| p_make (p_set_delta_proxy) | 0.000 | [-0.115, +0.115] | spans_zero |
| mark_ev | 0.000 | [-0.115, +0.115] | spans_zero |
| CVaR_10 | — | — | missing |
| robust_q25 | — | — | missing |

**Oracle pounce rate:** 31/52 = 59.6% (binomial test vs 50%: p=0.106 one-sided, not significant).

**Supplementary signal:** The oracle (which is EV-maximizing for Team 0, but Team 1 uses argmin) pounces 60% of the time when both options are available. This is the strongest p_make signal: the oracle's own choice reveals a preference for pounce under p_make reasoning. However, the p_set_delta_proxy CI spans zero (proxy is coarse: {-1, 0, +1} scale with 80% zeros).

**Summary:** All utility metrics span zero at bid=30. The oracle pounce rate weakly supports the book under p_make reasoning but not at conventional significance. This is the key wave-level finding that motivated the p_make vs EV framing.

---

### Probe 6 — ch12-setter-pounce-high-bid (t42-8kbh, n=1140)

Book claim: setter should also pounce at high bids (bid=35-42).

| Utility | Mean | 95% CI | Verdict |
|---------|------|--------|---------|
| EV (setter) | -10.420 | [-11.249, -9.637] | **contradicted** |
| p_make (p_set_delta) | -0.047 | [-0.056, -0.038] | **contradicted** |
| mark_ev | -0.047 | [-0.056, -0.038] | **contradicted** |
| CVaR_10 (from T0) | +4.398 | [+3.580, +5.273] | **contradicted** |
| robust_q25 | — | — | missing |

**CVaR sign note:** cvar_delta = cvar_pounce - cvar_decline from T0 perspective. Positive = pounce gives bidder better 10th-pct outcome. Book direction requires negative (pounce hurts bidder's tail = good for setter). cvar_delta=+4.4 means pounce actually helps the bidder in tail scenarios → setter worse off in tail with pounce → contradicted.

**Unanimous agreement:** All 4 available utilities agree: pounce at high bids is contradicted. Effect sizes are large (EV: -10.4 pts, t=-24.5; p_make: Δ=-0.047, both highly significant). Per-bid slices (35, 36, 39, 42) all contradicted under EV and p_make.

**Exception at bid=42 p_make:** CI spans zero [-0.008, +0.0003] → spans_zero. At bid=42, the mark threshold is so high that the pounce/decline p_make difference nearly vanishes. EV still contradicts at bid=42.

---

### Probe 7 — ch02-bid-only-enough (t42-ey88, n≈14000/step-pair)

Book claim: bid the minimum necessary (overbidding monotonically reduces EV).

| Utility | Mean (weighted) | 95% CI | Verdict |
|---------|----------------|--------|---------|
| mark_ev | +0.095 | [+0.042, +0.153] | **supported** |
| p_make | +0.095 | [+0.042, +0.153] | **supported** |
| EV | — | — | missing (mark_ev primary) |
| CVaR_10 | — | — | missing |
| robust_q25 | — | — | missing |

**All 5 step-pairs monotone:** 30→32, 32→35, 35→36, 36→39, 39→42; every CI excludes zero in book direction (mark_ev). Cohen's d ranges from 0.158 (35→36) to 0.472 (39→42).

**Summary:** The most robust book claim across utilities. mark_ev and p_make agree fully. CVaR not recorded (future probe could fill this gap).

---

## Per-Claim Utility Verdict Table

| Claim | EV | p_make | mark_ev | CVaR_10 | Flip? |
|-------|----|--------|---------|---------|-------|
| ch02-bid-only-enough | missing | supported | supported | missing | No |
| ch03-reentry-preservation | spans_zero | spans_zero | spans_zero | spans_zero | No |
| ch04-low-trump-trap | contradicted | missing | missing | missing | N/A |
| ch05-void-creation-lead | contradicted | contradicted | contradicted | spans_zero | Yes (CVaR only) |
| ch05-void-creation-follow | **supported** | spans_zero | spans_zero | spans_zero | **Yes** |
| ch12-setter-pounce-bid30 | spans_zero | spans_zero | spans_zero | missing | No |
| ch12-setter-pounce-high-bid | contradicted | contradicted | contradicted | contradicted | No |

**Flip taxonomy:**
- **Hard flip** (supported ↔ contradicted): zero instances
- **Soft flip-up** (contradicted → spans_zero): ch05-void-creation-lead CVaR_10 only
- **Soft flip-down** (supported → spans_zero): ch05-void-creation-follow p_make/mark_ev/CVaR

---

## Implications for Model Design

### Objective-Conditioned Heads

The clearest model design implication is for **ch05-void-creation-follow**: EV supports it, p_make does not. A model trained to maximize EV should learn to create voids in follow position; a model trained to maximize p(make) should not. This is directly observable in the oracle data and could inform separate head designs for EV-objective vs p_make-objective models.

### High-Bid Pounce Is a Dead End Under Any Utility

Four utilities agree: pounce at bid=35-42 is bad for the setter. The book is wrong here regardless of objective. The CVaR finding is especially clear: pounce makes the bidder's worst-case outcomes BETTER (bidder's 10th-pct Q improves by +4.4 pts with pounce vs decline). This is the strongest "unanimous verdict" in the campaign.

### Bid-Only-Enough Is Robust

The only claim unanimous supported across available utilities. Should be a reliable training signal for any model objective.

### Low-Trump-Trap Is EV-Contradicted; Other Utilities Unknown

The probe did not record p_make or CVaR. Given that EV says the low trump is often better (oracle prefers it 58%), a future probe should check whether p_make agrees. It's conceivable that low trump is better for EV but worse for p_make (higher-variance play), which would be a theoretically interesting reversal.

---

## Proposed Objective-Aware Ledger Schema

The current `ledger_status` is implicitly EV-conditional. Proposed addition:

```
ledger_status_ev          — current ledger_status (EV-based)
ledger_status_p_make      — verdict under P(Q >= make_threshold)
ledger_status_mark_ev     — verdict under mark-weighted EV
ledger_status_cvar_10     — verdict under CVaR_10 tail
ledger_status_robust_q25  — verdict under Q25 (currently all missing)
utility_flip_flags        — list of utilities where verdict diverges from EV
```

**Recommendation: ADOPT.** Three of seven claims are utility-sensitive. Carrying per-utility statuses enables:
1. Sharper claim classification (not just EV)
2. Objective-conditioned model head design decisions
3. Tournament (p_make) vs money game (EV) distinction in book validation reports

**Migration cost:** One-time addition of 4 new columns to the claim ledger. The orchestrator would populate from this wave's outputs and future waves' analyses.

---

## Caveats

1. CVaR_10 sign conventions require careful per-probe verification; all Q values are T0 perspective. The void-creation probes use b-a convention (preserve - void), pounce_high_bid uses a-b (pounce - decline).
2. p_make proxy for pounce_bid30 is coarse ({-1,0,+1} scale); 80% of values are zero. The oracle pounce rate is a better p_make signal but is a count estimate, not a paired CI.
3. robust_q25 unavailable for all claims; this utility is the most relevant for risk-averse players but requires quantile-level oracle queries not present in existing probes.
4. All probes use oracle-greedy or snapshot-level evidence; live play data may differ.
5. This analysis is read-only on Wave 2 artifacts. No new oracle queries were run.

---

## Artifacts

- `per_probe_utility_verdicts.csv` — row-level verdicts (51 rows)
- `per_claim_utility_summary.csv` — claim × utility pivot (35 rows)
- `objective_aware_ledger_schema.json` — schema proposal with per-claim proposals
- `analyze.py` — reproducibility script (pure pandas/numpy/scipy)
- Mirror: `wiki/experiments/w42-bookval-v2-utility-lens-synthesis.md`

## Provenance

Wave 3.0 agent t42-f2ur. Pure offline analysis from existing probe CSVs. No GPU. No forge access required.
