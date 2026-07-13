---
title: W42 Book Validation — Utility-Lens Synthesis (Wave 3.0)
kind: experiment
status: superseded
first_seen: 2026-05-03
last_updated: 2026-07-06
bead: t42-f2ur
parent_epic: t42-4zi6
---

## Superseded by Wave 4.0/4.1 — read this before the findings below

**This page's headline framing was inverted by the same campaign two waves
later.** [[w42-bookval-v3-utility-argmax-divergence]] (Wave 4.0) measured
argmax directly on the ch05-void-creation-follow snapshots and found EV
disagrees with p_make/mark_ev/CVaR_10 on 41-44% of them — an order of
magnitude above this page's implied "narrow" split — and, decisively, that
**p_make picks void MORE often than EV does**, the opposite of what this
page's tables below suggest. [[w42-lens-v1-utility-head-to-head]] (Wave 4.1)
then measured which utility actually wins games and found EV is the
**best**-scoring of four lenses while p_make is the **worst** — i.e. the
book's void-creation advice aligns with the worst-scoring utility on this
corpus, not the best. The schema recommendation below ("ADOPT") was
correspondingly downgraded to **"ADOPT-DEFERRED"** by the campaign tracker
([[w42-book-validation-campaign]], Wave 3.0/4.0 rows). This page's own tables
were never revised to reflect either reversal — read the Wave 4.0/4.1 pages as
the current frontier on which utility a play policy should optimize.

## Summary

Wave 3.0 re-analyzed all 7 closed Wave 2 probes under 5 utility lenses: EV, p_make, mark_ev, CVaR_10, and robust_q25. The central question: which book claims flip verdict depending on which objective a player optimizes?

**Key finding (superseded, see above):** Three of seven claims show utility-dependent verdicts. The most consequential is ch05-void-creation-follow: EV supports the book's void-creation advice in follow position, but p_make, mark_ev, and CVaR_10 all span zero. No claim flips from supported to contradicted (or vice versa) — all flips are between supported/contradicted and spans_zero.

**Most model-design-relevant:** High-bid pounce is contradicted under all 4 available utilities (unanimous, large effect). Low-trump-trap is contradicted under EV only (other utilities not recorded in that probe).

---

## Method

**Inputs:** paired_contrasts.csv from probes t42-v9lu, t42-jysl, t42-26j8, t42-z31l, t42-ntbe, t42-8kbh, and step_pair_deltas.csv from t42-ey88.

**Statistics:** Bootstrap CIs (n=2000, percentile method, seed=42). Verdict = "supported" if CI excludes zero in book direction; "contradicted" if CI excludes zero against book; "spans_zero" otherwise.

**Utility proxies:**
- EV: ev_delta or ev_delta_setter columns
- p_make: threshold_mass, p_set_delta, or p_set_pounce/decline columns
- mark_ev: ≡ p_make at bid=30 (confirmed Wave1.2 [[w42-bookval-v1-wave1-distribution-lens-reranker]]); positive-affine identity at all bids (Wave2.H [[w42-bookval-v1-wave2-ch10-action-level]])
- CVaR_10: cvar_delta columns; all Q values recorded from Team 0 (bidder) perspective
- robust_q25: not recorded in any probe (missing for all claims)

**Sign convention:** Each probe's delta column convention was verified algebraically. Void-creation probes use b-a (preserve minus void from T0); pounce probes use a-b (action_A minus action_B from T0 or setter perspective). See manifest.json for details.

---

## Per-Probe Re-Analysis Table

| Probe | Claim | N | EV | p_make | mark_ev | CVaR_10 | Utility Flip? |
|-------|-------|---|----|----|---------|---------|---------------|
| reentry_v2 | ch03-reentry-preservation | 222 | spans_zero | spans_zero | spans_zero | spans_zero | No |
| reentry_v2 (late) | ch03 late-game | 60 | contradicted | — | — | — | — |
| low_trump_trap | ch04-low-trump-trap | 257 | contradicted | missing | missing | missing | N/A |
| void_creation | ch05-void-creation-lead | 276 | contradicted | contradicted | contradicted | spans_zero | Yes (CVaR only) |
| void_creation_follow | ch05-void-creation-follow | 500 | **supported** | spans_zero | spans_zero | spans_zero | **Yes** |
| pounce_bid30 | ch12-setter-pounce-bid30 | 52 | spans_zero | spans_zero | spans_zero | missing | No |
| pounce_high_bid | ch12-setter-pounce-high-bid | 1140 | contradicted | contradicted | contradicted | contradicted | No |
| bid_only_enough | ch02-bid-only-enough | ~47K | missing | supported | supported | missing | No |

---

## Per-Claim Utility Verdict Table

| Claim | EV | p_make | mark_ev | CVaR_10 | robust_q25 | Flip? |
|-------|----|--------|---------|---------|-----------|-------|
| ch02-bid-only-enough | missing | **supported** | **supported** | missing | missing | No |
| ch03-reentry-preservation | spans_zero | spans_zero | spans_zero | spans_zero | missing | No |
| ch04-low-trump-trap | contradicted | missing | missing | missing | missing | N/A |
| ch05-void-creation-lead | contradicted | contradicted | contradicted | spans_zero | missing | Soft (CVaR only) |
| ch05-void-creation-follow | **supported** | spans_zero | spans_zero | spans_zero | missing | **Yes** |
| ch12-setter-pounce-bid30 | spans_zero | spans_zero | spans_zero | missing | missing | No |
| ch12-setter-pounce-high-bid | contradicted | contradicted | contradicted | contradicted | missing | No |

---

## Proposed Objective-Aware Ledger Schema

The current `ledger_status` field is EV-conditional (forge oracle E[Q] is the source). Proposed additions:

| New column | Definition |
|------------|-----------|
| `ledger_status_ev` | rename of current `ledger_status` |
| `ledger_status_p_make` | verdict under P(Q >= make_threshold) |
| `ledger_status_mark_ev` | verdict under bid × P(make) - penalty × P(fail) |
| `ledger_status_cvar_10` | verdict under 10th-percentile tail outcome |
| `ledger_status_robust_q25` | verdict under Q25 (currently all missing) |
| `utility_flip_flags` | list of utilities where verdict diverges from EV |

**Recommendation: ADOPT** (superseded — downgraded to **ADOPT-DEFERRED** by [[w42-book-validation-campaign]] after Wave 4.0 tripped the argmax-divergence gate; "defer adoption until the next 3-5 probes record full coverage"). Original rationale: (a) ch05-void-creation-follow is EV-supported but p_make-inconclusive — directly relevant to whether to train a model to do this; (b) ch12-setter-pounce-high-bid is unanimously contradicted, strengthening that verdict; (c) ch02-bid-only-enough is unanimously supported, also strengthening. The schema enables objective-conditioned model head design decisions.

**Against:** Adds complexity to ledger; most current probes don't record CVaR or q25; requires future probes to capture more quantile data.

---

## Implications for Model Design

**Strongest finding for model design:** ch05-void-creation-follow divides by objective. A model with an EV head should learn to create voids in follow position. A model with a p_make head should remain neutral. This is the only place where the two objectives give different training signals for the same situation.

**Unanimous contradicted:** ch12-setter-pounce-high-bid is wrong under every available utility. Any model objective should learn to decline rather than pounce at high bids (35-42).

**Unanimous supported:** ch02-bid-only-enough is right under every available utility. All model objectives should learn lower bids are better.

**Missing utilities problem:** ch04-low-trump-trap has only EV. Conceptually, low-trump is a higher-variance play (accepts lower EV for different risk profile). If p_make prefers the dominant trump more strongly than EV does, the claim would be reinstated under p_make. This is a priority for a future probe.

---

## Caveats

- CVaR_10 sign conventions verified per probe but complex; see manifest.json and analyze.py.
- p_make proxy for pounce_bid30 is coarse (mostly zero {-1, 0, +1} scale). Oracle pounce rate 59.6% is a better directional signal but not a paired-CI result.
- robust_q25 unavailable for all claims. This utility is most relevant for risk-averse play.
- All evidence is from oracle-greedy snapshots; live play data may differ.
- This wave is read-only on Wave 2 artifacts. The orchestrator decides whether to adopt the schema.

---

## Artifacts

- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/per_probe_utility_verdicts.csv`
- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/per_claim_utility_summary.csv`
- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/objective_aware_ledger_schema.json`
- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/utility_lens_synthesis.md`
- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/analyze.py` (reproducibility)

---

## Provenance

Wave 3.0 synthesis agent (t42-f2ur). Pure offline analysis from existing probe CSVs.  
No GPU. No forge access. Bootstrap statistics, pandas/numpy/scipy.

---

## Links

- [[w42-bookval-v1-wave1-distribution-lens-reranker]] — mark_ev ≡ p_make at bid=30 (Wave1.2 source)
- [[w42-bookval-v1-wave1-cross-ai-agreement]] — cross-AI agreement with p_make at 59%, CVaR at 83%
- [[w42-bookval-v1-wave2-pounce-window-bid30]] — ch12 bid=30 probe (t42-ntbe)
- [[w42-bookval-v1-wave2-pounce-high-bid]] — ch12 high-bid probe (t42-8kbh)
- [[w42-bookval-v1-wave2-ch02-multistep]] — bid-only-enough probe (t42-ey88)
- [[w42-bookval-v1-wave2-ch10-action-level]] — mark_ev identity at all bids (Wave2.H)
- [[w42-bookval-v1-wave2-reentry-v2]] — reentry-preservation probe (t42-v9lu)
- [[w42-bookval-v1-wave2-void-creation]] — void-creation lead probe (t42-26j8)
- [[w42-bookval-v1-wave2-void-creation-follow]] — void-creation follow probe (t42-z31l)
- [[w42-book-claim-synthesis-and-ai-directions]] — campaign synthesis
- [[w42-bookval-v3-utility-argmax-divergence]] — Wave 4.0, the argmax-divergence gate that inverted this page's void-creation framing
- [[w42-lens-v1-utility-head-to-head]] — Wave 4.1, the head-to-head that found EV best and p_make worst of four lenses
- [[w42-book-validation-campaign]] — the tracker recording the ADOPT → ADOPT-DEFERRED downgrade

---
