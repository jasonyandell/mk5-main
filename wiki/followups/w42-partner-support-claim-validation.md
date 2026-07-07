Reviewed against code on 2026-07-07 — no issues found.

- All six findings rows match `w42/partner_support_claim_validation/summary.json` and `claim_proxy_stats.csv` exactly (n, means, CIs, paired contrasts, verdicts).
- Cross-referenced numbers (+0.683 [+0.147, +1.232]; -8.428 [-9.268, -7.613]) match the deep-dive page; the replication page reports slightly different bootstrap CIs ([+0.151, +1.228] / [-9.285, -7.570]) which this page attributes correctly to the deep-dive, not the replication.
- Commit `c603a0d9` exists; `analyze_partner_support.py` does compute per-decision regret via best-of-class contrasts with seeded bootstrap CIs as described.
- Cheap next probe: the corpus dependency on `/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt` is a machine-local absolute path; recording its sha256 in summary.json would make the report reproducible from other checkouts.
