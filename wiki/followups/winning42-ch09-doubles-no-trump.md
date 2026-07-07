Reviewed against code on 2026-07-07 — no issues found.

- Numbers cross-checked: 66.1% / 76.2% (w42-phase4-doubles-notrump-regime-tests), 90% EV-lying and 38.9% CVaR disagreement (w42-bookval-v1-wave1-distribution-lens-reranker), 4-4 bidder_partner impact 47-49 (w42-bookval-v1-wave1-hidden-threat-impact-ranker).
- Two claim-ledger rows remain "underpowered" (low-double sacrifice, dynamic suit counting); a cheap next probe is oracle counterfactuals on Ch9-like hands with/without the low-double first lead.
- The "nearly double the corpus average" phrasing for doubles-trump CVaR disagreement is qualitative; the source table only gives per-detector rates, so a corpus-wide CVaR-disagree baseline would firm it up.
