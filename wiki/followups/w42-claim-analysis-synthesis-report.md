Reviewed against code on 2026-07-07 — no issues found.

Verified: all referenced scripts and artifact dirs exist (`w42/claim_analysis_synthesis/`, validation script paths); headline numbers match `w42/joined_claim_row_model_table/summary.json` (28,000 decisions, 75,079 action rows, mean regret 1.359796 → 1.125741) and `ablation_results.csv` (drop_sequence_seat is the dominant ablation, regret 1.3678); 111 GB corpus note matches `w42/claim_data_inventory/summary.json`; all [[wiki links]] resolve.

- A cheap next probe: `drop_hidden_public_proxy` actually beats the full feature set (1.1179 vs 1.1257) — the page notes hidden proxies "remain diagnostic," but a one-line pruning test could confirm whether they're net-negative as features.
