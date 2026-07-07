# iter3-comparison audit — 2026-07-07

## Corrections

- page cited "burl/SPIKE_REPORT.md"; the report lives at repo root `SPIKE_REPORT.md` (evidence: `git ls-tree -r dbadb5f` and current tree)
- variants table linked iter-3-v2's adapter to [[iter3-rules-adapter]]; iter-3-v2 is a distinct adapter trained on an 18-row corpus (iter-3-rules used 30 rows) with no wiki entity page (evidence: SPIKE_REPORT.md Phases 6-7 tables; wiki/entities/iter3-rules-adapter.md never mentions iter-3-v2)

## Verified

- Headline numbers (90% bot-match, 0 retry-exhausted at eval, 100% first-legal, 32% retry-exhausted for iter-3-v2, trick_winner_if usage up post-SFT 1.56 → 1.70) all match SPIKE_REPORT.md Phases 6-7 and the dbadb5f commit message.
- Three-mode `enable_primer`/`enable_rules_tools` flag matrix matches SPIKE_REPORT.md (default = trimmed primer + 42-framing; rules_tools = preamble; primer off = spike-v2 shape; incoherent combo raises ValueError).

## Follow-ups

- Note that at rollout level iter-3-rules had 1/50 (2%) retry-exhausted; the page's "0 retry-exhausted" is the N=10 eval headline. A one-word qualifier ("eval") would preempt confusion.
- iter-4-thoughts was byte-identical to iter-3-rules on N=10 — worth a sentence on this page since it bounds the value of the thoughts channel at this scale.
