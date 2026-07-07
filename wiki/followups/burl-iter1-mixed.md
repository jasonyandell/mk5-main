# burl-iter1-mixed — audit 2026-07-07

## Corrections

- Page said primer trimmed 1549 → ~500 words; the inlined `_TRIMMED_PRIMER` is 386 words and SPIKE_REPORT says ~400 (evidence: `git show 09b841e:burl/harness/agent_runner_native.py`; the ~500 figure came from the commit message, not the artifact).
- Page said Step 2 rollout "trace bodies ~3× larger than iter-0"; the report's 3× is wall-time per decision (~180 s vs ~56 s), not trace size — iter-1 eval token-out was actually smaller (2.9 K vs 5.6 K chars) (evidence: SPIKE_REPORT.md @ 09b841e, "Corpus N=30, not N=50" bullet).

## Verified

- 4-baseline table (n_completed, retry-exhausted, bot_match, mean_eq_delta, first_legal_rate, eq/trump histograms, costs) matches the commit message and SPIKE_REPORT @ 09b841e exactly, including the 5-completed footnote.
- Step 1 (framing-only: 0 wins, 50% exhausted at 6 decisions), Step 2 (43% K1, tool histogram is_legal 44 / trump_declared 4 / is_trump 1 / eq 2), adapter name, and iter-2 options all match the report.

## Not verifiable in repo

- `burl/data/star_iter1_corpus.jsonl` and `burl/eval/results/move4_iter1_*` are gitignored (noted as such in the commit); adapter lives on HF (private).

## Follow-ups

- The report notes eval ran at `--max-retries 3` while spike v2 used 7; the cheap "re-eval at max_retries 7" probe was never recorded here — worth checking whether iter3-comparison actually ran it.

## Review (second pass, 2026-07-07)

- Verified — corrections stand.
- Primer word count: `_TRIMMED_PRIMER` at `09b841e:burl/harness/agent_runner_native.py` is exactly 386 words (counted); SPIKE_REPORT @ 09b841e says "~400 words"; the old ~500 figure indeed traces to the commit message ("~500 words instead of 1549").
- The 3× claim: SPIKE_REPORT's "Corpus N=30, not N=50" bullet says "ran ~3× slower per-decision than iter-0 (180 s vs 56 s wall)" — wall time, not trace size; iter-1 eval mean_tokens_out is 2.9 K chars vs iter-0's 5.6 K, so "trace bodies ~3× larger" was backwards. Auditor's rewrite is accurate.
- Spot-checked the "Verified" section: 4-baseline table, rollout numbers (13/30 = 43% K1, 0 retry-exhausted, tool histogram is_legal 44 / trump_declared 4 / is_trump 1 / eq 2), Step 1 framing-only result, and adapter name all match SPIKE_REPORT @ 09b841e.
- Follow-up suggestion kept: wiki/experiments/iter3-comparison.md records no max_retries-7 re-eval of the iter-1 adapter, so the question remains open.
