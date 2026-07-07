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
