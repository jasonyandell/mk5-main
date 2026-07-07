Reviewed against code on 2026-07-07 — no issues found. All round-robin, sample-sweep, fp16-sanity, mark_ev-sanity, and disaster numbers match `w42/lens_v1/summary.json` and the CSVs in `w42/lens_v1/results/`; all referenced paths exist; `forge/eq/generate/actions.py::select_actions` confirmed to be p_make-argmax with E[Q] tiebreak, as the page claims.

- The still-open one-line switch of `select_actions` to ev-argmax + a Zeb-Large eval remains the cheapest high-value follow-up; the page already flags it as NOT applied.
- lens.py now also contains an `upside_10` utility not mentioned on the page — a cheap probe would be a 1000-hand upside_10 vs ev matchup and a one-paragraph addendum.
- N=10 vs N=100 at 1000 paired hands (the page's own noted out-of-scope question) is still unrun.
