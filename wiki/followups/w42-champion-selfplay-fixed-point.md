Reviewed against code on 2026-07-07 — no issues found.

The page appears to have already absorbed an era-6 audit pass: its caveats (tuned-knob pmake_scale, unmeasured 0.045 seed floor, unverified plunge/onyx export) match `champion/optimism_gap.json` findings and `champion/belief_bidder.py` code comments verbatim. The calibrated 4-round table matches `champion/evidence/run26_selfplay/RESULTS.txt` exactly (KL, margins, CIs, wins, acc).

## Follow-ups

- The as-is (raw) round table has no surviving artifact in the repo — only the calibrated run's log was preserved under `champion/evidence/run26_selfplay/`. The raw numbers are corroborated only by `wiki/log.md` (~−3.4 mean, ~9/80 wins, 0.116→~0.08). A one-line evidence pointer on the page would make provenance explicit.
- `scratch/champion-run/run_26_par.sh` is gitignored and no longer on disk; the RESULTS.txt header preserves its invocation shape, which is the only surviving record of the loop script.
- Cheap next probe (already named in wiki/log.md): back the belief bidder with a PIMC-calibrated value instead of double-dummy, since the ratio is bid-dependent (0.64–0.81) and no single pmake_scale can fit it.
