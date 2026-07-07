# gus-arena-pilot — audit note

Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers (10/20 vs 16/20 contracts, 21.2 vs 29.6 avg bidder points, −8.4/hand, 1.39 Q-pt regret, seeds 900020-900021) match the commit message at 1a2f67f verbatim, and `gus/eval/arena.py` / `gus/eval/play_visualizer.py` exist and behave as described (arena replaces `select_actions` for the student seat).
- The specific V_head +26 / π-pick −0.4 decoupling figures come from `scratch/BLUNDER_FORENSICS.md`, which is gitignored and absent from this checkout — unverifiable, but consistent with the commit narrative.
- Cheap next probe: rerun the pilot with the student at seats 1-3 (defender roles) to separate bidder-skill loss from general play loss; 20 games at one seat is a small sample for a 30pp claim.
