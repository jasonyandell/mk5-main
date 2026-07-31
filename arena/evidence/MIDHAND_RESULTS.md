# Mid-hand takeover: rob vs E[Q] from rob's exact window (2026-07-30)

Follow-up to `DROPPED30_RESULTS.md` (full-hand: E[Q] beats rob 6.5σ).
Hypothesis under test: rob is only exact once few enough tricks remain;
earlier he leans on the sigma rollout model. Does the full-hand deficit
survive when the measurement starts inside the exact window?

## Protocol

Driver: `arena/midhand_eval.py` (commit 2d7375bc). Dropped-30 hands
(forced 30, best pip trump), JudPlay (deterministic argmax heuristic, no
oracle) plays ALL four seats through the prefix, position frozen, then
played out twice from the byte-identical state: rob's team holding the
contract vs rob's team defending, E[Q] n=10 (`LensPlay(utility="ev")`) in
the other seats. Paired offense-vs-offense on identical positions.
Zero rules divergences (bridge cross-check on every rob decision).

## Results (256 positions per takeover trick)

| takeover | rob off made | E[Q] off made | discordant (rob/eq) | off pts/hand rob / eq |
|----------|-------------|---------------|---------------------|------------------------|
| trick 3  | 93/256      | 91/256        | 16 / 14             | 23.6 / 23.7            |
| trick 4  | 84/256      | 87/256        | 10 / 13             | 22.8 / 23.2            |
| trick 5  | 89/256      | 91/256        | 4 / 6               | 22.8 / 22.8            |

(The original 16-position trick-4 run: 16/16 concordant both ways —
`arena/results/midhand_t4_16/`.)

Every row is a statistical dead heat (|z| ≤ 0.6 on discordants); the
pooled make-rate gap across all 768 pairs is −0.4pp ± 1.7pp — nothing.
Points per hand agree to a few tenths.

## Reading

The full-hand 6.5σ deficit does NOT come from rob's exact window: from
trick 3 onward rob and the champion are indistinguishable on identical
positions, both making and defending. The entire dropped-30 gap
accumulates in tricks 1–2(–3) — the sigma rollout regime — and/or in the
divergent early trajectories it produces. "Ridiculous amount of headroom"
localized: the headroom is the opening, not the solve.

Caveats: positions come from a JudPlay prefix, not from rob's or E[Q]'s
own early play (the full-hand runs diverge by trick 4; these don't);
dropped-30 pip-trump distribution as before.

Artifacts: `arena/results/midhand_t{3,4,5}_256/midhand_results.json`.
