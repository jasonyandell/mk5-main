---
title: "Already-clean test 1b: the position-conditioned audit — Gus and the field net read on the same footing (Lem, #226)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). [[lem-gus-formation-2026-10-20]]
found that an untrained Gus's units are 0.91–0.94 clean for free. A transformer's decision token
reads its position, and the play count fixes the mover's seat in the trick. So the audit could
not tell a learned unit from one that reads position. This test conditions every unit on the
play count before reading it, and reads the field net the same way, so the two architectures can
be compared.

Scope: these nets, these inputs and list v2. An unnamed unit is "not named in list v2".

## Pre-registration

- **Instrument** (`--condition n_plays`, `lem.probe.audit.condition`).
  - Each neuron's pre-activation is z-scored within each play count (0–27) before its top 1% is
    taken. A unit that reads only the play count keeps nothing to rank by, and one with no
    variation left is recorded as unnamed.
  - A concept fixed by the play count falls to about its base rate in any top set, so no columns
    are blanked.
  - On/off and dead come from the raw activation.
  - Everything else is the sealed audit: precision ≥ 0.9 clean, 20 shuffles at each series' final
    net.
- **Gus.** The same 18 nets and 43,927 test-split rows as [[lem-gus-formation-2026-10-20]]: 15
  checkpoints of `pretrained-v0` and three untrained nets, with list v2.
- **The field net, for comparison.** The three 128×64 seeds of [[lem-dropout-order-2026-10-17]]
  and their three untrained twins, on the 434,535-row audit sample with list v2. Each is read
  both conditioned and unconditioned; the unconditioned reads are [[lem-naming-2026-10-17]]'s.
- **Predictions.**
  1. **The free signal goes.** Untrained Gus's conditioned clean share (mean of 3) is at most 0.30.
  2. **Trained Gus beats its floor.** At step 200,000 the conditioned clean share is at least the
     conditioned floor + 0.20.
  3. **The field net barely moves.** Conditioned, the 128×64 trained mean is within 0.10 of its
     unconditioned 0.638, and the untrained mean within 0.10 of its unconditioned 0.312.
  4. **Gus's units keep their family.** Under conditioning, family kept is at least 0.6 and at
     least twice the null.
  5. **It forms.** Gus's conditioned clean share at step 200,000 is above step 1,000's, and above
     the conditioned floor from step 5,000 on.
- **Cost.** 24 audits on the Mac's CPU through `scripts/cpu-slot`, about 1–3 minutes each.

## Results

Run 2026-10-20 as registered. Evidence is `lem/conditioned-audit-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/d08009bf4262423b4c2667591886e88da646bfbf/lem/conditioned-audit-2026-10-20):
24 audits, `grade.py`, `grade.txt`, `grade.json` and the runner. Every share was 0.000 on all 20
label shuffles at each series' final net.

| | unconditioned (earlier pages) | **conditioned on the play count** |
|---|---|---|
| Gus, untrained (3 nets) | 0.931 / 0.938 / 0.907 | **0.026 / 0.009 / 0.008** |
| Gus, step 200,000 | 0.517 [0.440, 0.595] | **0.275 [0.239, 0.333]** |
| field net 128×64, untrained (3 seeds) | mean 0.312 | **0.076 / 0.057 / 0.099** |
| field net 128×64, trained (3 seeds) | mean 0.638 | **0.385 / 0.420 / 0.412** |

Gus's conditioned curve: 0.026 at step 0, 0.046 at 1,000, 0.069 at 5,000, 0.135 at 10,000, 0.237
at 20,000, 0.252 at 100,000, 0.281 at 140,000 and 0.275 at 200,000. All 15 steps are in
`grade.json`.

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. untrained Gus conditioned ≤ 0.30 | mean 0.014 | holds |
| 2. trained Gus ≥ conditioned floor + 0.20 | 0.275 against 0.214 | holds |
| 3. field net within 0.10 of unconditioned (trained 0.638, untrained 0.312) | 0.406 and 0.077 | **wrong** |
| 4. Gus family kept ≥ 0.6 and ≥ 2× null | **0.981** against 0.266 (108 units) | holds |
| 5. Gus rises: final above step 1,000, above the floor from step 5,000 | 0.275 > 0.046; above from 5,000 | holds |

**Reading.**

- **On the same footing, both architectures are clean well above chance.** Conditioned on the
  play count, the untrained floors are near zero (Gus 0.014, field net 0.077). Trained Gus is
  0.26 above its floor and the trained field net 0.33 above its own. Gus's rows are pour A, all
  declarations, 43,927 test rows; the field net's are pip, 434,535 rows. So the gap between the
  two is not a measured architecture difference.
- **The field net carried position too, and prediction 3 was wrong about it.** Its unconditioned
  floor of 0.31 and about a third of its trained share were the play count. That is a correction
  to every earlier field-net share in the program: the squeeze, the drop-out order and the naming.
  - Those studies compared nets read the same way, so their *comparisons* stand: narrow against
    wide, trained against untrained under one instrument.
  - Their absolute shares and floors overstated how much the units name beyond position.
  - The conditioned read is the honest absolute. Re-reading the squeeze and drop-out families
    under it is [[lem-squeeze-conditioned-2026-10-20]] (#227).
- **Gus forms cleanly once position is removed.** The U-shape of the unconditioned read is gone,
  and the conditioned share rises from about 0.01 to 0.28 as the top-1 agreement climbs. Units keep
  their family as they form: 0.981 against a 0.266 null. That is the essay's formation signature,
  and it is stronger here than the field net's 0.82.
- **What Gus's clean units are named.** At step 200,000 the 244 clean units are 188 by a Silas
  concept and 56 by a raw bit:
  - the declaration (`trump-is-fives` 31, `trump-is-fours` 19);
  - the count bit for one tile (`x:count:14`, 35);
  - the coarse `late` 53 and `lead` 23.
- **A residual, stated plainly.** Conditioning removes each play count's mean and scale, not the
  shape of its tail. A unit whose variation is widest late in the trick still takes its top 1%
  from those rows, so `late` and `lead` survive. Read them as "this unit's variation is widest
  there", a weaker statement than the other names. A rank-within-stratum top set would remove
  this too.

Scope: these nets, these inputs and list v2. What we learned from these models, not all of 42,
yet.

## Correction, 2026-10-20: family kept

"Family kept" is compared with independent draws from the final labels. That null misses
continuity: a unit's label persists between checkpoints for reasons that have nothing to do with
concepts, and most of family kept is the same label staying. When a label does change, a
co-occurring concept, often in the same family, is the likeliest replacement
([[lem-family-statistic-2026-10-20]]). Read the number as "units keep their label as they form",
which is expected, and not yet as growth along a concept's facets. The re-check against a
shuffled-label run and a co-occurrence-weighted null is #229.

## Caution, 2026-10-20

The conditioning z-scores each play count separately. Where a unit is nearly silent, that promotes
noise into its top set ([[lem-lead-units-2026-10-20]]). For 12.5% of 16×8 and 25% of 128×64 layer-1
units, over a quarter of the conditioned top rows fire below 5% of the unit's maximum. The shares
here await a re-read under a fixed instrument, #233. Comparisons made under one instrument may
survive, since the artifact hits trained and untrained nets alike.

## Links

[[lem-gus-formation-2026-10-20]] · [[lem-naming-2026-10-17]] · [[lem-dropout-order-2026-10-17]] ·
[[gus-tiny-gpt]] · [[clean-neurons-42]]
