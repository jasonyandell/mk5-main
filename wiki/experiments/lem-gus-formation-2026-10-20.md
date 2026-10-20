---
title: "Already-clean test 1 on Gus: do a tiny GPT's own neurons come out clean, and do they keep their family as they form? (Lem, #194, #203)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]], [[lem-the-teacher]]). Every test so
far read the field net, an integer MLP over one position. [[gus-tiny-gpt]] is a different
architecture with different inputs: a two-block GPT that sees a seat's original hand and the
public play so far. This is the first audit of its own neurons against the teacher's vocabulary,
and the first formation series on a transformer. The split is #203's: Codex trains Gus, and the
Lem desk reads it. The audit is read-only against `lem/observer`'s own `encode` and
`Observer.forward(tokens, inspect=True)`, and nothing in Gus's code or Hub repo changes. Codex's
U65 study followed one selected unit's trajectory and made no all-neuron claim. This audit makes
the all-neuron reading.

Scope: this net, these inputs and this vocabulary.
- The net is Gus `pretrained-v0`: 107,392 parameters, next-tile imitation of Zeb's 500,000
  recorded deals, bid 30.
- Its inputs are one seat's original hand, the declaration, the bidder and the public plays of
  this hand. That is within-hand history, which the field net never saw. There is no
  cross-hand history.
- The vocabulary is concept list v2 (70 Silas concepts) plus the field-v3 input bits. Those bits
  are the field net's raw inputs, used here as a vocabulary of simple viewer-only facts, not as
  Gus's tokens.

An unnamed unit is "not named in list v2", never mixed.

## Pre-registration

- **Subject.** The `pretrained-v0` run's own checkpoints (Codex's scratch,
  `observer-corpus-squeeze`, seed 42, 200,000 steps). Its final `best.pt`, step 200,000, sha256
  `99fd106a…`, is tensor-identical to [jasonyandell/gus@pretrained-v0](https://huggingface.co/jasonyandell/gus/tree/pretrained-v0).
  It has 2 blocks of 256 GELU units: 512 neurons and 1,024 poles. A pole is a neuron's
  pre-activation (`+`) or its negation (`−`), at the decision token, the mover's seat token after
  the public plays. "On" means pre > 0.
- **Checkpoints, 15:** steps 0 (`initial.pt`), 250, 500, 1,000, 2,500, 5,000, 10,000, 20,000,
  40,000, 60,000, 80,000, 100,000, 140,000, 170,000 and 200,000.
- **The floor.** Three untrained nets: `initial.pt` (seed 42) and two untrained twins built by
  `Observer(Config)` under torch seeds 1 and 2. All are audited with the same rows and list.
- **Rows.** The pour A audit sample of G4 ([[lem-field-v3-audit-2026-10-16]]), restricted to
  Gus's held-out **test** deals by Gus's own split (`lem.observer.data.partition`). That leaves
  43,927 of 434,804 rows, with 37,906 pip and 6,021 doubles or no trump. Labels are
  `labels_v3_v2` (list v2, 70 concepts). The audit is the sealed one: top 1% (439 rows), clean at
  precision ≥ 0.9, partial at ≥ 0.7, and 20 label shuffles at the final checkpoint.
- **The one plumbing forward.** Before this page was written, one forward at step 200,000 was run
  without reading any pole. Gus's top tile matched the recorded move on 0.6669 of these rows.
- **Measures.**
  - the clean share at each checkpoint, with its interval, and the floor;
  - live, unnamed and dead counts;
  - the first-clean step and the family kept, with the same statistic and null as
    [[lem-formation-2026-10-17]] (`formation.kept_share`);
  - the order in which families become clean;
  - top-1 agreement with the recorded move at each checkpoint.
- **Predictions, before any pole is scored.**
  1. **Clean, above the floor.** At step 200,000 the clean share exceeds the mean of the three
     untrained floors by at least 0.20.
  2. **In the field net's range.** At step 200,000 the clean share is in [0.45, 0.75]. The field
     nets' v2 shares were 0.60–0.64 at 136–192 neurons.
  3. **Family kept.** Of the units clean at step 200,000 that have a first-clean step, the share
     whose first clean label is in the same family as their final label is at least 0.6, and at
     least twice the null.
  4. **The trick first.** The median first-clean step of units whose final label is in the
     position or trick family is earlier than that of units in the count or mastery family.
  5. **It forms.** The clean share at step 200,000 is above that at step 1,000, and every
     checkpoint from step 2,500 on is above the floor's mean.
- **Cost.** 18 audits of 43,927 rows each on the Mac's CPU through `scripts/cpu-slot`, about a
  minute each.

## Results

Run 2026-10-20 as registered, plus one exploratory arm, labelled below. Evidence is
`lem/gus-formation-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1579b9ae90b1994361cf3aaf84f4ddf87d77d816/lem/gus-formation-2026-10-20):
- both arms' 18 audits;
- the 43,927 rows with their tokens and labels;
- the blanked-column list;
- `grade.py`, `grade.txt` and `grade.json`;
- the runner.

The checkpoints are Codex's run, archived at
[`lem/observer-corpus-squeeze-2026-10-17` @ db83ef2e](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/db83ef2ea26e95cba36340e9dab87c3c055f7057/lem/observer-corpus-squeeze-2026-10-17).
Every share was 0.000 on all 20 label shuffles at step 200,000.

| step | live poles | clean share (list v2) | Gus's top tile = recorded move |
|---:|---:|---|---:|
| untrained, seeds 42 / 1 / 2 | 1,001 / 974 / 986 | **0.931 / 0.938 / 0.907** | 0.035 / 0.034 / 0.031 |
| 250 | 1,015 | 0.357 [0.282, 0.432] | 0.249 |
| 1,000 | 1,022 | 0.348 [0.279, 0.423] | 0.407 |
| 5,000 | 1,024 | 0.287 [0.208, 0.375] | 0.578 |
| 20,000 | 978 | 0.446 [0.372, 0.505] | 0.640 |
| 40,000 | 950 | 0.500 [0.431, 0.566] | 0.656 |
| 80,000 | 918 | 0.514 [0.449, 0.580] | 0.665 |
| 140,000 | 899 | 0.537 [0.461, 0.592] | 0.667 |
| 200,000 | 887 | 0.517 [0.440, 0.595] | 0.667 |

(Steps 500, 2,500, 10,000, 60,000, 100,000 and 170,000 are in `grade.json` and lie on the same
curve.)

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. final clean share ≥ floor + 0.20 | 0.517 against a floor of 0.925: margin **−0.408** | **wrong** |
| 2. final clean share in [0.45, 0.75] | 0.517 | holds |
| 3. family kept ≥ 0.6 and ≥ 2× null | 0.825 against a null of 0.291 (280 units) | holds |
| 4. position/trick units clean before count/mastery units (median first-clean step) | 80,000 (144 units) against 90,000 (30 units) | holds, narrowly |
| 5. final above step 1,000, and above the floor from step 2,500 | above step 1,000; never above the floor | **wrong** |

**Reading.**

- **The floor is the finding, and it is a tools finding (rung 1).** An untrained Gus's units are
  0.91–0.94 clean, and almost every one is named by the mover's place in the trick: `lead`,
  `second`, `third` or `last`: 859 of 932, 882 of 914 and 828 of 894 clean units across the three
  untrained nets. At the decision token an untrained
  transformer's units read mostly the position embedding. The number of tiles played fixes the
  seat in the trick, so top-1% precision hands every random unit a perfect position word.
  - The field net never had this: its untrained floor is 0.11–0.23, because random integer
    weights over input bits do not concentrate on one fact.
  - So the clean share is not comparable across architectures, and Gus's registered floor
    measures the instrument, not the net. Predictions 1 and 5 broke on the instrument.
- **Training dissolves the free units and builds new ones.** The curve is U-shaped:
  - from 0.93 at initialization it falls to 0.29 by step 5,000, while the top-1 agreement climbs
    from 0.03 to 0.58;
  - then it rises to 0.52–0.54 as agreement levels at 0.667.
  - At the end the clean units' words are no longer mostly position words (`lead` 112, `late`
    68, `third` 15, `last` 10). They include the declaration (`trump-is-fives` 25,
    `trump-is-fours` 15), the contract (`to-set-le-5` 26, `count-out-20plus` 13), the bidder's
    lead and the count bits.
  - This is the formation signature the essay describes: units that grow into concepts. Here
    they grow out of a random positional start.
- **Units keep their family.** 0.825 of the units clean at the end were first clean in the same
  concept family, against a null of 0.291. The field net's formation gave 0.82 against 0.28 on a
  different architecture, inputs and training. The free position words make this weaker than it
  looks for the position family; the blanked arm below removes them.
- **Exploratory, not registered: the blanked arm.**
  - The same 18 nets were re-audited with 10 Silas concepts and 75 raw bits blanked. These are
    the columns constant within every play count on these rows (`rows-nopos/blanked.json`:
    `lead`, `second`, `third`, `last`, `late`, `trick-1`, `trick-2`, `trick-7`, `hand-le-3`,
    `forced`, the trick-slot bits, and count bits that never fire here).
  - The untrained floor falls to 0.427–0.465, still mostly raw bits such as `gone` that move with
    how far the hand has gone.
  - Trained Gus ends at 0.389 [0.312, 0.479], 196 units by a Silas concept, with family kept 0.956
    against a null of 0.267.
  - Blanking the position words does not remove position from the units. A fair instrument has to
    condition on the play count, and that is #226.

**What this says and does not say.** It does not say Gus's units are mixed, or less clean than
the field net's. The instrument cannot yet tell a learned unit from one that reads position for
free. It does say:
- a transformer's untrained units carry a large free signal that our audit counts as clean;
- training replaces that signal with units named by the declaration, the contract and the count;
- those units keep their family as they form, as the field net's did.

Scope: this net (`pretrained-v0`), these inputs (within-hand history, no cross-hand history),
this vocabulary (list v2 and the field-v3 bits).

## Correction, 2026-10-20: family kept

"Family kept" is compared with independent draws from the final labels. That null misses
continuity: a unit's label persists between checkpoints for reasons that have nothing to do with
concepts, and most of family kept is the same label staying. When a label does change, a
co-occurring concept, often in the same family, is the likeliest replacement
([[lem-family-statistic-2026-10-20]]). Read the number as "units keep their label as they form",
which is expected, and not yet as growth along a concept's facets. The re-check against a
shuffled-label run and a co-occurrence-weighted null is #229.

## Links

[[gus-tiny-gpt]] · [[lem-formation-2026-10-17]] · [[lem-naming-2026-10-17]] ·
[[lem-field-v3-audit-2026-10-16]] · [[clean-neurons-42]]
