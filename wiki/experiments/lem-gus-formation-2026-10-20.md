---
title: "Already-clean test 1 on Gus: do a tiny GPT's own neurons come out clean, and do they keep their family as they form? (Lem, #194, #203)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
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

Not yet run.

## Links

[[gus-tiny-gpt]] · [[lem-formation-2026-10-17]] · [[lem-naming-2026-10-17]] ·
[[lem-field-v3-audit-2026-10-16]] · [[clean-neurons-42]]
