---
title: "Already-clean test 1b: the position-conditioned audit — Gus and the field net read on the same footing (Lem, #226)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
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

Not yet run.

## Links

[[lem-gus-formation-2026-10-20]] · [[lem-naming-2026-10-17]] · [[lem-dropout-order-2026-10-17]] ·
[[gus-tiny-gpt]] · [[clean-neurons-42]]
