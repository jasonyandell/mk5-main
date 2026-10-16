---
title: "The first field-v3 net on pour A, audited with concept list v2 (Lem, the paper's G4; #185)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

G4 of the paper's work order ([`lem/paper/README.md`](../../lem/paper/README.md); [[lem-the-teacher]])
and the pre-registration line #185 asks for before concept list v2 is run on any new net. Written
2026-10-16 while [[lem-pour-a-2026-10-16]] was still pouring (276 of 500 chunks); nothing below has
been trained or measured. Code: `lem/probe/` (CC0), `scenario.field_train` (the trainer builds
`field-v3`).

## Question

The shipped field net ([[lem-field-net-audit-2026-10-15]]) plays pip trumps only, and 43 of its 100
raw-feature poles were declaration detectors the sealed concept list had no word for. Pour A has
positions on all ten declarations with the book declaring. Train the first `field-v3` net on it and
ask: (1) does the student still form the teacher's concepts when the declaration varies, and does
the new list name what the old one could not; (2) what do the three kind bits become inside the net;
(3) is the student as faithful to its two teachers (the shipped net's field for pips, the one-rung
Walt for doubles and no trump) as the v2 net was to its one.

## Pre-registration

- **Data.** Pour A merged (`lem.pour.a merge`): the pip half and the doubles/no-trump half, both
  utility `make`, trained as one dataset (`--pour merged/pip merged/dtnt`). The trainer holds out
  deal % 10 == 0 as validation; the audit sample is exactly those rows (`lem.probe sample --every 10
  --layout field-v3` on each half, merged), confirmed disjoint from the training rows by deal id
  before any audit runs (if the halves' deal offsets break the modulus, the sample is rebuilt with
  the trainer's offset; recorded here if so).
- **Net.** `python -u -m scenario.field_train --target margin --device mlx --hidden 128,64`, the
  trainer's defaults otherwise, one seed: the same shape as the shipped net so the audits compare.
  Named `lem/nets/field_v3_pour_a_h128x64.npz`; published with the pour's data card.
- **Audits, a 2×2.** Concept list v1 (sha256 13e4d525…, 48 concepts) and v2 (sha256 2f9a3ddc…,
  70: v1 + the declaration `trump-is-<pip>`, `trump-is-doubles`, `no-trump-game`, the readers'
  conjunctions, off-suit and in-led voids) on the new v3 net; and list v2 on the shipped v2 net
  (its audit sample, 434,535 rows). So the list change and the net change are separated. The
  procedure is the sealed one (top 1%, clean ≥ 0.9 at base ≤ 0.5, 20 shuffles, Wilson per pole,
  the split-half replicate of #191 for the clean share).
- **Fidelity** as in [[lem-fidelity-by-concept-2026-10-16]]: regret under the teacher's plan on the
  sample, by concept, by declaration kind (pip / doubles / no trump) and by forced vs unforced deal.
- **Predictions, before training.**
  1. The shipped v2 net under list v2: the clean share rises from 0.597 to at least 0.70, and at
     least 30 of the 43 `count`-block declaration detectors become `trump-is-<pip>` Silas labels.
  2. The v3 net under list v2: clean share within 0.10 of the v2 net's under list v2; at least one
     clean pole each for `trump-is-doubles` and `no-trump-game` (the kind bits are read directly).
  3. The v3 net under list v1: a clean share below its share under list v2 by at least 0.05 (the
     list, not the net, was missing the declaration).
  4. Dead layer-2 poles: at most 12 (the v2 net had 12).
  5. Fidelity: mean regret on the pip positions within 0.002 of the v2 net's 0.0084; on doubles
     and no-trump positions higher (above 0.012): fewer deals, a different teacher, and the forced
     deals are ones the book would not have declared that way.
  6. Lead remains the highest-regret region on pips; on doubles and no trump the highest is the
     bidder near the contract (`to-make-le-10`).
- **Cost.** Training: one MLX run, tens of minutes on the Mac. Labels: 70 concepts at the v1 pass's
  ~21 ms a row over about 430k rows in slot-sized chunks (the same two-worker pattern as the v1
  audit), about two hours; the v2-list pass on the v2 net's sample the same again. CPU only.
- **What would be a surprise.** A v3 clean share far below the v2 net's (the declaration mixing
  costs concept formation), or the kind bits absent as clean poles (the net plays doubles and no
  trump without a unit that says so; then G3's causal tests go there first).

## Results

Not yet run (pour A in progress).

## Links

[[lem-the-teacher]] · [[lem-pour-a-2026-10-16]] · [[lem-field-net-audit-2026-10-15]] ·
[[lem-fidelity-by-concept-2026-10-16]] · [[scenario-player]] · [[strategy-fusion-fence]]
