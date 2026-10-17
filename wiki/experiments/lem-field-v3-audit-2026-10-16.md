---
title: "The first field-v3 net on pour A, audited with concept list v2 (Lem, the paper's G4; #185)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
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

Run 2026-10-17 as registered. Evidence (the net in `.npz` and its metrics, the audit sample,
labels v1/v2, the three new audits, the split-half, teacher values, fidelity, runners, logs,
`SHA256SUMS`): `lem/field-v3-audit-2026-10-16/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

**One deviation, the one this page anticipated.** The trainer numbers the doubles/no-trump deals
after the 217,376 pip deals, and 217,376 is not a multiple of ten. So that half's audit sample was
rebuilt at the trainer's offset: (deal + 217,376) % 10 == 0, 56,095 rows. The merged sample is
exactly the trainer's 434,804 held-out rows. The label pass ran at about 0.2 ms a row, against the
21 ms priced. The old price was the sample reader's column-decompression bug, fixed 2026-10-16.
The new pass reproduces the original v1 labels exactly on the shipped net's sample.

**The net.** `field-v3`, 128×64, margin target, 30 epochs on 3,915,156 rows of pour A. Validation
regret is **0.0087 [0.0086, 0.0088]** make probability (deal bootstrap), against uniform random
play's 0.0412. It picks the teacher's best move on 0.752 of rows, and the top decile of positions
carries 0.826 of its regret.

**The 2×2 audit** (live poles; shuffled-label clean share 0.000 in every cell):

| net | list | clean share | clean by a Silas concept | by a raw feature | dead |
|---|---|---:|---:|---:|---:|
| shipped v2 net (pips) | v1 | 0.597 | 122 | 100 | 12 |
| shipped v2 net (pips) | v2 | 0.602 | 155 | 69 | 12 |
| field-v3 net (all ten) | v1 | 0.628 | 106 | 125 | 16 |
| field-v3 net (all ten) | v2 | 0.628 | 118 | 113 | 16 |

The v3 net's split-half (#191) gives clean shares of **0.633 / 0.625** on disjoint deal halves.
Its row-Wilson intervals are calibrated here: \|z\| > 1.96 on 0.052 of 368 poles, sd of z 0.94.

**Fidelity by kind** (deal bootstrap):

| positions | rows | regret [95%] |
|---|---:|---|
| pip trumps | 378,709 | 0.0082 [0.0081, 0.0083] |
| doubles trump + no trump | 56,095 | 0.0120 [0.0116, 0.0125] |
| of those, forced deals | 53,985 | 0.0119 [0.0115, 0.0123] |
| of those, the book's own declarations | 2,110 | 0.0163 [0.0140, 0.0188] |

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. shipped net under v2: clean share ≥ 0.70; ≥ 30 of the 43 declaration detectors become `trump-is-<pip>` | | 0.602; 29 | **wrong** both, the second by one |
| 2. v3 under v2 within 0.10 of the v2 net; ≥ 1 clean pole each for `trump-is-doubles` and `no-trump-game` | | 0.628 vs 0.602; doubles 1, no trump 0 | first holds, second half wrong |
| 3. v3 under v1 below its v2 share by ≥ 0.05 | | 0.628 both | **wrong** |
| 4. dead layer-2 poles ≤ 12 | | 16 | wrong |
| 5. pip regret within 0.002 of 0.0084; doubles/no trump above 0.012 | | 0.0082; 0.0120 [0.0116, 0.0125] | first holds; second at the line, its interval straddles it |
| 6. lead the highest-regret region on pips; on doubles/no trump the near-contract bidder | | pips: `lead` 0.0102, highest; doubles/no trump: `lead-boss-trump` 0.0177, `can-ruff` 0.0161, `holds-sure-trump` 0.0155 | first holds, second wrong |

**Reading.**
- **A richer list renames clean units; it does not add them.** Concept list v2 turned 33 of the
  shipped net's raw-feature labels into Silas words (100 → 69 by raw feature, 122 → 155 by Silas).
  Most of those were declaration detectors, now `trump-is-<pip>`. The clean share barely moved
  (0.597 → 0.602), because a declaration detector was already clean under the raw `count` bit
  that encodes it. Predictions 1 and 3 assumed the list was missing clean units. It was missing
  their names. For the core thesis (#193) that matters: the clean share measures the net, not the
  dictionary, as it should.
- **One net plays all ten declarations as cleanly as the pip-only net plays pips** (0.628 vs 0.602,
  replicated 0.633 / 0.625), at the same pip fidelity (0.0082).
- **It names the declaration poorly.** One clean `trump-is-doubles` pole, no `no-trump-game` pole,
  and no pole labelled by a kind bit. It plays doubles and no trump worse, at 0.0120. It is worst
  where trump itself is the question: leading the boss trump, ruffing, holding a sure trump.
  Under doubles trump those mean the doubles. A field net reads "trump" through the pip-suit
  features it learned on pips, and the kind bit alone does not re-map them.
- **The book's own doubles and no-trump declarations are the hardest positions** (0.0163 vs
  0.0119 forced), with few rows (2,110). Those are the deals where a player chose doubles or no
  trump on purpose. Measuring them takes a pour that weights them.

Caveats: one seed; pour A's doubles and no-trump teacher is the one-rung Walt, a different teacher
from the pip half's; seat 0 bids 30; linear, top-1% labels as before.

## Links

[[lem-the-teacher]] · [[lem-pour-a-2026-10-16]] · [[lem-field-net-audit-2026-10-15]] ·
[[lem-fidelity-by-concept-2026-10-16]] · [[scenario-player]] · [[strategy-fusion-fence]]
