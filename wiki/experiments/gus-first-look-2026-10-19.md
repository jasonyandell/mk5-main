---
title: "Gus first look: current-lead attention routing into L1/U65"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does attenuating attention to the current lead tile reduce layer 1/unit 65's signed pre-GELU response more on matched double-blank-following rows than on other-lead rows?

This is a registered routing diagnostic for the history-conditioned tiny Gus, not a field-net cleanliness score or a semantic confirmation. All First Look reviewers reread all 943 lines of Cammarata's essay, including appendices. The corrected reader notebook notes that a row suffix is the viewer's decision index, not the viewer's within-trick slot; the initial cards also confound history length with turn, hand size, trick phase and legal-set size. Reader interpretations therefore remain candidate descriptions, and a change in model output cannot be attributed to one unit from those cards.

## Specimen and method

The circuit intervention reused the 512 immutable records from [[gus-blank-lead-2026-10-19]], selecting 48 complete own-actor prefixes: 12 matched physical-deal pairs in each of the existing development and confirmation panels. These panels had already been inspected; the second panel is not independent semantic confirmation. Matching fixed declaration, viewer decision number, within-trick slot, legal-action count and pre-settlement status. Original G0 focal/partner policies, pinned Zeb opponents, model weights, input contract and the existing **607,306** physical-root exclusions were unchanged. There was no training, new game generation, player search, or new strength evaluation.

The registered intervention multiplied the attention weight for the current lead-tile column by 0.5 before softmax renormalization at every causal query in both layers. A layer-0-only, layer-1-only, individual layer-1-head, actor-token and other-lead-token series served as diagnostics and controls. The primary quantity is the matched-pair difference in signed raw pre-GELU change (double-blank-following row minus other-lead row); intervals are descriptive 95% percentile intervals over 2,000 resamples of the reused physical pairs. The layers-and-queries intervention can alter earlier token representations that later reach the actor site. It is not a single edge cut at the actor query.

All 144 model/row units passed an independent explicit-forward audit bit-identically, including all MLP pre/post coordinates, logits and query attention. Exact zero-dose and empty-key shams were zero. The registration and complete evidence bundle were freshly verified by anonymous file and member hashes at immutable Hub revisions:

- [Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/c12ecbaeab8133ee03202e908c49b248a2c0dc1f/lem/gus-first-look-2026-10-19/circuit/registration) — SHA256 `905d8b7e2857f1fe648fbe8c0d57c78e032301b95dd85d5c64d462adf60cbbd5`.
- [Verified circuit evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7b9870110ce9029d177c22b28a3791d4d526ffa5/lem/gus-first-look-2026-10-19/circuit/evidence) — 8 files, 156 members, 54,039,633 bytes; verification receipt pins revision `7b9870110ce9029d177c22b28a3791d4d526ffa5`.

## Result

Attenuating current-lead tile attention reduced L1/U65 preactivation more on the double-blank-following row than its matched other-lead row in both panels:

| Reused panel | Matched pairs | Mean paired preactivation change | Descriptive 95% interval |
|---|---:|---:|---:|
| Development | 12 | −1.894 | [−1.964, −1.828] |
| Confirmation | 12 | −1.474 | [−1.818, −1.038] |

The layer-0-only effect was −1.875/−1.457 in development/confirmation; layer-1-only was −0.020/−0.016. The analogous current-lead actor-token control was near zero, while other current-trick and previous-trick tile controls had small positive effects. The original parent showed a similar response, so this routing sensitivity predates cooldown. The untrained coordinate is not a cleanliness comparison because its raw activation scale differs.

Attention attenuation also changed downstream output distributions on these rows. Legal-renormalized total variation averaged 0.0396/0.0116 for positive rows and 0.0514/0.0125 for matched negative rows in development/confirmation; legal argmax changed on 1/12 positive and 2/12 negative development rows, and on none in confirmation. These are static distribution changes after a broad attention intervention. They are not wins, move-quality evidence, or mediation through U65: many units can change, earlier queries are perturbed, and the computation has multiple routes.

## Interpretation boundary

The result supports sensitivity to current-lead-tile attention in this Gus checkpoint, on these original-own-seven/public-history inputs and reused fixed-field contexts. It does not establish an exclusive meaning for U65, distinguish the literal double-blank token from an abstract suit computation, or show that U65 mediates the downstream output changes. It adds no new semantic confirmation, global already-clean claim, training gain or playing-strength gain. The broader blank-suit hypothesis remains inconclusive under the prior study's rare-context coverage limit.

The interactive report is deployed at [gus-first-look.psychometrix.chatgpt.site](https://gus-first-look.psychometrix.chatgpt.site). It includes measured unit doses, a GELU illustration and the routing comparison. Desktop and 390-pixel phone layout and controls were exercised. The existing research heartbeat maintains this same report as verified findings accumulate.

## Next question

The next useful question is which layer-0 head/value directions carry current-lead information toward U65's input, with interventions that preserve explicit controls for relay and alternate paths. That is a proposed diagnostic only; this study did not select or measure individual layer-0 heads.

## Links

[[gus-native-cooldown-2026-10-19]] · [[gus-blank-lead-2026-10-19]] · [[gus-book-silas-tools]] · [[gus-tiny-gpt]]
