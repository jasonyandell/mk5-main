---
title: "Posterior-statistic labels for the decision quotient — routing is bought with reuse"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

[[gus-observation-labels-2026-10-21]] found that no tile-free label built from facts about the viewer's own hand routes the decision quotient of [[gus-decision-quotient-2026-10-21]]. The colliding blocks agreed in every fact; they differed in hidden tiles: which remain out, and who must play them. Issue #239 asked for the next family. These labels carry statistics of what the viewer can infer about the unseen tiles. Is there a lawful label, one that reads only the viewer's own information and names no tile, that routes while fresh roots stay seen states? The issue also asked for a decision-only interface that drops domain collisions. This is a count on saved data; no model is involved.

## Labels

Each observed play gets six new facts, computed once the play joins the public record. Unseen means not in the viewer's original hand and not yet played:

- `unseen_trumps`, `unseen_count`: trumps and count points still unseen;
- `voids`: per other seat, the suits it has shown void in;
- `exhausted`: per other seat, the suits of which it can hold no unseen tile;
- `outranked_by`: unseen tiles that would beat the play in its trick;
- `suit_left`: unseen tiles that still follow the trick's suit.

One fact is per block. `likelihood` is the observation's probability given the viewer's history and action (child mass over node mass), bucketed to within a factor of two. It reads the course's prior and its fixed field, which the viewer knows, and no world.

Labels: `unseen`, `voids`, `exhausted`, `outrank` (`outranked_by` + `suit_left`) and `likelihood`, each with `seat`; `posterior` is all six with `seat`; `posterior_likelihood` adds the bucket; `everything` is all fourteen facts plus the bucket. Two **solved** labels are a ceiling, not candidates. They name the next decision's answer: its shape output (`next_output`) or its fork class (`next_fork`). A label that keeps no play fact also hides how many plays there were.

**Decision-only** is the subset construction. The controller's state is the set of states its label history cannot tell apart, and a label sequence one of them never meets constrains nothing. A node fails when some reachable set answers two ways (menu size or maximizing slots). A TRAIN root *covers* a fresh root when it accepts every label sequence the fresh root does and the two sets answer alike there; a controller built for the TRAIN root is then exact wherever the fresh root can go. Dropping domains is not an equivalence relation, so this gives no state count.

## Result

| Label | States | Root ladder | Root alphabet | Fresh roots seen | Fresh discretionary seconds seen | Roots with a decision collision | Roots with a domain collision | Fresh roots covered, decision-only |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fork (no routing) | 144 | 10, 139, 139 | — | 56 of 128 | 3,338 of 3,338 | — | — | — |
| seat | 429 | 10, 173, 239, 239 | 4 | 7 | 3,267 | 192 | 91 | 10 |
| hand-relative (all eight) | 3,746 | 10, 247, 254, 254 | 1,597 | 2 | 1,647 | 47 | 206 | 2 |
| unseen | 1,780 | 10, 247, 254, 254 | 326 | 1 | 2,613 | 151 | 183 | 4 |
| voids | 2,094 | 10, 248, 255, 255 | 173 | 1 | 2,209 | 181 | 175 | 1 |
| exhausted | 2,772 | 10, 248, 256, 256 | 1,630 | 0 | 2,200 | 109 | 189 | 1 |
| outrank | 1,802 | 10, 245, 246, 246 | 383 | 3 | 2,682 | 99 | 173 | 3 |
| likelihood | 704 | 10, 237, 252, 252 | 20 | 3 | 3,177 | 187 | 161 | 5 |
| posterior | 5,927 | 10, 248, 256, 256 | 4,395 | 0 | 238 | 34 | 106 | 0 |
| posterior + likelihood | 6,012 | 10, 248, 256, 256 | 5,510 | 0 | 237 | 18 | 69 | 0 |
| everything | 6,943 | 10, 248, 256, 256 | 7,023 | 0 | 175 | 7 | 69 | 0 |
| solved: next output | 144 | 10, 139, 139 | 4 | 56 | 3,338 | 0 | 0 | 95 |
| solved: next fork class | 144 | 10, 139, 139 | 4 | 56 | 3,338 | 0 | 0 | 95 |

The root alphabet is the number of distinct observation labels read out of roots. The other hand-relative rows repeat [[gus-observation-labels-2026-10-21]] exactly. HALT and REJECT are additional in every row. As before, every failure is at a root.

**No lawful label in the family routes. Routing improves only as fresh roots stop being seen.** The finest lawful label (`everything`) cuts the roots with a decision collision from 47 to 7. Its alphabet is 7,023 labels at the roots and it sees 0 of 128 fresh roots. No posterior label sees more than 3. Every fact that splits colliding blocks also splits the states the course could reuse. The fresh discretionary seconds show it most: 1,647 of 3,338 are seen under hand-relative, 175 under `everything`.

**Decision-only drops exactly the domain collisions.** In every row, the roots that fail decision-only are exactly the roots with a decision collision. Dropping domains never rescues a decision collision, and it adds none. It does raise reuse where a label routes: under the solved ceiling, 95 of 128 fresh roots are covered, against 56 seen when domains are kept. That gap is the analog of the note's 11/15 against 23/31.

**The ceiling is four symbols.** Labelling a block with the next decision's answer routes and gives fork exactly: 144 states and 56 of 128 fresh roots. This holds in general. Any label's quotient refines fork, by induction on the signatures, since forgetting the label maps each labelled signature to the fork signature. So 56 of 128 bounds the fresh roots any label can make seen, and the solved label reaches the bound. On this course the second decision is the last real choice, so the next class is the next output. At a root, the routing label reduces to four symbols: the next choice is slot 0 (374 of the 10,302 root blocks), slot 1 (328), either (6,218), or a forced tile (3,382). `seat` also has four symbols and fails at 192 roots. A routing label is the next decision's advice. The question was never the label's size. It is whether a statistic declared before solving computes that advice.

**What tells the colliding blocks apart.** Under the hand-relative label, posterior facts separate 182 of the 193 consecutive decision-colliding block pairs. `exhausted` and `likelihood` do most of it (`exhausted` + `likelihood` 49, `likelihood` 40, `exhausted` 27); 11 pairs differ in nothing. Under `posterior`, `likelihood` separates 47 and `follow_slots` 24, and 17 differ in nothing. Under `everything`, all 11 colliding groups, at 7 roots, differ in nothing the family states.

**The collision that survives.** test-0038 again: sixes are trump, and the right-hand opponent leads 0-0 or 1-1 into the viewer's 4-3 and 5-4. The two leads agree in all fourteen facts and in the likelihood bucket. Both are the top of an off suit with no count. The same number of unseen tiles follows each, and every seat has the same voids and exhausted suits. After 0-0, 4-3 is better (5/18 against 2/9); after 1-1, both lose. In test-0797 the partner and the right-hand opponent play 1-1 and 2-2, or 2-2 and 1-1. The facts agree, yet the next choice matters one way (q 1 against 0) and not at all the other (1 and 1). The difference is joint, not marginal: which seat holds which of the unseen blanks or ones. No per-seat count and no per-play rank sees it.

## Reading

- The posterior family does what #239 hoped. It finds the information the hand-relative family missed, and with the likelihood bucket it leaves 18 roots with a decision collision, against 47. But it routes by fragmenting: thousands of labels, and no fresh root lands in a state the course holds. Inside these families, no label routes better than `seat` (192 roots with a decision collision, 7 fresh roots seen) without seeing fewer fresh roots.
- The ceiling says why. A label that routes this course must carry the next decision's answer, and a lawful label can carry it: the solved label is computed from the viewer's own information. It costs a solve, though. A cheap statistic would have to agree with that solve on every block, and the residual collisions show none of these does: the answer depends on the joint placement of hidden tiles.
- In the note's terms, the quotient depends on the kernel. A label is useful to a controller only if it is about as coarse as the decisions it routes, and the decisions here are coarse (four answers at a root) while the facts that decide them are fine. A learned label, trained to predict the next decision's class, is the natural next object. That makes it a question about the net ([[gus-decision-class-probe-2026-10-21]], #240) rather than about the family.

## Scope

One fixture: three own tiles, acting last in trick five, nine plays left. Fixed G0 partner, pinned Zeb, uniform legal-consistent root prior. The labels are hand-chosen families of six posterior facts and eight hand-relative facts, not every lawful label. A statistic of the joint posterior, such as the support pattern of the unseen tiles across seats up to relabeling, is untried. The decision-only audit counts failures and coverage but no states. The bound of 56 of 128 is on fresh roots *seen* with domains kept, for every label, because it is a statement about fork. Decision-only coverage is a different measure and is not bounded by it (95 under the solved label). Every count is exact and reproducible with `python -u -m lem.quotient <study> --labels`.

## Evidence

Tool `lem/quotient.py` (`--labels`: `posterior_facts`, `likelihood_bucket`, `decision_only`, `alphabet_by_stage`), tests in `lem/tests/test_quotient.py`. The run took 27 s under `scripts/cpu-slot` (4.8e11 instructions). The default report stays byte-identical, sha256 `14fde1ac…`. [labels.json](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1ffebf8ce06d8b02c784bd089779665025b59374/lem/gus-posterior-labels-2026-10-22), sha256 `66f4333b…`, verified on the Hub. Issue #239.
