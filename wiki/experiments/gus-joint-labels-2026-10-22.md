---
title: "Joint-support labels for the decision quotient — they route, and what routes is the observation up to pip relabeling"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

[[gus-posterior-labels-2026-10-22]] left 11 colliding groups, at 7 roots, that differ in nothing its fourteen facts state. Its reading was that the joint placement of the hidden tiles decides them. Issue #241, item 1, asks for the object that sees joint placement: the support pattern of the unseen tiles across seats, up to pip relabeling. Does a label built from it route the decision quotient of [[gus-decision-quotient-2026-10-21]], and does any fresh root land in a state the course holds? The issue predicted that it would route more and see fewer fresh roots. This is a count on saved data; no model is involved.

## Labels

**The support.** An unseen tile is one the viewer neither holds nor has seen played. A seat *may hold* an unseen tile when no shown void rules it out and the other unseen tiles can still fill every hand (Hall's condition on the hand sizes, one tile to a seat). These are exactly the pairs some consistent world uses. Together with the hand sizes they therefore determine the set of consistent worlds, and they are its canonical form: the support of the joint posterior under any prior that gives every consistent world mass. It is the rule posterior, not the policy posterior. The fixed partner and the pinned Zeb may give some consistent worlds no mass, and the support cannot see that.

**Up to pip relabeling.** Remaining tiles are edges of a multigraph on the seven pips, doubles being loops. Each edge carries the seats that may hold it, its trump flag and its count. A label keeps the least encoding over all relabelings of the pips, so it names no tile. Trump and count stay on the edges, so a relabeling that moves the trump pip or a count tile changes the encoding. The order of the pips is not kept, and the rules read it twice. The high end of a non-trump lead names the suit, and the pips rank tiles within a suit.

Each observed play gets five facts, computed once the play joins the record, each led by the other seats' hand sizes:

- `support_counts`: the multiset of (seats, trump, count) over the unseen tiles; no pips at all;
- `support`: the pip graph of the unseen tiles, up to relabeling;
- `support_hand`: the same graph with the viewer's own remaining tiles as seat 0;
- `shape_hand` (control): `support_hand` with every unseen tile's seats erased. It keeps which tiles remain and the viewer's hand, but not who may hold them;
- `support_named` (diagnostic): every unseen tile by name with its seats. It names tiles, so it is not a candidate.

One fact is per block. `support_end` (control) reads `support_hand` once, at the next decision, rather than after every play.

Labels: each fact with `seat`; `joint_everything` is the fourteen facts of the earlier families, the likelihood bucket and `support_hand`. `shape_everything` swaps in `shape_hand`, and `named_everything` swaps in `support_named`. The routing audit lists every decision collision of a joint label, not the first eight.

## Result

| Label | States | Root ladder | Root alphabet | Fresh roots seen | Fresh discretionary seconds seen | Roots with a decision collision | Roots with a domain collision | Fresh roots covered, decision-only |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fork (no routing) | 144 | 10, 139, 139 | — | 56 of 128 | 3,338 of 3,338 | — | — | — |
| raw (tiles named) | 8,257 | 253, 256, 256 | — | 0 | 14 | — | — | — |
| seat | 429 | 10, 173, 239, 239 | 4 | 7 | 3,267 | 192 | 91 | 10 |
| everything (posterior family's finest) | 6,943 | 10, 248, 256, 256 | 7,023 | 0 | 175 | 7 | 69 | 0 |
| support_counts | 2,371 | 10, 248, 256, 256 | 2,322 | 0 | 2,321 | 118 | 182 | 1 |
| support | 3,648 | 10, 248, 256, 256 | 4,668 | 0 | 1,652 | 16 | 50 | 0 |
| support_hand | 6,932 | 10, 248, 256, 256 | 9,865 | 0 | 381 | 4 | 31 | 0 |
| shape_hand (control) | 6,585 | 10, 248, 256, 256 | 9,713 | 0 | 498 | 10 | 54 | 0 |
| support_end (control) | 4,179 | 10, 256, 256 | 6,210 | 0 | 1,784 | 96 | 149 | 0 |
| joint_everything | 8,425 | 10, 248, 256, 256 | 10,029 | 0 | 132 | **0** | 13 | 0 |
| shape_everything (control) | 8,424 | 10, 248, 256, 256 | 10,029 | 0 | 132 | 0 | 13 | 0 |
| named_support (names tiles) | 5,417 | 10, 248, 256, 256 | 5,064 | 0 | 467 | 0 | 0 | 0 |
| named_everything (names tiles) | 7,165 | 10, 248, 256, 256 | 7,474 | 0 | 132 | 0 | 0 | 0 |

Fork, raw, seat and everything repeat [[gus-decision-quotient-2026-10-21]] and [[gus-posterior-labels-2026-10-22]] exactly. Without `--joint` both earlier reports are byte-identical. HALT and REJECT are additional in every row, and every failure is at a root.

Intervals are 95% Clopper–Pearson, reading the 128 fresh roots as a sample of roots like them. Every joint label sees 0 of 128 fresh roots, [0.0%, 2.8%]. The baselines are fork's bound of 56 of 128 (43.8% [35.0, 52.8]) and `seat`'s 7 of 128 (5.5% [2.2, 10.9]). `joint_everything` sees 132 of 3,338 fresh discretionary seconds (4.0% [3.3, 4.7]), `everything` 175 (5.2% [4.5, 6.1]) and `support_hand` 381 (11.4% [10.4, 12.5]). The decision-collision counts are exact on this course. `joint_everything` has 0 of 256 roots ([0, 1.4%] as a rate) against `everything`'s 7 (2.7% [1.1, 5.6]).

**A lawful label now routes every decision.** `joint_everything` leaves no root with a decision collision. Decision-only, it fails nowhere: it is the first tile-free label in these families that a decision-only controller can follow exactly on the whole course. Its 13 domain-collision roots (32 groups) are told apart only by naming tiles: test-0388, test-0663, test-0718, test-0966, test-1004, test-1167, test-1214, test-1665, test-1721, train-0387, train-10273, train-20345 and train-20482. With `support_named` in place of `support_hand` they go too, and the label routes as an interface.

**It routes the way the posterior family did, by fragmenting.** No joint label sees a fresh root, and none covers one decision-only, except `support_counts`, which covers 1. `joint_everything` has 8,425 states, more than the raw interface's 8,257, which names every tile. Its quotient is not coarser than reading the tiles. It is about as fine, and it shares fewer classes across roots (8 against the posterior family's 45). The prediction in #241 holds on both counts.

**The seats do not do the work.** `shape_everything`, the control that erases who may hold each unseen tile, gives the same row as `joint_everything` (8,424 states, 0 decision collisions, the same 13 domain roots). Alone, support does add something: `support_hand` leaves 4 roots with a decision collision against `shape_hand`'s 10. Next to the posterior family's per-seat voids and exhausted suits it adds nothing. What `everything` lacked was the shape of the remaining tile set relative to the viewer's hand, read after every play, rather than who may hold the tiles.

**The posterior at the decision does not route; the sequence does.** `support_end` reads the joint support once, at the next decision, the state the decision is made in. It leaves 96 roots with a decision collision. Read after every play, the same object leaves 4. The difference between consecutive patterns is the played tile's edge, so the per-play sequence names each played tile up to a relabeling of the pips. test-0797 shows it. The partner and the right-hand opponent play 1-1 and 2-2, or 2-2 and 1-1, and at the next decision the unseen tiles (2-0, 2-1 and the trump 5-4, each possible for any seat) and their support are identical. Only the pattern after the partner's play tells the two apart. So what routes is the observation up to pip relabeling, not a statistic of the posterior.

**What `everything` missed.** Under `everything`, the joint facts tell apart all 11 consecutive decision-colliding pairs that it could not: `support_hand` and `shape_hand` separate all 11, `support` 8, `support_end` 8. In test-0038 the right-hand opponent leads 0-0 or 1-1, with sixes trump. The unseen tiles all may lie anywhere, yet the pip graphs differ: after 0-0 the remaining double 1-1 sits on the pip without count, and after 1-1 the remaining double 0-0 sits on the pip whose suit still holds the 5-0.

**The four collisions that survive `support_hand` are pip order.** In each, the patterns where the two blocks differ are images under a relabeling of the pips that the rules do not respect. Each is told apart only by naming tiles (`support_named`) and by the likelihood bucket:

- test-0995 (fours trump): the right-hand opponent leads 3-1 or 5-3. The high end names the suit led. After 3-1 the viewer must follow with 3-2 (q 2/9). After 5-3 it is void in fives and may play 2-2 (q 2/3) or 3-2 (0).
- test-0393 (threes trump): the right-hand opponent plays 2-0 or 4-0 last, off suit. After the last play, swapping pips 2 and 4 maps one pattern onto the other and the viewer's 2-2 onto its 4-4. The rules do not respect the swap, because 4-2 led is a four. After 2-0, 2-2 is best (q 1 against 1/2). After 4-0, 4-4 is (1 against 0).
- test-0718 (fives trump): the right-hand opponent plays 4-2 or 6-2 last. 2-0 is worth 1/6 after 4-2 and 0 after 6-2.
- train-10273 (fours trump): the left-hand opponent plays 3-0 or 3-1 first. 0-0 is worth 0 after 3-0 and 1/4 after 3-1.

The other 28 roots with a collision under `support_hand` have domain collisions only: test-0112, test-0388, test-0663, test-0691, test-0897, test-0966, test-1004, test-1167, test-1214, test-1377, test-1380, test-1395, test-1567, test-1660, test-1665, test-1684, test-1721, train-0387, train-10278, train-20129, train-20345, train-20371, train-20411, train-20482, train-20492, train-20516, train-20545 and train-20659. `joint.json` lists every decision collision and counts the domain collisions. The domain roots here and above were named by re-grouping the same facts. A label up to pip relabeling cannot see these four by construction, since the rules read pip order. `joint_everything` separates them with the earlier families' facts: two pairs differ only in the likelihood bucket, one also in `outranked_by`, and one in the rank facts (`beats_all`, `follow_slots`, `beats_slots`, `outranked_by`, `suit_left`).

## Reading

- #241's object was the right next step. The joint support, read play by play, finds everything the posterior family missed, and with that family's per-seat facts it routes every decision. The label's own sequence of patterns, not the support at the decision, does the routing. Its seats add nothing to the family's voids and exhausted suits. The routing label is the observation itself, seen up to which pip is which. The same 13 domain roots remain until tiles are named.
- The ceiling of [[gus-posterior-labels-2026-10-22]] still governs. Any label's quotient refines fork, so 56 of 128 bounds fresh roots seen. A label routes this course by carrying the next decision's answer, four symbols at a root. The joint family carries it by carrying nearly everything, with 10,029 labels at the roots, so no fresh root is seen. A declared statistic that routes without that cost is still missing. #241's item 2, a learned label that predicts the next decision's class, is the open alternative and is parked for Jason.
- In the note's terms (`scratch/sources/42_minimal_decision_2026-10-21.md`), the kernel here is the observation up to the symmetries the rules respect. Pip relabeling is not one of them, because rank and the suit led read pip order. So "up to pip relabeling" is lawful but over-coarse in exactly four roots, and too fine everywhere it would need to merge.

## Scope

One fixture: three own tiles, acting last in trick five, nine plays left. Fixed G0 partner, pinned Zeb, uniform legal-consistent root prior. The support is the rule support, and the policies' zero-mass worlds are not removed. Relabeling permutes all seven pips and keeps trump and count as edge marks. A relabeling that kept pip order where the rules read it, or restricted to the symmetries of the declared trump, is untried. Each play's pattern is canonicalized on its own, and the relabelings are not tied across a block. The decision-only audit counts failures and coverage but no states. Every count is exact and reproducible with `python -u -m lem.quotient <study> --labels --joint`.

## Evidence

Tool `lem/quotient.py` (`--labels --joint`: `support`, `_closure`, `canonical`, `joint_pattern`, `JOINT_LABELS`), tests in `lem/tests/test_quotient.py` (18 pass), code commit `dc7c4214`. The run took 51 s under `scripts/cpu-slot` (9.6e11 instructions). Without `--joint` the reports stay byte-identical: default sha256 `14fde1ac…`, `--labels` sha256 `66f4333b…`. [joint.json](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1b9504a4d8434e8806fb685f1f3d064174d49d80/lem/gus-joint-labels-2026-10-22), sha256 `0e6a687f…`, verified on the Hub. Issue #241.
