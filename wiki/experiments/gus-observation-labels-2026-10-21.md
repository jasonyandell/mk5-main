---
title: "Tile-free observation labels for the decision quotient — none routes"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

[[gus-decision-quotient-2026-10-21]] found 144 fork states and 56 of 128 fresh roots in a state the training course holds. But fork cannot route: it does not say which observation leads to which next decision, so it is not a controller. Is there an observation label that names no tile, under which fresh roots are mostly seen states *and* the quotient still routes (every block from a state leads to one state)? If no label in the family tried does both, which collision breaks it? This is Night-Probe's second question for the night of 2026-10-21. It is a count on saved data; no model is involved.

## Labels

Each observed play gets facts that name no tile. They describe how the play stands in its trick and against the focal seat's remaining tiles, which are named only by slot (rank by tile id among them, as menu slots are):

- `seat`: relative seat (1 left opponent, 2 partner, 3 right opponent);
- `trump`: the tile is trump;
- `follows`: it leads, follows the led suit, or fails to follow;
- `count`: its count, 0, 5 or 10;
- `beats_all`: it outranks every remaining own tile in this trick;
- `winning`: it is winning the trick after it is played;
- `follow_slots`: for a lead, which own slots follow it;
- `beats_slots`: which own slots it outranks in this trick.

A block's label is the action's menu slot plus the sequence of play labels. The output is the shape output (menu size, maximizing slots). Successors are the set of states per (slot, label). A label **routes** when each such set holds one state. Bottom-up signatures and Moore refinement agree for every label, as they do for the three original interfaces. The original report stays byte-identical (sha256 `14fde1ac…`); the labels are behind `--labels`.

Each collision is a (slot, label) out of one node that reaches two states. It is classified as:

- a **decision collision**: the states answer differently, by menu size or by maximizing slots;
- a **domain collision**: they answer alike but accept different label sequences afterwards (the feasible-history domain the interface preserves, as in the note's tie-and-domain profiles).

## Result

| Label | States | Root ladder | Fresh roots seen | Fresh discretionary seconds seen | States that fail to route | Roots with a decision collision | Roots with a domain collision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| fork (no routing) | 144 | 10, 139, 139 | 56 of 128 | 3,338 of 3,338 | — | — | — |
| seat | 429 | 10, 173, 239, 239 | 7 of 128 | 3,267 of 3,338 | 216 | 192 | 91 |
| seat + follows | 872 | 10, 225, 245, 245 | 4 | 3,089 | 225 | 174 | 174 |
| brief: seat, trump, follows, count, beats_all | 3,062 | 10, 247, 254, 254 | 2 | 1,973 | 212 | 114 | 204 |
| brief + winning | 3,298 | 10, 247, 254, 254 | 2 | 1,863 | 210 | 107 | 203 |
| brief + follow_slots | 3,510 | 10, 247, 254, 254 | 2 | 1,752 | 210 | 51 | 209 |
| hand-relative: all eight facts | 3,746 | 10, 247, 254, 254 | 2 | 1,647 | 207 | 47 | 206 |

Raw has 8,257 states, 0 of 128 fresh roots seen and 14 of 3,338 seconds; shape has 477, 3 and 3,256. HALT and REJECT are additional in every row. Every failure is at a root. Below the root the next tile is forced, so a second decision's successors are a single state.

**No label in the family routes, and none makes fresh roots mostly seen.** Labels coarse enough to merge seconds (seat: 3,267 of 3,338) leave 192 roots with a decision collision. The finest label leaves 47 roots with a decision collision and 206 with a domain collision, and sees 2 of 128 fresh roots.

**The collision that breaks it.** Under the finest label, all 102 decision-collision groups and every colliding pair of blocks agree in all eight facts. No field of the family tells them apart. They differ in the identity of an opponent's or the partner's tile, and the next decision's maximizing slots differ with it. In test-0038, sixes are trump. The viewer sloughs 4-4 on trick five, and the right-hand opponent wins it and leads 0-0, 1-0 or 1-1. All three leads are off-trump with no count, beat both of the viewer's remaining tiles (4-3, 5-4) and are followed by neither. The choice between 4-3 and 5-4 then reads:

- after 0-0, 4-3 is better, 5/18 against 2/9;
- after 1-0, the two tie at 5/12;
- after 1-1, both lose.

In test-0348, the right-hand opponent's 5-3 or 6-1 under identical facts makes the next choice worthless or certain (0 and 0, against 0 and 1). The lead's identity decides which hidden tiles partner and left-hand opponent must follow with. Through the fixed partner and pinned Zeb, it also decides which hidden worlds remain. A fact stated against the viewer's own tiles sees neither.

Under coarser labels the facts do separate many collisions. Under brief, `winning` tells apart 394 colliding block pairs, `follow_slots` 238 and both 31; 2,027 differ in no fact. Adding `follow_slots` removes every menu-size collision: the next menu depends on which own tiles follow a lead.

## Reading

- Routing and reuse pull in opposite directions inside this family. Every fact added to route splits training states, and fresh roots stay unseen: 2 to 7 of 128, against 56 for fork, which routes nothing.
- What is missing is not a relation to the viewer's hand but information about the hidden tiles: which remain out, and who must play them. A label that could route would carry a statistic of the posterior, such as unseen trumps or count, suits exhausted per seat, or the field's likelihood of the play. That is the note's warning that the quotient depends on the kernel, seen from the label side. It is the next family to try (#239).
- Domain collisions (206 roots under the finest label) come from preserving which label sequences are feasible. A decision-only interface that accepts any label would drop them, as the note's 11/15 counts drop its 23/31 tie-and-domain profiles. The decision collisions remain either way.

## Scope

One fixture (three own tiles, acting last in trick five, nine plays left), fixed G0 partner and pinned Zeb, uniform legal-consistent root prior. The labels are one hand-chosen family of eight facts, not every tile-free label: a label built from posterior statistics is untried. Every count is exact and reproducible with `python -u -m lem.quotient <study> --labels`.

## Evidence

Tool `lem/quotient.py` (`--labels`), tests in `lem/tests/test_quotient.py`; run 9.5 s under `scripts/cpu-slot`. [labels.json](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/db97e13f4820958731b8ee6a4b17d9e2b4d489bd/lem/gus-observation-labels-2026-10-21) (verified by sha256). The companion question, whether Gus's representation already carries the class, is [[gus-decision-class-probe-2026-10-21]].
