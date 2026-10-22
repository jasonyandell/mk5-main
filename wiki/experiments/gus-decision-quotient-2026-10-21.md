---
title: "Gus decision quotient of the contingent course"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

How many decision states does the saved contingent-teacher course hold, and does a fresh root ever reach a state the course already contains? [[gus-historical-choice-wish]] adopts the formulation "what a sequence of choices means for the choices still ahead". Jason's note *A minimal comparison algebra for a declared 42 family* (2026-10-21) makes it exact: two histories are one state when every future output agrees, the coarsest right congruence of the decision interface, computed bottom-up and verified by partition refinement; for its 560-deal no-trump ending with a two-type partner it finds 767 decision states plus HALT and REJECT, and a class ladder of 1, 18, 31, 31 by block depth. This measurement applies the same quotient to the 256 eligible roots of [[gus-contingent-maskbreadth-2026-10-20]] (128 TRAIN, 128 TEST), read from their saved `teacher.json` graphs. No model is loaded, nothing is trained, and the research loop remains stopped.

## Interface

A state is a focal information node: the [[gus-tiny-gpt]] token history at an own decision. The output is the legal menu with the complete maximizing set, ties included. A block is an own action followed by the observed plays up to the next own decision; the final tile leads to HALT; absent blocks lead to REJECT. Values are not outputs, so a forced winning tile and a forced losing tile are one state. Three label interfaces are compared:

- **raw** — tiles as the student sees them: menu and maximizing set as tile ids, observations as (relative seat, tile) pairs.
- **shape** — no tile identity: menu size, maximizing slots by tile-id rank within the menu, and under each slot the multiset of successor classes.
- **fork** — as shape, but under each slot the *set* of successor classes: which next decisions the public evidence can lead to, not how many observations realize each.

Bottom-up signatures and Moore refinement from the output partition must produce the same partition; the refinement rounds are the block-depth ladder. All arithmetic is exact. The tool is `lem/quotient.py`, tested in `lem/tests/test_quotient.py`.

## Result

The course has 50,754 information nodes: 256 root decisions, 10,302 second decisions and 40,196 final tiles. Menus: 43,578 forced, 6,953 two-way, 223 three-way (three-way menus occur only at roots; a root that won trick five leads with two tiles). Discretionary unsettled nodes: 6,466. The 128 TRAIN roots reproduce the registered dataset exactly, 3,000 rows with 420 nonflat; TEST holds 3,466 with 412. Nonflat by stage: 130 of 256 roots, 702 of 6,210 second decisions.

| Interface | States | Ladder by block depth | Root ladder | States per root | Fresh roots in a TRAIN state | Fresh second decisions in a TRAIN state |
| --- | ---: | --- | --- | --- | ---: | ---: |
| raw | 8,257 | 753, 8,257, 8,257 | 253, 256, 256 | 6 to 121 | 0 of 128 | 14 of 3,338 |
| shape | 477 | 11, 462, 477, 477 | 10, 237, 252, 252 | 3 to 22 | 3 of 128 | 3,256 of 3,338 |
| fork | 144 | 11, 144, 144 | 10, 139, 139 | 3 to 6 | 56 of 128 | 3,338 of 3,338 |

HALT and REJECT are additional in every row. Under raw, the per-root compression is 10/7 to 323/7 nodes per state, median 569/115; the sum of per-root states is 9,031 against 8,257 course-wide, and 55 raw classes appear in more than one root, 28 of them the forced final tiles.

At the 223 three-way nodes all 13 weak orders of three values occur: 106 triple ties, 74 pair ties, 43 strict orders. At the 6,243 two-way discretionary nodes: 5,528 ties, 384 and 331 for the two strict orders.

## Reading

- Per-root controllers are small, as in the note: a 90-world ending with at most 121 raw states, median about five nodes per state, and 50,754 nodes to 8,257 states across the course. This is the analog of the note's 24,089 nodes to 767 states.
- The raw interface never repeats a root. No fresh root, and 14 of 3,338 fresh second decisions, land in a state the training course holds. A student that reads raw tiles cannot transfer by lookup; it must compute the class from the history. This is a statement about the interface, not a diagnosis of the failed transfer gates.
- The raw and fork ladders stabilize after one block; the shape ladder needs two (11, 462, 477, 477; at the roots 10, 237, 252, 252). A forced last tile does not by itself bound the ladder at one block: with at most n own tiles the bound is n − 1 blocks, here two, and a three-tile course can need both (proved in Lean, [[comparison-algebra-2026-10-21]]). The root ladders 253, 256, 256 and 10, 139, 139 have the shape of the note's 1, 18, 31, 31.
- The course holds 139 distinct fork structures among 256 roots, and 56 of 128 fresh roots share one with training. Fork is tile-free and ignores routing, which observation leads to which next decision, so it bounds merging from above. It is not a controller.
- All 13 weak orders are realized, and ties are the majority class: half the roots and nearly nine tenths of second decisions are flat. The flat row is a class of the hexagon, not an absence of information.

## Follow-ups

- [[gus-decision-class-probe-2026-10-21]]: does Gus's representation already carry the class? Not linearly at the query token on fresh roots; flatness yes, from pretraining (#240).
- [[gus-observation-labels-2026-10-21]]: a tile-free label that routes? None in an eight-fact family; the collision is a hidden tile's identity (#239).
- [[gus-posterior-labels-2026-10-22]]: posterior statistics of the unseen tiles route better (7 roots with a decision collision, not 47) but see 0 of 128 fresh roots; the only routing label that keeps 56 is the next decision's answer (#239).

## Scope and verification

One fixture (three own tiles, acting last in trick five, nine plays left), fixed G0 partner and pinned Zeb, uniform legal-consistent root prior carried through observations. The quotient depends on the kernel and the interface; the note says so for its own family. The tool rebuilds every tree from saved rows and audits them exactly: parent links, every legal action explored, children partitioning the cell mass, and Bellman consistency `q[a] * mass = sum over observations of child.mass * max child.q`. All 256 graphs pass. That re-verifies the saved DP from its rows; it does not replay the native field or any model. The note's realization hexagon and its mixing boundary are not exercised: nothing here mixes beliefs. Inputs are the per-root `teacher.json` files in the published maskbreadth [evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/762b5a71a622a1ff6df8fed47e50ebe5b2a43e44/lem/gus-contingent-maskbreadth-2026-10-20/evidence); the report is reproduced by `python -u -m lem.quotient <study> --out report.json`. The note's own bundle (atlas, tables, runtime) was not in hand; its figures are cited, not reproduced. Tools for extending the course are inventoried in [[gus-continuation-teaching-tools]]; the first teacher is [[gus-contingent-teacher-2026-10-20]]. Silas reproduces every count of the report independently, exactly ([[comparison-algebra-2026-10-21]]).
