---
title: O2 vs O1 on three-trick endgames — the first convention measured (2026-10-12)
kind: experiment
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

## Summary

Issue #113, the last item of the research-night list ([[research-night-2026-10-12]]
§ What we are missing mathematically, item 4). One Opus lab agent, one afternoon,
every run under five minutes. Verbatim report: [[endgames-o2-vs-o1-2026-10-12-report]].

**Question.** Every referent measured on research night held the other seats fixed or
moved them one at a time, so none could see a convention: two partners changing moves
they actually make, each change worthless without the other. In mini-42 the Walt
partnership map reached O2 (the common-information coordinator's optimum) in 1,840 of
1,840 priors; every gap stood on an unreached information set. Does that hold in real 42?

**Answer: no.** On real-42 three-trick endgames O2 beats the O1 fixed point, and about
half of the Walt-map gap stands on reached information sets. The gap is small in points.

## Setup

A random deal, random pip trump, bidder and bid; four tricks played by the deployed net
`margin_d250k_h128x64` at all four seats; then the last three tricks solved exactly. The
viewer V's three tiles are common knowledge, the prior is uniform over V's void-consistent
fiber (≤ 1,680 worlds), the partner P's hand is private, and the two opponents play the
deterministic net. Utility is the team's final margin (one team point = 2 margin points).
Four endgames per deal (both teams; V first or second to act in trick 5).

Referents: **O1 plain** (strict-improvement best-response dynamics from the net profile;
cannot cycle), **O1 Walt** (the C3 lab's partnership map: exact Walt at every information
set, coherent belief, fiber fallback off path, lowest-id ties; a fixed point certified only
when two levels agree at every reachable team information set), **O2** (the coordinator:
a tile at V's turn, a prescription from P's hand to a tile at P's turn; exact by branch
and bound; every O2 played back world by world realized its value, 533 of 533), **O2+**
(the coordinator also sees P's hand; an upper bound on O2).

## Numbers

566 endgames from 142 deals, all seven trumps; O2 exact on 533; 95% intervals from a
deal-resampling bootstrap.

| expected final margin | n | mean [95%] | share > 0 | max |
|---|---|---|---|---|
| O2 − O1 plain | 533 | +0.115 [+0.084, +0.154] | 73.2% | 6.40 |
| O2 − O1 Walt (fixed or cycle) | 484 | **+0.027 [+0.022, +0.033]** | 49.4% | 0.58 |
| O2 − O1 Walt (certified fixed) | 344 | +0.028 [+0.021, +0.037] | 46.5% | 0.58 |
| O1 plain − (net, net) | 533 | +1.64 | 93.6% | |
| O2+ − O2 | 533 | +0.46 | 78.2% | |

No O1 value ever exceeded O2. The Walt gap is larger when the partner acts first in
trick 5 (+0.036, at 60% of endgames) than when V does (+0.019, at 39%). Bidders and
defenders are alike. In team points the Walt gap is 0.014 per endgame.

## Where the gap stands

For each fixed point below O2, pin one partner's moves at every information set the fixed
point reaches and solve the rest exactly (pinning both always returns the fixed point's
value, 449 of 449).

| fixed point below O2 | closes with one partner changing on path (the mini-42 mechanism) | **no single-partner freeing gains anything: a convention** | share of gap the best single pin closes |
|---|---|---|---|
| Walt (171 diagnosed) | 27 | **78** | 29% |
| plain (278 diagnosed) | 72 | 29 | 72% |

Plain fixed points are stuck mostly on unreached sets, as in mini-42; an off-path repair
(keep the plain fixed point on reached sets, put Walt replies off path, rerun) reaches O2
in 87 of 390 gapped endgames and improves 229. Walt's own off-path replanning removes most
of those traps. **About 70% of what remains needs both partners to change moves they
actually make.** That is the first convention measured in this project.

**The counterexample** (`python -u show.py 421 def first` in `scratch/endgames-o2/`):
trump 2s, defenders, V = seat 1 holding {0-0 1-0 6-1}, P = seat 3, 64 worlds, 10 possible
P hands. At the Walt fixed point P trumps in with 4-2 on the first lead and the defenders
finish +40 in the two worlds where P holds {4-2 5-4 6-2}. Under O2, P sluffs 5-4 instead,
the net does not trump, V's 0-0 wins, V reads the sluff as "I hold the trumps, lead your
6-1", and the defenders finish +42. Pinning P alone gains 0; pinning V alone gains 0;
both moves must change. V's information set after the sluff is reached under Walt, because
other P hands sluff the 5-4 there, so V's current reply is a best reply to what the sluff
currently means. This is the sender/receiver structure of the theory report's §5.2, on
reached sets. Gap 2 worlds × 2 points / 64 = 0.0625.

## Cycles

The Walt map: 359 certified fixed points, 149 cycles, 58 runs at the 50 s cap. Every cycle
had period 2, every flipping information set was unreached at every level, and the profile
value never changed. Consistent with the Lean witness (`rung_seeded_no_settle` and the
coherent-belief cycle in `scenario/lean/Cycle.lean`) and with research night's T2: coherent
beliefs can cycle; here the cycles never touch play.

## What this means for Zeb

1. **O1 is the right referent for one seat and the wrong one for a partnership**, by a
   measured amount: 0.027 margin per three-trick endgame against the Walt map, half of it
   a convention. Small, but it is the first nonzero O2 − O1 number, and it says the
   partnership operator is where the next mathematics is, not the single-seat one.
2. **The partnership map's cycles are harmless** on these endgames: period 2, off path,
   value constant. Replanning everywhere, off path included, still removes most traps.
3. **Conventions cannot be learned from Zeb's own pour**: the pour records what the
   one-seat operator does, and the convention is a joint deviation that is worthless to
   either seat alone (a self-confirming equilibrium with a misspecified partner model, the
   same Berk–Nash shape as "data stops at parity").

## Not finished

33 of 566 endgames (5.8%) hit the coordinator's 20 s cap, all with 700–1,680 worlds, so
the sample leans small-fiber (bounds printed by `analyze.py`; a MILP via HiGHS at 1–2
min per endgame would close them). Signals run one way only (P to V) because V's hand is
common knowledge; the both-private prior (≤ 369,600 deals) needs prescriptions at both
seats. The prior uses voids only, not what the net's first four tricks reveal. Margin
utility only; no `make` variant. O3 not attempted.

## Links

[[research-night-2026-10-12]] · [[scenario-player]] · [[the-wall]] · [[strategy-fusion]]
· [[endgames-o2-vs-o1-2026-10-12-report]]
