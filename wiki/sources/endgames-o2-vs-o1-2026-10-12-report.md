---
title: O2 vs O1 on three-trick endgames — the lab's report (2026-10-12)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

The verbatim report of the Opus lab agent behind [[endgames-o2-vs-o1-2026-10-12]]
(issue #113), in two messages. Scripts and data are in `scratch/endgames-o2/`
(gitignored): `eg.py`, `run.py`, `analyze.py`, `show.py`, `probe.py`, `stage.py`,
`RUNS.md`, `out_r1.jsonl`…`out_r4.jsonl`. Part 1 was cut by the transport after
"Deal. Seed 421, trump 2s, seat 0"; part 2 resumes there.

## Part 1

O2 beats the O1 fixed point in real-42 three-trick endgames, and in about half of them the gap stands on reached information sets. That is the reverse of mini-42, where the Walt partnership map always reached O2. The gap is small in points: 0.027 of final margin per endgame against the Walt-map fixed point.

**Definitions implemented** (`eg.py` docstring)
- **Game.** A random deal with random pip trump, bidder and bid, played for 4 tricks by the deployed net at all four seats (`margin_d250k_h128x64`, rules and record from `scenario/`). The viewer V's 3 tiles are fixed and common knowledge. The prior is uniform over V's void-consistent fiber (1,680 worlds or fewer). Partner P's hand is private. Opponents play the deployed net, which is deterministic.
- **Utility.** The team's final point margin. One team point moved is 2 points of margin.
- **Sample.** Four endgames per deal: both teams, and V as the first or second team seat to act in trick 5.
- **O1 plain.** Strict-improvement best-response dynamics from (net, net). The incumbent is kept off-plan and on ties. It ends at a mutual best response and cannot cycle.
- **O1 Walt.** The C3 lab's partnership map: exact Walt at every information set, coherent belief, fallback to the fiber off path, lowest-id ties. A fixed point is certified only when two levels are equal at every team information set reachable under any team play.
- **O2.** The common-information coordinator: one tile at V's turn, a prescription (P's hand to tile) at P's turn. It is exact by branch and bound with the subadditive bound C(A∪B) ≤ C(A)+C(B). Every O2 was played back world by world as seat-local play and realized its value exactly (533 of 533).
- **O2+.** The coordinator also sees P's hand. This is an upper bound on O2.

**Numbers.** 566 endgames from 142 deals, all seven trumps. O2 is exact on 533. Intervals are 95%, from a bootstrap that resamples deals.

| quantity, expected final margin | n | mean [95%] | share > 0 | max |
|---|---|---|---|---|
| O2 − O1 plain | 533 | +0.115 [+0.084, +0.154] | 73.2% | 6.40 |
| O2 − O1 Walt (fixed or cycle) | 484 | +0.027 [+0.022, +0.033] | 49.4% | 0.58 |
| O2 − O1 Walt (certified fixed) | 344 | +0.028 [+0.021, +0.037] | 46.5% | 0.58 |
| O1 plain − (net, net) | 533 | +1.64 | 93.6% | |
| O2+ − O2 | 533 | +0.46 | 78.2% | |

- No O1 value ever exceeded O2.
- **When the partner acts first in trick 5,** the Walt gap is larger: +0.036, at 60% of endgames. When V acts first it is +0.019, at 39%.
- **Bidders and defenders** are about the same.

**Where the gap stands.** At each fixed point below O2, I pinned one partner's moves at every information set the fixed point reaches and solved the rest exactly. Pinning both partners always returned the fixed point's value, which is the expected sanity result (449 of 449).

| fixed point below O2 | one partner changes on path, the other only off path (the mini-42 mechanism) | no single-partner freeing gains anything (a convention) | share of gap the best single pin closes |
|---|---|---|---|
| Walt (171 diagnosed) | 27 fully closed | **78** | 29% |
| plain (278 diagnosed) | 72 fully closed | 29 | 72% |

- **Plain fixed points** are stuck mostly on unreached information sets, as in mini-42.
- **An off-path repair** keeps the plain fixed point on reached sets, puts Walt replies off path, and reruns the dynamics. It reaches O2 in 87 of 390 gapped endgames and improves 229.
- **Walt's own off-path replanning** removes most of the unreached-set traps. About 70% of what remains needs both partners to change moves they actually make.

**Cycles.** The Walt map gave 359 certified fixed points and 149 cycles, and 58 runs hit the 50 s cap. Every cycle had period 2. The flipping moves sat at information sets unreached at every cycle level, and the profile value never changed. The plain dynamics cannot cycle.

**Counterexample (convention on reached sets).** Run it with:
```
python -u show.py 421 def first
```
- **Deal.** Seed 421, trump 2s, seat 0 […cut; continued in part 2]

## Part 2

COUNTEREXAMPLE: a convention on reached information sets. Reproduce with `python -u show.py 421 def first`.
- Deal seed 421: trump 2s, seat 0 bid 30. After 4 tricks, team 0 has 0 points and team 1 has 39. Seat 1 leads trick 5.
- Initial hands: s0 {2-0 3-3 4-1 4-2 5-2 5-3 6-4}, s1 {0-0 1-0 1-1 2-1 2-2 6-1 6-6}, s2 {3-0 3-1 4-0 4-3 4-4 5-1 6-3}, s3 {3-2 5-0 5-4 5-5 6-0 6-2 6-5}.
- True endgame: s0 {4-2 5-2 5-3}, s1 {0-0 1-0 6-1}, s2 {3-0 4-3 5-1}, s3 {5-4 6-2 6-5}.
- Voids shown, unseen tiles only: s1 {3-0 4-3 5-3}; s2 {4-2 5-2 6-1 6-2 6-5}; s3 {1-0 3-0 4-3 5-1 5-3 6-1}.
- The team is the defenders. V is seat 1 holding {0-0 1-0 6-1}; P is seat 3. The fiber has 64 worlds and 10 possible P hands. Opponents seats 0 and 2 play the deployed net.
- Values, as expected final margin for the defenders: net/net 40.344; plain O1 40.594; Walt fixed point 40.625, certified at level 3; O2 40.6875, realized exactly; O2+ 40.9375.
- At the Walt fixed point, V's best response is 40.625 and P's is 40.625, so it is a mutual best response.

The moves. In both affected worlds P holds {4-2 5-4 6-2}. Seat 0 holds {5-1 5-2 6-5} or {5-2 5-3 6-5}, and seat 2 holds the rest.
- Walt: s1 0-0, s2 3-0, s3 4-2 (trumps in), s0 5-2 (overtrumps and wins). Then s0 6-5, s1 6-1, s2 4-3 or 5-1, s3 6-2. Then s3 5-4, s0 5-1 or 5-3, s1 1-0, s2 5-3 or 4-3. The defenders finish +40.
- O2: s1 0-0, s2 3-0, s3 5-4 (sluffs), s0 5-1 or 5-3 (the net does not trump; V's 0-0 wins). Then s1 leads 6-1, s2 follows, s3 4-2, s0 6-5. Then s3 6-2, s0 5-2, s1 1-0, s2 4-3. The defenders finish +42.
- The gap is 2 worlds × 2 points / 64 = 0.0625.

Team decisions that differ (Walt vs O2):
1. P holding {4-2 5-4 6-2} after 0-0 3-0: 4-2 vs 5-4.
2. V holding {1-0 6-1} after 0-0 3-0 5-4 5-1: 1-0 vs 6-1.
3. V holding {1-0 6-1} after 0-0 3-0 5-4 5-3: 1-0 vs 6-1.

Why no single-partner change closes it: the constrained coordinator measured both one-partner options exactly, and neither gains anything.
- Pin P at every information set the Walt fixed point reaches, with V free everywhere and P free off path: the result is 40.625, a gain of 0.
- Pin V the same way, with P free: also 40.625.
- V's information set after the 5-4 sluff is reached under Walt, because other P hands sluff the 5-4 there. So V's 1-0 is a best reply to what the sluff currently means.
- P's sluff pays only if V reads it as "I hold the trumps, lead your 6-1", and that reading pays only if P sluffs from exactly this hand. That is the S/R structure of the theory page's §5.2, on reached sets.

A second example closes by the mini-42 mechanism: `python -u show.py 119 bid second`, 30 worlds.
- Walt −30.800, O2 −30.733.
- Pinning P's reached moves gains nothing. Pinning V's reached moves reaches O2.
- P switches from 6-3 to 5-4 on the 2-2 lead. V must then answer 3-2 instead of 1-1 at an information set Walt never reaches, where Walt's fallback reply is wrong.

CYCLES
- The Walt map gave 359 certified fixed points, 149 cycles, and 58 runs that hit the 50 s cap, out of 566 endgames.
- Every cycle had period 2.
- In all 149, the flipping information sets were unreached at every level of the cycle, and the profile value was identical across levels.
- The plain dynamics cannot cycle, because every step strictly improves.
- No cycle touched play.

WHAT I COULD NOT FINISH, AND WHAT IT WOULD NEED
1. 33 of 566 endgames (5.8%) hit the coordinator's 20 s cap, so O2 is not exact there. All of them have 700 to 1,680 worlds, so the sample leans toward smaller fibers. Lower bound max(plain, Walt) and upper bound O2+ are printed by analyze.py. Closing them needs a tighter bound at P's prescription step, or a mixed-integer formulation solved with HiGHS (scipy 1.17 has milp). That would take roughly 1 to 2 minutes per endgame, which needs a written exception to the 5-minute run cap.
2. Signals run one way only, P to V, because V's hand is common knowledge in the prior. The both-private prior has up to 369,600 deals before voids and needs prescriptions at both seats. The same coordinator code handles it with V's turn turned into a prescription, but the branch-and-bound cost grows sharply. Try it first on small fibers late in trick 5.
3. The prior uses voids only and ignores what the net's plays in tricks 1 to 4 reveal, as the issue specified. A net-coherent prior is a filter on the starting worlds.
4. There is no make-utility variant: margin only. One team point is 2 margin points, so the Walt gap of 0.027 margin is 0.014 team points.
5. O3 (team maxmin) was not attempted.
6. 58 Walt-map runs hit the 50 s cap, all on the biggest fibers.

SCRIPTS, all in /Users/jason/code/mk5-main/scratch/endgames-o2/
- eg.py: game, exact solves, plain and Walt O1 dynamics, O2 coordinator (plus pinned variant), O2+, realization check, reached sets, off-path repair, where_gap_stands.
- run.py: the driver. `python -u run.py SEED0 NDEALS OUT.jsonl WALL WALT_CAP`, 16 processes.
- analyze.py: deal-clustered bootstrap aggregation.
- show.py: one endgame in full.
- probe.py, stage.py: timing probes.
- RUNS.md: the run log.
- Data: out_r1.jsonl to out_r4.jsonl. r1 has no pin fields. out_smoke.jsonl is the overrun smoke run.

RUNS.md WALL TIMES
- Probe: 59 s.
- Smoke run: overran, killed at about 300 s because the branch and bound had no effective cap. It was fixed after that.
- Stage diagnosis: 100 s.
- r1: 247 s, 152 endgames. This is the longest compliant run.
- r2: 188 s. r3: 181 s. r4: 193 s.
- show.py, inline pin test, analyze.py: under 10 s each.
- Caps inside each run: coordinator 20 s per endgame; Walt map plus repair 60 s in r1 and 50 s in r2 to r4; each pinned solve 10 s. Every run after the smoke run used timeout 295.
