---
title: Perf log — append-only field notes
kind: topic
first_seen: 2026-07-18
last_updated: 2026-07-18
status: active
---

Append-only field notes for performance work, per Jason's 2026-07-18
directive: this log replaces GitHub issues for the perf program. Entries are
terse but rich — what was tried, how the numbers moved, what died. Newest at
the bottom. Laws distilled from here graduate to [[perf]]; never edit old
entries (corrections are new entries).

## 2026-07-18a — program registration: the net-free kernel

**Decision (Jason):** the net-free kernel is the way. No per-experiment
justification needed — solving more of the game faster is a **stable eval**
(invariant of the game, not of our model generation) that will run forever.
Track in this log, not issues. Fable xhigh on the hardest parts.

**Program:** fork the wavefront engine ([[walt-spec]] §4) into a zero-torch
kernel (`walt/kernel/`): bitboard hands, LUT tricks, SoA waves, pluggable
profiles (compiled σ tables / stochastic profiles / regret-matching). Three
consumers, named per the distill-for-what rule: (1) **exploitability meter**
— exact single-seat best response vs any frozen profile (counter-walt lands
free); (2) **CFR+ reference profiles** at H4 + the frozen-root stable eval;
(3) H5/H6 horizon pushes. Honesty line, registered now: 42 is two-team
zero-sum with private hands — CFR's Nash guarantee is a 2-player theorem;
we claim "low-exploitability reference priced by exact BR," not equilibrium.
v1 exploitability = single-seat deviation (partner stays on profile);
team-pair deviation is harder and explicitly out of scope.

**Priors (registered before building):**
- P1: net-free BR on the 46-fixture suite ≤ 0.5 s total (≥17× vs the 8.5 s
  net wavefront), p50 per fixture < 2 ms.
- P2: σ-table compilation (one net pass per root subgame) costs about one
  current solve; after that, re-solves are net-free — CFR iterations
  amortize to ~free.
- P3: CFR+ reaches single-seat exploitability < 0.1 pt at a typical H4 root
  in ≤ 500 iterations, wall seconds/root.
- P4: exploitability of the deterministic walt-BR profile (counter-walt) is
  material — wide prior, 0.5–3 pts/hand at H4.
- P5: net-free H5-capped(512) BR p50 < 100 ms (kernel makes rung 1 of the
  ladder unnecessary; the ladder starts at H6).

## 2026-07-18b — kernel lane landed (commit 27a3b63b)

K1 46/46 (best_move, value, every root_value ≤1e-9, identical n_nodes).
**P1 PASS ×9**: 0.188 s/46 fixtures (45× vs the 8.5 s net wavefront, 666×
vs the recursion), p50 0.90 ms. **P2 confirmed**: compile_sigma = 0.79× one
net solve (5.2M rows), the only net-touching call; after it, deterministic
BR is backward-pass-only with payoff43 AND belief weights swappable per
re-solve (~0.9 ms). **P5 PASS ×28**: H5-cap512 BR p50 3.49 ms (uncached
rewalk 17 ms also under). 24.7 ns/node vs 1339 net-bound. Stochastic
full-support stress: blowup p50 526×, max 5219× (777009: 446M nodes, exact
via per-root-move chunk fallback, 141 s) — [[walt-spec]] §3's "×10²–10⁴"
now measured. numba DECLINED on profile evidence: deterministic BR is
bincount/reduceat-bound in pure numpy; keeping parity single-sourced beat
a speculative jit. Design note: the reachable tree is profile-dependent,
so the subgame is lazy and one generic wave engine serves six σ-providers
(net/table/rule/uniform/dict/full-width).

## 2026-07-18c — CFR lane landed (commit 27a3b63b)

CFR+ (RM+, alternating, linear averaging) verified: V1–V5 green — CFR ==
LP-exact values on 2p-izable toys (|d| ≤ 7.7e-4), make-payoff toy exact,
773 isets legality-audited, sign-symmetry exact. Cross-implementation
gate: same cfr_solve bitwise-identical (1e-12 traces) on the pure-python
reference vs the fast kernel; two independently-built br_solve agree
at 1e-9. **Measured fact**: ~500 toy seed/pin configurations, zero
required mixing — late-game 42 vs deterministic pins admits pure
equilibria everywhere searched (concealment, not game value, is where
mixing must earn — feeds #77). H4 scale estimate: full-width ~80k
nodes/world → CFR on world-capped roots (K≤512, ~40M nodes) is the
doctrine; full-u CFR exceeds memory by design.

## 2026-07-18d — integration first light: P3 split verdict, first rent number

Real H4 root (evalset 555091, 8 worlds, defense): **gap 0.0296 pts in 40
iterations** — P3's iteration prior holds with 10× room. But wall = 167 s
at EIGHT worlds (kernel BR at the same root: 0.29 ms): cfr.py's traversal
is python-recursion-bound, ~1 s/seat-traversal; cap-256 extrapolates to
hours/root. Killed the cap-256 run; re-tasked the CFR lane to vectorize
traversal + regret updates over the kernel's SoA tree (gates: toy-trace
parity vs the verified loop, 555091 rerun ≤5 s, one cap-256 datapoint).
**First exact rent number**: walt-vs-jud value 17.38 (defense orientation
gain +2.28 pts) vs CFR reference self-play value 19.65 at the same root —
the field-model rent priced exactly, single root, direction as predicted
by the #72 prior.

## 2026-07-18e — vectorized CFR (commit follows) + the 12-root mini-reference

CFR lane rewrote its traversal onto the kernel's SoA tree: **bitwise
parity** vs the verified loop (0.0 drift on traces AND exported profiles;
both engines kept, loop pinned in test_cfr_parity), 555091 rerun 166.9 s →
4.57 s (traversal itself ~110×), cap-256 median root 134 s / 7.4 GiB peak.
Remaining wall is BR-vs-mixed-profile gap pricing, not iteration (1.1
s/iter at ~15M slots, zero python per node).

**Mini-reference, 12 stratified evalset roots, cap 256** (P3 verdict:
PASS in full): every root reached gap ≤0.05 pts in ≤40 iterations —
**iteration count is world-scale-invariant** (10 → 33,740 true worlds);
wall 6–262 s/root, median ~115 s. **Rent distribution** (walt BR-vs-jud
value minus CFR reference self-play value, hero orientation): median
**+1.9 pts/root**, mean +2.2, range −0.08…+5.63, never materially
negative; two fully-decided roots at exactly 0. Reading: at the exact
level, ~2 pts/root of walt's H4 edge is jud-specific — the per-root
version of the #72 transfer question. Caveat kept visible: the comparison
conflates jud's exploitability with opponent-population difference; it is
the rent *indicator*, not the graded transfer test.

**Deliberately left on the table** (wall-clock discipline, 4 a.m.): the
full 200-root reference sweep (~4–7 h single-process; parallelism is
RSS-bound at 7.4 GiB/root until int32 narrowing — both queued); overnight
H4 corpus regen for the scar probe; bulk profile export vectorization
(~15 % of CFR wall); br_every tuning (halves gap-pricing cost).

## 2026-07-18f — promotion: the kernel is [[hoyt]]

Jason's call: the instrument gets its own name and entity. `walt/kernel/`
→ `hoyt/` (package + tests + the frozen eval anchor
`hoyt/evalset_h4_v1.jsonl`), entity page [[hoyt]] carries the full
synthesis. Named for "according to Hoyle" — the authority you appeal to
on games; ours is computed. Identity clarified by the split: walt is a
player (BR vs a modeled field), hoyt is the referee (no model in the
loop). Entries a–e above predate the rename and say `walt/kernel` —
historically correct, left as written (this log is append-only). 40
tests green post-move; K1 parity fixtures stay in `walt/tests/` (the
gate is a walt↔hoyt cross-check by nature).

## 2026-07-18g — CFR lane perf day: the RSS hog was python objects

Paired-bench anchor: evalset 555006, cap 256, 40 iters (gap 0.039), quiet
box (wall reproduced last night's 119.8 s at 120.9). **Anatomy first**
(law 6): the numpy arrays were innocent — stored waves 1.13 GiB, retained
iteration structure ~1.2 GiB — the 7-10 GiB peak was **python objects**:
`exp_entries` (2.75M tuples-of-tuples inc. path tuples) plus TWO resident
2.75M-entry dict profiles per solve, and ~5.5 s of the 7.8 s build was
the tuple-building loop.

**Landed** (every step value/gap/trace BIT-IDENTICAL on the anchor; gates
I1/I2 + P1/P2 + 21 pytest green throughout):
- `want_strategy=False` in gap pricing (BR strategy was extracted and
  discarded): br 43.9→41.6 s.
- **Columnar StochasticProfile** (the day's centerpiece): frozen columnar
  store keyed (seat<<28|hand, h1, h2) under a mixed 64-bit sort key with
  exact verification; `set_bulk` loads the whole CFR export in one call —
  hash keys straight off the walk's rolling 128-bit path hash, zero
  python objects per entry; `_DictProfileProvider` fully vectorized
  (searchsorted + segment assembly, identical per-entry composition).
  **wall 120.9→91.8 s, export 12.2→0.87 s (14×), br 43.9→28.6 s, build
  7.8→5.4 s, peak RSS 10.6→7.0 GiB.** Every mixed-profile BR (counter-walt
  #77 included) rides the same path.
- dtype narrowing: stored waves int32/int8, leaf_pts int16 (RSS
  6.98→6.76); PS/GID/uedge deliberately KEPT int64 — they feed
  bincount/fancy-indexing in the hot loop and numpy converts non-intp
  index arrays per call (narrowing would pessimize).

**Negative results** (append-only honesty): `_rm_plus_update`
add.at→bincount = wash (numpy 2.x ufunc.at is already fast on sorted
indices); `_wave_pass` reach-fusion (two fewer slot-size temporaries) =
wash. Iterate (56.9 s, now 62% of wall) is bandwidth-bound vectorized
numpy; the next iterate lever is a compiled kernel, declined again per
the single-source parity doctrine.

**BR anatomy** (kills a queued idea with data): hidden-hero BR walks
~19M nodes in ~3.8 s/seat vs hero-me 3.2 s at the same node count — the
~170-per-seat group loop costs only ~0.6 s/seat (~15% of br). Batching
groups into one walk (wave-0 seeding) is NOT worth the engine change;
the walk size is the cost. Logged, not built.

Net: cap-256 reference solve 120.9→~92 s (1.31×), peak RSS 10.6→6.7 GiB
(1.58×) — sweep parallelism on the 48 GiB box goes from ~4 to 5-6
workers with headroom.
