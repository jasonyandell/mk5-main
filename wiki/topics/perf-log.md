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

## 2026-07-18h — the cap-ledger cascade: refsweep + wall-budget verdicts

**Directive (Jason):** "when we're doing perf, it is for the sake of perf...
I want to optimize 4th-play eval count per wall clock and I'll tolerate cap
results as early-cap games are an interesting artifact to analyze on their
own and are available in bulk." Headline metric registered: **H4
evals/hour**.

**Lineage:** the gomoku VCT cascade labeler (gomoku wiki
`topics/vct-cascade-labeler.md` + `topics/mega-vct-solver.md`) — solver
returns result-or-cap as first-class verdicts, every item gets an explicit
ledger row (no absence-as-state), a budget ladder deepens only the
shrinking capped-survivor tail, and the deepening curve is itself the
artifact. 42's cap is STRONGER than gomoku's boolean `hit_cap`: a CFR stop
at any point is a QUANTIFIED verdict — the average profile plus its
exactly-measured single-seat BR gap, priced by exact BR. A gap_capped row
is a usable reference at gap g, not a failure.

**Landed** (working tree, worktree-perf-day; gates P1/P2 + I1/I2 + 34
pytest green, incl. 13 new):

- `cfr_solve(..., wall_budget_s=)` (hoyt/cfr.py): wall checked at every
  iteration boundary and after every gap measurement; on expiry the gap is
  measured once more and the solve stops with `CFRResult.capped=True` —
  quantified, never silent. Convergence beats the cap (a stop that meets
  target_gap is capped=False). Wave-only: the loop engine is the frozen
  parity mirror and wall-driven stops are timing-nondeterministic, so it
  raises instead of silently diverging. Default-off is behaviorally
  identical (parity gates stay 0.0e+00). Capped-gap exactness is tested:
  a wall_budget_s=0 run reproduces bit-for-bit the gap of an uncapped run
  stopped at the same iteration (impl=hoyt.reference, toys).
- `profile_value(..., slot_budget=)` (hoyt/br.py): mirrors br_solve's
  existing knob; a monster root that needed a raised budget in cfr_solve
  previously had no way to value its exported profile (hardcoded 32M would
  re-raise at every rung). Default identical.
- **`hoyt/refsweep.py`** — the cascade harness (production, not scratch:
  the stable eval reruns forever). Rung ladder (defaults, `--rungs` to
  override): r0 = iters 80 / gap 0.05 / wall 90s / 32M slots; r1 = 240 /
  600s / 64M; r2 = 1000 / wall inf / 128M — iters is a backstop only
  (18e: iteration count to gap 0.05 is world-scale-invariant). Verdicts
  per (root, rung): converged / gap_capped (usable reference at measured
  gap; `capped_by_wall` names the binding constraint) / slot_capped
  (KernelMemoryError; next rung's budget retries) / error (kept, carried
  forward like a cap). Resume unions all shard ledgers, verdict-aware,
  never re-enters a (root, rung) with a row. Merge = best verdict per
  seed → `reference_h4_cap256.jsonl` + deepening curve + evals/hour.
- **Sharding fixes the day's two observed sins**: the snake rank-aligned
  monster phases across workers AND let one 64M-slot root block ~20 cheap
  queued roots ~21 min. Now: sort ascending by n_worlds_sigma, deal
  round-robin (every queue spans the size range), rotate queue w by w/W of
  its length — monster phases land at staggered queue positions (measured
  on the 200-root evalset at W=8: positions 24, 21, 18, ... 3).
  Memory-aware worker default: 7 GiB/32M-slots (18g anchor) scaled by the
  rung's slot budget, 25% RAM headroom; explicit `--workers` is honored
  with a warning.

**Measured (smoke, 2 tiny roots, 1 worker, quiet-ish box during the
running sweep):** both converged rung 0 (555181: 10 worlds, 4.0 s, gap
0.016; 555038: 54 worlds, 4.8 s, gap 0.028, rent +1.94 — consistent with
18e's median +1.9). Full pipeline exercised: worker spawn, ledger rows,
30 s heartbeat, merge, deepening curve, evals/hour, resume (rerun on a
finished outdir: "retired=2", zero new rows).

**Deferred, deliberately:** full-scale run and its deepening curve (the
ad-hoc 200-root sweep was still running; parent validates via
`python -u -m hoyt.refsweep --dry-run` and the
`--seeds 555038,555181 --workers 1` smoke, then launches the real sweep);
whether the rung-1/2 wall+slot defaults are right for the real monster
tail (first full run will say); per-rung `br_every` tuning (still the
gap-pricing cost lever from 18e).

## 2026-07-18i — the exact grind killed by its own numbers; cascade takes over

The 8-wide exact 200-root sweep was stopped at **43/200** rows banked
(~50 min in): recent big-root walls had reached 1245–1837 s — **5–7×
their quiet-box cost**. Measured contention law (new corollary to law 3):
**bandwidth-bound monster roots barely parallelize** — 8 concurrent big
solves aggregate to ≈1.3× ONE quiet worker (workers at 50–74% CPU,
memory-bandwidth saturated), while the small stratum parallelizes fine.
Cold strata math said 3–4 more hours; Jason's directive (18h) says that
shape is exactly what we no longer run. Also observed live: the snake
sharding rank-aligned monster phases (three ~5-min windows with ZERO
completions across all 8 workers).

The 43 banked rows are all converged (gap ≤ 0.05) and skew heavy (they
were every shard's front); grafted into the cascade ledger as rung-0
converged rows (`rung0_shard99.jsonl`, rung_spec marks the provenance).
refsweep launched over the remaining 157 with the default ladder;
deepening curve + evals/hour land in the next entry.

## 2026-07-18j — reference_h4_v1_cap256: the anchor's reference line, 200/200 exact

The cascade's first production run closed the whole anchor:
`hoyt/reference_h4_v1_cap256.jsonl` (committed beside the frozen root set)
— **200/200 converged at gap ≤ 0.05, no capped residue**. Deepening curve:
rung 0 (90s/32M, 5 workers) 174/200 = 87%; rung 1 (600s/64M, 2 workers)
99%; rung 2 (128M, 1 worker) 100%. Total 12.4 machine-hours;
**rung-0 velocity 224 H4 evals/hour** vs the killed exact grind's ~45.

**Velocity/coverage curves measured** (the cap in win/lose/cap terms):
resolve-within-cap is three-regime — a cheap quarter (27% by 20s), a
plateau (nothing resolves 20–30s), a steep middle (34%→82% across
45–150s), and a thin hard tail that is **predictable a priori from
n_worlds_sigma** (8/10 of 90s-cap survivors were worlds_used=256). Capped
verdicts are barely degraded: gap at first measurement p50 0.067, 97%
≤ 0.2 — an "explore-mode" 30s rung yields ~750 evals/hour within ~0.1 pt.
Route, don't discover: schedule big-belief games to deep rungs (or to the
cap-artifact corpus) before paying for them.

**The wedge** (Jason's call: truncate, don't grind): 555090 slot-capped
32M AND 64M (42M slots at wave 23, 85M at 24); at 128M it converged at
**gap exactly 0.0, rent −0.0** — the biggest tree in the anchor is a
fully decided position (29.1 GiB peak; the wall was one exact BR pricing).
Filed as the first specimen of the hard-game stratum: tree size ⊥ decision
difficulty.

**Rent distribution, full population** (walt-BR-vs-jud minus reference,
hero orientation): median **+1.53**, mean +2.02, p90 +5.22, 146/200 over
+0.5. **The mini-reference's "never materially negative" is DEAD**: 18
roots < −0.5, min **−8.65** (max +15.49) — the 12-root sample missed the
negative tail. Reading: the rent indicator conflates jud's exploitability
with opponent-population difference (18e's caveat), and the tail shows the
population term can dominate with either sign. Rent is ~flat in belief
width (median +1.44 at ≥200 worlds vs +0.97 at <50). Feeds #72.

Run-notes: two externally-killed driver incidents mid-day — ledger resume
retired 227 and 228 rows respectively and lost only in-flight work; the
no-absence-as-state design paid for itself twice on day one.

## 2026-07-18k — next lever registered: the fused kernel (#82)

Filed [#82](https://github.com/jasonyandell/mk5-main/issues/82) (stacks on
PR #81) — design the iterate to the memory wall: forced-slot compression
(83% of slots are σ=1 dead weight; compressed arrays ~45 MB → SLC
territory), fp32 licensed by decision damage (law 5), fused
native-int32 kernel with the numpy path as the pinned mirror. **Priors
registered before building: P6 forced-slot ≥2× iterate; P7 fp32 gap
drift ≤1e-3 on the 200-root anchor.** Tuning = short capped refsweep
runs on the M5 only. B200/Modal port EXPLICITLY DEFERRED until H5 is the
live frontier (one planned burn, rungs pre-registered). Named consumer:
distill a student from hoyt reference values; lens:ev grading with the
reference as leaves.

## 2026-07-18l — the fused iterate landed: 7.1× iterate, bitwise, P6 confirmed / P7 refuted

The [#82](https://github.com/jasonyandell/mk5-main/issues/82) kernel, built
and gated in one session (branch `fused-iterate`, stacks on PR #81).

**Anatomy first (law 6) — and it inverted the registered picture.** On the
18g anchor (555006, cap 256): 15.9M isets / 18.8M strategy slots / 19.1M
edges; forced isets 82.8%, forced slots 70.1% (compress 3.35×), forced
edges 69.5% — the issue's estimates confirmed. But the phase split
surprised: `_rm_plus_update` (the slot-space RM+ block) was **72% of
iterate** (1.06 of 1.47 s/iter), the per-edge wave passes only 0.40 —
the "gather-multiply-bincount chains" the issue centered were the smaller
half. The compression lever aimed at exactly the bigger half.

**Landed** (`hoyt/iterkernel.py` + engine="fused", now cfr_solve's
default; wave stays the pinned pure-numpy mirror, loop the recursive
oracle):

- **Forced-slot compression, proven exact**: a forced iset is single-slot,
  so its regret update is cf − cfv ≡ 0 and its σ ≡ 1.0 *exactly*, forever
  — compression is a bitwise no-op, not an approximation. reg/avg/cf/xI
  live on non-forced slots only (18.8M → 5.6M), stable-sorted by seat so
  each seat's update is a contiguous slice (the old block also burned the
  other three seats' slots every update — 4× more dead work on top of the
  forced 3.35×).
- **Fused numba edge kernels**: int32 indices, int8 edge seats, forced
  edges (cgid = −1) skip the σ gather entirely, pr/pm/pu temporaries never
  exist. Single-threaded loops replicate numpy's accumulation order
  (bincount = ascending-edge adds; strict IEEE, no FMA contraction) —
  **fp64 bitwise parity by construction, verified**: P1/P2/P3 toys+pins
  0.0e+00, anchor trace/value/gap/exported-profile all exactly equal.
- Gates: 53 pytest + parity scripts green; jit warm-up runs in the build
  timing bucket.

**Measured (anchor, quiet box, triad detector 39–46 GB/s throughout):**

| variant | iterate s/iter | vs wave |
|---|---|---|
| wave (numpy) | 1.39 (pass 0.39 + update 1.00) | 1× |
| compression-only (numpy passes + compressed update) | 0.44 | **3.13× — P6 CONFIRMED** (prior: ≥2×) |
| fused (kernels + compressed update) | 0.196 (pass 0.15 + update 0.046) | **7.1×** |

Anchor solve wall 88.6 → 41.0 s (2.16×). Paired stratified trio (≤10 min
rule): 555080 2.37×, 555189 2.16×, 555043 1.91× — **1.98× aggregate**,
bitwise PASS on every root.

**P7 REFUTED, and deleted.** fp32 was built, measured, and removed in the
same session: gap drift 2.59e-3 on the anchor (over the 1e-3 license bar;
reference value moved 3.2e-2), AND **zero throughput win** — 0.194 vs
0.195 s/iter, because the fused loops are gather-latency-bound, not
float-bandwidth-bound. The 18k framing ("2× on all slot traffic") priced
bytes, but after fusion the iterate stopped paying in bytes. No speedup +
no consumer + drift over bar ⇒ the dtype knob is gone (receipts here; the
possible revisit is the *contended multi-worker* regime, where aggregate
DRAM pressure — not single-stream latency — is the wall, and only if a
tighter numerics story caps the drift).

**Amdahl bookkeeping (law 7):** iterate fell from 62% of solve wall to
~19%; **exact-BR gap pricing is now 65–71% of wall** (26.5 s of 41 on the
anchor; 84 of 117 on 555043). The next solve-wall lever is the
mixed-profile BR walk / gap-measure cadence (`br_every`, queued since
18e), NOT more iterate work. numba is now a runtime dep of the default
engine (CONTRACTS.md updated; wave/loop remain numba-free).

**Velocity, measured the honest way** (20-root stratified sample = 10% of
the anchor, 5 workers, rung-0 ladder, production `refsweep` sharding):
20/20 rows in 360 s — 17 converged, 2 gap_capped, 1 slot_capped (the
555090 wedge, as banked). Same-seed same-rung pairing vs the 18j
production ledger: **2,898 → 808 worker-seconds, 3.59×** on the 17
both-converged roots. The per-root spread (1.24×–9.8×) is the law-8
corollary in reverse: the banked contention victims (555008 954 s,
555045 986 s) fell 9–10× because five fused workers no longer saturate
DRAM — small roots kept completing straight through monster phases
(loadavg 3–5). Peak worker RSS 5.8 GiB on converged roots (was 7.0).
Projected rung-0 velocity ≈ 3.6 × 224 ≈ **~800 H4 evals/hour** (the
sample's naive 190/h is tail underutilization on a 20-root batch — 1,800
worker-s allocated, 1,231 spent; don't quote it). The projection gets
banked as a measured number at the next full production sweep (H5 ladder
or evalset v2) — rerunning the closed v1 line would only reproduce
bitwise-identical rows.

One verdict flip, by design: 555039 banked *converged* at 1,837 s (it
blew far past the 90 s budget before its first gap measurement — 
convergence beats the cap) but *gap_capped* here at 149 s, because the
faster engine reached a budget checkpoint first. Wall stops are
timing-dependent (exactly why engine="loop" refuses `wall_budget_s`);
the cascade carries the root to rung 1 unharmed.
