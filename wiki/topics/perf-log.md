---
title: Perf log — append-only field notes
kind: topic
first_seen: 2026-07-18
last_updated: 2026-07-19
status: active
---

Append-only field notes for performance work, per Jason's 2026-07-18
directive: this log replaces GitHub issues for the perf program. Entries are
terse but rich — what was tried, how the numbers moved, what died. Newest at
the bottom. Laws distilled from here graduate to [[perf]]; never edit old
entries (corrections are new entries). Cold readers: the shorthand is
decoded in [[hoyt-perf-primer]].

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

## 2026-07-18m — the Fable consult: stop walking — in-struct gap pricing (16×), the duplicate value, and three honest kills

A fable subagent was consulted for next levers after 18l. Its structural
finding beats everything the 18k issue listed: **gap pricing re-walked a
tree that was already resident** — `measure()` exported the profile and
`impl.br_solve` rebuilt the whole subgame per seat per measurement,
paying the stochastic-profile blowup (18b: p50 526×) each time. Exact
single-seat BR needs only a forward-reach + backward-argmax pass over
the `_WaveStruct` the solve already holds. The fix is not to fuse the
walk; it is to stop walking.

**Landed (L1)** — `_wave_br` in cfr.py: counterfactual reach forward
(hero edges at prob 1), backward with per-iset signed argmax at hero
info sets (chosen move's per-world child values propagate unweighted),
sigma-weighted expectation elsewhere. Ties → lowest move
(value-identical); zero-reach isets contribute zero (same exactness
argument as pinned support pruning). Export moved to the final stop.
The wave/fused engines share it; **engine="loop" keeps impl.br_solve
pricing as the standing oracle**, plus new gate P4 (in-struct gap ==
br_solve gap on the exported profile, ≤1e-9 — measured 0.0e+00 on
toys/pins, 3.6e-15 on the anchor). Anchor trace reproduces the banked
ledger row at 2.2e-07 (within its 1e-6 rounding license); fused-vs-wave
stays bitwise (shared pricing path). **P8 (registered: ≥10× on the
anchor's 26.4 s br bucket): PASS at 16×** — br 26.5 → 1.65 s; anchor
solve wall 41.0 → **15.9 s** (88.6 at session start: **5.6× today**).

**Landed (L2)** — the reference value was computed twice: measured on
ALL 200 banked rows, `value_selfplay == cfr_reference_value` with max
delta exactly 0.0 (both are exact expectations of the same profile over
the same worlds). refsweep now uses `res.value`; `profile_value`
survives as a deterministic 1-in-20 audit that raises on >1e-9 (P9:
audit never fires). Non-CFR phase walls (enumerate/σ-filter/compile/
walt-BR) — 6% of the 18j fleet spend and previously un-instrumented
inside `wall_s` — now land in every ledger row as `phase_s` (law 6).

**P10 sweep** (same 20 stratified roots, 5 workers, rung-0 ladder):
20/20 rows in **120 s elapsed** (was 360 s at 18l) — **19/20 converged
at rung 0** (was 17/20; 555039 and 555156 now beat the 90 s cap instead
of laddering). Same-seed worker-time: banked 2,898 → 18l 808 → **235 s
(12.3× vs banked, 3.43× vs 18l)** on the both-converged subset; on the
full-ladder accounting (19 seeds, banked needed rung-1 re-solves) it is
**277 → 19.5 worker-s per usable eval = 14.2×**. Old contention victims
fell 35× (555008: 954 → 27 s). Sample-naive velocity 570/h is
tail-bound (64% utilization on a 20-root batch); fleet-equivalent
projection: rung-0 **~2,000–2,800 evals/hour** (was 224) — P10 (≥1,600)
PASS by projection, to be banked at the next full production sweep.

**Kills, all measured or theory-checked (append-only honesty):**
- **BR-walk numba fusion** (my queued lever): consumer vanished under
  L1; the stochastic walk is argsort/unique-bound (~200 ns/node vs the
  fused iterate's ~10 ns/edge) and only #77-class offline work still
  pays it.
- **Regret-based gap bounds**: the 2p folk bound (exploitability ≤ Σ
  avg regrets) needs utility linear in ONE opponent's strategy; with
  three opponents the average of products ≠ product of averages, so
  seat u's average regret does NOT bound its BR gain vs the average
  profile. Dead on theory, and L1 removed the motivation.
- **Seat-subset early exit**: only helps failing measurements; a capped
  row's final_gap must price all four seats anyway.
- **L4 (more workers)**: killed by its own gate — monster-shard peak
  RSS is 16.3 GiB (L1 actually lowered it from 18l's 21.2: the mixed-BR
  walk was itself a big allocator) on the 48 GiB box; 5 workers is the
  right rung-0 default. GIB_PER_32M stays.

**Queued, not built** (Fable's L3/L5/L6): adaptive gap cadence
(measure every 5–10 iters, stop at first crossing — pricing is now
~0.4 s so the optimum inverts; worth ~25% of iterate); exact CFR state
checkpoint for rung survivors — serialize (sig_c, reg_c, avg_c, it) and
RESUME, bitwise-equal to an uncapped run, NOT profile warm-start (RM+
with zero regrets resets toward uniform; regret transfer is a numerics
license we don't need); numba `expand_full_width` — post-L1 anatomy
says build is now the top bucket (166 s of 386 on the P10 sweep = 43%,
iterate 38%, br 9%) but it is the shared parity surface under
compile_sigma/br_solve/CFR — a much bigger license, lever-after-next.

Amdahl after 18m: **build 43% / iterate 38% / export+br+value ~13% /
oracle phases ~6%** — the referee is no longer priced by its pricing.

## 2026-07-18n — the build lever: stop re-deriving what the walk held (pslot/actor resident, numba fill)

The registered lever was "numba expand_full_width" (18m: build 43% of
worker wall, but shared parity surface under compile_sigma/br_solve/CFR —
a big license). **Sub-anatomy first (law 6), and it shrank the license.**
Priors: PB1 walk ≤35% / post-walk ≥50% / _build_fused ≤15% of anchor
build; PB2 the unique + PS-reconstruction argsorts ≥50% of post-walk.
Measured (anchor, triad 39–42 GB/s): build 5.90 s = walk 2.28 (39%) +
post-walk 3.08 (52%) + _build_fused 0.54 (9%) — PB1 confirmed on
post-walk, hair over on walk. **PB2 half-REFUTED**: np.unique is 0.15 s
(numpy 2.4's hash-based unique — the argsort fear is stale); the real
cost is **searchsorted 2.04 s**, re-deriving per wave (a) the parent slot
of every child slot by (node, world)-key binary search and (b) the actor
per node via first-child search — both of which `run_engine` already held
as locals (`rows`/`gidx` ARE the parent-slot indices; `actor` is
computed per wave). The 18m lesson generalizes: the fix is not to fuse
the search; it is to stop searching.

**Landed, two parts, both output-identical by construction:**
- **A (structural, no numba):** `keep_slots` now also stores per-wave
  `pslot` (int32 parent-slot per child slot) and `actor` (int8 per node).
  `_build_wave` reads them; the searchsorted reconstruction is deleted
  (no-legacy). License: additive fields on the CFR-lane-only
  `keep_slots=True` path — br/compile_sigma call run_engine with
  keep_slots=False and never execute the new lines. Gate: old derivation
  replicated in scratch and asserted EXACTLY equal on all 16 anchor waves
  + a durable toys invariant in test_full_width_walk.
- **B (numba, provider-scoped):** `hoyt/buildkernel.py::fw_fill` emits
  (slot, move) pairs straight off the legal bitmasks in np.nonzero's
  row-major order, killing the per-wave (slots×28) bool matrix + 1.03 s
  of nonzero. Selected by `_FullWidthProvider(kernels=True)`, threaded
  as `_build_wave(kernels=fused)` — **the numpy provider stays the
  pinned mirror and the wave lane stays numba-free** (CONTRACTS.md), so
  the standing fused-vs-wave bitwise gates now cover the build too.
  Gate: kernels walk == mirror walk, every field of every wave + leaf,
  exact.

**Measured (anchor 555006, cap 256, 40 iters):** fused-lane build 5.90 →
**2.49 s (2.36×)**; solve wall 15.9 → **13.0 s** (88.6 at 18k start:
**6.8×**). Bitwise PASS (trace/value/gap/profile vs wave; P8 oracle
3.6e-15; banked row 2.2e-07 within the 1e-6 rounding license). All
parity gates green (P1–P4, 53 pytest).

**P10-style sweep** (same 20 roots, 5 workers — full cascade this time,
18m ran --rungs 0): paired both-converged **235 → 194 worker-s = 1.21×
vs 18m** — registered prior **P11 (≥1.25×) NARROWLY REFUTED**: small
roots pay fixed overheads (subgame build, jit-warm, worlds) that the
2.36× doesn't touch; fleet build bucket moved 165.8 → 94.6 s (1.75×),
not 2.36×. Banked/18n on the paired subset: **14.97×**. Per usable eval
(19 seeds, ladder accounting): 19.5 → **16.4 worker-s** (1.19× vs 18m,
**16.9× vs banked**); projected rung-0 velocity ~**2,400–3,300
evals/hour**. Bonus receipt: the 555090 wedge, slot_capped in every
rung-0-only sweep, laddered to rung 2 and **converged in 366 s** (banked
paid 1,799 s for that verdict) — peak RSS 27.0 GiB at cap 1024 on the
48 GiB box, fine solo but confirms rung-2 solves must not share the box
with 4 siblings (the L4 kill holds).

Amdahl after 18n (rung-0 fleet, wedge excluded): **iterate 50% / build
31% / br 12% / export 5%** — the wheel turns back to iterate. Next lever
per the queue: **L3 adaptive gap cadence** (pricing is 0.4 s; measure
every 5 iters and stop at first crossing — the tail iterations after
the gap crosses are pure waste, ~25% of iterate). expand_full_width
deeper fusion (run_engine's own gathers, 0.87 s/anchor) is now the
lever-after-next and still carries the shared-surface license.

## 2026-07-18o — gap_exit: the cadence economics invert when intermediates only answer yes/no

L3 as queued said "measure every 5–10 iters". The trace economics said
otherwise: a full 4-seat measurement costs 3.5–7 iterations equivalent
(measured per root, 18n sweep), so k* = √(2·T·c_meas/c_iter) ≈ 16–18 —
**br_every=20 was already near-optimal for FULL pricing**, and densening
it would have lost money. The unlock is structural: an intermediate
measurement only decides continue-vs-stop, and gap > target ⇔ SOME
seat's BR gain > target — so price seats in descending last-known-gap
order and **exit at the first crossing**. Intermediates then cost ~1
seat, the ratio drops to ~1.3–2, and k* ≈ 9–11.

**Landed:** `cfr_solve(gap_exit=True)` (wave/fused; the loop mirror
raises, wall_budget_s precedent) + refsweep `BR_EVERY 20 → 10`.
Exactness is by construction, not tolerance: convergence is only
declared after ALL live seats are priced; any measurement that can end
the solve without convergence (iters exhausted, wall budget) prices all
four (a capped row's final_gap stays a full verdict; a budget expiry
during a partial measurement defers the cap to the next, full, one). So
the stop iteration and final profile/value/gap are IDENTICAL to
gap_exit=False at the same cadence — only intermediate trace entries
change (certified-above-target partial maxima). **Gate P5** (parity
suite): stop iter + final result exactly unchanged on toys tuned to
force 23 intermediate exits. All gates green, 53 pytest.

**Measured (same 20 roots, 5 workers, full cascade):** **P12
(registered: 19-seed rung-0 wall 310.8 → ≤280 s) PASS at 279.1 s — by
0.9 s; quote it as 1.11×, not a triumph.** Mechanism receipts: 8/19
roots stop one cadence step earlier (555046: 20 → 10 iters, 61.4 →
52.3 s), iterate bucket −19% (151.6 → 123.2 s), br bucket flat
(36.8 → 37.7 s — the doubled cadence fully paid for by seat-exit),
zero verdict flips, every stopping gap a full-priced ≤0.05
certificate. The new rows stop less-converged (555258: gap 0.0499 vs
18n's 0.0286) — that is the protocol working: pay only for the
certificate, not for polish past it. Per usable eval (19 seeds, ladder
accounting): 16.4 → **14.7 worker-s = 1.12× vs 18n, 18.9× vs banked**;
wedge ladder 700 → 578 s cum. Projected rung-0 **~2,700–3,700
evals/hour**.

Amdahl after 18o: iterate 45% / build 33% / br 14% / export 6%. The
big single-root buckets are now build's run_engine gathers (shared
surface) and the iterate floor itself. Session total on the anchor:
88.6 → 13.0 s solve; fleet per-usable-eval 277 → 14.7 worker-s
(**18.9×**). Next candidates, none registered yet: adaptive cadence v2
(predict crossing from gap decay — saves another ~½ step), run_engine
gather fusion (the big license), or bank the line and spend the
velocity on H5/evalset-v2 instead.

## 2026-07-18p — walk elisions land small, a buggy anatomy copy refuted honestly, and the parallel-iterate lever registered

**PB5 registered** (walk 1.15 → ≤0.70 s, fused build 2.49 → ≤1.9 s on
the anchor), on the strength of a timed structural copy of run_engine
that ran the walk in 0.48 s. **PB5 REFUTED, both clauses — the copy was
buggy.** It used the hero_is_me hands line, so `col_arr[me] == -1`
silently negative-indexed the LAST world column instead of mymask —
wrong hands, wrong legality, an 8.2M-slot tree instead of the real
19.1M. The phantom 0.6 s of "headroom" was a smaller tree. Law-6
corollary, earned twice tonight: **an anatomy copy that does not
reproduce the tree is not an anatomy** — print the tree size before
believing the clock (the buggy script is kept in scratch as the
receipt).

**What actually landed (all value-identical, gates green):** lazy
`starts_node` (only the hero branch consumes it; the hero=-1 walk never
did), `_cat2` empty-side concatenate elision (the full-width walk
concatenated an empty hero side into every slot array every wave),
minority-write `hands` (in-place assignment on me-slots instead of a
full-width np.where), and the export's move emission routed through
`fw_fill` (killing the last big (isets×28) bool + nonzero in
_build_wave). Measured: walk 1.15 → **1.08 s (1.06×)**, fused build
2.49 → **2.20 s (1.13×)**, anchor solve 13.0 → **12.6 s**, bitwise PASS
(trace/value/profile vs wave; P8 3.6e-15; banked 2.2e-07). No sweep run:
1.13× on a 33% bucket ≈ 4% fleet — at the ±4% noise floor
([[perf]] law; perf-subset-5), the anchor receipts carry it.

**The real walk anatomy** (in-memory instrumentation of run_engine
itself, NOT a copy): slot-gather+mask 0.31 s / numba fill+legality
0.20 / prep+hands 0.11 / transitions 0.08 / sort+seg 0.10 / hash+stores
0.09. It is E-scale-fancy-gather-bound. A fused σ-branch kernel
(gather+mask+wt+world+pslot in one pass over rows) caps at ~0.3 s/anchor
≈ 3–4% fleet — **registered as a cap verdict, low priority; do not build
without a cheaper reason.**

**Registered next lever — parallel fused iterate (P13).** Iterate is
45% of fleet wall and single-threaded by the 18l bitwise doctrine. The
doctrine survives threading IF every thread owns a DISJOINT output
range with unchanged within-range accumulation order: fwd is a pure
map (trivially parallel); bwd's `v_p[p] +=` partitions at parent-slot
boundaries (edges are parent-sorted — disjoint outputs, ascending order
within each parent preserved ⇒ bitwise regardless of scheduling); the
`cf_c[g] +=` accumulation is the hard part (an iset's slots span
non-adjacent parents) — precompute a per-wave permutation grouping the
updating seat's edges by gid at build time (structure, cached in
_FusedLayout), turning cf into segmented sums parallel by group range
with ascending-edge order inside each group. **P13 prior: ≥1.8× iterate
on the anchor at 4–8 threads, bitwise vs single-thread fused, RSS flat.**
Risk: numba prange scheduling must not touch output ranges — chunk
boundaries must be precomputed structure, not runtime heuristics.

## 2026-07-19a — P13 lands: threaded iterate, bitwise by construction; refsweep threads=4

**Sub-anatomy first (law 6), and it refuted the premise before a line of
kernel code.** Registered PA1 (bwd ≥55% of anchor iterate) — REFUTED:
fwd 48% / bwd 28% / rm_plus 22% / loop 2%, so rm_plus threading is
mandatory, not optional. PA3 (PS parent-sorted, 18p's design premise) —
**REFUTED: PS is neither sorted nor contiguous-by-parent** (mono_frac
0.78–1.0, runs ≫ uniq on waves 0–11 of the anchor). The 18p design as
written dies; the surviving design is stronger and simpler: **group
edges by output owner with build-time stable argsorts** (structure in
`_FusedLayout`, never runtime heuristics) — by parent for `v_p`, by gid
for `cf_c`, by iset for the CFR+ update. Each numba `prange` iteration
owns a disjoint output range and folds its contributions in ascending
edge order (= np.bincount's fold), so results are **bitwise identical
at any thread count and any scheduling**. Trick-tail waves 12–15
(11.9M of 19.1M edges) are parent-bijections — pure-map fast path, no
permutation stored. Verified before believing: np.maximum returns +0.0
on ±0.0 ties (mirrored by the clamp), and a +0.0-seeded IEEE fold can
never produce −0.0, so register-accumulate-then-store is exact.

**Landed:** `cfr_solve(threads=N)` (fused-only; wave/loop mirrors stay
the pinned single-threaded references and raise), five kernels in
`hoyt/iterkernel.py` (`fwd_edges_par`, `bwd_v_map`, `bwd_v_seg`,
`cf_seg`, `rm_update_seg`), perm build in `_build_fused(par=True)`,
refsweep `--threads` (default 4). **Gate P13 added to the parity
suite** (threads 3/4 + gap_exit compose, exact-zero drift); P1–P5
green, 53 pytest green.

**Measured (anchor 555006, cap 256):** iterate **2.67× @4 threads,
3.12× @8** (prior ≥1.8× CONFIRMED, beaten), bitwise PASS at every
count; anchor solve 12.6 → **7.2 s**. RSS honestly **+0.21 GiB (+5%)**
in isolated runs — not strictly flat (perms + numba parallel runtime);
fleet worker maxima unchanged (28.5/14.4/11.2 → 27.9/13.4/11.4).

**Measured (same 20 roots, 5 workers, full cascade), the thread ladder
with per-config priors:**
- **P14 (t2: rung-0 wall ≤240 s) NARROWLY REFUTED at 245.9 s** — by
  5.9 s; quote it as 1.13×, not a triumph.
- **P15 (t4: ≤ t2 AND full-run ≤320 s) SPLIT**: rung-0 **221.8 s
  (1.26× vs 18o's 279.1)** confirmed; full-run 360 s REFUTED — the
  wedge's solo rung-2 is not iterate-bound (250 → 234 s, t2 → t4).
- **P16 (t8: rung-0 ≤210 s else t4 default) NARROWLY REFUTED at
  215.4 s** — t8/t4 = 1.03×, inside the ±4% noise floor at 40 threads
  on 18 cores. **threads=4 is the production default by P16's own
  decision rule.**

Zero verdict flips in any config; every stopping gap a full-priced
≤0.05 certificate. The anchor's 2.67× compresses to fleet 1.26×
because small roots pay thread overhead (555319: 0.62×) and iterate
was 45% of fleet wall. Fleet iterate bucket 207.6 → **101.8 worker-s
(2.04×)**; per usable rung-0 eval **14.7 → 11.7 worker-s (23.7× vs
banked)**; full-cascade root-wall-cum 578 → 473 s; projected rung-0
velocity **~3,400–4,650 evals/hour** (was ~2,700–3,700).

Amdahl after 19a (rung-0 fleet, t4): **build 48% / iterate 23% / br
17% / export 10%** — the wheel turns back to build, and the two big
build levers are known: run_engine's own gathers (the shared-surface
license, 18n's lever-after-next) and the per-root fixed overheads that
kept P11 honest (subgame build, jit-warm, worlds). Cap verdicts kept:
threads on the solo rung-2 wedge are marginal (234–250 s at any
count); σ-branch gather kernel stays capped at 3–4% fleet (18p).

## 2026-07-19b — big-root build anatomy: the fleet build bucket is half contention; PB8's counting sort built, measured, deleted

The P11 lesson applied to the anatomy itself: fleet build is dominated
by big-TREE roots (555046: 29.8 s build in the t4 sweep for 4.1 s of
iterate — it converges in 10 iters; 555156: 21.2 s), so the anchor is
the wrong specimen. Priors registered on 555046 (117.8M slots, solo,
threads=4): PB6 walk ≥45% of build — **CONFIRMED at 47%** (5.6 s);
PB7 _build_fused ≤15% — **REFUTED at 17%** (2.0 s). Post-walk 37%.

**The headline is neither bucket: 555046 solo builds in 11.9 s vs
29.8 s inside the 5-worker sweep — the fleet build bucket is ~2.5×
DRAM contention, not code.** (Law 3 and the law-8 corollary, measured
again at rung 0: bandwidth-bound monsters barely parallelize. The
refsweep already staggers monsters; a width-per-stratum schedule —
solve the 2–3 big-tree roots at low width first, then flood the small
tail at 5 workers — is the cheapest untaken build lever and needs no
kernel work at all.)

**PB8 registered and REFUTED same-session** (fp32 precedent: knob
built, measured, deleted). Hypothesis: P13's perm-building argsorts
tax big-root builds; a numba stable counting sort (O(n), provably
identical perms) would cut ≥0.5 s of 555046's _build_fused. Measured:
2.02 → 1.92 s — **0.1 s; numpy's stable int32 argsort is already a
radix sort.** This is PB2's np.unique lesson RE-LEARNED (18n: "the
argsort fear is stale knowledge") — twice now: **comparison-sort
intuitions do not price numpy 2.x integer sorts; measure before
building around them.** The counting-sort kernel was deleted; the
perm-equality gate script stays in scratch as the receipt
(perm_gate.py, PASS before deletion).

Next levers for the build bucket, in cheapness order: (1) the
width-per-stratum sweep schedule (pure scheduling, prior ~1.3–1.6×
on the fleet build bucket via decontention); (2) run_engine gather
fusion (0.87 s/anchor scale, shared-surface license); (3) _build_fused
base layout (1.9 s on the big root — casts and seat loops, threadable
behind the same P13 structure). None registered yet with bars — the
successor should anatomy (1) first since it is free.

## 2026-07-19c — the zero-diff width experiment: decontention is real and still loses to width; width-per-stratum dies before it was built

Prior P17 registered before the run (`scratch/fused-iterate/p17_prior.md`):
`--workers 3 --threads 6` (18 threads on 18 cores) paired vs
`sweep20_p13t4`, same 20 seeds. **P17a (rung-0 root-wall-cum ≤200 s vs
221.8) CONFIRMED, beaten: 156.2 s (1.42×)** — build bucket 93.4 → 62.8
worker-s, iterate 55.9 → 41.7, br 40.5 → 30.0; every big-tree root
improved 1.44–1.91×. Zero verdict flips; full-run wall 360 s unchanged
(the wedge's solo rung 1+2 dominates at any width). Small roots
regressed at t6 (555258 0.71×, 555124 0.84× — thread spin-up, P13's
555319 lesson again); worker RSS maxima rose to 15.3 GiB (fatter
queues: every worker eventually holds a monster). Box contaminant
logged: mediaanalysisd ~2 E-cores throughout — margins are far from
every bar, no re-run.

**Adoption DECLINED, against the letter of the pre-registered rule —
the rule was mis-specified, and the math goes on the record.** P17b's
elapsed bar ("20/20 rows by the 90 s tick") passed, but 30 s heartbeat
granularity cannot resolve elapsed at this fleet size, and the metric
the north star runs on is steady-state throughput (Little's law:
width / per-eval root-wall). w5t4: 5/11.67 = **0.428 evals/s**. w3t6:
3/8.22 = **0.365 evals/s** — a 17% throughput regression hiding under
a 1.42× worker-seconds "win". Decontention 1.42× < width given up
5/3 = 1.67×. Per-eval worker-seconds is a pricing tool, not the
objective; adopting by the rule's letter would have shipped the
regression to every production sweep.

**The same numbers kill the width-per-stratum lever (19b lever 1)
before a line was written.** Every stratum loses to width: monsters
sum 160.1 → 102.8 = 1.56× < 1.67×; top-2 monsters 1.44×; small tail
≤1.33×. Even the monster stratum packs better at width 5 (160.1/5 =
32.0 s elapsed vs 102.8/3 = 34.3 s). Exactly one root in 19 beats the
ratio individually (555039, 1.91×). Width-1 monsters would need ≥5×;
19b's solo anatomy caps build decontention at ~2.5× on ~half the
monster wall (~1.7× total) — dead at every width. **w5t4 stands as
refsweep's production default; law 8's corollary sharpens: width beats
decontention wherever decontention < width ratio, and on this box that
is every stratum measured.**

What the zero-diff run bought (why it was the right first move): the
contention ladder is now priced at three widths (per-eval 11.7
worker-s @5w / 8.2 @3w / solo floor from 19b), which re-prices lever
(2): **run_engine gather fusion saves DRAM traffic, and at width 5
every byte not moved pays twice** — once as the root's own wall, once
as decontention of four co-runners. That is the next lever; its
anatomy is free (`build_anatomy_big.py`).

## 2026-07-19d — walk gather fusion: built, measured at both scales, refuted at both bars, deleted; the P11 compression law reaches the build

Walk sub-anatomy on 555046 first (priors PW1–PW4,
`scratch/fused-iterate/pw_prior.md`): walk 5.64 s = expand 1.17 (21%)
+ **argsort 0.22 (4%)** + cat2 0.00 (the hero-less full-width walk
never concatenates) + remainder 4.25 (75%). PW1 (argsort ≥25%)
REFUTED — numpy's radix argsort wins a THIRD time (PB2 → PB8 → PW1;
stop pricing numpy integer sorts by comparison-sort intuition,
permanently). PW4 (fusable gather surface ≥35%) CONFIRMED at 75%.

**PW5 built**: `walk_gather` numba kernel — the σ-path's six
slot-scale passes (rows/skey gathers, sw gather + actor/col
double-gather + bit-clear scatter, swt, sworld) fused into one prange
map, pure gather-of-gather, disjoint outputs, no folds → bitwise at
any thread count by construction. Exact-parity gate PASS (three roots,
threads 1+4, every wave array + leaf + pslot), 53 pytest green.

**Measured, quoted flat: solo bar MISSED** — walk 5.64 → 4.22 s
(1.34×, bar ≤3.8), build 11.97 → 10.83. **Fleet bars REFUTED** (same
20-seed paired sweep, w5 t4): build bucket 93.4 → 87.4 worker-s
(1.07×, bar ≤80); rung-0 root-wall-cum 221.8 → **223.2 s — a wash**
(bar ≤208); zero verdict flips. The registered honesty clause fired:
the removed passes did not buy back co-runner decontention — **19c's
double-pay theory takes its hit**. Deleted same-session (fp32/PB8
precedent); receipts stay in scratch (walk_anatomy_big.py,
walk_gather_gate.py — the gate still validates fw_fill's mirror).

**The lesson with a number: solo walk wins compress ~4:1 at fleet** —
−1.14 s on the specimen became −6.0 worker-s across 19 roots because
the walk is only the monster-share of build. This is P11's law
reaching the build bucket: anchor-scale (now specimen-scale) wins
never fully cash fleet; the compression ratio is the monster-share of
the touched surface. Post-walk assembly (~4.6 s solo, the biggest
remaining build chunk) is PRE-PRICED by the same math: expect ~1.05×
fleet from a 2× solo win — not worth kernel work without a
structural elision that applies to every root.

**Kept from the spike**: `numba.set_num_threads` hoisted above
`_build_wave` in `_solve_wave` — it previously ran after the build, so
any future numba-parallel kernel reached from the walk would default
to all cores and oversubscribe a 5-worker sweep 5×18-on-18. Real bug,
zero-cost fix, landed.

## 2026-07-19e — the rung-1 stratum dissolves, the production model turns over, and the wedge is priced: 5× big, 2× memory-pressured

**PR1 (rung-1 pricing) refuted its own framing, which is the finding.**
Sample = banked rung-1 min/p25/p50/p75/max (555037/555304/555104/
555071/555095) through the normal cascade: **4 of 5 converge AT RUNG 0
now** (14.0–37.2 s; banked 109–453 s) — the banked 174/24/2 rung mix
is an artifact of banked-era speeds, not tree structure. Only the
banked max still ladders: 555095 rung-1 converged **77.4 s vs 1376.8
banked (17.8×)**; my mean-bar was ill-posed (one survivor), the
max-root bar (≤120 s) CONFIRMED, zero verdict flips.

**Production velocity model v1 (model, not measurement — quote it as
such):** extrapolating the sample, ~193/200 roots are rung-0-class
post-P13. Est. full-line: ~174×≲10 + ~19×21 + ~5×87 worker-s at
width 5, plus the two wedges ~2×250 s at width 1 (mem cap) ≈ **~17 min
wall ≈ ~700 evals/hour full-line — ~37× banked** (banked line: 10.8 h,
18.5/hr). The full post-P13 line has NOT been run end-to-end (>10-min
bench cap); the model's soft spot is the 174 banked-rung-0 roots' true
average. **The wedges are ~half the modeled wall, at width 1 — the
production Amdahl points at the wedge class, where solo wins cash 1:1
(19d's compression law does not apply).**

**Wedge anatomy (PW6–PW8 registered, one solo rung-2 solve):** build
118.6 s of 234 s wall. **PW6 CONFIRMED beyond its bar: 589.6M total
tree slots — 5.0× 555046** (the slot budget is per-WAVE; totals
diverge). Per-slot build 201 vs 102 ns/slot → a clean **2× residual**,
and **PW8 CONFIRMED: 32.7 GiB compressor churn** during the solve
(maxrss 26.4 GiB on the 48 GiB box; triad degraded 45.5 → 31.8 GB/s
across the run). The wedge is big×pressured, both terms now measured.
Walk shape at scale: 50.8 s = expand 13% / argsort 3% / remainder 84%.

**Successor lever (registered in spirit, needs a bar before code):**
narrow the walk's WORKING arrays — `sw` is (M,3) int64 holding 28-bit
hand masks (int32 fits exactly), `sworld` int64 for world indices
≤256. ~16 B/slot less transient churn ≈ ~9 GiB at wedge scale,
attacking the measured 2× pressure multiplier where it lives, and
"bytes-per-eval down" is the north star's second axis. Stored waves
narrowed in 18n already; this is the transient side. Dtype-promotion
traps (numpy silently upcasting mixed int32/int64 expressions) make
this a parity-gated structural change — workflow/subagent sized.

## 2026-07-19f — int32 walk working arrays: kept as a bandwidth win; the transient-pressure theory refuted flat (churn is ordering, not dtype)

Priors + bars registered before code
(`scratch/fused-iterate/int32_prior.md`). Change: `run_engine`'s
transient slot arrays narrowed — `sw` (M,3) and `sworld` int64 → int32
(28-bit masks / world ids ≤ 256); `snode` STAYS int64 (index array,
18g, and `snode<<5` keys pass 2^31 near the slot budget); the stored
per-wave copies become `astype(np.int32, copy=False)` aliases (the
walk rebinds sw/sworld per wave and never writes them in place —
`sw_sig` is a gather copy); the σ bit-clear goes through an int32
clear-mask LUT (`_NB28_I32`) because `&= ~(_I64_1 << mv)` would
promote; `worlds_i64` retired for `worlds_i32` (br.py's grouped BR
included). Parity gate: value-normalized hashes of every wave array +
leaf + counters vs HEAD's walk, 3 roots × kernels on/off — EXACT;
53 pytest green.

**Measured on TWO wedge pairs, both orderings — the reversal earned
its keep.** Compressor churn is position-determined, not
code-determined: first arm ~35.3/35.5 GiB, second arm ~31.7/31.1,
whichever code ran. **I1 (churn ≤0.70×) REFUTED FLAT: the coupling
from transient write traffic to compressor churn is ≈0.** The ~19
GB/build of removed transient writes never reach the compressor
(hot, short-lived pages); the churn lives in the RESIDENT stored
waves. 19e's "attack the 2× pressure multiplier via transients"
theory is dead — a future pressure lever must shrink resident
footprint, not transients. maxrss is position-determined too (first
arm 26.2/26.5, second 28.5/28.3 — inverted vs the naive read); I2
unmeasurable by this design. Wedge iterate swung 37.9 → 64.5 s
between back-to-back pairs at fixed code — **±25% ambient
sensitivity; wedge bars need paired same-session arms, always.**

**The mechanism-pure win, consistent across orderings**: walk 50.8 →
43.1 and 50.3 → 44.2 s (**1.16×**), build 1.07× both pairs; wedge
solo wall ≈0.97× position-corrected (I3 bar ≤0.95 MISSED, quoted
flat), in-sweep wedge 233.7 → 232.8 (wash). Fleet gate: 20-seed
paired sweep, zero verdict flips, rung-0 root-wall-cum **221.8 →
221.8 s — 1.000×** (19d's compression law, third sighting), build
bucket 93.4 → 88.7 worker-s, worker RSS max 28.3 → 27.5 GiB.

**Verdict: KEPT, with the primary bar refuted and the override named
(19c template).** I1 priced a coupling measurement puts at ≈0 — no
transient-dtype change could ever have passed it; it was a bar on a
wrong theory, not on the lever. Every decision variable that COULD
move passed: parity exact, wall non-regression (0.97–1.00×), fleet
gate green, transient bytes/build −~19 GB (the north star's second
axis). Unlike 19d's deleted kernel this is not a second lane — a
dtype in the single implementation, net −1 module attribute. The
residual liability is the alias invariant (stored waves must never
be written in place), documented at the store site.

## 2026-07-19g — the second wedge dissolves: 555212 converges at rung 0; the production model has ONE wedge

Prior registered (`scratch/fused-iterate/sw_prior.md`): SW1 =
still-wedge-class (slot-caps to rung 2, 180–320 s) vs SW2 = the
19e-shaped alternative (converges rung ≤1, ≤120 s). **SW2 CONFIRMED
beyond its bar: 555212 converged AT RUNG 0 — 72.2 s wall, 60 iters,
gap 0.039 (banked 1332.4 s = 18.5×, the same ratio class as 555095's
17.8×).** SW1 refuted; the banked-rung-2 stratum was half artifact.
Caveat carried forward: its rung-0 solve held 27.4 GiB maxrss — in a
width-5 fleet that co-residency is untested (the memory-aware cap
only guards rungs 1–2); the full-line run is the test.

**Production model v2 (model, not measurement):** the wedge share of
the full line halves — 19e's "two wedges ~2×250 s ≈ half the wall"
becomes ONE wedge (555090, 232.8 s in-sweep post-int32, ~30% of
wall) plus 555212 folding into the width-5 rung-0 pool (~72
worker-s, noise at pool scale). Est. full-line wall ~515 s width-5
pool + ~233 s wedge ≈ **~12.5 min ≈ ~960 evals/hour (~52× banked)**.
The soft spot is unchanged (the 174 banked-rung-0 roots' true
average) — which is the argument for measuring: the split design the
19e handoff named (two ≤10-min runs) replaces model v2 next.

## 2026-07-19h — the full line, measured: 200/200 converged in 17.0 min — 706 evals/hour, 38× banked; model v1 was right for wrong reasons

Priors FL1–FL4 registered (`scratch/fused-iterate/fl_prior.md`).
Design: the split the 19e handoff named — 200 evalset roots in two
interleaved halves, each a full production cascade (w5 t4), each
**510 s wall** (both under the 10-min bench cap; the split's 2×
spin-up + 2× rung barriers make it a conservative overstatement of
single-line wall).

**Headline: 1,020 s total for 200/200 CONVERGED — zero gap-capped
finals, zero errors (FL3 PASS beyond its bar) = 706 evals/hour
measured** (banked line 10.8 h ≈ 18.5/hr → **38.1×**). Per-eval:
2,953 worker-s / 200 = **14.8 worker-s/eval** — the full population
is ~26% heavier than the 20-seed subset's 11.7; the model's
174-root soft spot was real.

**FL1 (≤900 s) REFUTED at 1,020. FL2 (±20% of model v2's ~750 s)
REFUTED at +36% — and model v1's "~17 min ≈ ~700/hr" (19e) was
accidentally EXACT**: its phantom second wedge (~250 s that does not
exist, 19g) cancelled its underpriced rung-0 pool (measured ~631 s
elapsed vs ~515 modeled). Two wrong terms, right sum — a model that
validates by luck is still wrong; the measurement replaces both
models.

**FL4 fired, informatively: 555212 LADDERS in production.** Its solo
rung-0 convergence (72.2 s, 19g) is wall-cap-marginal against the
90-s rung-0 cap; width-5 contention pushed it over, it gap-capped
and converged at rung 1 (135 s, width 2, rss 13.4). The cascade
self-corrects at the price of one failed attempt. In-fleet rung mix:
**197 / 2 / 1** (rung 1 = 555095 138 s + 555212 135 s; rung 2 =
555090, 208 s, rss 27.6 — its best time yet). Measured line Amdahl:
rung-0 pool ~62% of wall, the wedge ~20%, barriers/tails the rest —
the pool is 19c's closed question, the wedge is 19a's cap verdict.
**At this altitude the line is priced by measurement; this is the
measured-exhaustion receipt for the perf push at H4-cap-256.**

## 2026-07-19i — the tail-idle lever: static shards die, pull-based dispatch lands; balance comes from the pull, not the cost model — line re-measured 727 evals/hour

The handoff's one unpriced move: rung-0 fleets dropped to 2/5 alive
near the tail. Priors TI1–TI5 + FL5 registered before every number
(`scratch/fused-iterate/ti_prior.md`).

**Stage A (simulation, free)**: the 19h line's 200 measured per-root
walls replayed under candidate schedulers. Static LPT packing is DEAD
ON ARRIVAL — by-sigma it *pessimizes* (0.61–0.73× vs the snake:
estimate error compounds in a fixed partition); even by banked walls
(rank-corr 0.889 with fresh walls) it's a wash. Pull-based dispatch
tracks the perfect-balance floor with any reasonable order: ceiling
1.117× (h1) / 1.282× (h2) of rung-0 makespan.

**Stage B (5 paired arms, h2-even 50 roots, rung-0 only, one
session)**: static 126.2/125.3 s; dyn-sigma 106.9/108.5 s; dyn-banked
117.6 s. **Keep bar (≥1.15×) passed in BOTH orderings: 1.181× and
1.155×.** The night's insight: **the sharper cost model LOST.**
Banked-wall ordering front-loads the true monsters into simultaneous
residency and pays +13.6% cum contention inflation; sigma's noisy
ranking (0.622) decorrelates the heavy phases (+3–5% cum) and the
pull self-balances regardless — every dynamic arm's shards landed
within ±1 s. The old snake's stagger insight, reborn inside the pull.
`--cost-file` built, measured, DELETED (no-legacy). Parity across all
arms exact (verdict/gap/iters/reference value per seed). Residual
quoted flat: sigma order can strand one ~37 s root (555233) in the
last-40 dispatches — ≤~30 s tail exposure, the sim's sigma-vs-oracle
gap; accepted, contention dominates.

**Ship + full-line re-measure (FL5, same split as 19h)**: h1 540 s +
h2 450 s = **990 s for 200/200 converged = 727 evals/hour (39.3×
banked), 15.1 worker-s/eval**. FL5a (≤960 s) REFUTED at 990 — quote
1.03×, not a triumph: h1's snake partition was already near-balanced
(1.037× realized) and sigma inflation ate +5.8% of its cum; h2's
pool, the imbalanced one, gave the real win (347.6 → 279.2 s
makespan, 1.245×, cum flat). Rung mix 197/2/1 reproduced with the
SAME three roots (555212 ladders again — its marginality is stable
under the new order), and **all 200 merged reference values are
bit-identical to the 19h line**: scheduling moved, nothing else did.
Code: `plan_shards` → `dispatch_order` + atomic claim files
(`hoyt/refsweep.py`), net simpler; 53 tests green. The
measured-exhaustion receipt now includes the scheduling family:
pools balance to ±1 s, so residual line wall is contention
inflation + the wedge + barriers — all previously priced.
