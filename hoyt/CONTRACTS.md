# hoyt — the net-free kernel (frozen interface, 2026-07-18)

2026-07-21 additive extension: [[full-metal-hoyt]] preserves every exact CPU
surface below and adds explicitly approximate Metal/sampling surfaces. Nothing
approximate is allowed to masquerade as `br_solve` or the fp64 default.

Program registration: wiki/topics/perf-log.md entry 2026-07-18a. The kernel
is the zero-torch solve substrate: bitboard hands, LUT tricks (from
`walt.tables`, never reimplemented), SoA waves. Consumers: exploitability
meter (exact single-seat BR vs frozen profiles), CFR+ reference profiles +
the frozen-root stable eval, H5/H6 horizon pushes.

## Shared objects (both lanes build against these; do not drift)

- **Payoff** = `np.float64[43]` leaf table indexed by final DECLARING-team
  points (0..42). `points` → `arange(43)`; `make(bid)` → step at bid;
  marks-race utilities compile to this table at the root. Hero maximizes
  `sign * payoff-expectation` with `sign = +1 iff hero_team == bid_team`,
  reported un-flipped (declaring orientation), same as walt.solver.
- **Subgame** = `build_subgame(root, worlds, weights)` — opaque, net-free.
  root: `walt.contracts.EndgameRoot`; worlds/weights per walt/worlds.py
  contract. Chance = the given world distribution; a world is consistent
  with a public action iff the acting hidden seat holds the tile (hero
  plays from root.my_hand). Internal representation is the kernel lane's
  design (public tree + info-set index).
- **Profiles**:
  - `SigmaTable` — deterministic: reachable (seat, hand_mask, node) → move.
    Built ONLY by `compile_sigma(root, worlds, oracle)` — the single
    net-touching call (batch via existing `FieldOracle` public APIs; one
    net decision per unique reachable info set, walt-solve-shaped domain).
  - `StochasticProfile` — same domain → probability vector over legal moves
    (used for exploitability of mixed/CFR profiles; also the export format
    of CFR average strategies).
- `br_solve(subgame, profile, payoff43, hero=root.me)` →
  `BRResult(value, best_move, root_values: dict[move,float], strategy)` —
  exact best response of hero; every non-hero seat plays `profile`.
  Tie rule: hero moves ascending, first strict `sign*v > sign*best` wins
  (walt parity).
- `cfr_solve(subgame, payoff43, iters=..., target_gap=...)` →
  `CFRResult(profile: StochasticProfile, trace: [(iter, gap)])` — CFR+
  (regret-matching+, linear averaging) over ALL FOUR seats' info sets,
  team payoff sign per seat; `gap` = max over seats of single-seat BR gain
  vs the average profile (measured with `br_solve`). Deterministic given
  seeds. Full-tree preferred; if H4 enumeration exceeds memory, STOP and
  escalate to the orchestrator before reaching for sampling variants.
  `engine="metal"` is the additive float32 dense engine: its iterate and
  in-structure gap passes are Metal-resident, while build/export remain on
  host. It is accuracy-calibrated and never bit-parity claimed.
- `WorldSampler.from_root(root)` exactly counts and samples physical hidden
  deals without enumeration; `sample_metal` runs one DP sampler per Metal
  thread. This samples the uniform physical-world distribution only.
- `SampledCFR(root, payoff43, capacity=...)` is the experimental shared-table
  four-seat external-sampling lane. It alternates updater seats, branches all
  updater actions, samples one current-policy opponent action, and exports a
  current regret-matched `SparsePolicy` candidate. The policy can be saved as
  `.npz`; unseen information fingerprints use an explicit uniform fallback.
- `SampledBR(...)` trains against uniform opponents or is created with
  `SampledCFR.fork_best_response(seat)` to reset one seat while freezing the
  other three. A `SampledBRResult` carries independent evaluation standard
  errors plus bounded-payoff empirical-Bernstein radii, not an exact
  `BRResult` and not by itself a gap upper bound.
- `audit_sampled_gap(...)` trains four frozen-policy BR candidates. Without a
  calibrated optimization-shortfall bound its upper gap is infinity and its
  verdict cannot be `converged`. With one, candidate and shortfall miscoverage
  budgets must be supplied separately and are union-bounded. All sparse tables
  fail closed on capacity.
- `metalcal.br_shortfall_upper` supplies the missing one-sided calibration
  primitive: exact gap minus independently evaluated sampled-BR upper bound.
  Production convergence requires candidate uncertainty plus this calibrated
  optimization-shortfall bound; otherwise the verdict is `unresolved`.
- `HoytPlay` (`hoyt/play.py`) is the experimental Arena-shaped player
  consumer. It projects each `ZebGameState` to an `EndgameRoot` containing
  only the acting seat's hand and public state, runs a fresh `SampledCFR`, and
  returns a legal zeb slot. Its default payoff is `payoff_make(bid)`; `points`
  is optional. It may only enter at H5/H6, schedules batch members serially,
  and exposes fixed-candidate action intervals without claiming calibrated
  optimization shortfall. The resumable table stores the full deal under
  `scratch/` solely for the zeb referee and never renders hidden hands.

## Gates

- **K1 parity**: `br_solve(subgame, compile_sigma(...), points)` vs
  `walt.solver.solve` on all 46 fixtures (`walt/tests/fixtures_h4.jsonl` +
  `_expected.jsonl`): identical best_move, value ≤1e-9 rel, AND identical
  root_values per move (≤1e-9).
- **K2 toys**: ≤2-trick games, ≤12 worlds — br_solve equals explicit
  enumeration over ALL hero pure info-set strategies (walt T2 pattern);
  CFR verified on 2-player-izable toys (two seats pinned to a known
  profile) against enumerated maxmin value; exploitability of the CFR
  average → 0 as iters grow.
- **K3 speed** (log to perf-log, priors P1/P3/P5 registered): fixture-suite
  BR ≤0.5 s total; report ns/node and rows/s equivalents.
- **M1–M3 Metal**: MLX GPU availability; float32 dense value/gap/profile error
  on calibration toys; finite-sample conformal value/gap/BR-shortfall math.
- **W1–W3 sampling**: DP population count equals exact enumeration; host and
  Metal draws are physical; empirical cell frequencies pass the registered
  uniformity smoke.
- **S1–S4 sampling**: exact H2/H3 BR values fall inside the registered smoke
  band and root moves agree; capacity fails closed; four-seat sampled CFR
  matches exact H2 value, visits all actors, and frozen-policy forks reset only
  their updating seat. Candidate policies round-trip as artifacts.
- **P1 playable boundary**: changing the three non-acting hidden holdings
  leaves `root_from_state` identical; session save/load is exact; the renderer
  shows only public history plus the human hand; forced moves do not construct
  a solver; complete H5/H6 continuations remain zeb-legal.

## Honesty line (registered)

Two-team zero-sum with private hands: CFR's Nash guarantee is 2-player.
Claims are "low-exploitability reference priced by exact BR." v1
exploitability = single-seat deviation (partner stays on profile);
team-pair deviation out of scope.

## File ownership (no collisions)

- KERNEL lane: `hoyt/{__init__,subgame,profiles,br,bench_kernel}.py`,
  `walt/tests/test_kernel_*.py`.
- CFR lane: `hoyt/{cfr,toys,reference}.py`,
  `walt/tests/test_cfr_*.py`. `reference.py` is a tiny pure-python
  implementation of THIS interface for toys only (correctness mirror, no
  perf goals) so the CFR lane never blocks on the kernel lane.
- Full-metal lane:
  `hoyt/{metalkernel,worldsample,sampled_br,metalcal,play}.py` and
  `hoyt/tests/test_{cfr_metal,worldsample,sampled_br,play}.py`. MLX is required
  for these surfaces (`burl/requirements-mlx.txt`); exact CPU defaults do not
  import it eagerly.
- Neither lane edits existing walt modules, this file, or the other lane's
  files. Python via `/Users/jason/code/mk5-main/.venv/bin/python -u` from
  the worktree root. numba is a RUNTIME dependency since the fused iterate
  engine (#82, `hoyt/iterkernel.py` — cfr_solve's default engine; the
  fused lane's structural build also uses `hoyt/buildkernel.py`, perf-log
  18n); install via `uv pip install --python
  /Users/jason/code/mk5-main/.venv/bin/python numba`
  (engine="wave"/"loop" stay numba-free).
  Benches ≤10 min wall. Temp files in scratch/.
