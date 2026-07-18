# walt/kernel — the net-free kernel (frozen interface, 2026-07-18)

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

## Honesty line (registered)

Two-team zero-sum with private hands: CFR's Nash guarantee is 2-player.
Claims are "low-exploitability reference priced by exact BR." v1
exploitability = single-seat deviation (partner stays on profile);
team-pair deviation out of scope.

## File ownership (no collisions)

- KERNEL lane: `walt/kernel/{__init__,subgame,profiles,br,bench_kernel}.py`,
  `walt/tests/test_kernel_*.py`.
- CFR lane: `walt/kernel/{cfr,toys,reference}.py`,
  `walt/tests/test_cfr_*.py`. `reference.py` is a tiny pure-python
  implementation of THIS interface for toys only (correctness mirror, no
  perf goals) so the CFR lane never blocks on the kernel lane.
- Neither lane edits existing walt modules, this file, or the other lane's
  files. Python via `/Users/jason/code/mk5-main/.venv/bin/python -u` from
  the worktree root. numba install allowed via
  `uv pip install --python /Users/jason/code/mk5-main/.venv/bin/python numba`.
  Benches ≤10 min wall. Temp files in scratch/.
