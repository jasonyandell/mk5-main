---
title: Research night 2026-10-12 — floor agent (verbatim)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

# FLOOR role — "the floor applies only when the worlds were sampled" (2026-10-12)

Branch `worktree-agent-a72be62a882667abe`, commits `134b253f` (the rule + tests) and `f4078fa4` (docstrings: where it reaches).
Not pushed, not merged.

## The rule as implemented

`Rung.sampled_floor: bool = False` (referee) / the 4th element of each rung tuple
(core). When true and the rung's invocation got the WHOLE fiber (enumerated because
`fiber_size <= n`), the invocation's floor is 1, i.e. never floored. Root never
floored (unchanged). The default keeps the old behavior bit for bit.

- `scenario/fiber.py:127-151` — `sample_worlds` returns `(worlds, exact)`, `exact =
  fiber_size(...) <= n` (the same test that already chose enumeration).
- `scenario/planner.py:88` — the field on `Rung`; `:313-321` — `_invocation` takes
  `floor = 1 if exact and rung.sampled_floor else rung.floor` and hands it to
  `_Invocation` (which no longer reads the rung's floor itself). `_floored` untouched.
- `scenario/core/src/fiber.rs:170-205` — `sample_worlds -> (Vec<[u32;4]>, bool)`.
- `scenario/core/src/planner.rs:323` (Config rung tuple gains `bool`), `worlds_for`
  returns the bit, `:445-466` `Inv::new` sets `floor: if exact && rungs[k].3 {1} else {rungs[k].2}`.
- `scenario/core/src/python.rs` — `Core(rungs=[(n, h, floor, sampled_floor)], ...)`.
- `scenario/compiled.py:48` passes the 4-tuple.
- Arena spec: bare **`x`** on a walt spec, e.g. `walt:k0,n512,F=<net>,x` = "exact cells
  are never floored". Applies to the live rung (the only rung with a floor in arena
  specs). I did not use `e` because `e=<eps>` already exists on the same spec.
  `arena/walt_play.py` docstring + `WaltPlay(sampled_floor=...)`.

Design note: the bit is carried by the invocation's floor, not by a new branch in
`_floored`; "an exact cell has floor 1" is the whole rule in one line per engine.

## Parity / bytes evidence

- `python -u -m pytest scenario/tests -q`: **92 passed** (90 + 2 new) against the new core.
- New tests: `scenario/tests/test_planner.py::test_the_sampled_floor_spares_an_enumerated_fiber_and_only_it`
  (two tricks left, n=160 > fiber of <=90, floor 161: without the rule every live
  interior floor test fires; with it none does; a sampled n=8 invocation mid-hand
  is still floored under the rule) and
  `scenario/tests/test_field.py::test_compiled_matches_the_planner_under_the_sampled_floor`
  (referee = core, `decide` and `analyze`, nodes equal, at 8 late positions each for
  `Rung(512, None, 16, sampled_floor=True)` + net field and a two-rung points config
  with the net under rung 0; asserts the rule changes q somewhere so it is not vacuous).
- Default bytes unchanged: `scratch/research-night/floor/bytes_check.py` digests
  (tile, q, nodes) over 120 positions. HEAD code + the shared installed core vs branch
  code + new core, rule off: `k0,n512,F=net` 5c887392ae57fca3 = 5c887392ae57fca3;
  `k1,n160,m8` 15d196fc8f91f292 = 15d196fc8f91f292. Rule on: ac04badb..., d41949bd...
- **The shared interpreter's `scenario_core` was NOT reinstalled.** The core's
  constructor signature changed (4-tuple rungs), so installing the new wheel globally
  would have broken every other agent running the main checkout's `compiled.py`. The
  wheel is installed with `pip install --target scratch/research-night/floor/site` and
  put first on `PYTHONPATH` for my runs only. When this merges, the hub's wheel recipe
  must be rerun (the old installed core rejects the new `compiled.py`).

## Arena

Protocol: `bid30` both sides, 1 mark/hand, games to 7, mirrored halves, 384 games,
seed 1 (`python -u -m arena.cli --team-a bid30+walt:<spec>,x --team-b bid30+walt:<spec>
--n-games 384 --base-seed 1`). A = rule on.

| block | games won (A) | marks/game (95% CI) | made A / B | wall |
|---|---|---|---|---|
| `k0,n512,F=v3 net` (x vs not) | 190/384 (49.5%; halves 54.2 / 44.8) | **−0.02 [−0.35, +0.29]** | 36.3% / 36.5% | 584 s |
| `k1,n512,m8` (champion shape, x vs not) | 192/384 (50.0%; halves 55.7 / 44.3) | **−0.01 [−0.33, +0.29]** | 36.6% / 36.7% | 1208 s |

Logs: `scratch/research-night/floor/arena.log`, `arena_k0/`, `arena_k1.log`, `arena_k1/`.

## Why it is null: the rule almost never fires (`changes.py`, `changes.log`)

k0,n512,F=net with and without `x`, 2,771 late pip positions (14+ plays down) from
random games: the chosen tile changed in **26 (0.9%)**.

| plays down | positions | tile changed | nodes off → on |
|---|---|---|---|
| 14 | 245 | 1 (0.4%) | 4536 → 4760 |
| 15 | 243 | 7 (2.9%) | 3290 → 3968 |
| 16 | 384 | 2 (0.5%) | 1722 → 1918 |
| 17 | 231 | 4 (1.7%) | 1001 → 1266 |
| 18 | 242 | 7 (2.9%) | 329 → 444 |
| 19 | 251 | 5 (2.0%) | 146 → 179 |
| 20–23 | 1,175 | 0 | identical |

The structure behind it: with two tricks left (where the fiber is always enumerated)
the viewer's only interior decision is its forced last tile, so the floor there was
already a no-op. The rule bites only where three or more tricks remain AND the fiber
fits in 512 (mid trick 6, or earlier with shown voids) — a sliver of the hand, about
one decision in a hundred of the late game, a few per game. A +/-0.3 marks/game
interval cannot see that. Arena marks over the whole k0 block: 2104 vs 2112.

## Reading, in Jason's units

- k1 (champion shape): −0.01 [−0.33, +0.29], made 36.6% vs 36.7%: the same null.
- k0 (the shipped Zeb shape): −0.02 marks/game [−0.35, +0.29], made 36.3% vs 36.5%:
  the same player to within noise, and the diagnostic says that is structural, not a
  lack of power: the two players disagree on ~1% of late decisions.
- The theory claim stands as theory (an exact cell is the posterior under the model,
  so a max there is a best response, not clairvoyance), and the rule is cheap and
  correct by construction. But it does not buy marks, and it costs a little search
  (up to +30% nodes at 17–18 plays, from tiny bases).
- Worth shipping? **As a correctness/aesthetics change, yes-leaning**: it removes the
  one place where the floor's own justification ("a thin cell is a sample artifact")
  is false, at no measurable cost in marks or wall. **As a strength lever, no**: it is
  a measured null and structurally can't be large. If Jason prefers fewer knobs, the
  honest alternative is to make it the unconditional behavior (drop the flag) — the
  arena says either is fine; I kept the flag so tonight's comparison was clean.
- Caveat for theory: "exact posterior" holds under the model (others = the field net,
  deterministic), the same assumption every cell already makes. If the field is wrong,
  a thin exact cell is still a max over a model-conditioned posterior; it is the
  sampling noise, not the model error, that the rule removes.

## What I could not do

- Did not reinstall the wheel into the shared interpreter (deliberately; see above).
- The tile-change diagnostic was run at k0 only; k1's arena block is the same null.
- No wiki update (research night: the orchestrator synthesizes).
