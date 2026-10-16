---
title: "Demon repository preparation: native extraction works; the auction policy must move too"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
---

**STRATEGY-FUSION FENCE.** [[the-demon]] remains a fenced opponent. Jason
requested preparation for a separate repository, with dependencies identified
before the move. This page records an extraction rehearsal, not a repository
cutover or a new strength result. Follow-up and coordinated heads-up:
[#188](https://github.com/jasonyandell/mk5-main/issues/188).

## Recommendation

Move the complete player into its own repository with immutable upstream Git
dependencies first. Keep the referee, tournament registry and research evidence
in their current homes. Separate the shared science later, in coordination with
the existing planner split [#158](https://github.com/jasonyandell/mk5-main/issues/158).
No new repository, production source removal, registry mutation or licensing
change occurs in this preparation.

The native engine can leave now. A complete player release should wait for the
auction-policy boundary and external-source tournament integration described
below. Copying only `tournament/demon/` would leave the strongest player's
early-40 behavior behind.

## Dependency boundary

The audit uses mk5-main `2b0632fe`; the engine export uses the immutable
[[d48909c4]] pin. The engine tree is identical at both revisions:
`26dd409acd3e71ac4bcab367fcd46906ffdea4f3`.

| Component | Actual dependency | First move |
|---|---|---|
| Native play and forecasts | `tournament/demon/`: sampling, declaration reader, search, bidding forecasts, JSON-lines CLI and native tests | Move these files intact to the new repository root. |
| Exact per-world search | `silas`, with defaults disabled and only `clairvoyant` enabled; typed dominoes, contract, position and solver | Retain Silas upstream at an exact Git revision. It is a standalone crate, but currently `publish = false`. |
| Rules and hidden-world construction | `scenario_core::rules`, `fiber::Fiber`, `planner::{Rec, SplitMix}`, `physics::ZebPhysics` and conversion helpers | Retain the same upstream revision. This is the main coupling to the larger project. |
| Actual auction and declaration decisions | `DemonSeat` in `tournament/demon_fenced_seat.py`: opening/overcall rules, early 40, partner pass, support thresholds, forecast cache, declaration choice, decision seeds and game lifecycle | Extract the policy into a portable module; keep a thin tournament bridge. A later Rust port is a separate behavior-preserving change. |
| Referee and seat adaptation | `arena`, `forge.zeb.game`, participant parsing, process cost collection | Keep upstream. The player receives its own hand and public observations through a documented protocol. |
| Reproducible participant materialization | `participants.py`, `artifacts.py`, `demon_fenced_seat.build` | Add explicit external-repository provenance for the new participant; preserve old records and their rebuild path. |
| Research evidence | Existing public Hugging Face datasets and wiki studies | Keep the immutable URLs; link from the new repository. |

The native executable needs no Python, Node, GPU, weights or model downloads.
It uses Rust/standard-library code and Cargo dependencies at build time, then
JSON lines on stdin/stdout. Rust edition 2024 is required; the rehearsal does
not establish a minimum supported compiler or validate every target.

The current Python seat eagerly imports `arena.cli.parse_bidder`, even for a
PIMC bidder. That import reaches `arena.match` and NumPy. Its play bridge uses
the pure Python game/referee types under `forge`; this does not imply a GPU
runtime. Copying the adapter unchanged would drag in the harness unnecessarily.

## Why scenario_core deserves attention

The required data structures are embedded in a broader planner crate:

- `Rec` replays the public record and void information; replacing it changes
  what the sampler knows.
- `Fiber` depends on `planner::SplitMix` and `ALL_TILES`. The completion-count
  sampler and exact RNG stream are part of same-seed behavior.
- `ZebPhysics` bridges declaration IDs and rules to Silas. It is shared rules
  infrastructure, not a call to Zeb's learned player.
- The crate also compiles field/convention/planner modules the Demon does not
  invoke as a player. Its `python`/PyO3 feature is not enabled here.

The source [license line](../../LICENSE) declares Demon and Silas CC0, but
`scenario/core/src/planner.rs` is `CC0-1.0 AND MIT`, and other core files contain
MIT Zeb code. The repository explicitly says to treat the mixed files as MIT
until #158 separates them. A Git dependency preserves that boundary; copying
the entire core and describing it as CC0 would not.

An eventual small science crate could own the rules, record, RNG, fiber and
physics bridge. That requires coordinated changes and parity checks in both
the Rust and Python consumers. It is not necessary to block separate Demon
repository ownership on that refactor. Replacing the sampler with another
constructor during the move would mix extraction with a policy/seed change.

## Native extraction rehearsal

The export flattens `tournament/demon/` to a local directory, carries its native
tests and CC0 notice, and changes only these dependency declarations:

```toml
scenario_core = { git = "https://github.com/jasonyandell/mk5-main", rev = "d48909c485dc647ac6bada3577000b5602b95f64", features = ["silas"] }
silas = { git = "https://github.com/jasonyandell/mk5-main", rev = "d48909c485dc647ac6bada3577000b5602b95f64", default-features = false, features = ["clairvoyant"] }
```

Both direct and transitive Silas dependencies must use the same source identity.
The resolved graph contains one Silas package, with only `clairvoyant`; no
PyO3 package or local sibling-path dependency is present. Cargo.lock changes
only by adding the Git source to `scenario_core` and `silas`. Registry crate
versions and all native source files remain unchanged. Git dependencies still
fetch this monorepo; this stage is separate ownership, not an independent
release stream for every dependency.

`cargo test --locked` passes all 21 native tests; the release build also uses
`--locked`. All **18,125 saved requests from 128 game traces** match the prior
root-reuse engine's replies, including node counts, with only descriptive `ms`
excluded: 16,772 plays, 1,207 support forecasts, 18 best-declaration forecasts
and 128 hellos. This does not test an extracted auction policy: it replays the
native requests the existing policy produced.
This is an Apple arm64 rehearsal with Rust 1.95.0, not Linux/MSRV certification.
Every build, test and replay uses `scripts/cpu-slot`; no quiet-machine wait or
wall-clock decision budget is introduced.

Reproduction uses the public `prepare.py` and `verify.py` receipts, Python 3.12+
for preparation tooling, and the pinned corpus in [[demon-instructions-2026-10-16]].
Place the scripts under `scratch/demon-move-prep/` in a checkout retaining the
source pin, run `prepare.py`, resolve/test once, then use `cargo build --release
--locked --manifest-path scratch/demon-move-prep/engine/Cargo.toml`. Save
`cargo metadata --locked --format-version 1` for that manifest as `metadata.json`
beside the scripts; `verify.py --corpus <prior-study-directory>` checks lock,
dependency identity, unchanged files and the native request corpus.

## Full-player cutover gates

1. **Carry the auction policy.** The binary currently exposes `hello`, `play`,
   `declaration_model`, `forecast`, `forecast_best` and `forecast_support`, not
   `bid`/`declare`. Extract the actual seat policy, including its exact floating
   cutoff comparisons implemented by integer bisection. Preserve seed derivation,
   hand numbering, per-team process lifetime and caches. Keep the confirmed
   configuration explicit: 100 play worlds, declaration weight 1, margin ties, 64 bid worlds, both
   cutoffs 0.875, bounded/support forecasts, bait 40, declarations 0–7 and 9.
2. **Define the external seat interface.** Document tile IDs, declaration IDs,
   own-hand/public-record inputs, seed ownership, auction state, errors and
   lifecycle. No hidden opponent hands cross the interface. Use a thin mk5
   adapter; do not import the tournament as the standalone player's library.
3. **Establish independent gates.** Native correctness tests, full auction and
   lifecycle golden traces, deterministic repeat, license notices, and local
   fence rules travel with the player. Keep whole-game legality and duplicate
   faceoffs against mk5's referee. Verify Linux and macOS before release.
4. **Add external-source provenance.** Today's `DEMON_SOURCE` pins three mk5
   trees (`silas`, `scenario/core`, `tournament/demon`); the builder fetches mk5
   `origin` and archives those trees. A new source repository needs an explicit
   immutable repo/commit/lock/config identity and integrity-checked materializer.
   Create a new participant ID; never reinterpret an old one. Retain
   `pins/demon-d48909c4` and older source/evidence required for rebuilds.
5. **Preserve the fence across repositories.** Keep [[strategy-fusion-fence]]
   annotations and tests. Adjust mk5's path-specific guard and inventory during
   the coordinated cutover. Walt, Zeb and Silas still must not depend on the
   Demon. A public repository does not widen the sanctioned use.
6. **Prove identity before promotion.** Require exact native reply parity,
   auction/declaration/lifecycle parity and full duplicate-game traces for the
   same deals. Compare retired instructions for any cost claim; CPU and wall
   remain descriptive. A changed source location alone earns no new win-rate
   claim. Remove the active in-tree copy only after external integration passes;
   historical Git pins remain the reproducibility mechanism.

## Evidence

The [public rehearsal receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/20aec66014f1c5947c5dac22414ac4a5885fd956/demon/2026-10-16-repository-prep)
contain the export recipe, source manifest, dependency/lock audit, all replay
replies, build/test/check logs and SHA-256 sums. Publication revision:
`20aec66014f1c5947c5dac22414ac4a5885fd956`. The report, hash manifest and first
reply file were downloaded again and matched byte for byte.

The local rehearsal is preserved under
`scratch/demon-repository-prep-2026-10-16/`, with the engine export in `engine/`.
The public recipe recreates it from the retained Git pin; build artifacts and
dependency checkouts are deliberately excluded from the public evidence.
No strength comparison or actual repository migration is part of this preparation.
