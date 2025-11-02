# Texas 42 Vision (mk5-tailwind)

Purpose: a living north star that states why we’re building this, what outcomes we’re aiming for, the non‑negotiable guardrails, and the next focused bets. It complements, not repeats, the reference docs.

Related: see CONCEPTS.md (reference), ARCHITECTURE_SYNTHESIS.md (synthesis), OPUS_CONCEPTS.md (mental models).

---

## North Star Outcomes

- Fair, deterministic multiplayer: same seed + actions → identical state across environments.
- Composable game modes: add special contracts and UX/policy features without touching executors.
- Shareable, teachable games: instant replay from URLs for debugging, coaching, and learning.
- Low‑friction clients: clients trust server truth, render projections, and stay lightweight.
- AI as peers: AI plays via the same protocol as humans with no privileged access.

---

## Non‑Negotiable Principles

- Event‑sourced, pure core: state is derived from history; transformations are pure and immutable.
- Server authority and trust boundary: clients never revalidate or refilter server decisions.
- Capability‑based access: permissions are tokens, not identity checks.
- Single composition point: rules (layers) and variants compose once in the kernel.
- Parametric polymorphism: executors call rules; they never inspect state to branch on modes.
- Zero coupling: engine is unaware of multiplayer, transport, persistence, or UI.
- Filter on demand: unfiltered state at rest; per‑request visibility filtering.

These mirror the Architectural Invariants and define regressions if violated.

---

## Key Bets (6–12 Months)

1) Transport evolution (Workers/Edge)
- Goal: run kernel/server on Web Workers and edge (e.g., Durable Objects) with identical semantics.
- Success: action round‑trip P95 < 250ms in‑region; < 500ms cross‑region; zero logic forks by transport.

2) Spectator mode + commentary
- Goal: many observers receive filtered state and optional commentary without impacting players.
- Success: 25+ spectators sustain stable CPU/memory; zero impact on action latency; toggleable commentary stream.

3) AI strategy progression
- Goal: add intermediate/expert strategies while preserving protocol equality and determinism.
- Success: intermediate beats beginner ≥ 60%; expert beats intermediate ≥ 60% in simulation; no privileged reads.

4) Composition explainability + compatibility
- Goal: make composed rules and variant pipelines inspectable; prevent drift.
- Success: “explain composition” tool for a config; matrix tests for Layer×Variant with goldens; invariant checks green.

5) Replay at scale (snapshotting + URL versioning)
- Goal: fast load of long histories and durable share links across versions.
- Success: snapshot every N actions with hash verification; replay time < 1s for 95th percentile games; URL version upgrader.

6) WebRTC pilot (exploratory)
- Goal: optional P2P for low‑latency local/lan play with server mediation fallback.
- Success: ≥ 90% successful connections in test harness; LAN round‑trip < 100ms; graceful fallback to server relay.

---

## Non‑Goals (For Now)

- Client‑side validation or duplicate game logic.
- Privileged AI with hidden state access.
- Storing filtered state at rest.
- Inheritance‑heavy, monolithic classes or global mutable state.
- UI‑embedded rules or transport‑specific logic in core.

---

## Now / Next / Later

Now (0–2 months)
- Composition explainer (CLI/devtool) and Layer×Variant compatibility tests.
- Snapshotting + URL versioning; migration path for old links.
- Worker transport prototype and AI lifecycle integration.

Next (2–4 months)
- Edge (Durable Objects) adapter with soak tests and latency SLOs.
- Spectator + commentary capabilities and delivery pipeline.
- Intermediate AI strategy with evaluation harness.

Later (4–6 months)
- WebRTC pilot with TURN/STUN fallback path.
- Expert AI exploration (tree search/heuristics) under deterministic constraints.
- Mobile/web accessibility refinements and spectator UX polish.

---

## Risks & Trade‑offs (with Mitigations)

- Composition drift (Layers vs Variants): add explainer tooling; goldens for representative combos; keep invariants tests mandatory.
- Override precedence opacity: surface effective rule sources and order in the explainer output.
- Per‑request filtering cost: profile; memoize projections opportunistically; budget latency SLOs.
- Concurrency/consensus semantics: write transport‑agnostic tests for contention and ordering; document guarantees.
- URL/replay migration: embed version; provide upgrader and deprecation windows; snapshot to cap load time.

---

## Success Metrics (Scorecard)

- Determinism: property tests pass across environments; seed+history reproduces state byte‑for‑byte.
- Latency: action execute→broadcast P95 within SLOs per transport; client render idle to interactive < 50ms per update.
- Stability: memory/CPU stable under 4 players + 25 spectators for 30 minutes.
- Quality: invariant test suite 100% green on main; zero regressions on “must‑not‑break” scenarios.
- Adoption: number of shared URL replays opened; tutorial/spectator session minutes per week.

---

## Working Agreements

- Keep this document short; review quarterly in planning.
- Record major changes as ADRs (docs/adrs/ADR-YYYYMMDD-<slug>.md) and link them here.
- If a change conflicts with Non‑Negotiable Principles, treat it as an ADR with explicit trade‑offs.

---

## Pointers

- Concepts reference: docs/CONCEPTS.md
- Synthesis and core components: docs/ARCHITECTURE_SYNTHESIS.md
- Mental models: docs/OPUS_CONCEPTS.md
- Multiplayer protocol: docs/protocol-flows.md, src/shared/multiplayer/protocol.ts

