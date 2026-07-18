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
