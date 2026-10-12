---
title: Research night 2026-10-12 — the brief (verbatim)
kind: source
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

# Research night, 2026-10-12: does Zeb find the optimal move in finite time?

Jason (the author, asleep; family heritage game, Dijkstra lineage, "beautiful and
correct above all"): "the group of you try to mathematically prove that zeb finds the
optimal move in finite time. or counterexample that and try to find what we'd need to
prove that. if we can prove it in lean fantastic. nice not required. everyone's
experiments are limited to 5 minutes with exceptions granted by [the orchestrator],
watching the clock for 7am."

Hard stop: all agents report by 06:00 CDT. The orchestrator synthesizes by 06:45 and
pushes. Any single experiment run: 5 minutes wall, unless the orchestrator grants an
exception in writing. Always `python -u`. Everything you write lives under
`scratch/research-night/<your-role>/` (gitignored); you do not touch the repo outside
scratch except as your brief says. Code may be READ from ~/code/plunge and
~/code/texas-42 but never copied across repos.

## The object

Read `wiki/entities/scenario-player.md` (the hub: construction, levers, measured
results, "The atlas", "The net in the field", the 250k pour, Zeb the package) and the
two notes in `notes/` (Jason's math discussion from Oct 10 and Oct 12; they are the
prior art for tonight and already contain several counterexamples; build on them, do
not rederive them).

Walt (the ingredient): a seat samples N hidden-hand worlds uniformly from the fiber of
deals consistent with the public record and shown voids (`scenario/fiber.py`, exact
uniform via counting DP), then solves one contingent plan over the whole cell: max at
its own turns over actions, with one action chosen per information set across all
worlds that share it; sum (belief-weighted) at other seats' turns where the other
seats are played by a declared model F (the "field"); leaves scored by a utility
(`make` = indicator of the contract, or `points` margin). Plays the plan's first move.
Replans from scratch every turn with a fresh sample. Deterministic tie rule (lowest
tile id). Rungs: F may itself be a Walt one rung down (k1,n512,m8 = champion) or, in
Zeb, the FIELD NET: a small deterministic integer MLP (343->128->64->28) trained on the
pour of Zeb's own self-play, playing the other three seats at rung -1 (`walt:k0,n512,
F=<net>`). A cell-size FLOOR (f=16): when fewer than 16 live worlds share an interior
information set, the viewer stops optimizing and plays as the rung below would (the
net), with a fresh sample; the root is never floored. The splat: per root move, the
leaf (final bidder-team margin) each world reaches under the plan; pmake(b) is a tail.

Python referee: `scenario/planner.py` (`Walt`, `Config`, `Rung`), `scenario/record.py`,
`scenario/rules.py`, `scenario/field.py`; Rust core bit-identical: `scenario/core/`
(`CompiledWalt` in `scenario/compiled.py`; `decide_batch`). Pours: `scenario/atlas.py`
(`pour`, `load`, `Position`, `makes`, `holding_predictors`). Arena: `arena/`. The shipped
net: `scratch/field_net/v3/margin_d250k_h128x64.npz`. A five-deal pour of the shipped
player with the whole splat at every planned position: `scratch/viz/pour_viz_5/`
(re-pour in 2 s: `python -u -m scenario.atlas pour --deals 5 --spec k0,n512,F=<net>
--seed 424242 --out <dir>`). Tests: `python -u -m pytest scenario/tests -q` (87, ~40 s).

Measured facts to respect: the uniform belief beat every belief tried on the seam
(auction, dice, field-tilt reweighting; the field tilt HURT via effective-sample-size
collapse: "a strong g must be sampled from, not reweight 160 uniform draws"); the
floor calibrated the forecast and bought marks; 512 root worlds beat 160; the net
field is level with the two-rung champion at a quarter of the cost; the data lever
stopped at parity. Beliefs on the seam are PAUSED, not dead.

## The precise questions (the orchestrator's sharpening)

"Optimal" needs a referent. Three candidates, increasing in strength:

  O1. Best response for one seat to a FIXED field F, under the posterior beta^F that
      F induces (worlds weighted by the likelihood that F would have produced the
      observed plays from them). With F deterministic (the net), likelihood is 0/1:
      a world is F-consistent or it is not.
  O2. Best partnership response (a pair of seat-local policies) to fixed opponents.
  O3. Team-maxmin / equilibrium of the hand phase (two teams, zero-sum, hidden hands).

Claims to settle tonight:

  T1 (expected TRUE, prove it): exact Walt (all worlds of the fiber, exact evaluation,
      no floor, no sampling) with belief beta^F is an O1 best response at the root, and
      the computation is finite. State it for a finite extensive-form game with perfect
      recall; the 42 hand phase is an instance.
  T2 (PROVED in Lean, 01:33, sorry-free): under FIXED beliefs (not recomputed from
      the rung below), with P^{-1} = U and P^k = R[P^{k-1}], coordinate c settles for
      all k >= h(c) (dependency height), i.e. after h(c)+1 applications of R; the 42
      hand phase settles by P^{23} = the 24th application (note 2, section 2). The
      fixed point is UNIQUE under a uniform height bound. The bound is attained
      (an operator can hold the root answer for n applications and change at n+1).
      CORRECTION: the brief's first draft said "k >= h(c) applications"; Lean proved
      that false (a coordinate with no dependencies still needs one application).
  C1 (expected FALSE for deployed Zeb): uniform-over-fiber is not beta^F in general;
      exhibit a position (toy 42-like game, and if possible a real 42 position with the
      real net) where Walt-uniform's move is not an O1 best response to F. MEASURE in
      real positions: fraction of uniform worlds that are net-consistent (replay the
      record with the net deciding each modeled seat's plays from that world's hands);
      does planning over consistent worlds only change the move / pmake / the plan?
      This is the belief repair of note 1 section 2 with zero new machinery.
  C2: sampling (finite N), the floor, and the tie rule each break O1 in constructible
      cases; characterize each gap and the certificate that would close it
      (Delta > 2*delta; the early-stopping certificate of note 2 section 2.4).
  C3: O1 fixed points are not O2/O3 optimal in general (note 1 section 4); give the
      42-flavored instance if one is cheap; otherwise state the barrier.
  Q:  the ledger "what we would need to prove it": for each of O1/O2/O3, the exact
      list of changes to deployed Zeb (coherent beliefs via consistency filtering;
      certified sampling; floor certificate; double oracle with certificates for O3)
      and which are affordable. Note 1 section 7 proposes Delta_execution = V_plan -
      V_execute; measuring it on exactly solvable small positions is in scope.

Deliverables per agent: a markdown report at `scratch/research-night/<role>/REPORT.md`
with: claims, proofs or counterexamples (with the exact game and the exact numbers),
experiments (command, wall time, result), what you could not do. Precise, no
hand-waving, numbers in tables, every claim either proved, measured, or marked
conjecture. Write for Jason at 7am with coffee: short sentences, his units (tricks,
hands, marks, pmake).

## Roles

  theory      — T1, T2, C2's gap bounds, the Q ledger. Pure math, markdown with LaTeX.
  counter     — C1/C2/C3 counterexamples in a tiny exact Python lab (enumerate all
                policies; exact Walt operator with uniform vs coherent beliefs).
  empirical   — C1 measured on real 42 positions with the real net and the real core;
                Delta_execution on small exactly-solvable late positions if time.
  lean        — T2 (and T1 if it falls out) formalized in Lean 4 in
                scratch/research-night/lean/ (elan/lake installed; mathlib allowed; the
                setup is exempt from the 5-minute rule, builds are not experiments).
  fable2      — a separate Claude Code session, Fable 5.1: independent attempt at the
                whole question from the brief alone, then (second phase, on request) a
                cold critique of the other reports.

The orchestrator (Fable 5.1, this session) merges everything into
`scratch/research-night/SYNTHESIS.md`, writes the wiki, and pushes.
