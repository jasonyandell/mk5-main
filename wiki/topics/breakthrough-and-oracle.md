---
title: Breakthrough and Oracle (Dec 24, 2025 – Jan 8, 2026)
kind: topic
first_seen: 085ffa71
last_updated: b89ff635
status: complete
---

## What this era is

The fortnight the project stopped being a web game with a decent minimax and became a
machine that computes ground truth. It opens Christmas Eve on a wall of its own — PIMC over
full bidding is intractable ("21 choose 7. yikes," 085ffa71, 2025-12-24) — and closes eleven
days later having built the tools that would eventually meet the *next* wall. [[suit-algebra]]
fixed the representation; [[the-oracle]] (three solver generations, then Crystal Forge) made
perfect-information ground truth cheap; [[strategy-fusion]] named, with mathematical
precision, why that ground truth cannot simply be averaged into an imperfect-information
bidder; [[the-analysis-epic]] spent three days characterizing what the oracle had built
without yet finding a mechanism to consume it as a plan.

The stated ambition, from the first night: "we are gonna get us an AI that's decent. been
dreaming about this since I was 12 literally. making a 42 ai. it ain't easy!" (085ffa71,
2025-12-24). The hobby-economics aside a day later: "I'm at like 1200 bucks in to this 42
game and it's the best 1200 bucks I've spent on a hobby in years, maybe decades." (4493b30d,
2025-12-25). And the recurring note on why any of it mattered, mid-failure: "this is a
crystal palace in the sky we're building and it's for FUN." (653a0e68, 2025-12-29) — the
naming that became [[forge]]'s own self-description a day later.

## The arc

1. **PIMC hits a wall, backward induction is born as a hunch** (Dec 24). Live PIMC over full
   bidding was too slow to matter. In the same breath: "I wonder if play itself can be
   bootstrapped from pimc starting at late game and bootstrapping earlier?" (085ffa71) — the
   idea that reorganizes the whole era.
2. **The algebra** (Dec 24–27) — see [[suit-algebra]]. Absorption vs. power, S₇ symmetry, the
   τ-encoding, "called suit." Fixed the representation before anything was built on it.
3. **The solver, three generations in a day** (Dec 27) — see [[the-oracle]]. CPU
   regret-tables → first GPU solver → solver2, with the score-removal compression that made
   the whole thing fit on a consumer GPU.
4. **Distillation's first honest wall** (Dec 28–29). A raw-domino-ID MLP memorized seeds
   instead of learning structure (test loss 0.040 vs. target <0.02, [[suit-algebra|τ-encoding
   diagnostic]]); a transformer built the same week closed the gap (86.35%/82.89%, 1.03x
   generalization gap vs. the MLP's 1.8x). "We have PERFECT play here. it's just too much data
   to be practical. using a transformer to shortcut is awesome but is it a valid shortcut?"
   (653a0e68, 2025-12-29). A parallel **policy-network** epic (t42-vvvz) and the **ValueMLP /
   Confidence Ladder** track — a DP-solver → Value-MLP → fast-PIMC → transformer pipeline
   ClaudeAI proposed in Dec 2025 (`docs/claudeai-mlp.md @ 233b7dc5`) — both specced a full
   in-browser AI pipeline the same week, running without referencing each other — see
   "Corrections" below.
5. **The naming, and the sun** (Dec 30–31). `forge/` (ML) and `core/` (engine) named — "the
   crystal forge sounds so badass. I can say 'yeah over in the forge' and I feel like a cool
   dude." (bec8b3d4, 2025-12-30). H100s rented for the first time ("EXCITING OMG," b00686d3,
   2025-12-31); a 94.5% accuracy plateau broken by a 817K-param model at 97.1% acc / 0.11
   q-gap (`2ea5cc1`), then 97.8% / 0.072 with a value head (`fc3acc7`) — whose own commit
   message names the era's cleanest engineering conclusion: **"bidding needs simulation, not
   regression."** That redirected the project to `forge/bidding/`, built Jan 1 and vectorized
   135x the same night.
6. **Strategy Fusion** (Jan 3–5) — see [[strategy-fusion]]. The era's most durable theoretical
   result: naively averaging perfect-information oracle rollouts overstates achievable
   imperfect-information value. Jason had already felt the bug precisely — "it thinks 2-2 is
   'just as safe' to lead as your high trump even when someone else might have a trump"
   (adb6de51, 2026-01-03) — before the formal name arrived.
7. **The analysis epic** (Jan 6–8) — see [[the-analysis-epic]]. 131 epic-scoped commits,
   folk-wisdom refuted six ways, a self-caught-and-retracted overclaim, and a mechanical
   epistemic-audit closeout.

## Ruled in / ruled out

The wall this era's tools would eventually meet (late Jan's founding condition: E[Q] became
the undefeated champion but "has no plan, no strategy, no feel; it computes p(make) and EV
and stops") did not yet exist as a stated problem. What this era actually settled:

**Ruled in (the machinery is sound):**
- The oracle is trustworthy and cheap enough to build on — 10.3M states in 22s, 97.8% policy
  accuracy, a 2M-state solve fitting in 4GB VRAM ([[the-oracle]]).
- Simulate-then-count beats value-regression for bidding (working P(make) vs. the 7.4-MAE
  regression failure).
- A seed-invariant encoding (τ / trump_rank) is a precondition for any of this to mean more
  than memorized seeds ([[suit-algebra]]).

**Ruled out (as consumption strategies):**
- Naively averaging perfect-info rollouts as a bidding evaluator — [[strategy-fusion]] proves
  it's an upper bound, "systematically advise you to bid too high." The sharpest ruling of the
  era.
- Count-centric abstraction: "just not a good abstraction. dead end. didn't correlate with
  good play at all." (conv 33bdd626, 2026-01-08).
- Six traditional folk-heuristics and a three-axis feature decomposition (R²=22.8%) as feature
  bases for good play ([[the-analysis-epic]]).
- Shipping a distilled net as an in-browser player — ruled out by abandonment, not by a stated
  negative (see "Corrections" below).

**What it did not touch:** consumption *at the level of strategy*. This era solved
**evaluation** — computing p(make)/EV correctly and correcting the bias in how it's
aggregated. It never approached **strategy** — forming a plan, modeling the opponent,
concealing, signaling. The Strategy Fusion fix (aggregate E[Q] per action, then argmax) is
real and durable, but "argmax a better expectation" is still evaluation without a plan.

## What was missed at the time

The 2-2-lead pathology named on Jan 3 — "it thinks 2-2 is 'just as safe' to lead as your high
trump even when someone else might have a trump" — is not primarily a Strategy Fusion bug.
Strategy Fusion is about bidding value being biased high because per-world best play got
averaged; its fix is aggregation order. The 2-2 lead is about the player having no strategic
model at all — no concealment, no read of who holds what, no plan beyond in-isolation EV.
Fixing the aggregation gives a better number; it does not give the player a plan. The era
conflated these two, declared victory on the aggregation half, and by mid-January the whole
thing had been mentally filed as "well past, that's the whole reason for Stage 2" (conv
23d674a1, 2026-01-17) — but the pathology named in one sentence on Jan 3 is, almost word for
word, the "no plan, no strategy, no feel" wall named months later. It was visible then, named
precisely, and then optimized past.

Separately: the browser AI the whole era was nominally in service of — "build some ai that
can do PIMC, then distill that into a policy network etc for the web game" (085ffa71,
2025-12-24) — got fully specced and then quietly abandoned once the research current pulled
toward the oracle and the analysis epic. `src/game/ai/actionSelector.ts` today still offers
only `'beginner' | 'random'` — no neural player has ever shipped in the actual game. That
trade (spectacular oracle, no playable AI) was never made as an explicit choice.

And a real piece of unrelated cleanup work — `actionsEqual`, capability-builder dedup,
scoring consolidation, GameState-factory unification, 22 commits on the short-lived `mk9`
branch (Dec 25–27) — was merged *into* `mk9` from the project's actual mainline (`mk8`, which
is what became today's `main`) but never merged *forward*. None of it is an ancestor of
today's `main`; `actionsEqual` exists only in `origin/mk9`'s tree. That cleanup is simply
gone — a real, if minor, loss.

## Corrections to the record

Several claims that had crept into prior narrative framing do not survive direct
re-verification against commits, beads, and the raw conversation corpus:

- **The suit algebra was not "later extended to the Sevens variant."** [[suit-algebra-spec]] (then `docs/theory/SUIT_ALGEBRA.md`)
  explicitly excludes Sevens ("Nothing in this algebra applies"). The extension bead
  (t42-d2ia) sat `pending` and was closed only by an automated stale-sweep four months later,
  not by an implementation.
- **No CFR work happened in this era.** The DP/backward-induction solver ([[the-oracle]]) is
  the only solver technique in-window; CFR belongs to a pre-era count-abstraction line Jason
  later called a dead end.
- **No Lean / formal-proof-assistant exploration exists anywhere in-window** — treat any such
  claim as asserted-unverified.
- **The "3blue1brown visualization" shipped as a static LaTeX PDF, not an animation.** Real
  artifact ("look at it it's beautiful. worked great. holy cow," fdac8110, 2026-01-01), but no
  Manim/video was ever built.
- **The domino-tables test count is 11, not 28.** "28" was the count of *dominoes* tested in
  the first test's name, not test blocks — it propagated from the commit message.
- **Jan 7 was 131 epic-scoped commits, not 286.** 286 is the repo-wide count including `bd
  sync:` auto-commits.
- **The `mv0-mv6 → q0-q6` rename is dated Dec 30 (`77f3823`), not Dec 31.**
- **The Strategy Fusion "~0.1 pts/game" magnitude is external-domain** (Skat/Bridge, Long et
  al. 2010), not measured in Texas 42 — the research doc itself lists an in-domain measurement
  as unaddressed. See [[strategy-fusion]].
- **The policy-network / ValueMLP pipeline was mostly never built.** Both tracks specced
  Python modules (`scripts/solver2/{features,model,train,evaluate,export_onnx}.py`,
  `scripts/mlp/{model,train,encoding,dataset,export}.py`) via imperative bead task-text; direct
  `git log --diff-filter=A` shows none of those files were ever created. What did run was two
  real `train_mlp*.py` scripts under an unplanned path. The surviving model lineage runs
  through the Crystal Forge transformer, not either named MLP track.
- **`mk9`'s cleanup work was genuinely lost, not redone elsewhere** — confirmed via
  `git merge-base --is-ancestor` on all 22 unique commits and a direct symbol-presence check
  (`actionsEqual` exists only on `origin/mk9`).
- **The "solve the whole traditional game" framing was rhetorical.** The real, same-session
  scope was explicit: "I just want to solve a few seeds completely, not the whole entire game
  for every possible deal." (conv 7343283b, 2025-12-27).
- **Bead-driven planning generated far more plan than artifact, and stale-bot closes overstate
  completion.** Dozens of this window's beads show `status: closed` only because an automated
  sweep on 2026-05-03 closed them four to five months later — "closed" in this window's
  tracker does not mean "done."

## What it did well

The project polices its own claims, and that is a capability, not a footnote. A "19% skill,
81% luck" variance-decomposition framing was struck the same day it was noticed to be false:
"both components are determined by the random deal — the oracle plays perfectly with no human
decisions measured." (`d29a317`, 2026-01-07, see [[the-analysis-epic]]). Six traditional
domino folk-heuristics were tested against the oracle and refuted, not confirmed. Dead ends
were named plainly: "count centric was just not a good abstraction. dead end. didn't
correlate with good play at all." (conv 33bdd626). A tempting shortcut was refused on
principle: "tempting but those are heuristics. always the siren song of heuristics in this
game but it defies them." (conv c1917412, 2026-01-07, on reaching for particle filters from an
unrelated side project — redirected into a likelihood-ratio opponent model instead of built).

## Links

[[suit-algebra]] · [[the-oracle]] · [[strategy-fusion]] · [[the-analysis-epic]] ·
[[forge]] · [[forge-analysis]] · [[pimc]] · [[rank-vs-price]] · [[champion]] · [[gus]] ·
[[sources/claude/era2-breakthrough-oracle|conversation digest]]
