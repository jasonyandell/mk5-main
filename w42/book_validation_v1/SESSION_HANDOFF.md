# W42 Book Validation — Session Handoff

**Session:** w42-claude
**Last update:** 2026-05-03 (orchestrator continuing autonomous run)
**Orchestrator:** Claude Opus 4.7 (1M context)

## State at handoff

### Closed (committed + pushed)

Wave 0:
- baseline ledger snapshot + agent rules of engagement (`t42-snwe`)

Wave 1 (5 offline analyses on existing data):
- 1.1 distribution-lens reranker (`t42-ybo6`)
- 1.2 mark-utility transform (`t42-c6sa`)
- 1.3 hidden-threat impact ranker (`t42-c2y9`)
- 1.4 cross-AI agreement matrix (`t42-m2i7`)
- 1.5 independent ledger audit (`t42-1nmm`)

Wave 2 infra:
- 2.A state-injection harness (`t42-rwdj`, commit 19fc675)
- 2.A.2 oracle-greedy snapshot corpora (`t42-y8b5`, commit 77a8511)
- 2.B bid-aware E[Q] driver smoke (`t42-6j3k`, commit 0c802d4)
- 2.B.2 full 50-seed MPS sweep (`t42-7eop`, commit 720aa83): 259,618
  action rows

Wave 2 probes:
- 2.A.3 reentry v2 (`t42-v9lu`, d9f8dbf): `context-limited` —
  late-game (n=60) contradicts book in book direction
- 2.D low-trump-trap (`t42-jysl`, 29b86b8): `context-limited` —
  count subgroup symmetric
- 2.E pounce bid=30 (`t42-ntbe`, 79b5b7d): `context-limited` —
  oracle pounces for p_make but EV says decline
- 2.C void-creation lead (`t42-26j8`, 8507b11): **`contradicted`** —
  campaign's first contradicted (lead-to-self-void only)
- 2.C.2 void-creation follow (`t42-z31l`, 37a5804):
  `context-limited` in book direction — position-dependent reversal
- 2.G ch02 multi-step bid-only-enough (`t42-ey88`, 6918aa9):
  **ch02-bid-only-enough promoted to `supported`** — all 5 step pairs,
  85/85 slice cells, monotone Cohen d 0.16 → 0.47, transitive
  cumulative additive within 0.81%
- 2.H ch10 mark-multiplier action-level (`t42-8na4`, 6918aa9): no
  status change; 71% action-flip at bid=42; threshold-q is the real
  mechanism, not multiplier scalar
- 2.E.2 high-bid pounce snapshot probe (`t42-8kbh`, 0b7fb01):
  **ch12-setter-pounce-high-bid-off DEMOTED to `contradicted`** —
  reversed Wave 2.B.2's aggregate-proxy promotion; triggered new
  promotion-guard rule in `AGENTS.md`

### In flight

(none)

### Most recently closed

- **Wave 4.1 / Lens v1** (`t42-4ouu`) — utility head-to-head round-
  robin. Closed 2026-05-03. Built Lens (1-step Q-greedy player wrapped
  by utility), reused Zeb's parallel-hand simulator pattern, ran
  {ev, p_make, cvar_10, robust_q25} round-robin × 1000 paired-seed
  hands × 6 pairings in 7 minutes wall. **Verdict: ev wins decisively.**
  Total ordering ev > robust_q25 ≳ cvar_10 > p_make, all 6 CIs exclude
  zero, ev beats p_make by +5.42 pts/hand ([+4.03, +6.81]).
  
  This **inverts the natural reading of Wave 4.0**: EV's "third-
  option discards" turn out to be point-winning, not noise. The book
  aligns with the worst utility (p_make) on this corpus. Two open
  interpretations: (a) book is locally right but globally suboptimal;
  (b) p_make is the wrong meta-objective for bid=30 contracts where
  p_make is near-saturated and tie-breaking arbitrarily.
  
  Sample-sweep at N ∈ {10, 50, 100} confirms N=10 is the right
  operating point. fp16 sanity passed (≥99% argmax match) but round-
  robin ran fp32 (MPS doesn't autocast inside model forward).
  
  **Production-code action item:** `forge.eq.generate.actions.select_actions`
  is essentially Lens(p_make) — the worst utility tested. Switching
  to ev-argmax is a one-line change predicted to improve E[Q] vs
  Zeb-Large win rate. Filed as a separate follow-up bead.

- **Wave 4.0** (`t42-hmjr`) — utility-argmax divergence (architecture-
  decision gate). Closed 2026-05-03. Computed argmax-under-utility for
  ALL legal actions on the 500 ch05-void-creation-follow snapshots.
  **Verdict: gate TRIPPED.** EV vs p_make disagree on 41.2% (CI [36.8%,
  45.6%]). p_make / mark_ev / CVaR_10 / robust_q25 are the utilities;
  p_make ≡ mark_ev exactly at bid=30 (0/500 disagreement, confirms
  affine identity). **Wave 3.0's framing inverted:** p_make picks
  void MORE than EV (38.6% vs 29.4%); EV is the outlier preferring
  third-option discards. The book's void advice aligns with risk-aware
  utilities, not with mean-EV. Recommendation: scope rung-2 utility-
  tunable searcher (decision held for user input on whether to build,
  what to call it, and what scope).

- **Wave 3.0** (`t42-f2ur`) — utility-lens meta-analysis. Closed
  2026-05-03. Re-processed all 7 closed Wave 2 probes through 5 utility
  lenses. Initial read narrowed the p_make/EV split hypothesis to one
  claim. Wave 4.0 broadened it back at the policy-action level (Wave
  3.0 was working at paired-contrast magnitudes; Wave 4.0 at argmax).
  Schema decision still ADOPT-DEFERRED until more probes record full
  utility coverage. AGENTS.md amended with utility-coverage requirement.

### Deferred (filed but not launched)

- **2.F (84-throwaway, `t42-wikw`)**: design pass complete (Option 3
  hybrid: filter Wave 2.B.2 bid=84 generations to bidder-seat 4+-doubles
  hands at endgame; expected ~170 paired contrasts). Held until Wave 3.0
  lands so the utility-lens slate can inform the per-utility verdict
  shape (84 is likely a sharp p_make vs EV split case).
- **Detector hygiene** (`t42-v0m5`, `t42-2yb5`, `t42-btpg`): P2,
  unclaimed.

### Ledger movement summary

3 promotions, 1 demotion, 0 forced reconsiderations:

| claim_id | path |
|---|---|
| ch10-point-system-skill-signal | underpowered → context-limited (Wave 1.5 audit absorption) |
| ch10-timed-marks-advancement-objective | not-yet-tested → context-limited (Wave 1.5) |
| ch02-bid-only-enough | not-yet-tested → context-limited (Wave 2.B.2 paired bid=32 vs bid=30) → **`supported`** (Wave 2.G all 5 step pairs) |
| ch12-setter-pounce-high-bid-off | underpowered → context-limited (Wave 2.B.2 aggregate proxy) → **`contradicted`** (Wave 2.E.2 snapshot-level) |

Status counts after Wave 2.E.2:

| status | count | delta vs baseline |
|---|---:|---:|
| supported | 24 | +1 |
| context-limited | 14 | +2 |
| underpowered | 19 | -2 |
| not-yet-tested | 4 | -2 |
| contradicted | 3 | +1 |

The Wave 2.G `supported` promotion is the campaign's first non-trivial
movement out of `context-limited`/`underpowered` to `supported`. The
Wave 2.E.2 demotion triggered a new promotion-guard rule:
**aggregate proxies do not qualify for promotion**, only paired
same-decision contrasts on the relevant action shape.

### Major emergent thread: utility-objective split (post-Wave 4.0)

Three iterations of this thread:

1. **Initial (Waves 1.4 + 2.E + 2.E.2):** "book may encode p_make-
   optimized advice at the contract threshold; EV-optimal play differs."
   Projected from action-ranking 59% disagreement and 2 probe verdicts.
2. **Wave 3.0 narrowed:** at the paired-contrast / ledger-verdict level,
   only ch05-void-creation-follow shows a true objective-dependent
   verdict (1 of 7 claims). The other apparent splits dissolved into
   "all utilities agree" or "all utilities are unresolved."
3. **Wave 4.0 broadened back, in a corrected direction:** at the
   argmax-action level on those same 500 snapshots, EV disagrees with
   p_make / mark_ev / CVaR_10 on **41-44%** of decisions. The split is
   real and substantial. Direction: p_make / CVaR / robust_q25 pick
   void *more often* than EV; EV is the outlier preferring third-
   option discards.
4. **Wave 4.1 (Lens v1) inverted again:** when the four utilities
   actually play games head-to-head, **EV wins decisively** (every
   pairing's CI excludes zero; ev beats p_make by +5.42 pts/hand).
   EV's "third-option discards" are not noise — they win games. The
   book aligns with the *worst-scoring* utility (p_make) on this
   corpus.

Conclusion: **the multi-utility architecture is real (utilities pick
different actions ~40% of the time)** but **a single-objective EV head
is the strongest fixed-utility choice on this corpus by a wide margin.**
Multi-objective architecture would only beat fixed-EV if utility
selection is *state-conditioned* — which is exactly what t42-nwuu
(Lens v2 future) tests. The book's chapter structure is plausibly an
implicit state→utility lookup table that no fixed utility captures.

Open architectural questions for the user:
- Build rung-2 (utility-tunable searcher over forge engine) at all?
- If yes, what name (Zeb is taken in the AlphaZero sense; candidates:
  Sift, Tally, Burlap, Drey)?
- Scope: pure rollout MCTS (~400 LOC, 2-4h) vs Gus-belief-conditioned
  rollouts (richer, larger build)?
- Validation corpus: ch05-follow-only first (replicates Wave 4.0 with
  search depth) or 10K mixed corpus (tests Wave 1.4 at scale)?

### Pending decisions for the user when resuming

- **2.F (84-throwaway)**: design pass on bead complete (Option 3
  hybrid). Decision pending Wave 3.0 results: if 84 lands as p_make-clear
  in the utility lens, the throwaway-ladder probe is highest-value;
  otherwise it's a one-off claim probe.
- **Detector hygiene beads** (`t42-v0m5`, `t42-2yb5`, `t42-btpg`):
  could roll into a single Wave 2.X cleanup pass when Wave 3.0 lands.
- **Rung-2 build decision**: gate tripped by Wave 4.0. Whether to
  build, what to call it, and what scope are the live open questions.
  Per user discussion: skipping the new model entirely would leave
  measurement against EV oracle n=10 only, which is suboptimal — but
  that's what Wave 4.0 just did with full Q-distribution comparison.
  So rung-2 (a searcher) is the bridge that lets us test whether
  utility-conditioned policies hold up at search-depth, before
  committing to rung-3 (a learned net).
- **Wave 4 follow-on candidates** (rung-2 design dependent):
  - Replicate Wave 4.0 on a 10K mixed-corpus snapshot pool to test
    whether the 41% disagreement holds at scale and across positions
    (currently bid=30 setter-only).
  - Wave 2.F (84-throwaway, t42-wikw): test mark_ev divergence at
    bid=84 where mm=2 and the affine identity breaks. Design pass
    already on file.
- **Wave 3 (auction policy + opponent population)**: deferred from
  earlier sessions; multiple Wave 2 probes flagged that real auction
  strategy testing needs a bid-policy simulator. Wave 2.G's
  `supported` promotion of bid-only-enough is "supported on
  same-hand bid-margin" — cross-contract bid choice still needs Wave 3
  scope.

### Files the user might want to look at first when resuming

- `wiki/experiments/w42-book-validation-campaign.md` — live status
- `wiki/experiments/w42-book-claim-synthesis-and-ai-directions.md` —
  what the book taught us, with all wave findings absorbed
- `wiki/experiments/w42-bookval-v1-wave2-pounce-high-bid.md` — the
  contradicted high-bid pounce (and methodology lesson)
- `wiki/experiments/w42-bookval-v1-wave2-ch02-multistep.md` — the
  first `supported` promotion
- `w42/book_validation_v1/AGENTS.md` — agent contract, with new
  promotion-guard rule
- `w42/book_validation_v1/wave2/bid_aware_atlas/power_analysis.csv` —
  raw evidence for Wave 2.B.2 promotions
- `w42/book_validation_v1/wave3/t42-f2ur_utility_lens_synthesis/`
  (after Wave 3.0 lands) — the master utility-conditional ledger view

### Branches and remotes

- `forge` branch is up-to-date with `origin/forge`. No stranded work.
- All probe agents have either self-committed or had their outputs
  committed by the orchestrator. No agent has touched the central
  ledger directly.
- No stash, no detached HEAD, no in-progress merge.
