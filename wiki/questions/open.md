# Open Questions

Questions raised during replay that the sources at that time do not yet answer. Each entry is tagged with the sha that raised it. When a later ingest resolves a question, move it to `resolved.md` with the resolving sha.

Format:

```
- **Q:** <question>
  - Raised: `<shortsha>` ([[source-page]])
  - Context: <one line>
```

---

- **Q:** Does switching forge's production `select_actions` from Lens(p_make) to ev-argmax raise the Zeb-Large win rate above the current 55.7% (E[Q] N=100)?
  - Raised: `afd4802` ([[w42-lens-v1-utility-head-to-head]])
  - Context: `forge/eq/generate/actions.py::select_actions` hardcodes p_make-argmax-with-EV-tiebreak — effectively Lens(p_make), the **worst** of the four utilities in the Lens v1 round-robin (Lens(ev) beats Lens(p_make) by +5.42 pts/hand). The recommended one-line switch to ev-argmax is **NOT applied** as of `afd4802` (verified against the current file plus the Zeb-protocol play path in `forge/zeb/eq_player.py`) — open two months on.

- **Q:** Does jud v2 — a bigger leaf on per-move targets (E[Q] distilled as a bootstrap value) plus opponents-in-rollout — close the play gap the v1 hand-level MLP could not?
  - Raised: `afd4802` ([[w42-jud-v1]], [[jud]])
  - Context: jud v1 held the unification at the auction but was mechanism-limited at play. `judsearch` recovered two-thirds of the play gap oracle-free (JS1 PASS +2.28) but neither more worlds (JS2 below band) nor a better-calibrated head (JS3 falsified) closed the rest. v1's diagnosis: the wall is per-move discrimination — a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's per-move oracle.
  - *Narrowed 2026-07-13 ([[jud-target-granularity]]):* per-move targets **at fixed v1 capacity** do not close it — parent-side dense aux and child-state value forms both graded marks-null, 3× corpus volume moved calibration but not ranking or marks, and ranking-label agreement was shown not to order play strength. The surviving question is the interaction: bigger leaf × per-move targets × on-policy loop data ([[w42-jud-v1|r4]]'s five-round corpus out-ranks fresh corpora), plus opponents-in-rollout — not per-move labels alone.

- **Q:** Which era-5 gestation designs (the IDEATED generation — Harl, LLem, walker, and the rest) are worth resurrecting, and which did the built LEM/Burl/Gus/jud line already subsume?
  - Raised: `afd4802` ([[the-gestation]], [[ideated-not-built]])
  - Context: The 52-day zero-commit gestation designed a whole generation out loud that mostly never shipped. The archaeology catalogued the names (classified IDEATED, never claimed as artifacts per anti-rot rule 4) but did not adjudicate which remain live options vs. which the built line already answered.

- **Q:** Does the `contradicted` verdict on high-bid pounce dissolve under an imperfect-information defender (belief/PIMC, no hand visibility), scoped to bid ≥ 35 and bidder ≤ 2 offs?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §4, [[w42-bookval-v1-wave2-pounce-high-bid]] caveat 0)
  - Context: The book's clause (p. 112, "regardless of whether you know who will win the trick") is an imperfect-information hedge; the probe's E[Q] arms evaluate under sampled-complete-worlds where the oracle always knows the trick winner — a regime in which "pounce regardless" is worse by construction. Prediction: the contradiction dissolves toward neutral-or-positive. Every other `contradicted`/`context-limited` row where the book hedges against *not knowing* deserves the same audit.

- **Q:** Do the book's bid→hand posteriors hold empirically (35 ⇒ two offs/one five-count; 31 ⇒ ≥1 double; shuffler-last-at-30/31 uninformative), and does [[jud]]'s head_8 respect the {30,31,35,36} bid lattice ("if you can bid 32, then you can bid 35")?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §1, proposed experiment 1)
  - Context: The auction decoder is the highest-value missed cluster — bids as messages, not risk budgets. Testable from bid-aware corpora as P(hand features | bid, seat, who-raised); the lattice is a dominance claim over the whole bid action space with one stated exception (last bidder raising a standing 31).

- **Q:** What does the missing book text say — pages 181–182 and 185–186 are absent from the OCR, the ch 16 four-trump table is truncated (37%/52% conditionals cut mid-row), and worked hands 1–14 are image-only diagrams with no prose reconstruction?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §Proposed experiments, item 9)
  - Context: Re-scan via the Kindle CDP pipeline (`scratch/winning42/`); the hand diagrams need prose reconstruction to be replayable as probes.

- **Q:** Can an explicit [[w42-book-second-pass|Winning 42]] convention overlay plus a learned fallback become a complete, calibrated blueprint whose partner-decoding gain remains positive after opponent decoding and full-match evaluation?
  - Raised: `1a4482fe` ([[convention-aware-blueprint-search]])
  - Context: Installing the same book overlay on sender and receiver supplies coordinated initialization before unilateral search. The proposed attribution is sender overlay x partner reader x opponent reader; double-dummy value isolates direct technique but does not classify away convention value. Latent opponent book-likeness extends the same design into four-seat action-derived inference. No contrary experiment has been run.

- **Q:** Can belief-weighted information-set MCTS extend JudSearch's demonstrated `+2.28` search gain far enough to beat `lens:ev`, with an additional marks gain attributable specifically to mid-tree belief updates or the shared convention blueprint?
  - Raised: `f6b691da` ([[belief-weighted-jud-mcts]])
  - Context: J0/J1/J2/J3/J4 separates current-trick JudSearch, root belief weighting, deeper determinized MCTS, information-set node sharing with mid-tree belief updates, and the book/learned blueprint. JS1 supports the search/leaf pairing; JS2 says more flat world samples are not the lever; Zeb and LAMIR did not test this combination.

- **Q:** Can tied-strategy rollouts — one action across belief-sampled worlds until the player's own observations differ — price the junk-retention economy (guards and walkers, the insurance/lottery value per-world E[Q] structurally zeroes via [[strategy-fusion]]), and does a [[count-fate-ledger]]-derived retention/discard policy beat the current best player in paired marks?
  - Raised: [issue #49](https://github.com/jasonyandell/mk5-main/issues/49) (conversation 2026-07-13→14; no raising sha — conversation-sourced, see [[count-fate-ledger]])
  - Context: All existing attribution is first-order `(domino, holder)` vs baseline; no tool ties one strategy across worlds, so guard/walker retention value has never been measured. Joint-world artifacts already retain `q_per_world` + `world_hands`; the missing mechanism is the tied-strategy evaluator, not new generation.
  - Narrowed 2026-07-15 ([[otis-v0]] P7): the cost half is ANSWERED — `otis/tiedroll.py` prices retention at ~1.0 s/decision (M=50, MPS), fusion gap +2.99 on the best replicable cell. The remaining half (does a ledger-derived retention policy beat the best player in paired marks) is [issue #53](https://github.com/jasonyandell/mk5-main/issues/53).
