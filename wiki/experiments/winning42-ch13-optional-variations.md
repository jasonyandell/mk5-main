---
title: Winning 42 Ch13 Optional Variations
kind: experiment
first_seen: local-2026-05-01
last_updated: afd4802
status: retired
---

**Phantom plan.** This one-session chapter harvest (2026-05-01) registered 7
measurement targets and got zero empirical follow-up in the two months
since — none of its detectors (`rule_variant_gate`,
`straight_auction_legality`, `first_lead_exception_bucket`,
`high_bid_scoring_invariant`, `variant_legal_set_diff`,
`preplay_information_leak`) were ever built, and none of its claim ids appear
in the central 64-row ledger ([[w42-phase4-final-claim-audit]]). The
project's actual frontier moved through the phase-4 closure and on to the
[[champion]]/[[jud]] ladder.

## Summary

This page is the bead-backed work surface for `t42-ni1l.13`: Chapter 13, "Optional
Variations." It harvests ruleset-gate and contamination-guard concepts from Winning
42 into measurable hypotheses for [[gus]], [[burl]], and [[forge]].

The chapter is less a strategy chapter than a ruleset hygiene chapter. Its main
measurement value is to keep straight [[texas-42]] corpora separated from optional
variants, then use the variants as adversarial legality and reward-function tests.
The source repeatedly frames Nel-O, Sevens, and Plunge as optional departures from
"pure" or "straight" 42, with tournament play prohibiting them in favor of
skill-preserving rules (Winning 42 Ch13, lines 6237-6314). That makes the chapter a
good home for `rule_variant` tags in [[winning42-strategy-measurement]] and for
variant-aware action features in [[gus-strategy-tags-probe]].

## Source Slice

- Source: `scratch/winning42/winning42.with_figures.md`, lines 6226-6825.
- Grounding pages: [[winning42-strategy-measurement]], [[gus-strategy-tags-probe]],
  [[gus]], [[burl]], [[forge]].
- Chapter focus: ruleset gates, variant contamination guards, optional-bid legality,
  high-bid scoring, first-lead exceptions, table-talk boundaries, and direct 84
  legality.

## Measurable Concepts

| Concept | Source claim | Detector/state inputs | Metric/test | Likely data source | Priority | Implementation notes | Readiness |
|---|---|---|---|---|---:|---|---|
| Straight-42 ruleset gate | The chapter distinguishes straight 42 from optional variants and says major tournaments prohibit Nel-O, Sevens, and Plunge (lines 6250-6299). | `ruleset_id`, allowed bid types, declaration mode, scoring mode, tournament flag. | Contamination rate: fraction of straight-42 corpus records containing variant bids, variant follow rules, or variant scoring. | Game generator configs, serialized game records, Burl trace prompts. | P0 | Add `rule_variant=straight42|nel_o|sevens|plunge|splash|folk_rule` to strategy tags; default generated corpora should assert `straight42`. | Enumeration-ready; Gus-ready; Burl-ready. |
| Variant bid exclusion in straight corpora | Nel-O, Sevens, and Plunge are "very optional" loopholes and not part of the straight strategy substrate (lines 6256-6263, 6288-6299). | Auction record, bid type, declaration field, bid amount. | Invalid-bid detector precision/recall on generated and hand-authored examples; zero-tolerance straight-corpus audit. | Forge generated games, synthetic auction fixtures. | P0 | Treat variant bids as separate regimes, not rare ordinary bids; hard fail if a straight-42 dataset contains them. | Enumeration-ready; Gus-ready. |
| Nel-O objective inversion | Nel-O bidder tries to take no tricks and must bid at least 42, unlike straight 42's trick-and-point objective (lines 6316-6327). | Bid type, bidder seat, trick winners, bidder trick count. | Nel-O make/set correctness: bidder makes only with zero tricks; compare to ordinary scoring confusion rate. | Synthetic Nel-O deals; optional-variant simulator if added. | P2 | Do not route through ordinary E[Q] until variant reward is explicit; otherwise labels invert. | Enumeration-ready after rules engine extension; oracle-needed. |
| Nel-O low-protection shape | A Nel-O hand can contain a high tile if it also has a low tile in that suit for protection (lines 6346-6366). | Bidder hand, pip/suit ranks, lowest tile per suit, opponent lead possibilities. | Protection recall: can the detector identify high tiles that are protected by lower same-suit exits? | Exhaustive hand enumeration under Nel-O rules. | P2 | Useful as a contrast against straight-42 `off_risk_and_protection`, where high/off protection has different value. | Enumeration-ready after variant rules; Gus-ready only if variant corpus exists. |
| Nel-O doubles-mode fork | Some Nel-O groups treat doubles as a separate suit, some let the bidder choose high/low doubles, and others treat doubles normally (lines 6363-6373). | Variant flags for doubles-as-suit, doubles-high-low, led double, legal follow set. | Legal-action divergence between doubles modes; detector flags positions whose legal set changes. | Synthetic legal-action fixtures. | P1 | This is a pure ruleset dimension; it should never leak into straight doubles-as-trump examples. | Enumeration-ready; Burl-ready. |
| Nel-O trade prohibition | The chapter rejects partner domino trading before Nel-O as unfair and outside 42 (lines 6406-6425). | Pre-play hand mutation, partner exchange event, hand-size invariant. | Corpus invariant: dealt hands remain fixed except for legal plays; zero exchange events. | Game logs, parser fixtures, any human-imported transcripts. | P0 | This is a data-integrity check, not a strategy feature. A hand exchange should invalidate the record for straight training. | Enumeration-ready; Gus-ready; Burl-ready. |
| Sevens distance regime | In Sevens, there are no trumps or suits; tile distance from sum seven determines play and winner relation (lines 6427-6480). | Bid type, tile pip sums, absolute distance from seven, tie handling. | Legal/winner correctness under Sevens; confusion rate with suit-follow legality. | Synthetic Sevens fixtures; optional simulator. | P2 | Requires a separate `rank_key=abs(sum-7)` rule; ordinary suit/trump tables are intentionally wrong here. | Enumeration-ready after variant rules; Burl-ready for legality prompts. |
| Sevens forced closest play | Every player must play the closest-to-seven tile left and cannot save one for later (lines 6460-6496). | Hand tiles, distance ranking, played tile. | Forced-play violation detector; late-save impossibility tests. | Exhaustive per-hand Sevens legal sets. | P1 | This is a good adversarial test for models that rationalize normal strategic holding in a forced-play variant. | Enumeration-ready after variant rules; Burl-ready. |
| Sevens set conditions | The bidder is set if an opponent plays closer to seven than bidder, or has key seven/six/eight concentration (lines 6476-6491). | Bidder tile distances by trick, opponent distances, tie flag, hand distance histogram. | Make/set classifier accuracy; pre-bid set-risk by distance histogram. | Exhaustive hand enumeration under Sevens. | P2 | Ties do not set the bidder; this is a crisp edge case. | Enumeration-ready after variant rules; oracle-needed. |
| Plunge/Splash eligibility | Plunge requires at least four doubles and an automatic 168 bid; Splash requires three doubles and three marks (lines 6512-6541). | Bid type, bidder hand doubles count, automatic bid amount, mark value. | Eligibility validator; invalid Plunge/Splash rejection rate. | Synthetic auction fixtures; human transcript import. | P1 | Keep as optional bid grammar. Straight bidding should not infer "four doubles means 168" unless variant enabled. | Enumeration-ready; Burl-ready. |
| Plunge partner-declares leak | Plunge lets the bidder reveal a strong helping hand before partner declares trump and leads, which the chapter compares to table talk (lines 6517-6578). | Bidder doubles count, partner hand, trump chosen by partner, first lead seat, pre-play information channel. | Information-leak magnitude: partner trump EV with/without knowing Plunge signal; table-talk leakage bucket. | Counterfactual oracle rollouts or exhaustive small fixtures. | P1 | This bridges Ch11 table-talk logic: variant-specific pre-play information must be explicit, not silently available in straight traces. | Oracle-needed; Burl-ready for trace audit. |
| Desperation 84 alternative | The chapter says a trailing team can legally bid 84 and raise rather than use Plunge (lines 6579-6592). | Score state, team deficit, opponent near-win, auction order, 84/raise sequence. | Score-pressure escalation detector; compare legal 84-raise EV/tail risk to Plunge-like reward. | Straight-42 generated games with score context; oracle rollouts. | P1 | This belongs in score-aware bidding buckets: legal desperation differs from variant communication. | Oracle-ready; Gus-ready; Burl-ready. |
| Forced fourth-seat 30 | Some groups force the fourth bidder to take 30 after three passes; the chapter frames it as a teaching variant (lines 6624-6650). | Auction position, first three passes, fourth-seat hand, forced-bid flag. | Forced-bid impact: make rate, regret, partner-help dependency, training curriculum value. | Synthetic auctions; generated games under forced-bid mode. | P2 | Useful as a curriculum bucket for marginal 30-point contracts, but should be flagged as variant. | Enumeration-ready; oracle-ready; Gus-ready. |
| 84 raise increment mode | Groups vary between 42-point and 84-point increments for raising 84; tournaments mostly use 42-point increments (lines 6652-6689). | Bid history, raise amount, match/game threshold, scoring mode. | Auction-legality divergence across increment modes; terminal Game-bid correctness. | Auction fixtures; tournament-mode configs. | P1 | The model must know which auction grammar applies before interpreting 126/168/Game. | Enumeration-ready; Burl-ready. |
| First-lead nontrump legality | The bidder is not required to lead trump first and may lead any domino, often to shed a dangerous off (lines 6691-6705). | Bidder first lead, trump declaration, tile trump flag, off-risk tags. | False-illegal rate for nontrump first leads; regret of off-first exception. | Straight generated games; oracle rollouts over first trick alternatives. | P0 | This is straight-42, not a variant. Add an eval bucket for legal but surprising first leads. | Enumeration-ready; oracle-ready; Gus-ready; Burl-ready. |
| Small-end lead disallowance | A nontrump first lead cannot choose either pip as the suit to follow; the chapter rejects "small-end lead" as a loophole (lines 6707-6725). | Led tile, declaration, led-suit resolver, attempted suit override. | Legal-set divergence; zero-tolerance detector for suit-override traces. | Engine legal-action fixtures; Burl trace parser. | P0 | This is a high-value contamination guard: legal first off lead, illegal arbitrary suit selection. | Enumeration-ready; Burl-ready. |
| Doubles-trump fallback-follow variant | Some groups require a player unable to follow a led double to play the double's pip suit; the chapter says decide up front (lines 6727-6756). | Doubles-as-trump flag, led double, hand has double?, hand has pip suit?, legal set. | Legal-action divergence by flag; bidding-shape impact when count can be pulled this way. | Synthetic legal fixtures; oracle rollouts if variant engine exists. | P1 | Separate from standard doubles-as-trump Ch9 buckets; use a distinct `doubles_fallback_follow` flag. | Enumeration-ready; oracle-needed. |
| High-bid scoring rule | Setting or making 42/84/higher earns the bid amount, not captured points plus bid (lines 6758-6769). | Bid amount, captured points, made/set, score update. | Score-update invariant for high bids; bug detector for bid-plus-points scoring. | Engine score fixtures; generated game logs. | P0 | This is straight scoring and should be covered by unit tests before model training labels are trusted. | Enumeration-ready; Gus-ready; Burl-ready. |
| Trump announcement as table talk | Players need not announce trumps when played, and others may not draw attention to them (lines 6771-6787, 6796-6798). | Trace text, tool observations, played trump, explicit verbal cue. | Trace leakage audit: model thought/tool text should not depend on illegal prompt hints. | Burl traces, prompt fixtures, Ch11 table-talk buckets. | P1 | Treat explicit trump callouts in human transcripts as contamination unless the prompt is a rules-teaching scenario. | Burl-ready; Gus-ready for corpus filter. |
| Direct 84 legality | A player may bid 84 directly without a prior 42 bid; bidding 42 is only score-risk management when one mark wins (lines 6802-6812). | Auction history, first bid amount, score-to-win, set penalty. | False-illegal rate for direct 84; score-aware 42-vs-84 choice regret. | Auction fixtures; score-aware oracle rollouts. | P0 | This is a clean bid-legality and score-context detector for both Gus and Burl. | Enumeration-ready; oracle-ready; Gus-ready; Burl-ready. |

## First Detectors

1. `rule_variant_gate`: validates that every record declares its ruleset and that
   straight corpora contain no Nel-O, Sevens, Plunge, Splash, trading, small-end,
   or fallback-follow events.
2. `straight_auction_legality`: checks direct 84 legality, high-bid raise increments,
   forced fourth-seat 30 if enabled, and variant bid exclusion if disabled.
3. `first_lead_exception_bucket`: separates legal nontrump first leads from illegal
   small-end suit selection, then measures oracle regret for off-first plays.
4. `high_bid_scoring_invariant`: verifies that 42/84/higher makes or sets score the
   bid amount only, not bid plus captured points.
5. `variant_legal_set_diff`: enumerates positions whose legal action set changes under
   Nel-O doubles modes, Sevens forced closest play, or doubles-trump fallback-follow.
6. `preplay_information_leak`: flags Plunge/Splash or trace text that gives partner-hand
   information before ordinary play evidence could reveal it.

## Readiness Notes

- Enumeration now: straight ruleset gates, auction grammar, direct 84 legality,
  first-lead legality, small-end disallowance, hand-size/no-trading invariants,
  high-bid scoring, and trace-text leakage checks.
- Oracle rollouts: off-first exception value, legal desperation 84/raise under score
  pressure, forced-bid teaching value, and variant bidding value once optional
  simulators exist.
- Gus analysis: `rule_variant`, `direct_84_legal`, `first_lead_nontrump_legal`,
  `small_end_illegal`, `high_bid_score_mode`, and `forced_bid_mode` are cheap
  public-state/action tags. Variant modes should be held out of the main straight
  policy unless explicitly training a variant-aware model.
- Burl traces: best immediate targets are false-illegal direct 84, false-illegal
  nontrump first lead, invented small-end permissions, illegal trump callouts, and
  using Plunge-like information in straight play.

## Claim Ledger

No empirical run was performed for this chapter harvest. Every row below is a
measurement target rather than a result.

| Claim | Status | Next check |
|---|---|---|
| Straight corpora should exclude Nel-O, Sevens, Plunge, Splash, trading, small-end, and fallback-follow variants. | context-limited | Audit generator configs and serialized games for `rule_variant` coverage. |
| Direct 84 is legal without a prior 42 bid. | untested | Add auction legality fixture and Burl prompt-trace bucket. |
| Nontrump first lead by the bidder is legal. | untested | Add engine legality fixture and oracle bucket for off-first exceptions. |
| Small-end suit selection on a nontrump first lead is illegal. | untested | Add legal-set fixture that rejects suit overrides. |
| High bids score only the bid amount when made or set. | untested | Add scoring invariant tests for 42/84/126/168/Game. |
| Sevens and Nel-O strategy labels should not be mixed with straight strategy labels. | context-limited | Keep variant simulators and corpora separate until reward/legality are explicit. |
| Plunge acts as pre-play partner-hand information. | untested | Compare partner-declaration EV with and without the Plunge signal. |

## Links

[[winning42-strategy-measurement]] | [[gus-strategy-tags-probe]] | [[gus]] |
[[burl]] | [[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cheapest revival probe: the two P0 straight-42 items (`high_bid_scoring_invariant`, direct-84 legality) are pure engine unit tests requiring no variant machinery, landable as fixtures independent of any chapter-harvest revival.
- Source `scratch/winning42/winning42.with_figures.md` is gitignored, so line citations trace only through the main checkout, not worktrees.
