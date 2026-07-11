---
title: Winning 42 Ch01 In A Nutshell
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.1`: Chapter 1, "In a Nutshell."
It harvests foundational rule and state-accounting concepts from Winning 42 into
measurable hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 1 is mostly not "strategy" yet. Its value for the strategy-measurement workstream
is that it defines the invariant substrate that later strategy claims must sit on: what
counts as a suit, how trump removes a tile from its secondary suit, how count is captured,
how the bidder makes or fails a contract, how lead control moves, and when a lower tile
becomes a walker. The chapter should therefore produce deterministic detectors first,
then sanity buckets for [[gus]] policy/regret and [[burl]] tool traces.

Source: `scratch/winning42/winning42.with_figures.md` lines 479-734. The source slice covers
the Chapter 1 rule/term summary from "In a Nutshell" through the first scoring examples.

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 479-734.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

## Grounding

- [[winning42-strategy-measurement]] frames the book as a hypothesis generator, not an
  authority to hard-code.
- [[gus-strategy-tags-probe]] shows cheap public-state and action-local strategy tags
  already improve a tiny Gus-like policy, making Chapter 1's rule tags useful feature
  substrate.
- [[gus]] should use these as concept buckets for policy/regret and belief sanity checks.
- [[burl]] should use these as trace-faithfulness and tool-use buckets: the model should
  ask or reason about the same state facts the detector says are load-bearing.
- [[forge]] provides the rule engine, E[Q] framework, and oracle-generated corpora that can
  validate deterministic accounting before any model-facing analysis.

## Measurable Concepts

| Concept | Source-backed claim | Detector/state inputs | Metric/test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---|---:|---|---|
| Hand point total | A hand has seven tricks worth 1 each plus 35 count points, totaling 42. | Completed trick records, captured count tiles, trick ownership by team. | Assert every completed hand totals 42 points; assert each trick contributes exactly 1 point; assert count sum is 35 over the full domino set. | Exhaustive domino constants; forge generated games. | P0 | Enumeration, Oracle, Gus, Burl | This is the root accounting invariant for all later bid/regret buckets. |
| Count identity and value | Count dominoes are 5-5, 6-4, 4-1, 3-2, and 5-0; their values are face sums. | Domino id -> pip pair; count lookup table. | Unit test exact count set and values; compare captured-count totals from game logs against detector. | Forge constants and game traces. | P0 | Enumeration, Oracle, Gus, Burl | Emit action-local features: `is_count`, `count_value`, `count_at_risk`, `count_donated`. |
| Suit membership | There are seven suits; non-doubles normally belong to two suits and doubles to one. | Domino pips, declared trump, led tile. | Unit test membership for all 28 tiles under non-trump context. | Enumeration over domino set. | P0 | Enumeration, Oracle, Burl | Later chapters depend on off-risk by both pips of a tile. |
| Trump exclusivity | When a suit is trump, all tiles in that suit are trump and do not also belong to their secondary suit for that hand. | Declaration, tile pips, led suit, candidate play. | Exhaustive legality/winner table under each trump suit; compare to forge trick logic. | Forge/oracle trick tables; generated decisions. | P0 | Enumeration, Oracle, Gus, Burl | This is the most important Chapter 1 legality trap for feature generation. |
| Follow-suit obligation | Players must follow the suit of the first tile in a trick unless void in that suit. | Led tile, declared trump, hand mask, legal action mask, play history. | Legal-mask exact match: all and only follow-suit tiles are legal when present. | Forge legal-action masks; generated games. | P0 | Enumeration, Oracle, Gus, Burl | Burl trace bucket: does the model mention follow-suit or call `is_legal` when constrained? |
| Trick winner | A trick is won by highest tile in led suit unless one or more trumps are played; highest trump wins. | Four played tiles in order, leader seat, declaration, led suit. | Exhaustive four-tile trick winner agreement with forge; adversarial cases with trump secondary-suit ambiguity. | Enumeration sampled from legal trick states; forge traces. | P0 | Enumeration, Oracle, Gus, Burl | Needed for `partner_currently_winning`, donation windows, and pounce buckets. |
| Lead control | The winner of each trick leads the next trick; winning bidder leads the first trick. | Bid winner, trick winner sequence, next leader. | Assert state transition leader[t+1] == winner[t]; first leader == bidder. | Generated game records. | P0 | Enumeration, Oracle, Gus, Burl | Useful Burl trace check: does it know who can shape the next trick? |
| Bid make/set threshold | Bidder's team makes by reaching or exceeding bid; otherwise bidder scores 0 and opponents score failed bid plus their captured points. | Bid amount, team captured points, opponent captured points. | Score formula tests from chapter examples: bid 30/take 33 -> 33 vs 9; bid 35/take 33 -> 0 vs 44; bid 35/take 30 -> 0 vs 47. | Unit tests; generated hand summaries. | P0 | Enumeration, Oracle, Burl | Separate hand points from match scoring. Later tournament scoring may alter objectives. |
| Bid floor and auction order | Minimum bid is 30; each player has one chance; if all pass the hand is redealt; shuffler bids last. | Auction state, dealer/shuffler, bids/passes. | Auction legality tests and corpus audit of bid/pass ordering. | Bidding module traces if available; simulated deals. | P1 | Enumeration, Oracle | Chapter 1 gives protocol, while Chapter 2 provides strategy. |
| Count capture by trick winner | Count tiles score for the team that wins the trick containing them. | Trick winner, team mapping, tiles in trick. | Per-trick captured-count attribution test; compare team totals after each trick. | Forge generated games. | P0 | Enumeration, Oracle, Gus, Burl | Feeds live count, count liability, and safe donation labels. |
| Off tile definition | An off in the bidder's hand is not trump and not a double, and is often the opponent's route to setting the bidder. | Bidder hand, declaration, double flag, trump flag. | Label bidder offs and bucket later outcomes by off count/shape and set rate. | Generated deals plus bid/declaration. | P1 | Enumeration, Oracle, Gus | Chapter 1 only defines the term; later chapters add strategic claims. |
| Walker | A mid- or low-level tile led while highest still unplayed in its suit is a walker and functions like a double when led. | Led tile, suit, played tiles, player hand visibility if available, remaining higher tiles, trump state. | Detector precision by exhaustive remaining-suit state; outcome/regret bucket for walker recognized vs missed. | Generated trick states; oracle rollouts; Gus decisions. | P1 | Enumeration, Oracle, Gus, Burl | First strategic detector from Chapter 1. Needs "when led" and "highest still unplayed" guards. |
| Visible trick piles | Won tricks remain intact and visible; one teammate collects team tricks. | Public trick history, captured tiles by team. | Corpus invariant: public history should reconstruct scoring without hidden memory. | Generated game records; Burl prompt/tool state. | P2 | Enumeration, Burl | This is a UI/state-representation requirement more than a policy feature. |
| Game endpoint | First team to 250 wins; if both reach 250 on same hand, the bidding team wins. | Match score before/after hand, bidder team, hand score. | Match scoring unit tests including simultaneous reach case. | Match simulator. | P2 | Enumeration, Burl | Chapter 10 will supersede with tournament scoring variants. |

## Highest-Value First Detectors

1. `rule_accounting_core`: count identity/value, seven trick points, hand total equals 42.
2. `legal_follow_suit_with_trump_exclusivity`: legal action mask under led suit and trump.
3. `trick_winner_core`: led-suit vs trump winner resolution for all trick shapes.
4. `contract_make_set_scoring`: bidder make/set threshold and failed-bid score formula.
5. `lead_control_transition`: bidder opens; trick winner leads next trick.
6. `effective_walker`: led tile is highest still-live tile in its non-trump suit.

These should become the Chapter 1 smoke-test suite before adding softer strategy tags. The
first five are deterministic invariants; the walker detector is the first place where
"rule vocabulary" turns into a strategy bucket.

## Analysis Routes

### Enumeration-ready

- Count table, suit membership, trump exclusivity, trick winner, follow-suit legal masks,
  lead-control transitions, contract scoring examples, and walker identity can be checked
  by deterministic enumeration or table-driven unit tests.

### Oracle-ready

- Off count/shape, walker recognition, follow-suit constrained choices, and count capture
  can be bucketed over E[Q] decisions once the deterministic detectors agree with [[forge]].
  The useful metrics are paired regret, tail regret, bot-match, near-tie rate, set rate,
  and count donated/captured.

### Gus-ready

- Add Chapter 1 concept buckets to [[gus-strategy-tags-probe]] reporting: constrained
  follow-suit decisions, trump-in decisions, count-donation actions, current-winner relation,
  bidder off labels, and walker leads. These are public-state or action-local features and
  should not require hidden oracle information except for evaluation labels.

### Burl-ready

- Use Chapter 1 as trace-faithfulness scaffolding. For a constrained play, [[burl]] should
  either call `is_legal`/`is_trump` or reason correctly from the visible state. For a scoring
  or contract judgment, it should distinguish hand points, bid threshold, and match score.
  Failure buckets: illegal follow-suit rationale, wrong trick winner, treating trump as also
  its secondary suit, missing current leader, or wrong make/set arithmetic.

## Claim Ledger

[[w42-phase4-laydown-rule-accounting]] now covers the deterministic rule
substrate for this chapter with 23 rule assertions and zero failures. Remaining
model-facing rows still need Gus/Burl corpus audits, but the core accounting
claims have executable Forge-backed fixtures.

| Claim | Status | Why |
|---|---|---|
| Chapter 1 deterministic rule accounting can be fully verified without oracle rollout. | supported | `t42-br7n.4` verifies count identity, hand total, suit membership, trump exclusivity, follow-suit masks, trick winner, lead control, count capture, and contract scoring. |
| Trump exclusivity is a high-value model sanity bucket. | supported-as-rule, model-untested | The deterministic fixture passes against Forge; no Gus/Burl bucket report was run. |
| Walker recognition deserves an early concept bucket. | Underpowered | The book defines it as rule vocabulary with strategic value; no oracle regret comparison was run. |
| Bid make/set scoring should be split from hand point accounting and match scoring. | supported-as-rule | `t42-br7n.4` verifies the Chapter 1 scoring examples. Chapter 10 owns mark-vs-point objective drift. |
| Burl traces can be audited for Chapter 1 rule faithfulness before deeper strategy. | Underpowered | The tool surface supports legality/trump/state checks, but no trace corpus was inspected here. |

## Next Implementation Notes

- Start with a tiny `strategy_tags` Chapter 1 report over generated games: decision count by
  follow-suit constrained/unconstrained, trump-available, partner currently winning, count
  in legal actions, and walker candidate.
- Add table-driven tests before modeling metrics. If these detectors disagree with forge,
  every downstream strategy bucket is suspect.
- Treat Chapter 1 concepts as low-level tags, not final strategy. Later chapters decide
  whether to spend trump, donate count, lead off, or exploit walkers.

## Links

[[winning42-strategy-measurement]] · [[gus-strategy-tags-probe]] · [[gus]] · [[burl]] · [[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Source slice verified against `scratch/winning42/winning42.with_figures.md` lines 479-734; the file lives in gitignored `scratch/` in the main checkout only, so the slice is not reproducible from the repo.
- Beads are retired (bd → GitHub issues); `t42-ni1l.1` and `t42-br7n.4` are historical bead identifiers — a cheap pass could annotate them as archived-bead IDs.
- The "Underpowered / model-untested" ledger rows (walker regret bucket, Burl trace audit) remain the obvious next probes.
