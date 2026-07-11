---
title: Winning 42 Ch06 Concentration Style
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.6`: Chapter 6, "Concentration
and Style." It harvests attention, inference, and style concepts from Winning 42 into
measurable hypotheses for [[gus]], [[burl]], and [[forge]].

The chapter treats concentration as public-state inference. Every play reveals something:
failure to follow establishes a void, bids and failed bids constrain likely hand shape,
forced count donations reveal partner weakness or voids, trump must be remembered even
when the non-trump pip looks visually tempting, and late-hand suit exhaustion can promote
a low-looking domino into an effective double. The style section adds a population layer:
aggressive, cautious, and match-score-conditioned players should be modeled as
predictable deviations from percentage play.

Sources: `scratch/winning42/winning42.with_figures.md` lines 2841-3061, especially
the concentration frame at 2852-2882, bid-derived inference at 2893-2910, partner count
donation inference at 2912-2923, renege/trump-memory warnings at 2925-2942, low-trump
and promoted-tile traps at 2953-2988, unsafe count donation after trump-in at 2990-3011,
and style/score framing at 3023-3058. The source slice ends at the start of an unfinished
style example before Chapter 7 begins, so style claims here stay deliberately
context-limited.

## Grounding

This chapter plugs into [[winning42-strategy-measurement]] as the attention and
inference chapter. It overlaps the existing analysis catalog entries for `void_evidence`,
`key_tile_owner_belief`, `partner_donation_window`, `setter_pounce_window`, live count,
effective walkers, style-conditioned bidding, and trap-family learning curves.

For [[gus]], the chapter is mainly a belief-calibration and concept-bucket problem:
does the public sequence let the model update hidden ownership, recognize promoted
tiles, and avoid high-regret count donations? For [[burl]], it is a trace-faithfulness
problem: do tool calls and thoughts cite the public evidence that justifies the play,
or does the agent merely commit a plausible domino? For [[forge]], it is an oracle
bucket problem: how much regret or set/make swing is attached to each attention failure?
The promoted [[gus-strategy-tags-probe]] result says cheap public-state/action tags
already improve a small policy student, but this chapter asks for finer concept buckets
rather than aggregate regret.

## Measurable Concepts

| Concept | Detector / state inputs | Metric / test | Likely data source | Priority | Implementation notes | Readiness |
|---|---|---|---|---:|---|---|
| Trick-by-trick concentration | Full public play history, current trick, legal actions, trick winner, count already captured | Full-history policy vs last-N-tricks ablation; regret and illegal/renege exposure deltas | Generated games plus oracle labels; Burl traces for thought-memory checks | P1 | Build an eval slice where the correct action depends on a reveal older than the current trick | Oracle-ready; Gus-ready; Burl-ready |
| Void inference from failure to follow | Led suit, played domino, legal-follow mask, unplayed dominoes in that suit, player seat | Hidden-owner Brier/log loss before and after void event; entropy reduction | Gus belief corpus with true hands; Forge generated state records | P1 | Deterministic void facts should become explicit tags and calibration buckets | Enumeration-ready for legality; Gus-ready |
| Bid-derived hand inference | Bid amount, bidder seat/order, final bid, declaration, hand shape labels, offs/count/doubles | Mutual information between bid bucket and hidden hand features; belief calibration by bid bucket | Bidding corpus, generated deals, Gus belief labels | P1 | Chapter claims 35 often signals two offs with a five-count; 30/31 last-seat bids need separate bucket | Enumeration-ready; Gus-ready |
| Failed-bid signal | Losing bids, bidder/pass order, partner/opponent identities, hidden doubles/off strength | Does failed-bid evidence improve partner/opponent hidden-shape prediction and later bid/play regret? | Auction logs from generated games; future Burl prompt traces | P1 | Failed 31 is a public signal, not dead auction noise; keep partner/opponent use cases separate | Enumeration-ready; Gus-ready; Burl-ready |
| Partner count donation implies weakness/void | Bidder led double, partner forced to play count in same suit, partner hand truth, remaining suit tiles | Precision/recall that donation was forced; belief update on partner void/weakness | Generated trick records with true hands; Gus belief labels | P1 | This is a clean public-evidence detector and a strong Burl trace target: the rationale should cite the forced donation | Enumeration-ready; Gus-ready; Burl-ready |
| Renege / retrospective follow-suit detection | Prior void claims, later same-suit play, trump declaration, led-suit resolution under current rules | Illegal sequence detection rate; reconstruction accuracy after later trick | Engine logs, rule-checker tests, variant-gated generated games | P2 | Separate genuine renege from trump/pip confusion; useful as data QA and table-talk boundary | Enumeration-ready; Burl-ready |
| Trump-suit retention / low-trump disguise | Trump suit, led domino with trump pip lower than other pip, legal follow choices, count exposed | Trap regret: count donated or wrong suit followed after disguised trump lead | Oracle eval buckets; Burl forced-commit traces; hand-authored trap states | P1 | Example: treys trump, bidder leads 3-5, inattentive defender treats it as fives and donates double-five | Oracle-ready; Gus-ready; Burl-ready |
| Late-hand promoted low tile | Suit exhaustion, remaining higher tiles, played tiles, current leader, pounce candidate count | Recognition precision/recall for "low-looking but boss" tiles; opponent pounce-regret rate | Generated late-trick positions; oracle action deltas | P1 | Same substrate as `effective_double_or_walker`; tag should expose highest-live-in-suit and higher-live-count | Enumeration-ready; Oracle-ready; Gus-ready |
| Unsafe count donation after opponent trump-in | Current trick winner after each play, partner/opponent relation, legal same-suit alternatives, count value | Catastrophic count-dump rate and regret when partner appears to win but opponent trump already wins | Oracle eval buckets; generated game traces; Burl thought traces | P1 | This is action-local: the bad play can be legal and suit-following but strategically blind to current trick winner | Oracle-ready; Gus-ready; Burl-ready |
| Mistake recovery by trap family | Repeated trap labels, model/player identity, prior failures, later decisions in same family | Learning-curve slope; repeated-blunder rate by trap type | Multi-run Burl evaluations; curriculum probes; future human-style sims | P3 | Chapter frames mistakes as practice-driven disappearances; this becomes curriculum effectiveness, not a single-state detector | Gus-ready after tags; Burl-ready |
| Aggressive/cautious/wild bidding style | Repeated player id, hand-strength estimate, bid amount, pass/raise thresholds, make/set outcomes | Style residual after hand strength; exploitability of overbid restraint vs raise | Simulated populations; future self-play player profiles; tournament logs if added | P2 | Needs player identity across hands; do not leak hidden hand features into live policy tags | Oracle-ready for synthetic profiles; Burl-ready later |
| Score-conditioned bidding style | Match score, deficit/lead, candidate hand strength, bid/pass/raise decision, terminal objective | Bid aggression shift by score bucket; regret under point-EV vs match-EV | Generated matches with score context; marks/points variants from later chapters | P2 | Source says match status can justify caution with a lead or aggression from a deficit; full test needs match-level reward | Oracle-ready; Burl-ready |

## Highest-Value First Detectors

1. `low_trump_disguise_trap`: finds leads where the trump pip is visually lower than the
   other pip and the defender has a tempting count/follow mistake. This is a compact
   hand-authored plus generated eval bucket for [[burl]] and action-local [[gus]] tags.
2. `unsafe_count_after_trump_in`: checks whether a candidate play donates count after an
   opponent has already become current trick winner by trumping. This should be cheap,
   high-signal, and close to the existing action-feature substrate.
3. `promoted_low_tile`: tags low-looking tiles that are currently highest live in their
   suit. This connects Chapter 6 attention to the `walker_candidates` and
   `effective_double_or_walker` family.
4. `failed_bid_shape_signal`: measures how much a failed bid changes beliefs about doubles,
   offs, and partner-help potential. This is a direct bridge from public auction history
   into Gus belief calibration.
5. `partner_forced_count_void_signal`: detects count donations that imply the partner had
   no safe choice. This is a good Burl rationale test because the explanation should name
   the public forcedness, not hidden knowledge.
6. `score_style_bid_shift`: buckets bid/pass/raise behavior by lead/deficit and player
   style residual. This is lower-level than tournament ecology but captures the first
   style-conditioned decision surface.

## Readiness Notes

Enumeration can check the deterministic state facts immediately: legal follow, void
events, prior void contradictions, current trick winner, trump-vs-native suit identity,
highest-live-in-suit, count value, and whether a count donation was forced.

Oracle rollout is needed for the value of attention errors: the regret of falling for
low-trump disguise, pouncing on a promoted low tile, donating count after an opponent
trump-in, or raising an aggressive bidder instead of trying to set him.

Gus analysis is ready once the chapter tags are emitted over generated games. The first
report should include belief Brier/log loss around void events, failed bids, partner
forced donations, and promoted-tile states, plus policy regret by the four P1 trap
families.

Burl analysis needs traces. The useful check is not just whether Burl chooses the oracle
move, but whether its tool calls and rationale cite the relevant public evidence:
trump suit, current trick winner, previous failure to follow, failed-bid evidence, or
highest-live-in-suit.

## Claim Ledger

No empirical run was performed for this bead. Every claim below is therefore a
source-backed hypothesis with a `context-limited` ledger status until the detector
and eval bucket are run.

| Claim | Source backing | Proposed empirical check | Ledger status |
|---|---|---|---|
| Failure to follow gives public void information that should guide later play | Lines 2876-2882 | Void-event belief calibration and regret before/after update | context-limited |
| Bid size and auction position reveal likely hand shape; 35 often implies two offs with one five-count suit | Lines 2893-2899 | MI and calibration between bid buckets and hidden hand features | context-limited |
| Failed bids remain useful public signals for partner and opponent decisions | Lines 2901-2910 | Belief/policy delta when failed-bid history is included vs ablated | context-limited |
| A forced partner count donation after bidder leads a double can reveal partner void/weakness | Lines 2912-2923 | Forcedness detector precision/recall and partner-hand belief shift | context-limited |
| Remembering trump prevents visually tempting low-trump disguise mistakes | Lines 2939-2978 | Trap bucket regret and Burl trace faithfulness to trump declaration | context-limited |
| A low-looking late-hand tile can become as strong as a double after higher tiles are gone | Lines 2980-2988 | Highest-live-in-suit detector and pounce-regret bucket | context-limited |
| Count should not be donated when an opponent has already trumped and is winning the trick | Lines 2990-3011 | Action-local count-dump regret under current-winner tracking | context-limited |
| Player style and match score can justify deviations from percentage play | Lines 3023-3058 | Style residual and score-conditioned bid/pass/raise analysis | context-limited |

## Links

[[winning42-strategy-measurement]] / [[gus-strategy-tags-probe]] / [[gus]] / [[burl]] / [[forge]] / [[regret-eval]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cited source line ranges verified against `scratch/winning42/winning42.with_figures.md`; the file is gitignored scratch, so citations do not survive a fresh clone — promoting the text slice or a digest into wiki/sources/ would fix that if the chapter series stays load-bearing.
- `unsafe_count_after_trump_in` is action-local and cheap over the existing action-feature substrate — a good first detector to actually run, converting one claim-ledger row from context-limited to measured.
- The bead reference `t42-ni1l.6` is historical (beads retired 2026-06; tracker moved to GitHub issues).
