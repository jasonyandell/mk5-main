---
title: W42 Book Second Pass — what the first extraction missed
kind: experiment
first_seen: 2026-07-07
last_updated: 2026-07-11
status: active
---

## Summary

A second close reading of Winning 42 (2026-07-07, four parallel readers over the
full OCR text in `scratch/winning42/text/`), run with the finished
[[w42-book-validation-campaign]] as the lens: knowing what the 64-claim ledger
tested and what the campaign's own retrospective flagged (the single-decision
blind spot, [[w42-book-claim-synthesis-and-ai-directions]]), what is in the text
that the first extraction flattened or never saw?

Raw reader reports (quotes, page numbers, per-finding experiment sketches) are
preserved at `wiki/sources/book-second-pass-2026-07-07/`: [[reader-A-report]]
[[reader-B-report]] [[reader-C-report]] [[reader-D-report]].

Headline: the first pass extracted the book's **tactics** and missed the book's
**information theory**. The auction decoder, the action-choice inference catalog,
the signaling conventions, and the reputation mechanism are all in the text and
none became ledger rows — and they map one-to-one onto the gaps the campaign
itself named as untested (auction policy, opponent response, partner-bid
intelligence, population effects). One existing `contradicted` verdict is
probably a category error.

## 1. The auction decoder lives in the style chapter (missed entirely)

The Bidding chapter (ch 2) is the *encoder* — bid what your own hand risks. The
extraction harvested it fully. The *decoder* — what a bid tells everyone else —
is stated in ch 6 ("Concentration and Style," pp. 53–56) and ch 12, where no
bidding-claim extractor looked:

- **Bid → hand shape posterior** (p. 54): "A 35 bid usually means the bidder has
  two offs, one of them from a suit with a five-count." A 31 bid ⇒ ≥1 double —
  *even a failed 31 bid* leaks this to both sides. Exception clause: shuffler
  last-to-bid at 30/31 is informationless.
- **Partner's bid raises your own ceiling** (pp. 14, 20): partner entering the
  auction is explicit Bayesian license to bid a step higher.
- **Auction-dynamics conventions** (p. 22): raises over 31 cluster at 35;
  interior bids are wasteful.
- **The bid lattice** (pp. 118–119): "there is no reasoned mathematical basis for
  a 32 or 33 bid… The most common, reasoned bids in 42 are 30, 31, 35, and 36…
  **If you can bid 32, then you can bid 35.**" A dominance claim over the whole
  bid action space, with one stated exception (last bidder raising a standing 31).
- **Who-bid asymmetry** (pp. 97–98): the same hand does not raise partner's 30
  but may raise an opponent's 30 (score-conditioned).
- **Match-score-conditioned bidding** (pp. 92, 129): risk 84 when "way behind…
  120 points or three marks"; and the desperation convention — at opponents'
  match point, first team bidder bids 84 and partner raises *regardless of
  hands*.

Why it was missed: the material lives outside the bidding chapter, and the
first pass modeled bids as risk budgets, not as messages. This is the highest-
value cluster given [[jud]]'s auction-first frontier — the campaign's own
synthesis ([[w42-book-claim-synthesis-and-ai-directions]]) names real auction
policy, opponent response, and partner-bid intelligence as untested.

## 2. The action-choice inference catalog (belief updates from choices, not voids)

The first pass extracted void inference. The worked hands (ch 12, hands 28–31;
ch 6) contain a richer class: inference from *what a player chose among legal
plays* —

- count donated onto a losing trick ⇒ the donor's played trump was his **only**
  trump (p. 106);
- *which* count partner donated ⇒ what partner **lacks** ("if your partner had
  [the double-five], he would have given it to you instead of the blank-five,"
  p. 106);
- the bidder's off-lead read symmetrically by partner *and* opponents ("he has
  the six-four, otherwise he would never have led the six-five," pp. 107–108);
- count-remaining arithmetic bounding a hidden holding (p. 107);
- count-on-opponent's-double ⇒ donor void in that suit (p. 53).

Each is a testable calibration assertion for the [[gus]] belief head: does the
prescribed posterior match the oracle's true-holding distribution at that node?
And each is a human-teachable "reason" — the [[belief-trajectory]] /
legibility surface.

## 3. Signaling conventions (the concealment/signaling material the harness is blind to)

- **The top-unplayed-trump lead is a protocol, not a pull** (p. 25): only the
  provably-winning trump lead makes "I win this trick" common knowledge, which
  is what *unlocks* partner's count donation; any lower trump forces partner to
  withhold. The first pass held "trump pull" and "donation windows" as separate
  claims and lost the coupling. The same passage prices the lead's
  information-gain leg ("you have nonetheless learned that all the count not in
  your hand is in your opponents' hands").
- **Donate-highest convention** (pp. 101–106): donating the highest useful count
  is a two-way code — the *absence* of a donation is decoded.
- **Keep-priority tier 5** (p. 72): the 84-defense keep algorithm's last tier is
  literally "low dominoes that will help your partner decide what doubles and
  domino pairs to keep" — discarding as signaling.
- **Dump-to-inform** (p. 118, hand 35): "opponent 1 dumped the ace-blank early
  so that opponent 2 would know not to save the double-ace… expert team playing."
- **Plunge/Splash as a legal one-bit signaling contract** (pp. 128–129):
  Roberson's own framing — the bid *is* the message ("≥4 doubles; you pick
  trumps"), and he condemns it precisely because it transmits information. The
  engine already supports these contracts: a bounded, ready-made signaling
  testbed.
- **Deception**: bait-a-trump (p. 107), suit-confusion trump lead (p. 55),
  walker-bait (p. 55), keep-low-fours-so-the-double-holder-saves-it
  disinformation (p. 114), deliberate double-void for a future trump-in
  (p. 108).

None of this can register in oracle-vs-oracle PIMC play — the harness
blindness Fable named ([[candlewax]] era, [[pimc]]). It is exactly the jud **v2 opponents-in-rollout** material: the
referee gap (oracle EV − V_realized EV) is where these conventions would show
up as they emerge.

## 4. A ledger verdict is probably a category error: high-bid pounce

`ch12-setter-pounce-high-bid-off` was demoted to `contradicted` under all
utilities (Wave 2.E.2 / Wave 3.0). The second pass re-read the source
(pp. 112–113): "pouncing when you can, **regardless of whether you know who
will win the trick**… Play it or lose it! This is even more critical against
high bids of 35 or 36."

The operative clause is an **imperfect-information hedge**: dump count onto a
live trick because you cannot know a better placement will come. A
perfect-information oracle always knows who wins the trick, so it only plays
count when it lands — under that regime "pounce regardless" is strictly worse
by construction. The probe tested the right words in the wrong information
regime. Proposed re-adjudication: re-run with a belief/PIMC defender that does
not see hands, scoped to bid ≥ 35 and bidder ≤ 2 offs. Prediction: the
contradiction dissolves toward neutral-or-positive. Until then the
`contradicted` row should carry an information-regime caveat — and every other
verdict where the book's advice hedges against *not knowing* deserves the same
audit.

## 5. Flattened conditionals (the "unless Z" clauses that got dropped)

- **Off-first envelope** (p. 24): leading offs before trumps is defensible
  **iff exactly 1 off ∧ ≥4 trumps**; outside that box it is condemned.
- **At-risk algebra** (pp. 11–12): per-off risk = max(sides), dedup shared
  count across offs, and **leading an off collapses its risk to the high side**
  — the bid literally prices the intended play plan. Bidding and play are
  entangled in the book's own arithmetic; single-decision extraction cannot
  represent that.
- **One-off inversion** (pp. 14, 21): with one off, assume opponents hold the
  setting tile; with two+ offs, partner-help probability rises and per-point
  aggression is licensed — off-count switches *which distributional assumption
  you make*, not just the total.
- **Count-commit switch by role × order** (pp. 45–46): setter playing before
  partner must commit count under uncertainty; helper commits only under
  certainty. Same action, opposite trigger, stated as opposites in the text.
- **Setter-seat asymmetry is in the book** (p. 51): left-of-bidder must
  trump-in speculatively (threshold scaling with the rank of the double led);
  last-to-play withholds. The campaign's empirical setter asymmetry finding
  assumed the book was seat-symmetric — it is not.

## 6. Multi-step plans, now with the book's own numbers

- **Strip-the-protector** (p. 92 vs p. 94): lead a low double (calls no count,
  locally worthless) to strip an opponent's guard so a later high double
  forces the count out — and the book shows the *same tile* flipping from
  load-bearing to "worthless" purely on opponent layout. The cleanest
  fully-specified discriminator between one-step E[Q] and plan-aware play in
  the whole text.
- **Double-ahead-of-off carries published probabilities** (p. 26): 53% forced
  on the double, ~60% with the partner leg, 33% if you lead the off first — a
  two-trick plan with author-supplied win rates to validate the oracle against.
- **Crisis management** (pp. 93–94): a named plan *branch* — trigger (opponent
  holds the outstanding trump, you're out), re-plan (abandon trump drawing,
  force count while he must follow), with tile-keyed ordering.
- **Two-defender 84 set** (hands 34–35, pp. 114–118): dump-to-inform +
  withhold-to-protect-a-pair + deliberate disinformation sets "about as good an
  84 hand as you could draw." Every move is EV-neutral in isolation.
- **The domino-pairs doctrine** (pp. 71–73, 114–115): a 5-tier keep algorithm
  with a hard override ("never forsake a double for domino pairs") and a
  release rule keyed to partner's observed discard. The first pass stored the
  categories, not the priority ordering.
- **84 set-condition booleans** (p. 116): "only two ways for the bidder to be
  set…" — an exact, enumerable predicate for laydown-adjacent checking.

## 7. Population, reputation, and the market gap

- **Reputation-induced overbidding** (p. 121): a tight-bidding reputation
  *scared an opponent into a wild bid* — second-order inference over a
  cross-game prior, structurally invisible to fixed-opponent evaluation.
  "I have beaten teams 7-1 and 7-2 without ever playing a bid" — setter
  primacy stated flatly.
- **Roberson on 42 apps** (pp. 160–161): existing apps "are not programmed to
  be good setters… ignore almost every skill and tactic I teach in chapter 5."
  Independent testimony that defense is the market-wide weakness — and the
  campaign independently found setter seats are where detectors over-fire,
  advice diverges, and attribution is asymmetric. The two agree: **setter
  defense is the differentiator.**
- **Tournament facts** (pp. 164–166): first tiebreaker is **total marks**
  (supports the margin-based [[jud]] objective); the laydown rule is
  adversarially provable ("any possible way the bidder can be set ⇒ forfeit")
  — a certified-win oracle analogue; 25-min qualifiers make tempo a bounded
  resource.

## 8. The quantified-prior calibration table

Priors the book states as numbers, mostly not in the ledger: 2:1 opponents
hold the double behind a straight off (pp. 64, 73); ~2/3 set rate bidding 84
with a straight off vs good players; 40% an opponent holds ≥3 of your trumps
(p. 93); ~80% first-trick void when holding 5 doubles-as-trumps missing one
(p. 80); 2:1 a shown count-suit tile was a singleton (pp. 94, 124); the
53/60/33 double-ahead-of-off trio (p. 26); four-trump double-up conditionals
37% and 52% (p. 183, OCR-truncated). One measured divergence between book
prior and oracle frequency is itself a finding *and* a teaching correction.

## Proposed experiments (ranked)

1. **Auction decoder v0** — build P(hand features | bid, seat, who-raised) from
   bid-aware corpora; test the book's three posteriors (35 ⇒ two offs/one
   five-count; 31 ⇒ ≥1 double; shuffler-last uninformative); test the bid
   lattice ("32/33 never uniquely optimal"; mass on {30,31,35,36}) against
   [[jud]]'s head_8; feed the posterior as an opponent-response feature.
   Cheap, falsifiable, and squarely on the auction-first frontier.
2. **Pounce re-adjudication under imperfect information** — the §4 category
   error. Also audit other `contradicted`/`context-limited` rows for
   perfect-info-regime testing of imperfect-info hedges.
3. **Match-score features for jud** — add marks-to-go; check overbid-at-match-
   point and the p. 129 desperation convention emerge from the wp head.
4. **Strip-the-protector paired scenario** (p. 92/94) — the E[Q]-vs-plan
   discriminator; a natural first target for jud v2's per-move-target leaf +
   search.
5. **Defender-pair coordination lift** — fixed bidder, independent-optimal
   defenders vs signal-decoding defenders; measure 84 set-rate lift. The
   opponents-in-rollout precursor, aimed at the agreed weakest surface.
6. **Belief calibration battery** — encode §2's inference rules + §8's priors
   as assertions over oracle deals; score Gus's posteriors against them.
7. **Plunge/Splash signaling testbed** — blind-partner trump choice with and
   without belief; bounded, engine-supported.
8. **Reputation meta-experiment** — opponent bid threshold as a function of
   advertised tightness (longer-horizon; needs a population harness).
9. **Housekeeping** — re-scan book pages 181–182/185–186 (ch 16 truncated) and
   the image-only hand diagrams (hands 1–14 need prose reconstruction to
   replay); verify the engine's doubles-trump re-rank (six-X promotion) and the
   high-bid set-scoring payoff (p. 131: points-originally-bid, not
   bid+captured).

## Links

[[w42-book-validation-campaign]] · [[w42-book-claim-synthesis-and-ai-directions]] ·
[[winning42-strategy-measurement]] · [[jud]] · [[gus]] ·
[[belief-trajectory]] · [[pimc]] · [[w42-lens-v1-utility-head-to-head]] ·
[[w42-bookval-v1-wave2-pounce-high-bid]]
