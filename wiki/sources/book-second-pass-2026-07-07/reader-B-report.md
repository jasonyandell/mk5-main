---
title: Book second pass 2026-07-07 — reader B raw report
kind: source
first_seen: 5e3f3245
last_updated: pending-this-ingest
status: active
---

# Second-Pass Report: Winning 42, pp. 59-108 (chapters 7-12)

## OCR quality note
Prose is clean throughout; quotes below are lightly de-noised. The domino grid diagrams in the worked-example hands (Hands 21-31, esp. pp. 97, 100, 103, 106, 108) are garbled pip-art (`ecco`, `See|ece`, `cee`) - the *tiles* of those hands are largely unrecoverable from text, but the reasoning narration is fully intact. Every finding rests on prose, not grids. Note the chapter mix here is NOT "Partner Support / Concentration" - pp. 59-108 are actually 84 bidding (ch.7), Setting 84 (ch.8), Doubles-as-trumps & No-trumps (ch.9), Tournament scoring (ch.10), Talking Across the Board (ch.11), and Advanced Bidding & Playing (ch.12, worked hands 26-31).

---

## TIER 1 - Highest actionability

### 1. The "strip-the-protector" maneuver: a setup lead that is locally worthless but flips a later double from dead to lethal
Cleanest instance of the "single-decision blind spot" in the range; likely not captured first pass.

> "You should lead the double-ace next. Why? Because if your opponent has the five-blank and five-ace, or trey-deuce and trey-ace, you want to get that ace out of his hand now so that, when you lead the double-five and double-trey, he is forced to play the count domino. You will have eliminated his 'protection' of those count dominoes from your doubles. This is very important." (p. 92)

Same hand, different opponent layout, value flips:

> "Note that, in this case, your double-ace is worthless to you. It doesn't call for count, whereas in the previous scenario it was very important to your strategy. Yet you have the same hand in both cases. The difference is your opponents' hands." (p. 94)

- Why missed: leading double-ace calls for no count and wins nothing immediately - one-step greedy / single-decision-contrast scores it as a wasted lead. Its value is a 2-ply combination (lead A strips a guard -> later high double forces count out), conditional on inferred opponent holdings.
- Experiment: purpose-built E[Q]-vs-greedy discriminator. Build the p.92 layout (opponent holds count + same-suit guard). Score the double-ace lead under one-step EV vs 2-3 ply lookahead that models "force the guard out now, cash the double later." Predict greedy ranks it near-bottom, lookahead top. Then run p.94 layout (guard absent) and confirm the same tile drops to worthless. The value swing on a fixed tile across two belief states is exactly the belief-conditional action value jud/E[Q] wants.

### 2. A catalog of explicit negative-inference belief updates (worked hands 28-31)
First pass got void-inference and bid-derived inference, but ch.12 worked hands contain a richer, mostly-unextracted class: inference from what a player chose AMONG legal plays - negative inference, donation-choice inference, lead-choice inference.

- Count-donation => singleton trump: "Opponent 2 would not have played the deuce-trey count domino unless it was his only trump. That means opponent 1 has two trumps left." (p. 106)
- Which count partner donated => what partner lacks: "one of your opponents has the double-five - if your partner had it, he would have given it to you instead of the blank-five." (p. 106)
- Bidder's off-lead => bidder's holding (read by BOTH sides): partner: "You also know now that the bidder has the six-four. Otherwise he would have never led the six-five." (p. 107); opponent independently: "you also sense that the bidder has the six-four, because he led the six-five earlier." (p. 108)
- Count-of-remaining => bound on hidden holding: "You know there are still four unplayed trumps, so the bidder can have no more than three in addition to his six-four." (p. 107)
- Singleton probability drives lead choice: "Odds are two-to-one that the deuce or trey he played was his only one, so he can trump your double if you lead that suit on the next trick." (p. 94)

- Why missed: these live inside narration of complete hands, not tactical chapters; a claim-extraction pass reading headers skips them. Each is an update rule keyed on an ACTION, not a void.
- Experiment: encode each as a belief-update assertion; run the oracle over sampled deals reaching that node; measure whether the prescribed posterior matches the oracle's true-holding distribution. Directly testable calibration checks for Gus/belief brain; double as human-teachable "reasons" for jud.

### 3. "Crisis management" is a named conditional plan branch with explicit trigger and re-plan - not a single move
> Trigger: "After the first two tricks are played, one trump is still out and your opponent has it... You're out of trumps. What do you do? The object now is crisis management. You need count. Your goal is to get in enough count dominoes to reach your bid; meanwhile, the opponent with the trump must follow suit and is unable to use his trump effectively." (p. 93)

Branch has its own conditional ordering rule keyed on a specific observed tile:

> "Look to see which suit - deuces or treys - has more dominoes out and then lead that double next. Or, if the opponent with the trump played the five-deuce on that last trick, then play the double-trey. If he played the five-trey on that last trick, then lead the double-deuce." (p. 93-94)

- Why missed: "detect opponent has extra trumps" is trivially a one-line claim; the value is the abandon-trump-drawing -> force-count-before-losing-lead re-plan and its tile-keyed ordering. Single-decision testing can't see a plan-switch.
- Experiment: plan-vs-plan harness. At the trigger node compare (a) continue drawing trumps vs (b) crisis re-plan (force count via doubles ordered by remaining-suit-count). Score by p(make bid), not one-step EV. Tests "a plan that succeeds" directly on the project goal.

---

## TIER 2 - Strong, structural

### 4. The prioritized throwaway/keep algorithm + generalized pair-protection
Ch.8 states an explicit 5-tier ranked keep-priority - an actual defensive-discard algorithm - plus generalizes "double protects off" to any two tiles of a suit.

> "1. Doubles with dominoes from that suit still out (except the double-blank). 2. Domino pairs with dominoes from that suit still out. 3. Dominoes that protect your domino pairs. 4. Non-winning dominoes. 5. Low dominoes that will help your partner decide what doubles and domino pairs to keep." (p. 72) ... "priority three always takes precedence over priority five." (p. 73)

Generalization: "even without any doubles, you can still set the bidder ... try to hold onto any dominoes in which you have two from the same suit." (p. 71) ... "you want to protect your domino pairs in the same way that a double protects an off." (p. 72)

Working-memory load quantified: "That's seven dominoes you kept track of in your head for five tricks." (p. 71) - closest thing to the working-memory spec the task asked about: track (off-suit remaining counts) + (specific tiles each of your doubles/pairs can catch).

- Why missed: "protected offs" collapses tiers 1-3 into one idea and drops tier 5 entirely - and tier 5 is a signaling instruction (play low tiles precisely so partner can read your kept doubles).
- Experiment: implement the 5-tier keep-order as a discard heuristic for setter play; A/B against EV/CVaR discard on setter seats. Separately instrument "tracked-set size"; book claims ~7 tiles - check whether decision difficulty (oracle regret variance) correlates.

### 5. Off/double ordering is CONDITIONAL; the conditions were flattened away
- Two offs same suit: "you would need to lead the trey-deuce first if the trey-ace has not been played by that time." (p. 63)
- Save highest-coverage double for a LATER trick to force offs: "It would be wise to save the double-four for the third trick to help your chances of forcing in the four-six or four-ace." (p. 81)
- No-trumps last-two-tricks ordering keyed on live counts: "when you lead the fourth trick, you're going to play the double that has the fewest remaining dominoes out from that suit, hoping to make that off a ... walker." (p. 81)
- Partner's abandon condition: "If, after five tricks or at any time earlier, all the other dominoes of that suit have been played, then throw away that double. You know your partner's off is not in that suit." (p. 65)

- Experiment: encode each ordering as a policy; test that the conditional beats the fixed order on deals where the guard-tile state actually varies. The p.65 abandon rule is a clean belief-driven prune worth its own test.

### 6. Signaling conventions carried entirely by tile choice (which count you donate)
First pass has "safe count donation"; book makes the CHOICE of which count a legible convention.

> Partner donates the highest useful count to the trick-winning partner: "Give him the five-blank, which sets the bidder on the first trick!" (p. 101); "Donate the blank-five to your partner." (p. 105) - and the ABSENCE is read: "if your partner had it, he would have given it to you instead of the blank-five" (p. 106). Withholding double-five signals you don't have it.

- Why it matters: a real signaling protocol - legibility earned purely through card choice, decodable convention (donate-highest => not-donating-X means not-holding-X). First pass treated donation as EV help, not a two-way signaling code.
- Experiment: model the donation convention as a signaling policy; measure whether an opponent/partner that DECODES it (updates on which count was/wasn't donated) gains over one treating donation as noise. Direct belief-legibility win metric.

### 7. Doubles-as-trumps and No-trumps are re-rank transforms + distinct play-plans, not just bid options
- Rank transform: "the double-five is not a five - it's a double. The highest domino of each of the other suits is the six instead of the double. The six-five is both the highest five and the highest six." (p. 75) - same physical tile changes suit membership and rank under contract. The p.76-77 re-evaluation ("Wait a minute, no you don't! Your six-five is the high six as well as the high five...") is a worked chain whose implicit rule is never stated: under doubles-trumps, each six-X becomes its suit's protector.
- No-trumps play grammar: "you cannot get out of the lead and be guaranteed of getting back in the lead. That's what trumps do for you. So you must play the hand like an 84 hand ... The difference ... is that it's possible to lose the last trick or two and still make your bid." (p. 80)

- Experiment: verify the engine's action-generation/eval applies the correct rank transform under doubles-trumps (six-X promoted, double demoted to trump). Test the "no-trumps ~= 84 with a point target" plan on no-trumps deals. If jud's value net trained without seeing these contracts' re-rank, likely blind spot worth an audit.

---

## TIER 3 - Worth logging

### 8. "Talking Across the Board" (ch.11) is the legibility CHARTER, not just etiquette
Likely skipped first pass as rules/etiquette, but states the project-relevant invariant: the only legal information channel is the tiles you play.

> "any type of verbal or physical cue from one partner to another ... is cheating." (p. 89) ... "The true and full extent of challenge, excitement, strategy, and reward ... can only be realized when no one knows anything specific about another player's hand during the bid process." (p. 89)

- Why it matters: the book's own justification for why belief/signaling must be EARNED through play - directly supports the belief-legibility thesis (a human-teachable "reason" is legitimate only if inferable from public plays). Cite in jud/legibility framing rather than test.

### 9. Bait-a-trump deception (induce an opponent to waste a trump)
> "Since you have the six-four, you could play the six-five. If opponent 1 does not have a six, he will trump it on the chance that one of the other players has the six-four." (p. 107)

Deliberate lead of a tile whose partner-tile you hold, to induce a wasted trump; later payoff on p.108. Experiment: test whether EV/lookahead ever finds this bait or whether it requires opponent-modeling in the rollout (the v2 "opponents-in-rollout" direction).

### 10. Deliberate double-void to create a future trump-in (seat-dependent)
> "throw away the five-four because that will leave you void in both fours and fives, which contain ten-count dominoes. If one of those two suits is led later in this hand, you can trump in." (p. 108)

Confirms/extends first pass's "void-by-discarding SUPPORTED" split - here a DOUBLE void created on purpose, only sensible for the seat that still gets to act. Ties to setter-seat asymmetry.

### 11. Quantified priors - a calibration table the book hands you
| Prior (verbatim) | Value | Page |
|---|---|---|
| Opponents hold the double behind a straight off | 2:1 (~67%) | 64, 73 |
| Set rate bidding 84 with a straight off vs good players | ~2 of 3 | 64, 73 |
| An opponent holds >=3 of your trumps ("beaten before you start") | 40% | 93 |
| With 5 doubles-as-trumps missing one, an opponent void on trick 1 | ~80% | 80 |
| A count-suit tile an opponent showed is his only one | 2:1 | 94, 124 |

- Why it matters: first pass extracted many as absolute rules; the numbers are the book being explicitly probabilistic. Experiment: measure oracle's actual frequencies against these five; divergences are findings AND legibility caveats (teach the true number).

### 12. Bidding decisions are seat/who-bid asymmetric
> "you shouldn't raise a 30 bid that's already been made." (p. 97) ... "as it's your partner with the 30 bid, why not just help him make his bid? It's not much fun raising your partner's bid with a risky one and then getting set." (p. 98) ... but "If you were well behind ... you could justify raising a 30 bid by your opponent." (p. 98)

Same hand bids differently depending on whether the standing 30 is partner's or an opponent's (and on score). First-pass eligibility claims likely flattened to a hand-strength threshold.

---

## Synthesis: the one experiment to run first
Finding #1 (strip-the-protector, p.92 vs p.94) is the highest-leverage single test: a fully-specified two-layout scenario where one fixed tile swings from load-bearing to worthless purely on inferred opponent holdings, and where greedy EV is structurally guaranteed to misrank the setup lead. Exercises E[Q], belief-conditioning, and human-teachable "why" in one deal - exactly the jud/champion frontier. #2 (negative-inference catalog) supplies the belief-update rules those plans depend on; #3 (crisis-management branch) is the plan-switch that "a plan that succeeds" is meant to model.

Source files: /Users/jason/code/mk5-main/scratch/winning42/text/p00XX__Page_YY_of_196.txt (all page refs are book pages).
