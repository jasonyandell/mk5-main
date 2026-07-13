---
title: Book second pass 2026-07-07 — reader C raw report
kind: source
first_seen: 2026-07-07
last_updated: 2026-07-11
status: active
---

# Second-Pass Report: "Winning 42" pp. 109–156 (files p0142–p0199)

## Scope note — read this first

**The page range does not contain the chapters the brief anticipated.** The brief expected chapters 8–11 (Setting 84 / Doubles-No-Trump / Tournament Scoring / Table Talk). What actually lives in pp. 109–156 is:

- **pp. 109–121** — the tail of the *Advanced Play* chapter: four fully-worked advanced hands (Hands 32–36), a bidding-lattice argument, and a reputation/table-image passage.
- **pp. 121–133** — **Chapter 13, "Optional Variations"**: Nel-O, Sevens, Plunge/Splash, plus a numbered list of miscellaneous rule variants.
- **pp. 133–156** — **Chapter 14, "Lone Star Domino Phenomenon"**: pure oral history and player biographies. Almost no strategy; mined below for the few embedded gems.

So the "Tournament Scoring" and "Table Talk" surfaces the first pass may have mapped to this range are **not here** — they're at earlier page numbers. A re-scoped pull is needed to cover Ch 10/11 proper.

That said, this range contains the single most actionable auction-frontier claim in the book (the 32/33 bid-lattice argument) and the richest two-defender coordination examples (Hands 34–35). Ranked below by actionability against the jud/auction-first frontier.

---

## TIER 1 — Directly actionable on jud's auction head

### 1. The bid-lattice claim: 32 and 33 are *never* reasoned bids; "if you can bid 32, you can bid 35"
**Quote (p. 118):** "I have encountered players who often bid 32 or 33. I find this strange, because there is no reasoned mathematical basis for a 32 or 33 bid. … If you bid 32, it means you expect to lose five tricks, plus one five-count. In other words, you're going to capture 32 points in only two tricks, and that's it? That's absurd."
**Quote (p. 119):** "The most common, reasoned bids in 42 are 30, 31, 35, and 36. I occasionally come across a justifiable 34 or 41, but that's it. … **If you can bid 32, then you can bid 35.** The only reason to bid 32 is when you are the last bidder and you only need to raise a 31 in order to win the bid."

**Why the first pass missed/flattened it:** Not a per-hand tactic — a claim about the *shape of the entire bid-value action space*. A first pass cataloguing per-decision contrasts has no slot for "this region of the action lattice is dominated." Exactly what jud's auction head exists to test.

**Experiment:** With the bid-aware corpus, compute the oracle-optimal bid distribution over {30,31,32,33,34,35,36,41,42}. Test: (a) 32/33 are almost never uniquely optimal (only when the sole legal raise over a standing 31 is 32); (b) mass concentrates on {30,31,35,36}. "If you can bid 32 you can bid 35" is literally a dominance relation between two actions given the same hand — check whether make-probability at 35 conditioned on "34-makeable hands" exceeds threshold. If jud ever prefers an interior 32/34, that's a real refutation or a corpus artifact — cheap, falsifiable test of the bid head against expert doctrine.

### 2. Reputation / table-image drives opponent overbidding — the named population/reputation gap, stated as mechanism
**Quote (p. 121):** "the opponent admitted that one reason he took the bid was because he felt sure that my partner, who is not known to bid wildly, had a rock solid bid. So our reputation for not being wild bidders actually scared him into bidding wild on his own hand. … If we did not have such a reputation, then this opponent might have chosen to pass in hopes that our own bid was risky or wild, and he might have set us and won the match."
**Quote (p. 121):** "I have beaten teams 7-1 and 7-2 without ever playing a bid!"

**Why the first pass missed/flattened it:** A single-decision extractor cannot represent a *cross-game reputation prior* — the opponent's bid is a function of belief about the author's bidding distribution accumulated over prior hands, not of this hand's tiles. This is the population/reputation effect the brief names as untested, and it is structurally invisible to a fixed-opponent or PIMC-vs-PIMC jud harness — the same concealment/signaling blindness Fable flagged.

**Experiment:** Meta-experiment, not a leaf test. Model an opponent whose pass/bid threshold is a function of a belief over the bidder's aggressiveness parameter. Predicted: a tight-bidding reputation *raises* opponents' bids and *lowers* their set-attempts against you. Quantify whether a fixed-opponent evaluation systematically mis-rates a tight bidder relative to an adapting population. "Winning 7-1 without playing a bid" is the extreme: EV can be maximized purely through the *setting* half of the game, which a bid-EV-only objective never surfaces.

---

## TIER 2 — Multi-step defensive plan grammar (setter/defender surface)

### 3. Hand 35: expert two-defender coordination sets a near-maximum 84 — full plan with disinformation and timing triggers
**Setup (p. 116):** the 84 bidder holds "four fours for trumps, two doubles, and a protected high off. This is about as good an 84 hand as you could draw." He is nonetheless set.
**Coordination, verbatim (p. 118):** "It was very important that **opponent 1 dumped the ace-blank early so that opponent 2 would know not to save the double-ace.** And **by not throwing away the five-ace early, opponent 2 had his six-five protected** from the bidder's double-five late in the hand. This was expert team playing, which was necessary to set such a fabulous 84 hand."
**Deliberate disinformation (p. 114, Hand 34, same doctrine):** "You should try to keep the low fours for a while so any opponent who has the double-four will save it, **thinking the bidder has a low four.**"

**Why the first pass missed/flattened it:** Canonical "single-decision blind spot." Every individual discard is EV-neutral or looks like a throwaway in isolation; the set exists only as an emergent property of a 5-trick coordinated sequence where (a) defender-1's early discard is an information signal that (b) reconfigures defender-2's saving policy, and (c) a tile is deliberately exposed to plant a false belief in the bidder. Tested as single contrasts, every move scores neutral and the plan evaporates.

**Experiment:** "Defender-pair coordination" counterfactual: hold bidder policy fixed, compare (i) two independently-optimal defenders vs (ii) two defenders sharing a signal channel (defender-1's discard order observable to defender-2's policy). Measure set-rate lift on the subset of 84 hands the bidder makes under independent defense. Hand 35 predicts a strictly positive lift unreachable by any single-defender policy — the concrete, measurable version of "setter defense is the richest untapped surface." The two named triggers (dump-to-inform, withhold-to-protect-a-pair) are the exact features to instrument.

### 4. The "domino pairs" theory of 84 defense — a compact, enumerable defender doctrine with a hard override rule
**Quote (p. 114):** "You have no doubles to save. So you are **strictly looking for domino pairs** to hold on to until the last two tricks. You have at least two dominoes from these suits: aces, deuces, treys, and sixes. These could help you set the bidder if he plays the double-ace and ace-blank on the last two tricks; double-deuce, deuce-blank; double-trey, trey-blank; or double-six, six-deuce."
**Override rule (p. 115):** "you know you should **never forsake a double for domino pairs.** You are forced to give up on fours at this time."
**Belief-update from partner (p. 114):** "Your partner played the deuce-blank, one that you were looking for. **You can now safely throw away a deuce.**"

**Why the first pass missed/flattened it:** First pass extracted "live doubles / same-suit pairs / protectors" as *static* categories. What it flattened is the **decision procedure and its priority ordering**: (1) a double you hold beats any pair (hard override); (2) else retain one same-suit pair per suit to catch the bidder's forced last-two-trick lead of {double-X, X-low}; (3) release a saved tile the instant your partner's discard proves that suit dead. An "if-double-then-keep-double, else-keep-pairs, unless-partner-signals-dead" conditional collapsed into a flat list.

**Experiment:** Encode as an explicit defender heuristic, score vs oracle-optimal defense on 84 hands: (a) does "never forsake a double for a pair" match the oracle? (b) how much of optimal set-rate does the pure pairs-doctrine recover? Also a legibility win — a teachable defender policy with a stated priority order.

### 5. Hand 34/35 name the exact set-conditions for a strong 84 as a boolean over opponent holdings
**Quote (p. 116):** "There are only **two ways for the bidder to be set**: (1) if an opponent had the five-six and another five or (2) if an opponent had two fours that could be saved until the last two tricks."

**Why it matters / was missed:** A fully-specified, checkable predicate for "is this 84 hand actually cold?", stated as a disjunction over the distribution of two specific tiles across the two opponents. First-pass "protected off" language captures the intuition but not the precise combinatorial condition. Ready-made oracle assertion: enumerate deals consistent with the bidder's hand, verify the set-set equals {opp holds 5-6 + another 5} ∪ {opp holds savable double-four + low four}.

### 6. Unconditional pounce doctrine — "play it or lose it," pivoting on a single ten-count
**Quote (p. 110):** "This hand is a classic lesson in **pouncing when you can, regardless of whether you know who will win the trick.** If opponent 1 had failed to play the ten-count on the trey-ace, the bidder would have scored 41 points. Play it or lose it! This is even more critical against high bids of 35 or 36."
**Single-tile counterfactual (p. 112):** "The only difference between this hand and the previous one is that opponent 1 did not have a ten-count to play on the bidder's off. Otherwise, the bidder would have been set."

**Why partially missed:** First pass likely captured "play count when you can't win." The load-bearing phrase is **"regardless of whether you know who will win the trick"** — count goes on the bidder's off even when the defender cannot see his partner will take it, because expected loss of hoarding dominates. The book frames Hands 32/32-variant as a matched-pair counterfactual differing by one tile — a natural paired-corpus item. "More critical at 35/36" claims pounce-value scales with bid height (thinner make margin → each ten-count more pivotal) — testable.

---

## TIER 3 — Bid-calibration counterexample

### 7. Hand 36: a bid-worthy hand with *no doubles and four offs*, with the EV arithmetic shown
**Quote (p. 119–120):** "Earlier in the book I promised you a bid-worthy hand with no doubles and four offs. Here it is." Trump risk "**Total trump risk: 1 point.**" Off risk "**Total off risk: four tricks and two five-counts = 14 points.**" "By counting on your partner for help on **just one measly trick**, you can easily bid 30 on this hand."
**Same anti-heuristic, from a champion (p. 149, St. Clair):** "So many people play regimented 42. **They won't bid without the double to their trumps, or they never bid more than 30 with a four, five, or six off. I don't play that way.**"

**Why it matters:** A labeled counterexample to the most common human bidding heuristic ("need the double to your trump"), with the author's own risk decomposition (15 pts at risk → needs ~1 trick of partner help → 30 is safe). Calibration probe: does jud's bid head bid ≥30 on the no-doubles-four-off hand class? If it inherits a "need doubles" bias from a naive corpus, this class is where it visibly underbids, and the book gives the exact reasoning to check against.

---

## TIER 4 — Action-space / rules variants (affect engine, corpus, special-contract modeling; not jud directly)

Chapter 13. Matter for what moves the generator emits and how special contracts/scoring are modeled.

- **Bidder need not lead trump; six-five-off opening convention (p. 132):** "A bidder may lead any domino… It is most common for a bidder to lead a **six-five off on the first trick to eliminate it as a threat in the suit of fives.**" — a concrete opening-book heuristic, testable as an oracle-preferred first move on trump-in-fives hands.
- **Small-end-lead prohibition (p. 132):** leading six-ace and declaring it "an ace" is disallowed — "one step away from trading dominoes." Legal-move generator must key led suit to the *high* end of a non-trump lead.
- **Set-scoring at high bids (p. 131):** "Whoever sets or makes a 42, 84, or higher bid merely gets the **points originally bid**" — NOT bid+captured. Directly relevant to mark_ev at bid≥42; interacts with the "mark_ev ≡ p_make" affine-identity note. Confirm the engine's high-bid payoff matches this.
- **Doubles-as-trump follow variant (p. 131):** if doubles are trump and you can't follow a led double, one variant forces you to play the *suit* of that double. Roberson notes it "could make a difference in how some hands would be bid." A Ch-9 declaration-regime knob.
- **Forced bidding (p. 130):** if first three pass, the fourth *must* take it for 30. Downside quantified in bios (p. 151): "sometimes a player caught no tricks and went down 72 points… Conrad Stone went down 72 points twice in an hour on forced bids."
- **84-raise increments (p. 130):** tournaments raise in 42s (84,126,168,Game); some circles in 84s. Fourth "Game" bid is really 210. Affects the legal bid-ladder above 84.
- **No verbal trump identification; exploiting opponent inattentiveness (p. 131→132):** players may not announce trumps; a sharp player exploits opponents who lose track. The passage on the bidder inferring an unplayed trump is **cut off mid-sentence at the p.131/132 boundary (OCR)** — see garble note.
- **84 needs no prior 42; bid-42 defensive purpose (p. 133):** "The only reason to bid 42, or one mark, is if that's all you need to win the match. That way, if you get set, your opponents only receive half as many points." — the one mark-state-dependent bid-choice in range, closest to the tournament-scoring surface the brief wanted.

**Special contracts (Nel-O / Sevens / Plunge / Splash), pp. 121–129 — mechanics for engine modeling:**
- **Nel-O (p. 124):** bid ≥42, take zero tricks. Variant: **doubles as a separate suit**, bidder optionally declares doubles high or low. "It is possible to have a higher domino or two… still bid Nel-O, as long as you also have a low domino from those suits for protection." Domino-trade variant condemned as illegal.
- **Sevens (p. 125–127):** no suits, no trumps; every player must play their tile closest to 7 each trick, cannot save. Inventory: three tiles sum 7 (6-1, 5-2, 4-3), four sum 6, three sum 8. Set-conditions enumerated.
- **Plunge (p. 127) / Splash (p. 128):** Plunge = ≥4 doubles, auto-bid 168 (4 marks), **partner declares trump and leads**. Splash = 3 doubles, 3 marks. Critique doubles as a table-talk boundary statement (Tier 5). Legit alternative given (p. 128): if opponents are one hand from winning, first team bidder bids 84 and partner raises, regardless of hands.

---

## TIER 5 — Table-talk boundary (the one Ch-11-adjacent gem in range)

The explicit legal/illegal information-channel boundary, stated while condemning Plunge:
**Quote (p. 128):** "The only way a player can **legitimately** learn about his partner's hand is **through what he plays during the hand.** The only way you can learn the slightest thing before play starts is from your partner's bid (if he makes one, which likely means he has at least two or three doubles)—and this is just a vague assumption."

**Why it matters:** Precise, verbatim boundary for belief-modeling and multiplayer capability-filter design — the only two legal inference channels are (a) played tiles and (b) the coarse prior implied by a partner's bid. Everything else is out of bounds. If the belief model or the multiplayer filter ever lets a partner condition on more than {played tiles, bid level}, it has crossed the canonical line. Note the quantified prior: a partner's bid "likely means he has at least two or three doubles" — a testable partner-hand prior for the belief net.

---

## Low-value biographical range (pp. 133–156), gems only

Chapter 14 is oral history; no strategy sections. Fragments worth one line each:
- **p. 153 (Hencerling, 3× state champ):** "Size up your opponents, realize **your team controls fourteen dominoes**, and then draw damn good dominoes." — the 14-tile-partnership lens.
- **p. 137 (Anderson):** "42 is an **offensive** game, not a defensive game… if you don't bid, you can't win." — a stated bidder-aggression prior, in tension with the author's own "win 7-1 without bidding"; a genuine disagreement within the source.
- **p. 143 / p. 150:** a five-year-old "proved her mettle… by **holding the double-five to set her grandfather's 84 bid**"; the Steve Dugan / family-reunion lineage anecdote is on p. 150 — the double-as-setter motif again.

---

## OCR garble log
- **Hand 33 diagram (p. 111, file p0143)** and **Hand 34 diagram (p. 113, file p0145)** — domino-layout ASCII art unrecoverable; surrounding play-by-play is clean, tile identities reconstructable from narration.
- **Nel-O sample-hand diagram (p. 124, file p0158)** — layout garbled; text says "three blanks and three aces, highest is the deuce-four," enough to reconstruct.
- **p. 131/132 boundary (files p0168 = Page 131, p0167 = Page 132 — filenames are page-swapped):** the passage on the bidder inferring an unplayed trump is cut off mid-sentence across the split; the completion is not in range. Note the filename/page-number swap when citing these two.
- Scattered "ay," artifacts (running-footer bleed) at the foot of many pages — ignore.

---

## Bottom line for the frontier
The two findings that most repay immediate work, given jud's auction-first focus, are **#1 (the 32/33 bid-lattice claim** — a sharp, cheap falsification test of the bid head against stated expert doctrine) and **#2 (reputation-driven opponent overbidding** — the mechanism behind the named population/reputation gap, and another instance of the fixed-opponent-harness blindness Fable identified). Behind those, **#3–#4 (Hands 34–35 two-defender coordination and the domino-pairs doctrine)** are the richest untapped *setter-defense* surface in the range, measurable as coordination-lift experiments and teachable as legible policies. Everything in Tier 4 is engine/corpus hygiene rather than frontier movement.
