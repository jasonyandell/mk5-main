---
title: Book second pass 2026-07-07 — reader D raw report
kind: source
first_seen: 5e3f3245
last_updated: e2171816
status: active
---

# Second-pass report: Winning 42, "chapters 12–16"

## META-FINDING (read first): the page-range brief was wrong

The task briefed pages 157–193 (files p0200–p0244) as "chapters 12–16: Advanced
Bidding, Variations, History, Celebrities, Statistics." That mapping is off by
several chapters. What actually lives in pages 157–193:

- **pp.157–170** — Chapter 14, *Lone Star Domino Phenomenon* (where 42 is played,
  domino manufacturing, tournaments, online, tournament rules, 45-year winners list)
- **pp.171–182** — Chapter 15, *42-Playin' Texas Celebrities*
- **pp.183–184** — Chapter 16, *Statistical Odds*
- **pp.187–193** — Index + About the Author

**Chapter 12 (Advanced Bidding and Playing) is pages 91–122; Chapter 13 (Optional
Variations) is pages 123–133.** The pounce passage the lead most wanted is on
**page 112–113 (file p0145–p0146)**, not in the assigned range. I read chapters 12
and 13 directly so the deliverable is complete. Everything below is verbatim
(lightly de-OCR'd), with page/file cited.

**OCR gaps to flag:** book pages **181–182 and 185–186 are entirely absent** from
the OCR set. Page 183's FOUR TRUMPS section ends mid-sentence ("Your bid…") and
page 184 resumes on a different section header (OTHER STATISTICS), so **the middle
of Chapter 16 — the third/fourth four-trump configurations and any three-trumps /
doubles-as-trumps / no-trump odds tables — is lost.** Also missing: pages 122, 134.
Confirmed TOC (p.vii): 1 In a Nutshell · 2 Bidding · 3 Play Your Hand After Winning
· 4 Helping Your Partner · 5 How to Set the Bidder · 6 Concentration and Style ·
7 Taking Every Trick · 8 Setting the 84 · 9 Doubles/No-Trumps · 10 Tournament
Scoring · 11 Talking Across the Board · 12 Advanced Bidding & Playing · 13 Optional
Variations · 14 Lone Star Phenomenon · 15 Celebrities · 16 Statistical Odds.

---

## THE POUNCE PASSAGE (lead's #1 ask) — verbatim, with the condition the first pass likely got wrong

**Book page 112 (p0145), closing the worked "Hand 32":**

> "This hand is a classic lesson in pouncing when you can, **regardless of whether
> you know who will win the trick.** If opponent 1 had failed to play the ten-count
> on the trey-ace, the bidder would have scored 41 points. Play it or lose it!
> **This is even more critical against high bids of 35 or 36.**"

**Book page 113 (p0146), the chapter-12 thesis restatement:**

> "To become an 'advanced' 42 opponent, you must learn which dominoes to save and
> which ones to throw away—and why. **And you must pounce with count whenever
> possible.**"

**Why the first pass's contradiction is probably a category error, not a wrong book
claim.** Read the exact conditions: (a) this is a **defender/opponent** heuristic,
not a bidder rule and not "the setter pounces to raise the bid"; (b) the operative
clause is *"regardless of whether you know who will win the trick"* — it is
explicitly an **imperfect-information hedge**: dump your count onto a live trick now
because you cannot guarantee a better placement later and saved count often dies in
hand; (c) the "high bids of 35 or 36" qualifier means *the bidder committed high →
he has few offs → the chance to unload count on one of his offs may not recur*, so
pouncing is "even more critical."

A perfect-information oracle **knows** who wins each trick, so it only plays count
when it actually lands — under that regime the book's "play it regardless" rule will
look strictly ‑EV, producing a spurious "contradicted under all utilities" result.
The extraction likely quoted the words correctly but tested them against an oracle
in the wrong information regime. **Suggested experiment:** re-run the pounce test in
an *imperfect-information* rollout (belief/PIMC defender that does not see opponents'
hands), scoped to defenders holding count when the bidder bid ≥35 and has ≤2 offs.
Predict: pounce becomes EV-neutral-to-positive there. Same PIMC-blindness point
already in memory (belief-value-is-legibility).

---

## HIGHEST-ACTIONABILITY FINDINGS FOR THE JUD / AUCTION-FIRST FRONTIER

### 1. Match-score-conditioned bidding — a named-gap hit (auction policy)

**Page 92 (p0122):** on a 35/36 hand —
> "If you were way behind in the match, say 120 points or three marks, **it's even
> worth risking an 84 bid** because you have the double ahead of your off."

**Page 129 (p0165), the desperation partnership convention:**
> "Simply have an understanding with your partner: **If the opponents are way ahead
> and only one hand away from winning the match, then the first bidder of your team
> on that hand will bid 84 and the other will raise him. This is done regardless of
> how good your hands are.** You're simply gambling that you will luckily match each
> other on the suits of your doubles and offs."

**Why the first pass missed it:** embedded inside a Plunge-is-unnecessary argument
and a worked hand, not stated as a bidding rule. The bid decision here is
**conditioned on match state (marks/score), not on hand EV** — exactly what an
auction-first value net trained on per-hand E[Q] will *not* learn. **Experiment:**
add match-score (marks-to-go per team) to jud's auction features; check whether the
win-prob head recovers "overbid when facing match point." Directly the lead's
flagged "auction policy" gap.

### 2. Reputation-induced overbidding — second-order opponent inference

**Page 121 (p0156):**
> "the opponent admitted that one reason he took the bid was because he felt sure
> that my partner, **who is not known to bid wildly,** had a rock solid bid. So
> **our reputation for not being wild bidders actually scared him into bidding wild
> on his own hand.** … If we did not have such a reputation, then this opponent
> might have chosen to pass in hopes that our own bid was risky or wild, and he
> might have set us."

Also page 121: *"I have beaten teams 7-1 and 7-2 without ever playing a bid!"* —
setter-primacy: matches winnable purely on defense.

**Why flattened:** first pass logged "reputation/style priors" generically. The
**mechanism** here is specific and second-order: a *tight* reputation makes
opponents *over*-bid (they read your partner's pass as concealing strength and feel
forced to outbid). **Experiment:** model opponent bid thresholds as a function of
your table reputation; test whether a jud that advertises tightness induces
exploitable opponent overbids. Opponent-adaptation, another named-untested gap.

### 3. Plunge / Splash as a *legal signaling contract* — feeds belief-legibility directly

**Pages 128–129 (p0164–p0165):**
> "To bid in Plunge, the bidder must have at least four doubles, and the automatic
> bid is 168, or four marks. … **The bidder's partner must declare trumps and begin
> play** … the partner has no idea which four doubles are in the bidder's hand …
> There is even a variation … called SPLASH, which requires only three doubles and
> a 3-mark bid."
>
> "The thing that's so unfair about this tactic … is that **it allows a player to
> tell his partner something very important about what's in his hand, as in talking
> across the board.** … The only way a player can legitimately learn about his
> partner's hand is through what he plays during the hand."

**Why missed:** first pass took Plunge/Splash as rules only. Roberson explicitly
frames them as **legal information transmission** — the bid *is* the signal ("I hold
≥4 doubles; you pick trumps"), and the partner then coordinates trump choice
**blind**. Engine already supports these contracts. Clean bounded testbed for the
signaling/belief-legibility thread. **Experiment:** does a belief-aware partner pick
trumps better than a hand-only partner in Plunge, and is the bidder's E[Q] legible
from the bid alone?

### 4. Multi-trick "strip the protection" sequencing — the single-decision blind spot, in the raw

**Page 92 (p0122)**, a 3–4 ply forced plan stated as prose:
> "You should lead the double-ace next. Why? Because if your opponent has the
> five-blank and five-ace, or trey-deuce and trey-ace, **you want to get that ace
> out of his hand now** so that, when you lead the double-five and double-trey, he
> is forced to play the count domino. **You will have eliminated his 'protection' of
> those count dominoes** … After leading the double-ace, you lead the double-trey
> and double-deuce. You're practically guaranteed of getting in the trey-deuce…"

The canonical multi-step plan the second pass says gets tested as a single contrast.
The principle — *lead an off-double to strip the low domino guarding a counter, THEN
lead the trump-double to force the counter out* — never stated as a named rule; only
appears inside Hand 32/33. **Experiment:** encode "protection-stripping" as a
candidate leaf plan; test whether E[Q]-greedy play discovers the same double
ordering.

---

## CHAPTER 13 VARIATIONS — exact mechanics the first pass likely reduced to "rules"

Engine-supported contracts; several are *forced-play* (no branching), cheap to
enumerate/verify.

**Nel-O set conditions (page 125, p0160), fully specified:**
> "In the sample hand, you have three blanks and three aces. The highest domino in
> your hand is the deuce-four. You would lead one of your lowest dominoes first …
> **the only way the bidder can be set is for one of the opponents to lead the
> deuce-blank or four-blank, with the other opponent void in deuces or fours** …
> the bidder's deuce-ace or four-ace would win the trick and set him."
> "It is possible to have a higher domino or two … and still bid Nel-O, **as long as
> you also have a low domino from those suits as well for protection.**"
> Doubles handling varies: separate suit; declarer may choose doubles high or low;
> or normal.

**Sevens mechanics + set conditions (pages 126–127, p0161–p0162):** no trumps, no
suits, score = distance from 7. Forced play: *"on each trick every player must play
the domino closest to seven left in their hand … You can't save one for late in the
hand."* Set two ways: (1) bidder holds only one of the three seven-dominoes and an
opponent holds the other two; (2) an opponent holds more sixes/eights than the
bidder. Ties don't set. **Fully deterministic forced-play contract — trivially
enumerable.**

**Plunge/Splash:** see finding #3.

**Forced bidding (page 130, p0166):** if first three pass, fourth seat may be forced
to take it for 30. Roberson: *"a great teaching tool … you are challenged to be
creative and strategic … Advanced players need not bother with it."*

**84-raise increments vary (page 130):** some raise in 42s (84,126,168), some in 84s
(84,168); tournaments use 42s.

---

## CHAPTER 16 STATISTICS — every quantitative claim present, with what's enumerated vs. missing

**Four Trumps (page 183, p0232) — TRUNCATED (only 2 of ≥3 configs survived OCR):**
- 27 ways the other three trumps distribute among the three opponents.
- Four trumps *incl. double but not 2nd-highest*: opponent can double-up on trumps
  in **10/27 (37%)** → **lead the double first, don't budget a lost trump trick;
  works 63% of the time.**
- Four trumps *incl. double but not the next two highest*: opponent doubles up in
  **14/27 (52%)** → "statistics just barely against you" → lead a **low** trump
  first and budget the loss; *but* if a count domino is on that trump, ~50/50, may
  lead the double to capture the count.
- **MISSING (pages 185–186 absent):** any third/fourth configuration, three-trump
  odds, doubles-as-trumps, and no-trump odds. Flag for re-scan.

**Hand-composition marginals (page 184, p0234) — 1,184,040 possible 7-domino hands:**
- Suits represented: 7 suits **41%**; 6 suits / one void **48%**; 5 suits / two
  voids **10%**; 4 suits / three voids **1%** (four is the minimum).
- Doubles: none **10%**; one **32%**; two **36%**; three **18%**; four **4%**; five
  **<0.5%**; six **1/50%**.
- Joint claim worth testing (not a marginal): *"the most common hand … two doubles
  and one void suit,"* and *"the odds of a hand containing two, three, or four
  doubles are at least 36% … this is why, when bidding with more than one off, you
  should count on your partner for help. There's a decent chance he has at least two
  doubles."*

**Enumeration check:** first pass reportedly verified "hand-count, suit/void,
double-count, four-trump configs." The **four-trump 37%/52% double-up figures are
conditional probabilities over opponent distributions** and should be re-verified
specifically (they're the interesting joint claims, and OCR truncated the table).
Suit/void and doubles marginals are straightforward to confirm against 1,184,040.

---

## CHAPTER 14 TOURNAMENT ECOLOGY — clock/tempo as resource, and the champion objective

**Official N42PA/State rules (pages 164–166, p0210–p0212):**
- Qualifying: **7 games, 25-minute per-game time limit**, random draw into **groups
  of 8**, round-robin within group; **best 32 win-loss records advance**; rest play
  consolation.
- Championship/consolation: single-elim, best-2-of-3, **1 hour 15 minute** limit;
  finals double-elim; one bye allowed.
- **First tiebreaker for bracket standing is TOTAL MARKS.** (Supports the champion
  objective being *margin*, not just win/loss.)
- Minimum bid 30; after 84, raises in 42s; **bids cannot be changed.**
- Lay-down rule: *"any bidder who believes he can make the bid may declare a
  lay-down at any time … However, if the opposition can demonstrate any possible way
  the bidder can be set, the bidder forfeits."* (Adversarial provability condition —
  interesting as a "certified win" oracle check.)
- No Nel-O, no forced bidding, no signaling (auto-DQ), no slow play, stacking
  permitted on ≥42 bids.

**Clock as resource:** 25-min qualifier + 75-min elimination caps + "deliberate slow
play will not be allowed" make **tempo a bounded resource** — the lead's clock/tempo
gap. The book offers no heuristic beyond "play quickly"; only the constraint.

**Weak-AI corroboration (pages 160–161, p0205–p0206):** Roberson states existing 42
apps *"are not programmed to be good setters … ignore almost every skill and tactic
I teach in chapter 5 for setting the bidder … Usually these apps are also risky
bidders."* Independent testimony that **setter/defensive play is the market-wide
weakness.**

---

## CHAPTER 15 CELEBRITIES — style aphorisms (already in first pass, for completeness)

Crippen: *"Watch what the other players are doing."* B.J. Thomas: *"Just play what
you've got. Don't over-bid. Have patience and don't be afraid to go low."* Robert
Earl Keen: *"Make certain you count the right amount of dots."* Already-extracted
watchfulness / overbid-restraint / dot-count / go-low priors; no new mechanism.
**Note:** the chapter documents genuine cultural pull for Nel-O (B.J. Thomas's
father "especially loved to bid Nel-O") — worth weighting how often a population
model's opponents actually choose low bids, even though Roberson disdains it.

---

## Ranked summary of what to test next

1. **Match-score-conditioned auction policy** (pp.92,129) — add marks-to-go to jud
   features; expect "overbid at match point" to emerge. Named auction-policy gap.
2. **Pounce under imperfect information** (pp.112–113) — re-run the contradicted
   pounce test with a belief/PIMC defender, scoped to bidder-bid≥35, ≤2 offs; expect
   the contradiction to dissolve.
3. **Reputation-induced opponent overbidding** (p.121) — model opponent bid
   thresholds vs. your advertised tightness; test exploitability.
4. **Plunge/Splash as legal one-bit signal** (pp.128–129) — belief-aware blind
   partner trump selection; bounded signaling testbed the engine already supports.
5. **Protection-stripping double-ordering** (p.92) — encode as a leaf plan; check
   E[Q]-greedy rediscovers the book's double sequence.
6. **Re-scan / re-enumerate Chapter 16 four-trump conditionals** (p.183, truncated)
   — the 37%/52% double-up figures; recover missing pages 185–186.
7. **Nel-O/Sevens forced-play enumeration** (pp.125–127) — cheap deterministic
   verification of set conditions.
8. **Total-marks tiebreaker + lay-down forfeit rule** (pp.164–166) — corroborates
   margin-based champion objective; lay-down "any possible set → forfeit" is a
   certified-win oracle analogue.
