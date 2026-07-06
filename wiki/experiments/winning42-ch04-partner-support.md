---
title: Winning 42 Ch04 Partner Support
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.4`: Chapter 4, "Helping Your
Partner Make a Bid." It harvests partner-support concepts from Winning 42 into
measurable hypotheses for [[gus]], [[burl]], and [[forge]].

The chapter's central move is to treat the bidder's partner as an active agent, not a
passenger. The partner should try to win tricks, lead doubles or effective boss tiles,
donate count only when the bidder is guaranteed to win, and avoid leads that expose count
or disrupt the bidder's trump plan. These claims come from the book preview slice on
pages 37-41 (`scratch/winning42/winning42.with_figures.md` lines 2058-2296).

This makes Chapter 4 a natural source of concept buckets for the first
`strategy_tags` work package in [[winning42-strategy-measurement]]. It also targets the
same action-local surface that helped [[gus-strategy-tags-probe]]: whether a candidate
play follows, wins, donates count, exposes count, or changes control. For [[burl]], the
chapter gives trace-level checks: does the model notice partner safety before dumping
count, and does it reason about low-damage leads instead of generic "play a double"
rules?

## Source Claims

- The partner should actively help the bidder, because captured points belong to the team
  and the bidder's offs can make the contract vulnerable.
- The first-order support plan is: try to win a trick, lead doubles when in control, and
  give count to the bidder whenever the bidder is guaranteed to win the trick.
- Count donation is unsafe unless the partner cannot lose; low trump leads can be traps
  intended to pull higher trumps, and count dumped there can swing the hand.
- If the partner wins lead but has no double, the next lead should minimize expected count
  exposure. The book's examples compare leads by whether they can call live five-count or
  ten-count dominoes.
- A middle-ranking tile can become an effective double when all higher tiles in that suit
  have already been played.
- Trump leads by the bidder's partner are usually disruptive; the chapter gives only a
  narrow exception when every off lead risks major count and the partner probably has more
  than one trump left.

## Concept Table

| concept | detector/state inputs | metric/test | likely data source | priority | implementation notes | readiness |
|---|---|---|---|---:|---|---|
| `partner_support_regime` | bidder seat, current player team relation, contract target, points already captured, remaining bid margin | Bucket all partner-of-bidder decisions; compare regret and set rate inside vs outside support regime | Generated games plus `forge` E[Q] records | P0 | Gate every Chapter 4 detector on "current player is bidder's partner" and ordinary non-84 regime unless explicitly extended | enumeration-ready for role/state; oracle-ready for regret; Gus-ready as tag; Burl-ready as trace bucket |
| `lead_capture_for_support` | legal actions, current trick winner, can-follow status, action rank/trump status, partner/opponent current winner | Missed opportunity rate when partner could win a trick and obtain lead; regret of winning vs sloughing | Oracle decision corpus; full-game rollouts for downstream lead value | P1 | Needs a "win now and lead next" event label, not just immediate trick value | enumeration-ready for can-win; oracle-ready for value; Gus-ready; Burl-ready |
| `safe_partner_count_donation` | partner currently winning, highest outstanding trump/suit tile, remaining trumps, action point value, later seats still to act, void evidence | Good donation rate; unsafe donation tail loss; regret when count is held despite guaranteed partner win | Generated games with per-action E[Q]; strategy probe action features | P0 | Extend existing `partner_donation_window` with guarantee strength: certain, probable, unsafe | enumeration-ready for public guarantee; oracle-ready; Gus-ready; Burl-ready |
| `low_trump_trap_against_count_dump` | partner led trump, led trump rank, outstanding higher trumps, later opponents with possible trump, candidate count action | Count-dump blunder rate on non-boss trump leads; tail regret when opponent overtrumps | Oracle corpus plus adversarial sampled states | P0 | This is the chapter's sharpest exception: "partner winning now" is insufficient if partner led a low trump | enumeration-ready for obvious higher-trump risk; oracle-ready; Gus-ready; Burl-ready |
| `lead_away_from_count_damage` | candidate leads, led suit(s), live count tiles callable by each lead, count already played/in-hand/trump, expected winner relation | Expected count exposed by lead; catastrophic count-call rate; regret of minimum-liability lead vs chosen lead | Forge analyzer over legal lead decisions where partner is bidder and player has lead | P0 | Compute a count-liability surface for every legal lead; include five-count vs ten-count severity | enumeration-ready; oracle-ready; Gus-ready as action tag; Burl-ready |
| `effective_double_highest_remaining` | candidate tile, suit led by candidate, all higher same-suit tiles dead/played/trumped/in hand, trumps outstanding | Recognition precision/recall; regret when model fails to lead effective boss tile | Public play history and hand; oracle decision records | P1 | The chapter's "stretching doubles" idea becomes a virtual-boss detector | enumeration-ready; oracle-ready; Gus-ready; Burl-ready |
| `avoid_disruptive_partner_trump_lead` | current player has lead, partner is bidder, candidate is trump, off-lead count liability, partner likely trump length, bid margin | Trump-lead regret; exception precision when trump lead is least bad | Oracle corpus with counterfactual action Qs | P1 | Do not hard-code "never trump"; model the exception where off leads are worse and partner likely still controls trump | enumeration-ready for candidate class; oracle-ready for exception value; Gus/Burl-ready |
| `bidder_off_vulnerability_support` | bidder prior off leads, trump remaining, bidder has not yet shed offs, partner count in hand, current lead source | Probability partner donation prevents set; set conversion delta when partner captures count early | Full-game rollouts and oracle counterfactuals | P2 | Hidden bidder off shape needs belief or sampled-world context; public proxy can still tag pre-off support windows | oracle-ready; Gus belief-ready; Burl-ready; partially enumeration-ready |
| `count_already_safe_or_dead` | count tiles played, count tiles in current player's hand, declaration/trump suit, candidate lead suit | False-risk and false-safe rates for count-liability detector | Deterministic replay over generated games | P1 | The book examples rely on knowing count is already played, in hand, or made trump | enumeration-ready; Gus-ready; Burl-ready |
| `support_attention_memory` | trick history, highest remaining by suit, live count by suit, partner/bidder role | Error rate on states requiring remembered played tiles; compare early vs late decisions | Concept-bucket eval over generated games; Burl trace audit | P2 | A broad bucket for Chapter 4's warning not to become passive when not bidder | enumeration-ready for labels; oracle/Gus/Burl-ready |

## First Detectors

1. `safe_partner_count_donation`: label every legal count play by whether the bidder/partner
   is guaranteed to win the trick, then report good donation, unsafe donation, and regret.
2. `lead_away_from_count_damage`: for partner-support lead decisions, compute live count
   callable by each legal lead and compare the chosen action to the minimum-liability lead.
3. `low_trump_trap_against_count_dump`: split donation windows where bidder led boss trump
   from low-trump force leads; this should catch high-tail mistakes quickly.
4. `effective_double_highest_remaining`: detect "virtual double" leads when a non-double is
   the highest remaining tile in its suit.
5. `avoid_disruptive_partner_trump_lead`: measure ordinary trump-lead harm and learn the
   narrow exception where all non-trump leads expose worse count.
6. `bidder_off_vulnerability_support`: connect early safe count donation to later bidder
   off-risk, using oracle rollouts or Gus belief buckets.

## Empirical Plan

Enumeration can implement the public-state labels immediately: role regime, legal winner,
guaranteed partner win, live/dead count by suit, highest remaining tile, and trump-lead
candidate class. These labels do not need hidden hands except for "probable" guarantee
strength.

Oracle rollouts are needed for value claims: whether minimum-liability leads actually
reduce set rate, whether donating count now beats saving it, and whether trump-lead
exceptions are real. The right metrics are paired regret, unsafe donation tail loss,
catastrophic count exposure, and contract make/set delta.

Gus analysis should use the labels as concept buckets over the existing policy and belief
heads. Useful reports: regret by donation guarantee strength, belief calibration on who
can overtrump a low trump lead, and whether action-local tags learn live-count exposure.

Burl analysis should inspect traces for source-faithful reasoning. The model should ask or
infer whether partner is guaranteed to win before committing count, should mention live
count liability when choosing a lead, and should avoid generic trump-leading rationales
when helping the bidder.

## Claim Ledger

[[w42-phase4-sequence-handshape-tests]] now gives Chapter 4 direct
public/action-local paired contrasts. It supports count donation only in a
sharply gated closure setting and warns against broad partner count exposure.
Private guarantee and exact low-trump trap claims still need richer state fields.

| claim | source basis | empirical status | next test |
|---|---|---|---|
| Partner support is an active role that can decide whether the bidder makes the bid | Chapter summary rules and team-point framing on pages 37-38 | context-limited support | `t42-br7n.1` finds exact closure count donation strongly positive but earlier count donation weak. |
| Count should be donated only on guaranteed partner-won tricks | Count-donation warning on page 38 | supported as gated timing rule | Closure count donation is `+3.143` Q across 65 pairs; count before closure is only `+0.308` Q with CI crossing zero. |
| Low trump leads can be traps where count donation is bad | Page 38 warning about low trump forcing a higher trump | context-limited / blocker | The proxy is weak and uncertain; exact low-trump trap needs private outstanding-trump state. |
| Leads should minimize live count exposure when partner's bidder cannot be directly helped | Pages 38-40 lead-away-from-count examples | supported directionally | Partner count-liability leads are `-3.425` Q across 526 paired lead decisions. |
| Highest remaining suit tile can substitute for a double | Page 39 four-trey example | underpowered | Detect virtual-boss leads and compare regret when chosen or missed |
| Partner should usually avoid leading trump while helping bidder | Page 40 warning and exception | underpowered | Measure trump-lead regret in partner-support regime and learn exception boundaries |

## Wave 2 Findings (Book Validation v1)

[[w42-bookval-v1-wave2-low-trump-trap]] tested the book's "low trump
into a count-bearing trick is a trap" claim with 257 paired contrasts
on oracle-greedy snapshots. The overall delta is `-1.95`
(CI `[-2.94, -1.07]`), but this aggregate is dominated by positions
where hoarding the dominant trump is broadly correct, not the specific
trap the book warns about. **In the count-bearing subgroup (n=86,
the actual book scenario)**, mean delta is `-0.02` with CI
`[-2.16, +2.15]` - symmetric, no signal. The trap fires in 42.4%
of cases (severe in 7.4%) but is offset by the larger correct-hoard
class. The probe correctly distinguishes the book's narrow claim from
the broader contrast pulled by the detector vocabulary. Status:
`context-limited`.

## Wave 1 Findings (Book Validation v1)

The hidden-threat impact ranker in
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] surfaced a finding
this chapter does not currently formalize:

- **In low-pip declarations, the highest off-suit double substitutes for
  trump as the load-bearing hidden tile**. The single highest-impact
  load-bearing tile in the corpus is `5-5` in a twos declaration
  (impact 56.8) - despite `5-5` not being trump in twos. Partner-support
  thinking in low-pip declarations should therefore weight knowledge of
  the highest off-suit double in the same way it weights trump knowledge.
  The chapter's "virtual boss tile" framing (Page 39 four-trey example)
  generalizes: the virtual boss in twos may be a hidden boss double, not
  a high tile of called suit.

The cross-AI agreement matrix in
[[w42-bookval-v1-wave1-cross-ai-agreement]] also confirms the chapter's
gated stance on count donation: detector hygiene work is needed to keep
`ch05_reckless_count` from over-firing as the partner-support analog (it
overfires on setter seats in 2,300 cases), but the within-pair contrast
between donation-on-closure and donation-before-closure is robust.

## Links

[[winning42-strategy-measurement]] · [[gus-strategy-tags-probe]] · [[gus]] · [[burl]] · [[forge]]
