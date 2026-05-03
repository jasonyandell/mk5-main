---
title: Winning 42 Ch05 Setter Defense
kind: experiment
first_seen: local-2026-05-01
last_updated: local-2026-05-01
status: active
---

## Summary

This page is the bead-backed work surface for `t42-ni1l.5`: Chapter 5, "How to Set
the Bidder." It harvests setter-defense concepts from Winning 42 into measurable
hypotheses for [[gus]], [[burl]], and [[forge]].

Chapter 5 reframes defense as a positive-EV skill rather than a passive role. The
book's central claim is that the bidder has priced a small number of losses into the
contract, usually one or two offs, and the setters' job is to force losses beyond that
risk budget by capturing tricks and count dominoes
(`scratch/winning42/winning42.with_figures.md` lines 2329-2341). This belongs in the
[[winning42-strategy-measurement]] workstream as a source of concept buckets, not as a
hard-coded playbook: pounce windows, void creation, extra-count pressure, trump-set
recognition, count-protection discards, and position-sensitive trump-ins should become
detectors and eval slices for Gus/Burl.

## Work Surface

The chapter slice is `scratch/winning42/winning42.with_figures.md` lines 2297-2840.
Completion requires source-backed concepts, detector inputs, metrics/tests, likely
data sources, implementation notes, and readiness labels for enumeration, oracle
rollouts, Gus analysis, or Burl traces.

Grounding pages:

- [[winning42-strategy-measurement]] defines the book harvest as a strategy-hypothesis
  catalog and requires chapter pages to emit detector inputs, metrics, data sources,
  implementation notes, and readiness labels.
- [[gus-strategy-tags-probe]] already shows that explicit public-state/action strategy
  tags reduce regret for a tiny Gus-like policy, but that the next useful question is
  concept buckets such as count-dump risk, donation windows, and pounce windows.
- [[gus]] supplies the belief/policy/regret target: Chapter 5 concepts should become
  belief-calibration buckets, action-local tags, and regret slices.
- [[burl]] supplies the reasoning/tool-use target: Chapter 5 concepts should become
  public-evidence questions Burl can ask and thought/tool faithfulness checks, without
  giving it direct E[Q] answers.
- [[forge]] supplies the oracle infrastructure: E[Q] action values, full-game decisions,
  and perfect-information rollouts that can test whether the book's defensive claims
  actually improve expected outcome.

## Source-Backed Claims

- Setting has asymmetric upside: the setters score the bid plus points captured, so set
  hands often score in the 50s or 60s while made bids commonly score only the bid value
  or the low 30s/40s (`scratch/winning42/winning42.with_figures.md` lines 2314-2327).
- The setter's strategic target is the bidder's calculated loss allowance, especially
  the bidder's one or two offs (`scratch/winning42/winning42.with_figures.md` lines
  2329-2341).
- Early trump leads create discard opportunities for defenders with few or no trumps;
  the book recommends using those forced/free plays to create voids that can later
  attack the bidder's off (`scratch/winning42/winning42.with_figures.md` lines
  2361-2388).
- The pounce often arrives on the third or fourth trick, when the bidder finally leads
  an off; defenders should have preserved doubles and count dominoes for that moment
  (`scratch/winning42/winning42.with_figures.md` lines 2390-2398).
- The setter often needs extra count beyond the count the bidder priced into the bid;
  a ten-count on the off trick may set a bidder who only priced a five-count loss
  (`scratch/winning42/winning42.with_figures.md` lines 2409-2441).
- Setter count donation deliberately differs from partner-support donation: when setting,
  a defender should often play count on the bidder's off before knowing whether partner
  wins, because the pounce window may not recur (`scratch/winning42/winning42.with_figures.md`
  lines 2443-2500).
- After defenders win a trick, the book says to keep attacking count by leading doubles,
  count dominoes, or count-calling dominoes (`scratch/winning42/winning42.with_figures.md`
  lines 2502-2513).
- Count-calling leads are conditional: doubles that call five-counts can be good if they
  force the bidder side to contribute count, but low count-calling leads may help the
  bidder shed an off without extra count (`scratch/winning42/winning42.with_figures.md`
  lines 2522-2540).
- Against 35/36 bids, especially overbidders with ten-count offs, the book recommends
  being willing to lead fours, fives, or sixes to attack ten-count exposure
  (`scratch/winning42/winning42.with_figures.md` lines 2561-2599).
- A setter should usually lead only count-calling dominoes after gaining lead, unless
  only one more point is needed to set the bidder (`scratch/winning42/winning42.with_figures.md`
  lines 2601-2616).
- Throwaway choices should protect count dominoes by preserving same-suit protectors
  when possible, preventing the bidder from pulling count with a double-ahead lead
  (`scratch/winning42/winning42.with_figures.md` lines 2641-2672).
- When a defender has many of the bidder's missing trumps, a trump set can flip control:
  the bidder's assumption that missing trumps are spread fails, and the partner should
  recognize the concentration after only one opponent follows suit
  (`scratch/winning42/winning42.with_figures.md` lines 2692-2728).
- In trump-set positions, the trump-rich setter's policy depends on position and trick
  content: trump in on likely count if acting early, hold the trump if last to act and
  no count is present, or lead high trump when remaining trumps are strong enough to
  strip bidder control (`scratch/winning42/winning42.with_figures.md` lines 2730-2824).

## Concept Table

| Concept | Detector / state inputs | Metric / test | Likely data source | Priority | Readiness | Implementation notes |
|---|---|---|---|---|---|---|
| Setter role and bid risk budget | Seat relation to bidder, bid value, declaration, points already won/lost, bidder team point allowance | Defensive action regret by `points_needed_to_set`; set conversion by deficit bucket | Generated full games, E[Q] decision records | P0 | enumeration, oracle, Gus, Burl | Base tag for every Ch05 slice. Compute `bidder_allowed_loss = 42 - bid` for ordinary point contracts, then track current loss and remaining count. |
| Pounce window on bidder off | Bidder leads a non-trump/off-suit candidate after early trump/double sequence; defender void or has winning double; live count in defender hand | Missed pounce rate; E[Q] delta for count play vs safe discard; held-count-never-used rate | Forge decision records plus oracle action values | P0 | oracle, Gus, Burl | Requires identifying bidder offs approximately from public play and known hand for training labels; public detector can use "bidder leads non-trump after pulling trumps." |
| Deliberate void creation | Defender has no trump on early trump lead or can slough on bidder double; candidate throwaway removes last tile of a suit | Later pounce probability, count-capture delta, regret of void-creating discard vs alternative discard | Generated games with full hands; counterfactual oracle rollouts | P0 | enumeration, oracle, Gus | Enumeration can count hand-shape opportunity frequency; oracle rollouts needed for action value of each discard. |
| Extra-count-to-set | Bidder expected count loss on off trick versus additional count needed for set; count played by other defender | Set conversion when extra count is donated; false-positive count dump rate | Forge game logs, action values, full trick records | P0 | enumeration, oracle, Gus, Burl | Directly mirrors the book's "bid priced five, ten-count sets" claim. Needs live count and current bid-margin accounting. |
| Count-before-certainty donation | Defender acts before partner on bidder off, cannot follow suit, has count, partner winner unknown | Regret distribution of count play vs non-count play; outcome split by actual partner ownership of winner | Oracle paired action values; public trace for Burl thought audit | P0 | oracle, Gus, Burl | This is the sharpest Ch04/Ch05 contrast: helping bidder waits for guaranteed trick; setting often risks count now. |
| Count-calling defensive lead | Setter has lead after winning trick; candidate lead is double/count/calls five-or-ten count; current points needed to set | E[Q] of count-calling leads vs non-count leads; set probability by lead class | Forge decision records and rollouts | P1 | enumeration, oracle, Burl | Must distinguish double leads that call count from low leads that call count but cannot win. |
| Low count-caller trap | Candidate low four/five/six lead calls count but depends on partner holding high tile; bidder/partner may hold doubles | Tail regret and bidder escape rate for low count-calling lead, stratified by bid >=35 | Oracle action values, full-hand labels | P1 | oracle, Gus, Burl | The book warns that low count-calling leads can help the bidder shed an off unless targeting known overbid ten-count exposure. |
| Overbid ten-count attack | Bid 35/36; inferred or actual bidder has ten-count off; defender can lead four/five/six | Set rate and regret of ten-count attack line by bid bucket and hand archetype | Oracle rollouts; later style-conditioned corpora | P1 | oracle, Gus, Burl | This is partly style/opponent modeling. Start with full-hand labels, later convert to public inference. |
| One-point-to-set exception | Defenders have already captured enough that any trick point sets bidder; candidate lead is non-count double | Regret of ordinary count-calling rule vs exception; set completion rate | Generated games; simple state accounting | P1 | enumeration, oracle, Burl | Cheap deterministic detector once points captured and bid threshold are known. Good contrast-pair candidate. |
| Count-protection throwaway | Defender has count tile plus same-suit protector(s); free discard choice could expose count to bidder double-ahead lead | Future forced-count loss rate; regret of discarding protector; count survival over tricks | Full-hand generated games, oracle action values | P1 | enumeration, oracle, Gus | Fits strategy tags as action-local `protects_count` and global `count_liability`. Needs suit/pip mapping under current trump. |
| Lower-of-two non-count discard | Defender has two non-count tiles in a suit during follow/free discard; can preserve higher future winner | Later trick win/capture probability; regret of low vs high discard | Generated games, oracle action values | P2 | enumeration, oracle, Gus | Lower priority but easy to compute as a local discard-order feature. |
| Trump-set recognition | Bidder leads trump; only one opponent follows; full/estimated missing trump concentration on that opponent | Recognition precision/recall; set conversion; partner count donation after signal | Full-hand labels, public sequence, Gus belief outputs | P0 | enumeration, oracle, Gus, Burl | Public signal is strong and early. Gus can be evaluated on hidden trump ownership after the first trump reveal. |
| Trump-rich setter policy | Defender holds multiple bidder trumps; current trick has count or likely count; defender position before/after partner; relative trump rank | Regret of trump-in vs hold vs high-trump lead; count captured; bidder crisis escape rate | Oracle action values; concept-sliced Burl traces | P0 | oracle, Gus, Burl | Split into scenario A bidder wins first trump trick and scenario B defender wins it. Position sensitivity is central. |
| Partner recognition of trump-rich setter | Partner sees only one opponent follow bidder trump; partner has count or count-calling lead opportunity | Partner donation/tool-call faithfulness; missed support rate after public signal | Generated games; Burl traces when available | P1 | Gus, Burl, oracle | Tests whether the non-trump-rich setter reacts to the same public evidence the book says everyone should notice. |

## Highest-Value First Detectors

1. `setter_pounce_window`: bidder-side non-trump/off lead, defender cannot follow or has
   winning double, live count available, bid margin says extra count matters. This is the
   cleanest Ch05 eval bucket because the book explicitly says the chance is rare and must
   be taken.
2. `count_before_certainty`: a special pounce subcase where the defender acts before the
   partner and cannot know whether partner will win. It should produce high-value contrast
   pairs against the Ch04 "only donate on guaranteed partner trick" rule.
3. `trump_set_recognition`: public early-trump evidence that one opponent holds the
   bidder's missing trumps. This is excellent for Gus belief calibration and for Burl
   trace faithfulness because the signal is public and strategically loud.
4. `trump_rich_setter_policy`: when the defender owns the bidder's missing trumps, evaluate
   whether to trump in, hold, lead high trump, or attack count first based on position,
   rank strength, and count on the trick.
5. `count_protection_throwaway`: detect whether a free discard protects or exposes a count
   domino against a double-ahead lead. This should be cheap to compute and likely useful
   as an action-local strategy tag.
6. `count_calling_lead_quality`: after defenders gain lead, compare doubles/count-calling
   leads with non-count leads and with low count-caller traps, with a one-point-to-set
   exception bucket.

## Readiness Notes

- Enumeration-ready: opportunity frequencies for pounce windows, void-creation chances,
  count-protection shapes, one-point-to-set exceptions, missing-trump concentration, and
  bid/point threshold states.
- Oracle-ready: action regret for count-before-certainty, count-calling leads, low
  count-caller traps, trump-in versus hold, high-trump continuation, and protector
  discard choices.
- Gus-ready: belief calibration for hidden winner ownership on bidder offs, hidden trump
  concentration after early trump reveals, and action-local features such as
  `creates_void`, `protects_count`, `donates_extra_count`, `spends_trump_on_count`, and
  `attacks_bid_margin`.
- Burl-ready: thought/tool faithfulness checks where the model should mention public
  evidence: bidder risk budget, pounce chance scarcity, whether the trick has count,
  partner position, only-one-opponent-followed-trump, and one-point-to-set exceptions.
  Burl should not receive direct best-move or E[Q] tools; it should reason from engine
  state, legal moves, trump declarations, unseen tiles, void audits, and outcome
  distributions consistent with its current tool contract.

## Claim Ledger

[[w42-phase4-sequence-handshape-tests]] now supplies direct paired contrasts for
several Chapter 5 rows over the 75079-action table. It supports count-calling
leads, immediate pounce count-taking, and the warning against reckless count
into bidder control. Exact private void-creation, count-protection, and
trump-rich recognition remain blocked by missing state fields.

| Claim | Source | Status | Next empirical check |
|---|---|---|---|
| Setter upside justifies aggressive defensive pressure because sets score bid plus captured points. | lines 2314-2327 | supported directionally by scoring/accounting substrate | Phase 4 verifies scoring mechanics and setter pounce value, but full point EV of successful sets by bid context remains a follow-up. |
| Bidder usually exposes one or two off windows, and defenders should concentrate count pressure there. | lines 2329-2341 | context-limited / bounded | Phase 4 supports pounce/count pressure in reached row-local states; exact bidder off-window concentration needs richer private hand/state fields. |
| Creating voids during early trump/double leads improves later pounce chances. | lines 2361-2398 | blocked by missing causal state injection | The row table cannot attribute later pounce windows to earlier void-creating choices; needs paired multi-trick state injection. |
| Playing count before partner certainty on bidder off is often correct when setting. | lines 2443-2500 | context-limited support | `t42-br7n.1` finds pounce take-count-now `+2.681` Q, while broader before-certainty pressure is weak/uncertain. |
| Count-calling defensive leads are preferred after winning lead, except for low-lead traps and one-point-to-set cases. | lines 2502-2616 | supported directionally | Setter count-calling/count leads are `+3.028` Q across 1866 paired lead decisions. |
| Protecting count dominoes with same-suit throwaways reduces bidder ability to pull count with double-ahead leads. | lines 2641-2672 | blocked by missing state fields | Current row table cannot test count-protection throwaways or future forced-count loss. |
| Only-one-opponent-followed-trump is a strong public trump-set signal requiring partner support. | lines 2692-2728 | blocked by missing belief/state fields | Needs hidden-trump belief update and partner donation decisions after the signal; not available in the current row table. |
| Trump-rich setters should choose between count attack, trump-in, hold, and high-trump lead based on rank strength, count on trick, and seat position. | lines 2730-2824 | blocked by missing state fields | Current row table lacks exact trump-rich recognition and private remaining-trump state. |

## Implementation Hooks

- Extend `strategy_tags` with role/regime labels:
  `role_regime=setter`, `points_needed_to_set`, `bidder_loss_allowance`,
  `bidder_off_window`, `setter_pounce_window`, `trump_set_signal`, and
  `trump_rich_defender`.
- Extend action-local tags with:
  `creates_void`, `breaks_void_plan`, `donates_count_before_certainty`,
  `protects_count_tile`, `exposes_count_tile`, `calls_count`, `low_count_caller`,
  `one_point_to_set_exception`, `trumps_in_on_count`, and `holds_trump_no_count`.
- Report concept buckets before aggregate scores:
  paired regret, tail regret, near-tie rate, set conversion, held-count-never-used rate,
  belief Brier/log-loss for key hidden owners, and Burl forced-commit/tool-faithfulness
  errors by Ch05 bucket.

## Wave 2 Findings (Book Validation v1)

[[w42-bookval-v1-wave2-pounce-window-bid30]] — paired-contrast probe on
52 oracle-greedy snapshots (filtered from 500 by 1-legal-move and
setter-led-trick exclusions). The book's setter-pounce instruction is
right under `p_make` (oracle pounces 59.6%) but **wrong under scalar EV**
(decline better in 65.4%). 10-point count cases show EV delta `+15.68`
with CI `[+1.60, +29.76]` (decline strictly better) yet oracle still
pounces 80% of the time. Status: `context-limited` for the bid=30 slice;
the high-bid scope (`ch12-setter-pounce-high-bid-off`) waits for Wave
2.B.2's bid-aware corpus (bead `t42-8kbh`). The chapter's pounce
recommendation encodes implicit `p_make` reasoning at the contract
threshold; under tail-aware utilities (CVaR, robust_q25) it would
recommend a different action substantially more often.

[[w42-bookval-v1-wave2-void-creation]] — paired-contrast probe on 276
oracle-greedy snapshots. Status: **`contradicted`** for the
**lead-to-self-void** slice at bid=30. Both primary metrics have CIs
excluding zero in the direction OPPOSITE to the book claim:
`p_set` delta `-0.0155` (CI `[-0.028, -0.003]`), EV delta `-2.63`
(CI `[-3.42, -1.84]`). Voiding by leading a singleton off-suit makes
the setter's situation worse, consistently across phase and count-exposure
subgroups. **Critical scope caveat**: this probe captures setter in
a leading position, not the book's canonical scenario where setter is
following a non-trump trick and chooses to discard their last tile
from a held suit. The broader Ch 05 void-creation framing remains
untested; bead `t42-z31l` will mine follow-position candidates and
re-run.

This is the campaign's first contradicted-status finding. The narrow
scope matters: a future ledger reconciliation should NOT mark the
broader Ch 05 void-creation concept as contradicted — only the
lead-to-self-void sub-scenario.

## Wave 1 Findings (Book Validation v1)

The cross-AI agreement matrix in [[w42-bookval-v1-wave1-cross-ai-agreement]]
and the hidden-threat impact ranker in
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] both turned the
setter-seat surface from "broadly important" to "concretely structured":

- **Setters dominate the divisive-decision pile**: 83 of the 100 most-divisive
  decisions in the spot-check pack are setter seats (45 left, 38 right).
  Detectors, the row model, and oracle EV most often disagree with each
  other in setter-seat positions.
- **Setter seats are asymmetric**: right-setter load-bearing hidden tiles
  are ~30% trump doubles; left-setter is ~37% plain tiles. The book's Ch 5
  treatment is symmetric; the data is not. A future detector pass should
  emit a `seat_pos` feature and not rely on `role_regime=setter` alone.
- **`ch05_reckless_count_to_bidder_control` overfires**: mean regret 9.18
  over 2,300 fires; the worst three cases all sit on trick-0 setter
  closures with 38-43 EV-point regret. The detector cannot distinguish
  the qualifying sub-condition from the broader regime. Refine via bead
  `t42-v0m5`.
- **`ch05_setter_pressure_regime` is a regime label, not an action label**:
  fires on all setter-regime candidates simultaneously. Reclassify as a
  regime feature; consider a sibling action-level label for specific
  pounces. Tracked as `t42-btpg`.

These findings do not retire any Ch 5 source-backed claim. They sharpen
the implementation hooks: setter-defense work needs seat-aware features,
gated reckless-count detection, and a regime-vs-action separation in the
detector schema.
