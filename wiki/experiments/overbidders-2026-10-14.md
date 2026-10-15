---
title: The book's own overbidders — its bidding against the two kinds it names, at power (#163, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Summary

Issue #163, the [[book-authority]]'s next measurement of assumption A6 ([[book-assumptions]]).
[[deviant-field-2026-10-14]] seated the book's bidding against a field that overbids by a fixed
step, and its cost shrank as a null at MDE 0.52. That field overbid by rote. The book names two
overbidders of its own (catalog rows, in our words): one raises a weak hand over an opponent's bid
because he doubts he could set it (S-0155-a, the set-bidding fallacy); the other habitually bids
35 holding offs in the ten-count suits (S-0070-a). The book's answer to both is restraint: pass
and set him (S-0080-c).

*Results follow once every registered batch has run.*

## Design (pre-registered before any measured batch; see `scenario/overbidders.py`)

- **Players.** Every seat plays Zeb 0.3.0. The field (team B) bids by `populations/set-bid.silas`
  (**S**: bid-only-enough, except over an opponent's standing bid it raises up to six points past
  the hand's worth), `populations/ten-count-35.silas` (**T**: 35 on a suit of three with an off
  that exposes a ten-count tile), or the style-free heuristic (**H**). Both populations are held to
  their readings in `silas/tests/bidding.rs`. Team A bids by the heuristic (**H**, the control),
  the book's `risk-budget` (**R**), or plays as the book player (**B**, block c).
- **Arms.** Marks: HS, RS, BS, HT, RT, BT, RH. The book's game: PHS, PRS, PHT, PRT, PRH.
- **Primary.** X_f = (R_f − H_f) − RH for f ∈ {S, T}: the book's bidding against its overbidder,
  relative to the style-free field. **The book predicts X_f > 0.**
- **Secondaries.** C_f = R_f − H_f; the field against the control, H_f (the book predicts
  overbidders set themselves); the book player B_f − H_f; the book's game; mechanism.
- **Size.** 576 game pairs per marks arm (seeds 2601–2648 × 12), 192 per points arm (2701–2716 ×
  12): MDE 0.39 at SD 3.3. 416 batches through `scripts/cpu-slot`, two at a time.
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]].

## Links

[[book-authority]] · [[book-assumptions]] · [[deviant-field-2026-10-14]] ·
[[book-head-to-head-2026-10-13]] · [[silas]] · [[strategy-fusion-fence]]
