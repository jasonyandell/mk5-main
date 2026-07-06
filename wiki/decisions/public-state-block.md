---
title: Post-Trick Public State Block
kind: decision
first_seen: 7f1994e
last_updated: 7f1994e
status: active
---

## Decision

After every trick, [[narration]] prompts include a structured public-state block showing:
- All dominoes played so far (N/28)
- Count domino status (taken by which team, or still out)
- Narrator's remaining hand

Cost: ~60 tokens per trick, ~300 extra tokens per prompt.

## Why

> A 2B model shouldn't reconstruct game state from prose any more than a human should memorize 28 dominoes. In real 42, this information is public and visible at the table. Some variants even exploit the difficulty of tracking it (stacking on 84 bids).

— commit message, [[sources/7f1994e]]

## Design philosophy

State that is public and available at the table is the narrator's job to provide, not the model's job to reconstruct. Model capacity should be spent on strategy, not bookkeeping. The ~300-token overhead per prompt is considered a "tiny cost for a massive reduction in the bookkeeping burden on the model."

## Relationship to prior decisions

**Complements [[scratchpad-validation]]:** scratchpad validation held the model accountable for reconstructing state, which proved too strict without format bootstrapping (see [[experiments/scratchpad-v2-iter0]]). The public state block removes the reconstruction problem entirely — state is given, not inferred. When [[scratchpad-validation]] is eventually re-enabled, the HAND section of the scratchpad becomes a copy task rather than a recall task.

**Relates to [[learned-by-playing]]:** learning from play is most efficient when model attention can focus on strategy rather than state-tracking. Providing public state in the prompt aligns with how humans actually play the game at the table.

## What this does NOT address

Private state (opponents' hands, inferred from bidding and play history) and strategic reasoning remain the model's responsibility. The state block covers only information that would be visible to any player at a real 42 table.

## Related pages

[[narration]] · [[lem]] · [[learned-by-playing]] · [[scratchpad-validation]] · [[experiments/scratchpad-v2-iter0]] · [[sources/7f1994e]]
