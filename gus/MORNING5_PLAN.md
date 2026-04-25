# Gus — Morning 5 Plan: Roll-forward belief + attention trajectory

**Date**: 2026-04-22 (end of session)
**Status**: Plan, not yet built. Handoff for next session.

---

## TL;DR

Reframe: **the reasoning is the product, not the play.** Build a 2-day spike
that rolls Gus's belief head and attention forward through a single game,
decision by decision, and renders the resulting trajectory as a readable
timeline. The question is not "does Gus play well" (answered: 0.551 Q-pt
regret, §20) — it's "does Gus's forward-pass trajectory, rendered legibly,
look like the reasoning we care about?"

No new training. No new data. Uses `v3_consistency_10000g.pt` + existing
corpus. Minimum viable deliverable on day 2.

---

## Why this — the reframe that resolved the ranking loop

Five days of work landed a distilled player that plays competently but not
strongly (76% bot-match, 0.551 regret). The MORNING4 session then spent
a day-plus on LAMIR-1 look-ahead and concluded distilled V/Q can't serve as
look-ahead leaves (§20). Post-LAMIR, several "what now?" options surfaced —
PPO self-play, LAMIR-done-right, meta-strategy head, detect-and-route.
Repeated stress-testing across 5+ rounds of check-in kept deflating each
of them.

The drift source, named explicitly by the user: **the project's real goal
was always the reasoning, not the bot strength.** OVERVIEW's kickoff said
"neural as player, LLM as narrator" — but what that actually names is a
system whose *output* is legible reasoning about hidden information,
*not* a strong action selector.

User quote: *"It's a lot to take in. Who has what, how did that change,
what does that mean, updated domino upon domino until the play. Ultimately
it's the reasoning I care about."*

That reframe decides everything downstream:

- **Inference, not search** (user-stated preference). Reasoning happens in
  the forward pass and is read from internals (attention, belief), not
  from tree exploration.
- **Belief trajectory is the spine, attention is the legible reasoning,
  V is summary.** §19 probe 2 already showed Gus's attention evolves
  layer-by-layer through a coherent sequence (DECL anchor → survey high
  non-trump → risk reassessment → commit). That's reasoning, visible in
  activations, we just never wired up its extraction as the primary
  output.
- **§21 unlocks it.** Gus's belief head is at the Bayes ceiling on our
  corpus (39.184%). That means **the belief trajectory Gus produces IS
  the correct one given available information.** We are not building a
  new reasoner — we are extracting reasoning that Gus already does in
  a forward pass.

---

## What this resolves

Outstanding issues from MORNING4_STATUS and the stress-test conversation:

| issue | resolution |
|---|---|
| "PPO or LAMIR next?" | Neither as first move. Both train action selection / planning; neither produces legible per-decision reasoning. Reclassified as downstream-depth, not first-move. |
| LAMIR-1 ceiling (§20) | No longer blocking. Look-ahead matters only if bot strength is the goal. Not the goal. |
| qMAE plateau (§18) | Q_head is a summary statistic, not legible reasoning. Its noise is only load-bearing if you use Q for planning. Not first-move-blocking. |
| Belief at Bayes ceiling (§21) | This is the enabling fact, not a dead end. A correctly-calibrated belief updater is exactly the base object we need. |
| LEM-as-narrator data problem | The narration target isn't "explain this action" (no natural corpus). It's "describe this belief+attention update object." Templated narration on structured objects is tractable; data problem becomes "write 5-10 reasoning-pattern templates," not "generate text from nothing." |
| Drift from kickoff | Named. OVERVIEW.md will be updated in a follow-up to reflect reasoning-as-product framing. |

---

## What we are NOT doing (and why)

- **PPO self-play**: trains action selection. Zero contribution to reasoning-
  as-product. Also, Zeb already demonstrates self-play works on 42 at
  ~1M games. PPO on top of distilled Gus would be skill-build, not research
  progress. Reclassify: "implement once for toolkit fluency, when there's
  time."
- **LAMIR done right**: trains planning via CFR+. Could eventually inform
  *deeper* narration ("I considered the line where R-opp has 6-6 and
  rejected it because..."), but the planning machinery doesn't produce the
  per-decision reasoning object we want first. Downstream, not first.
  Also, 4-player partnership may not port cleanly from the paper's 2-player
  setting — scoping risk unresolved.
- **Meta-strategy head (§22)**: adds more action selectors. Ruled out as
  first move; *adjacent* and useful later — the selector's reason ("hedging
  because belief is flat and outcome-variance is high") IS narration,
  just conditioned on an action category rather than on a belief shift.
- **Q-head multi-world variance reg (§18)**: a plausible fix for Q noise,
  but Q is summary not reasoning. Not first-move-blocking.
- **Exploitability tool**: standalone infrastructure, useful for any
  future grading, not reasoning-relevant directly. Build when there is
  a specific thing to grade.

None of these are *bad*. They're just not the first move, given the goal
is reasoning-as-product.

---

## The spike — minimum viable roll-forward

Total effort: ~2 days for a readable trajectory on a single game.

### Day 1 — extraction

Script: `gus/eval/roll_forward.py` (new).

For one game from `gus/data/corpus_eval_20.pt`:

1. Load `v3_consistency_10000g.pt` adapter. (§6-9 receipts; loads via
   existing `student.py` machinery.)
2. For each of the game's 28 decisions:
   a. Build the state for the seat-to-play using existing tokenizer
      (`gus/model/tokenize.py`).
   b. Forward pass through `StudentTransformerFullVoids`, **capturing**:
      - `belief_head` output: per-domino `[3]` posterior over {L, partner, R}
      - `v_head` output: scalar E[Q] estimate
      - attention weights at all encoder layers (use hooks; pattern from
        §19 probe 2)
      - `pi_me` output: softmax over legal actions
   c. Record: `(decision_idx, public_event_at_t, belief_t, V_t,
      attention_per_layer_t, policy_t, action_taken_t, oracle_truth_t)`.
3. Serialize the full trajectory as JSON + a companion numpy `.npz` for the
   attention tensors.

Key implementation notes:
- Attention hook pattern: register `forward_hook` on each
  `TransformerEncoderLayer.self_attn` module; capture `attn_output_weights`
  from the forward call.
- Use the same seat-symmetric tokenizer as training (§19 symmetry-checker
  verified) — seat gets encoded into the state, don't double-rotate.
- Join with oracle truth from `world_hands` field in the corpus for
  comparison.

### Day 2 — render

Script: `gus/eval/render_trajectory.py` (new). Renders to `scratch/trajectory_game_N.html`.

Per-decision panel (one row per decision, 28 rows):

- **Left column — public event**: "P0 led 1-1", "P1 played 2-1 (offsuit,
  VOID on 1s)", etc. Derivable from the game's play history.
- **Middle-left — belief distribution**: per-unseen-domino bar chart.
  3 bars per domino (L / partner / R). Color-shift across rows shows
  which posteriors moved between t-1 and t. Rank-annotate the top-3
  shifted dominoes with their KL.
- **Middle-right — attention, final layer**: bar chart over state tokens
  (DECL, MINE[0..6], history). Shows where CLS attention is at this
  decision. §19 showed this pattern is legible.
- **Right — value + action**: V before/after, π_me softmax over legal
  actions (top-3), action taken, oracle argmax for reference.

Render as a single HTML page with one row per decision, readable top-to-
bottom as the hand progresses. The point of this rendering is **not**
polish — it's to let you look at a full hand's worth of Gus's
forward-pass reasoning in one place and ask: *"does this structure
look like the reasoning I care about?"*

### Deliverable shape

End of day 2:

- `gus/eval/roll_forward.py` + `gus/eval/render_trajectory.py` checked in.
- 5-10 rendered trajectories under `scratch/trajectory_*.html` (held-out
  games from the eval corpus — include one ordinary hand, one nightmare
  hand like the §19 seed 900000 position, one endgame-heavy hand).
- Brief notes in a new receipt: `gus/PRACTICALITIES.md §23` — "what the
  trajectories look like, and what's surprising."

---

## Decision point — what to do based on what we see

**If the trajectories read like reasoning** (belief shifts map cleanly to
public events, attention patterns track the decision being made, the timeline
is legibly interpretable):

→ Invest in the templated narration pipeline. 5-10 reasoning-pattern templates
(void inference, count inference, trump-out, signaling, endgame forcing),
pattern matchers, rendering. Week-scale. Produces English narration per
decision that describes the belief+attention trajectory. LEM as narrator
becomes a fine-tune on top: stylize template output into fluent text.

→ Causal attribution pass (§19-style ablations: "what would belief be if
R-opp's void hadn't happened?") to strengthen the *why* of belief shifts.
~2-3 days.

→ OVERVIEW.md update to declare reasoning-as-product as the explicit goal,
demote action-selection improvements (PPO, LAMIR, meta-strategy) to
"possibly useful later."

**If the trajectories read poorly** (belief shifts don't track events,
attention is diffuse or incoherent, the timeline doesn't form a
narratable structure):

→ Specific learning: Gus's forward-pass reasoning has limits we didn't see
from aggregate metrics. Could point to:
- Belief head trained on truth-target doesn't produce human-interpretable
  trajectories (the §15 distribution-target belief head might).
- Attention is mostly "shortcut" encoding, not reasoning (a known failure
  mode in transformers).
- Reasoning is smeared across layers in a way that doesn't narrate
  cleanly, and we need a different architectural choice (e.g., Chain-of-
  Thought-style inline reasoning tokens, or a separate rationale decoder).

Any of those redirect the project concretely. The spike is informative
either way.

---

## Pointers to existing infrastructure

| thing | path | why it matters |
|---|---|---|
| Best adapter | `gus/adapters/v3_consistency_10000g.pt` | Base model to roll forward |
| Architecture | `gus/model/student.py` → `StudentTransformerFullVoids` | Has belief + V + π_me + Q + world_encoder heads |
| Tokenizer | `gus/model/tokenize.py` | Seat-symmetric; §19 verified |
| Eval corpus | `gus/data/corpus_eval_20.pt` | 20 games, 560 decisions, includes oracle truth |
| Attention-probe precedent | §19 probe 2 in `gus/PRACTICALITIES.md` | Pattern for extracting per-layer attention weights |
| Belief-ceiling diagnostic | `gus/eval/belief_ceiling.py` | Grounds "belief is at Bayes ceiling" claim; use to re-verify on any corpus touched |
| Joint-world tensor | in each corpus file (`world_hands`, `q_per_world`) | Ground-truth posteriors for checking belief shifts |

---

## Why this plan beats the alternatives

Six rounds of "check in on each" kept deflating every ambitious option and
kept re-surfacing the same small concrete items. This plan is the smallest
move that:

1. Directly addresses the user's stated goal (reasoning-as-product).
2. Uses only existing artifacts (no new training, no new data).
3. Produces a readable deliverable in 2 days.
4. Has informative outcomes in both directions (validates the reframe or
   redirects it with specific learning).
5. Sets up the downstream investment (templated narration + LEM-narrator)
   if validated.

The previous planning rounds were skipping this because it's small — it
doesn't feel like "moving the project forward." But moving the project
forward by committing weeks or months to an unscoped plan is how the
MORNING4 LAMIR detour happened. A 2-day validation before the next big
investment is the correct shape.

---

## Open questions (flag, don't block)

- Is Gus's forward-pass reasoning richer than attention + belief? If yes,
  the rendering will undersell it. Possible additions: per-layer state
  embedding trajectory (where does the CLS embedding move through the
  hidden-state space over layers?). Probe-level, out of scope for the
  2-day spike, possibly useful in the "invest further" follow-up.
- Do we need counterfactual narration ("if partner had X...") in the
  reasoning? If yes, the spike's inference-only approach needs extending
  to light ablative passes per decision. Out of scope for day 1-2; can
  layer on later.
- Does the trajectory shape generalize across hand types (ordinary /
  nightmare / endgame-heavy)? The 5-10 rendered trajectories should
  sample across types to surface this early.

---

## Commit trail expected from this spike

1. `feat(gus/eval): roll_forward — extract belief + attention trajectory`
2. `feat(gus/eval): render_trajectory — readable per-decision HTML timeline`
3. `docs(gus): PRACTICALITIES §23 — trajectory rendering receipts on N games`
4. (conditional on validation) `docs(gus): OVERVIEW — reasoning-as-product reframe`

No training commits. No adapter changes. No corpus changes. Pure
extraction + rendering on existing artifacts.
