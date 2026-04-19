# Rules-as-tools — design + scaffold (T1)

Status: design + scaffold landed for iter-2 primer replacement. Not wired.

## Wiring sketch (for later integration, not this task)

- **Register** in `burl/harness/agent_runner.py::build_tool_registry`: add four
  entries (`count_dominoes_remaining`, `trick_winner_if`, `what_beats_what`,
  `contract_progress`) alongside the engine tools; imports go to
  `from burl.tools import rules as rule_tools`. The tools already match the
  `ToolProtocol.__call__(game_state, **kwargs)` contract — shim with `_Tool`.
- **Publish schemas** in `burl/harness/agent_runner_native.py::TOOL_SCHEMAS`:
  append four JSON-schema dicts (see tool signatures below) so the native chat
  template renders them into Gemma's `<|tool>…<tool|>` block.
- **Swap primer** in the same file: replace `_TRIMMED_PRIMER` (~400 words) +
  the “how trump works / led suit / following / winning / count / partners
  communicate” prose with the ~650-byte *How to ask* preamble drafted below.
  Keep `_render_42_framing` verbatim — iter-1 evidence says it is net
  positive (`SPIKE_REPORT.md` §Layer 1, §iter-1).

## Why rules-as-tools (context from SPIKE_REPORT)

iter-0 trained on Layer-1 traces and regressed (60% bot-match vs 88.9% base
spike v2). iter-1 trimmed the primer and recovered quality on the decisions
that completed (80% on 5/10) but broke commit discipline on the rest (retry
exhausted 5/10). The diagnosis was that the primer was doing two jobs at
once: *teaching rules* (net negative at 2 B scale — long prose eats
attention, suppresses `eq_outcome_distribution`) and *signalling commit*
(net positive — tells Gemma “you have enough, pick a play”).

Rules-as-tools separates those jobs. Rules become on-demand answers — a tool
call returns a structured fact instead of a paragraph the model has to keep
re-reading. Commit signalling moves to a one-line protocol directive in the
compact preamble. The 42-framing block stays as the pre-parsed scaffold that
worked in iter-1.

## Primer inventory (1,549 words → where each section goes)

| Primer section | Words | Where it goes in iter-2 |
|---|---|---|
| Players and equipment (28 dominoes, 4 players, 7 each) | 45 | **Drop.** Implied by the hand/decl/seat fields in the user prompt; every Burl decision already sees its own hand. |
| Suits (pip membership, double-is-high) | 140 | **Drop.** `is_trump` + `trump_declared` already answer trump-membership; non-trump suit membership is implicit in `can_follow` via `is_legal`. |
| Bidding abbreviation (bid is already resolved) | 85 | **Drop** from prose; **move to `contract_progress()`** — bidder seat, bid target, bidder team. |
| How trump reshapes the deck (3 trump kinds) | 130 | **Keep as one line of preamble** (“declarations are pip-suit, doubles-trump, or notrump”); structural facts move to `what_beats_what()` via `trick_rank`. |
| Led suit rule (trump-if-trump, higher-pip otherwise) | 110 | **Move to `trick_winner_if()` / `what_beats_what()`** — both return `led_suit` so the model sees the rule's *output* for this state instead of reading the rule. |
| Following suit (must-follow-if-you-can) | 70 | **Keep in `is_legal()` reason string** (already there: “must follow suit X”). Drop from prose. |
| Winning the trick (highest trump, else highest of led) | 80 | **Move to `what_beats_what()`, `trick_winner_if()`.** These literally simulate the rule. |
| Count dominoes (5-5, 6-4 = 10pt; 5-0, 4-1, 3-2 = 5pt) | 90 | **Move to `count_dominoes_remaining()`** — returns live 5-pt / 10-pt list with ids + pip labels. |
| Scoring / contract math (35 + 7 = 42; make vs. set) | 140 | **Move to `contract_progress()`** — captured vs target, margin, tricks remaining, status tag. |
| Communication (no signalling) | 25 | **Drop.** Doesn't change behaviour; if anything the model hallucinated partner-reasoning anyway. |
| Facts-this-primer-commits-to (encyclopedic tail) | 480 | **Drop entirely.** Restates the above in bullet form; Gemma re-reads it per call with no new information. This is the single biggest win in token count. |

Net: every load-bearing fact from the primer has a tool home. The two
explicit “keep” items (trump families in one line; “must follow” reason
string) already live outside the primer in iter-2 draft.

## Target replacement prompt (draft — 645 bytes)

```text
# Texas 42 — how to ask

You have rule-answering tools. Prefer calling them over recalling rules.

Engine:
  is_legal(d), is_trump(d), unseen(), void_audit(seat, suit), trump_declared()

Rules:
  what_beats_what(a, b)        — which domino wins under the current lead
  trick_winner_if(d)           — simulate playing d into the current trick
  count_dominoes_remaining()   — live 5-pt and 10-pt dominoes
  contract_progress()          — captured vs bid; offense/defense; margin

Outcome distributions:
  eq_outcome_distribution(play, n_samples=10)
  conditional_outcome(play, assumption, n_samples=10)

Three declaration families exist: pip-suit trump, doubles-trump, notrump.
When done reasoning, call commit_play(domino_id) with an integer from your
hand. If the engine rejects as illegal you get another turn with the
rejection shown and may commit_play again.
```

At 645 bytes (∼120 tokens), this is roughly **1/8** of the 4.2 KB the full
LEM primer contributed and ∼**1/3** of the 1.9 KB trimmed primer. The
42-framing block (∼1.5 KB) stays unchanged, so the total system prompt
shrinks from ∼6–7 KB to ∼2.1 KB.

## Tool signatures (scaffolded in `burl/tools/rules.py`)

All four accept a duck-typed `game_state` identical to the contract already
documented in `burl/tools/engine.py` (required fields: `decl_id`, `hands`,
`played`, `play_history`, `current_trick`, `trick_leader` or `leader`, and
optionally `bidder`, `bid_state`, `team_points`). All return JSON-serialisable
primitives only (ints/strs/bools/lists of dicts), matching `engine.game_summary`'s
contract so the harness observation formatter round-trips cleanly.

### `count_dominoes_remaining(game_state) -> dict`

- `unplayed_5pt`: list of `{domino_id, pip_label, count_value}` (3 entries at
  start of hand)
- `unplayed_10pt`: same shape (2 entries at start)
- `played_5pt`, `played_10pt`: same shape, already captured
- `loose_count_points`: `int`, sum of count values still in live hands
- `my_team_captured`, `opp_team_captured`: `int`, read from `team_points`
  (0 when absent)

Thin wrapper over `forge.oracle.tables.DOMINO_COUNT_POINTS` and the
game-state `.played` set.

### `trick_winner_if(game_state, domino_id) -> dict`

If my current play would complete the trick (`position_in_trick == 4`),
simulate the full resolution via `forge.oracle.tables.resolve_trick` and
report the winner seat (absolute + role), whether my team takes the trick,
and the points at stake. If the trick would remain partial, report the
currently-leading play (after mine), the led suit, and how many plays
remain. If I'm leading a fresh trick, report the led suit my lead would set.

Rationale: the primer's longest sections (led-suit rule, winning-the-trick
rule, count-of-this-trick) collapse to one look-ahead tool call. Gemma
already wants to know “if I play 21, who takes it?” — this answers exactly
that, with the rule applied, rather than asking Gemma to apply it.

### `what_beats_what(game_state, domino_a, domino_b, lead_domino=None) -> dict`

Comparison under current declaration. If a trick is in progress, `lead_domino`
defaults to that trick's lead; else callers must supply one explicitly. Returns
`winner ∈ {"a","b","neither"}`, a one-line `reason`, and full
`{rank, is_trump, can_follow}` breakdown for each of A and B.

Thin wrapper over `trick_rank` + `can_follow` + `is_in_called_suit`.

Rationale: takes the *symbolic* rule (“highest trump wins, else highest of
led suit, else off-suit”) and returns its *applied* answer for the two plays
the model is actually choosing between. Removes a whole class of
rules-comprehension misfires — e.g. the iter-0 “is 6-6 trump under twos?”
error documented in `SPIKE_REPORT.md` Phase 4.

### `contract_progress(game_state) -> dict`

Captured vs target; `my_team_role ∈ {"offense","defense","unknown"}`;
`bid_target`, `offense_captured`, `offense_count_still_needed`,
`defense_points_to_set_bid`; `loose_count_points`, `trick_points_remaining`,
`tricks_completed`, `tricks_remaining`, `hand_points_remaining`;
`status ∈ {"contract_in_play","offense_has_made_bid","defense_has_set_bid","bidder_unknown"}`.

Uses `team_points`, `bidder`, and `bid_state.high_bid` with tolerant defaults
(the field is sometimes absent during early scaffolding states — the tool
reports `"bidder_unknown"` and a synthesised `bid_target=30` floor instead
of raising).

Rationale: the scoring + contract section of the primer is where the model
has to do arithmetic it is bad at. The tool does it once, concretely, for
this exact state.

## Correctness delegation

No rule is reimplemented inside `rules.py`. Every behavioural call routes
through existing forge helpers:

- `trick_rank` / `resolve_trick` / `led_suit_for_lead_domino` / `can_follow`
  / `is_in_called_suit` — the trick-resolution algebra.
- `DOMINO_COUNT_POINTS` / `DOMINO_HIGH` / `DOMINO_LOW` — the scoring tables.
- `DOUBLES_SUIT` / `NOTRUMP` / `has_trump_power` — the declaration flags.

The only logic `rules.py` contributes is *presentation*: what primitives to
return and how to label them. If the engine is right, the tools are right;
if the engine is ever wrong, it is wrong uniformly across every Burl code
path at once.

## Open questions (iter-2 follow-ups, not this task)

- Should `trick_winner_if` return a `points_at_stake_if_partner_wins` / `..._if_opp_wins` breakdown to save the model a second tool call? Probably yes, but waiting on iter-2 spike traces to confirm that query actually shows up.
- `what_beats_what` currently requires an explicit `lead_domino` when no trick is open (e.g. “I'm about to lead — if I led 21, would 15 beat it in reply?”). iter-2 might want a `hypothetical_lead` shape that picks the best opponent reply automatically; deferred until we see whether the base tool is used.
- `contract_progress.offense_count_still_needed` uses `captured - target` and does *not* factor in tricks-still-to-be-won (a 0-captured offense with 5 tricks left and 2-count still loose is mathematically cooked but the tool currently reports `"contract_in_play"`). `status` catches the hard cases; the counter is the naïve difference. Revisit if Gemma starts over-reasoning the margin.
