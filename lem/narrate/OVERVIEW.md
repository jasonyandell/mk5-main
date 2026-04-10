# lem/narrate — Game Narration Generator

Turns a Texas 42 game into conversational prose from one player's point of view.

The first consumer is us, eyeballing whether the voice is any good. Later consumers are
Gemma's parsing sanity check and the Stage 1 STaR prompts. Design now with those later
uses in mind, but don't build for them yet.

## What it does

```
seed ──┐
       │
       ▼
  Deal hands (deterministic from seed)
       │
       ▼
  Run bidder → (bid_value, decl_id)
       │
       ▼
  Play game with E[Q] policy, N=10      ← forge.eq.generate.pipeline
       │
       ▼
  Walk the 28 decisions in order, from narrator's seat
       │
       ▼
  Emit prose → stdout
```

## CLI shape (first cut)

```
python -m lem.narrate \
    --seed 42 \
    --narrator 3 \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    [--decl 5]        # override bidder, force a declaration
    [--show-eq]       # include E[Q] counterfactuals on real choices (default off)
```

- `--seed` is the only required run parameter. Everything else has a default.
- `--narrator` defaults to 3.
- `--decl` is an escape hatch; normally we let the bidder pick.
- `--show-eq` is wired but off. We want to see the clean voice first.

No batch mode, no file output, no HF upload. One seed, one game, printed to stdout.

## Output shape

First-person prose from the narrator's seat, sectioned:

```
[setup]     You are playing Texas 42. You are Player 3, partnered with Player 1.
            Your hand: 6-6, 5-4, 4-2, 3-3, 2-1, 1-0, 0-0.
[bid]       Player 0 bid 30 and called fives as trump.
[trick 1]   Player 0 led the 5-5. Player 1 followed with the 5-2. You were void
            in fives and sluffed the 0-0. Player 2 played the 6-5. Player 0 won
            the trick — 10 points to their team.
...
[trick 7]   <forced play for everyone>
[closing]   Final: your team 12, their team 30. Player 0 bid 30 and made it.
```

Voice rules:
- **Second person** for the narrator ("You play the 6-6").
- **Third person** for the other three players ("Player 1 follows with the 5-2").
- Name relationships explicitly on first reference ("your partner, Player 1").
- State facts, not strategy. No "you should have played X" commentary in the baseline
  version — that's what `--show-eq` will add later.
- Count points at the end of every trick.
- Use domino names in `H-L` form (`6-6`, `5-3`). Match how the viewer formats them.

## Pieces we'll borrow

| What | Where | Why |
|---|---|---|
| Hand dealing from seed | `forge/eq/generate/deals.py` | Deterministic starting state |
| Bidder (broken but reasonable) | `forge/bidding/` | Bid value + declaration |
| Game play with E[Q] | `forge.eq.generate.pipeline.generate_eq_games_gpu` | 28 decision records in one call |
| Stage 1 oracle checkpoint | `forge/models/domino-qval-large-*.ckpt` | Needed by the pipeline |
| Domino ID → pips | `forge/eq/viewer.py::domino_id_to_pips` | Formatting |
| Declaration names | `forge/eq/viewer.py::DECLARATION_NAMES` | "fives", "doubles", "no trump" |
| Suit / trick logic | `forge/oracle/tables.py` | `can_follow`, `led_suit_for_lead_domino`, `trick_rank` — needed to write lines like "you were void in fives" or "partner trumped" |

The whole script should be one file (`lem/narrate/__main__.py`) plus maybe a small
`render.py` if the prose rendering gets thick. No premature package structure.

## What we'll learn from running it

- Does the voice feel natural?
- Is the narration self-contained enough that someone with no other context could
  reconstruct the game?
- Are there facts that matter for reasoning that we're *not* stating (e.g. who's void in
  what, count-so-far, etc.)?
- Do we like first-person second-person or does it need adjustment?
- Is the output too long? Too terse?

Answers to these re-shape the voice rules above. We re-evaluate against the LEM goals
once we have a narration in hand.

## Deliberately out of scope for v0

- **Multiple games / batch mode.** Add when we like the voice.
- **File output.** Stdout is fine for now.
- **Truncation at a decision point.** Needed for STaR inputs. Design the renderer so it's
  a trivial addition (walk decisions, emit prose, stop condition) but don't wire a flag.
- **`--show-eq` counterfactual lines.** Flag exists, body doesn't.
- **Parsing-check question generation.** Separate concern for Stage 0.
- **HuggingFace upload.** Stage 1 concern.
- **Narrations for all 4 perspectives from one game.** Trivial later. Not today.

## Open questions we'll answer with the first output

1. Is the first-person-second-person voice right, or should we switch to pure second
   person throughout ("You watch Player 1 play the 5-2") or pure third person ("Player 3
   plays the 6-6")?
2. How much trick-state to restate each trick: just the plays, or a running tally of
   counts and voids?
3. Should the [setup] include the full declaration rules ("fives trump — the 5-5, 6-5,
   5-4, 5-3, 5-2, 5-1, 5-0 are trumps in descending order") or assume Gemma knows trumps
   from the rules adapter we haven't built yet?
4. Closing line — is "your team X, their team Y" enough, or do we want a trick-by-trick
   score recap?

We'll know after we look.
