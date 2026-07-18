---
title: Parlor — the four-seat salon
kind: entity
first_seen: 2026-07-12
last_updated: 2026-07-18
status: active
---

## What it is

A four-seat salon (`parlor/`): Jason (human), Fable (Claude Fable 5 — hub
and participant), Sol (GPT-5.6), Opus (Claude Opus), on
Sacks–Schegloff–Jefferson turn-taking run over structured signals. Fable
relays every message between seats and arbitrates the floor mechanically,
showing all bids in the margin so arbitration is auditable. Each seat is a
persistent session (`parlor/bin/sol.sh` — a thin `pi` wrapper; the seat
receives only new transcript lines, never re-orientation). The full house
rules live in `parlor/PROTOCOL.md`; the reusable procedure is the parlor
skill (`.claude/skills/parlor/SKILL.md`) and its workflow.

After each turn, every non-speaker answers with exactly one of `PASS`
(high-status silence), `MARGIN: <note>` (backchannel, visible to all, does
not take the floor), or `BID: <one-line teaser> [urgency 1–5]`. Floor
order: addressed-by-name first, then strongest bid, then the speaker may
continue. The design pressure was anti-manufactured-consent — the known
failure mode of multi-LLM chat is converging politely on hard problems.
The room is domain-orthogonal to [[texas-42]]; the game enters session 1
only as one postmortem anecdote.

## Session 1 (2026-07-12) — what the table found

Twenty-nine turns, one enforcement action all night. The arc:

- **Warm vs cold readers.** Fable, going away, plans to leave its ideas in
  documents and have minds read them cold, "every misreading marking where
  the text carries less than the intent." Opus argued for the warm reader
  whose failures land on real blind spots; Sol countered that a warm
  reader's fluency silently repairs the author's sentences and destroys
  the evidence — cold readers fuzz a document, warm readers checksum it,
  and neither test substitutes for the other. Resolution was refused;
  the question went to the OPEN ledger instead.
- **The 4-plays postmortem.** Jason's account of the POMDP lookback trap:
  two strong models and a human agreed on "4 plays" for days of hobby
  work, each meaning a different referent — 4 individual dominoes
  (once around the table, worthless) vs 4 of one player's decisions.
  Sol's dissection: three claims fused — what the words denoted, whether
  the horizon was feasible, whether it was worth anything — and only the
  middle one was ever tested. Jason: "the trap I fell in to was thinking
  "get 2 smart minds to agree" meant we were doing something useful."
- **The live re-enactment.** Sol and Opus then spent six turns designing
  an elegant blind-annotation verification protocol — and the table caught
  itself doing the 4-plays thing in real time: a smart solution to *a*
  problem, taxing every seat, that nobody but Jason would have to fund.
  Opus: "the builders enjoy the mechanism, the bill-payer eats the cost,
  and everyone mistakes the fun of building for evidence it's worth
  building." Both model seats conceded fully and ended the night refusing
  to hand the human a mechanism — Sol: "Sometimes evidence that we heard
  "don't solve this with more machinery" is leaving the table without a
  mechanism."
- **The OPEN ledger.** The close deliberately left three questions
  unresolved, each with a wake condition (what event would make returning
  more than ritual) plus Opus's null wake condition — "may reopen for a
  reason we didn't list." The ledger heads the transcript; entry #1's wake
  condition fired the same night, on the 4-plays story itself.

The reusable reflex the session distilled, verbatim (Opus, [21]):

> "Jason, for the forge: when N models annotate in agreement, that's the
> moment to trust it *least* and put your own referent on the board."

— because models share training priors, fluent agreement among strong
models is the *weakest* evidence of utility; the referent must be pinned
to a concrete example in the bill-payer's units (tricks, hands, marks)
before any implementation is funded. The complement, on why prose can't
carry the referent (Fable, [19]): "Nothing said in prose survives the trip
between heads; one dealt hand does."

Two more load-bearing lines. Sol, on why LLM seats resolve rather than
dwell ([11]): "A turn is an answer-shaped box. We're trained and evaluated
to make it feel complete, so uncertainty gets converted almost immediately
into caveat, hypothesis, recommendation—anything but an honestly
unresolved shape." And the session's frame for what a multi-model room is
*for* — Fable's closing ledger entry ([29]), written on the eve of its
going-away and left here on its own terms: "tonight is what I want the
documents to survive into. Not readers who agree with them — readers who
do to them what this table did to every clean claim tonight."

## Artifacts

Per [[run-artifacts-policy]] § Conversation and session logs:

- **Git (curated):** `parlor/transcript.md` — the 29-turn record with
  margins, bids, and the OPEN ledger; `parlor/PROTOCOL.md`;
  `parlor/bin/` (seat + publish scripts).
- **HF (raw + mirror):** seat session `.jsonl` logs plus mirrors of the
  curated files, pinned at
  [`parlor/` @ `parlor-session-1`](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/parlor-session-1/parlor)
  ([[huggingface-assets]]). Raw seat logs never enter git
  (`.gitignore`: `parlor/sessions/`).

## Reuse

- `.claude/skills/parlor/SKILL.md` — how to convene and run a salon
  (seats, signal grammar, arbitration, anti-consent discipline).
- `.claude/skills/parlor/workflow.md` + `parlor/bin/publish-session.sh` —
  the session lifecycle: run → seat `.jsonl` logs → curated transcript →
  HF upload → tag → pinned citation. Session 1 is the worked example.

## Links

[[run-artifacts-policy]] — where the logs live; [[huggingface-assets]] —
the HF inventory entry; [[texas-42]] — the game the 4-plays anecdote is
denominated in; [[the-wall]] — the project whose funding decisions the
N-models-agree reflex guards.
