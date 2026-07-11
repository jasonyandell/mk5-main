---
title: The book enters — Winning 42 as citation, extraction stayed dormant
kind: topic
first_seen: de7b7c71
last_updated: 2ab1a825
status: retired
---

## The question

What is the authoritative, complete statement of Texas 42's rules, and can it
be mined for concrete test data? Two distinct asks, three weeks apart:

1. **2025-07-26** — *"give me the complete rules of Texas 42... this needs to
   be checked and cross referenced and authoritative. tests: when can you bid
   3 marks? when can you plunge?"* Could an assistant-compiled, web-sourced
   formal specification stand in as ground truth for a from-scratch rules
   engine?
2. **2025-08-18** — *"I want to process the entire document as I need to get
   as much documented as possible to test my game... the ultimate outcome
   would be one file per example in the correct format."* Could a strategy
   treatise (primarily the Glynn Hill web treatise, with Roberson's book
   named as a Phase-3 supplementary source) be mechanically mined into
   individual, schema-conformant test-case files?

## What happened

**2025-07-26.** Jason requests a "formal, detailed specification"; the
assistant produces an 11-section document citing pagat.com, Austin42.org,
Texas Monthly, Wikipedia, and — as its second-listed authoritative source —
*"Winning 42: Strategy & Lore of the National Game of Texas" by Dennis
Roberson (Texas Tech University Press, 1997-2009) - Definitive published
rulebook.* This is the book's first documented appearance in the project.
The citation is substituted into `docs/rules.md` wholesale by commit
`5b5e97c` (2025-08-09) and survived verbatim there until the docs→wiki consolidation — it now lives in [[rules-of-42]] §Provenance
(`docs/rules.md:26`). Sibling files `docs/rules-gherkin.md` and
`docs/rules-tournament.md` do not carry the citation forward.

**2025-08-18.** Four same-day conversations converge on a phased extraction
protocol (`GH001`-`GH130` numbering, an `Example ID / Type / Source /
Original-text / hands / bidding / tricks-completed / analysis` schema,
`p0`-`p3` player mapping, `[6|5]` domino notation). Phase 3 of the protocol
explicitly names Roberson's *Winning 42* as an additional source to mine,
citing a specific target: *"Hand 36 (page 108) - controversial 30 bid with 3
trumps."* A citation format is even specified: `"Winning 42, page [X]"`,
"Book content, fair use excerpts only."

No extraction output exists anywhere on disk from this era — no `GH###`
files, no Roberson-hand files. All four conversations are plan-only.

## Terminal status

**RENAMED / RETARGETED, not carried forward as built infrastructure.** The
book's *citation* is BUILT and durable — carried forward into [[rules-of-42]], unchanged
in substance since the 2025-07-26 draft. The book's *extraction protocol* —
the actual mechanism for turning Roberson's hands into test data — stayed
IDEATED-only for the rest of this era: a detailed, iterated schema and a
named target hand, but zero extraction output, zero beads, zero later-era
commits referencing it.

The eventual consumer of this thread — the empirical Winning-42
strategy-validation campaign, and the Kindle-CDP page-grab that produced
`scratch/winning42/` — lands outside this window, in April/May 2026. Within
Era 1, treat "the book enters" as: citation built and durable; extraction of
content ideated and dormant for the remainder of the era.

## What it ruled in / out

Nothing about the wall directly — this predates the wall's articulation by
many months. It establishes only that Roberson's book was on the project's
radar as an authority six months before anyone asked what to *do* with
information extracted from it — the extraction protocol's ambition (turn
expert strategic hands into structured, machine-checkable test cases) is a
proto-form of the later "distill for what" question, aimed at a
rules-conformance test suite rather than a play policy.

## Related pages

[[web-game]] · [[texas-42]] · [[sources/claude/era1-web-game-prologue|conversation digest]]
