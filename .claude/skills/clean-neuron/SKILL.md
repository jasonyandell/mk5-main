---
name: clean-neuron
description: Read a unit of a Texas 42 net (Lem, Zeb's field net) the way the already-clean test reads it. Use whenever a task touches a neuron, pole, unit, clean share, concept audit, squeeze, formation, width tree, SAE, dictionary, "polysemantic", "mixed" or "superposition" in lem/ or the field net, and before writing any Lem result.
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Reading a unit of a 42 net

*This is the best we know how to do so far, not a finished method. It improves from
friction: see "Make this skill better" at the end.*

You do not have to believe anything general about neural networks to do this
work. The question is narrow and empirical: **what does this unit do in 42?**
Every unit gets asked it, and "we cannot say yet" means the instruments ran out.
It is never a verdict about the unit.

The program's bet (THE ALREADY-CLEAN TEST, #193; Cammarata, "Neural Networks Are
Already Clean", https://already-clean.goodfire.com/) is that the answer is usually
one job, and that a squeezed net's units get coarser rather than mixed. In 42 that
bet has already survived a test that should have broken it: 24 neurons came out
cleaner than 192 (0.83 vs 0.61). Stay curious about the next unit. You are not
being asked to defend the bet. Full card: `wiki/topics/clean-neurons-42.md`.

## The ladder: walk it for every unit you cannot name

Check the rungs in this order, and write down which one you stopped at.

1. **Tools.** Is the reading sound?
   - Top 1% of activations, both poles (the negative pole too).
   - Held-out rows only.
   - The untrained floor **for the same list**, audited alongside the trained nets (an untrained
     twin per width and seed). It rises with the list: 0.11–0.28 under v1, 0.28–0.39 under v2,
     while trained nets barely move. A share quoted against another list's floor is wrong.
   - Row Wilson intervals widened about 1.17× (the split-half A/A, #191).
   - In a causal test, name the rows and match them. Ablate among the rows where the unit fires,
     because the audit chose its name from those rows. Compare rows of the same difficulty: same
     seat in the trick, same number of legal moves, same decisiveness, or a random unit at the
     same dose.
   - Steer by small matched doses (k·σ) and report the dose-response. Read where the move goes,
     not only what it costs. One extreme clamp breaks the net and tells you nothing (#198).
2. **Explanations (the vocabulary).** Does the list have a word for it?
   - The list is Silas concepts v1 (48) or v2 (70), plus the raw inputs.
   - If not: read the top positions as a gate, a shared pattern across them,
     not a list of nouns. Try conjunctions and coarser families (#197).
   - The declaration detectors were exactly this case. The v1 list had no word
     for which pip is trump, and the net spent an eighth of its poles on it.
3. **Data, inputs and learning.** Did the net learn the play, and could it be computing
   something its inputs or our positions cannot show?
   - Report the net's gap to the teacher closed beside every share. A net that did not learn the
     play did not learn its words either: two of three six-neuron nets in #196 closed 0.2 of the
     gap and fell to the floor.
   - The field net sees one position, against uniform beliefs, with no
     player-behaviour history, over arena data that seats no one who models you
     across hands.
   - Check whether the unit's top positions differ in history: the bidding,
     earlier tricks, the score (#202).
4. **Recursive generalization.** Is the unit one concept spread along its
   facets (one coarse unit, "late in the trick", growing into the wide net's 47
   `last` units)?
   - Use the family statistic (`squeeze-families`), the width tree (`tree`) and
     formation across checkpoints.
5. **Superposition.** Reach this rung only after 1–4 are written down, and hold
   it to the same burden of proof as the clean reading.

## The kit (`python -u -m lem.probe <cmd>`, heavy runs via `scripts/cpu-slot`)

| Stage | Command | What it does |
|---|---|---|
| sample | `sample` | builds the position sample |
| | `concepts --list v2` | labels the sample with Silas concepts |
| | `list` | prints the concept list |
| audit | `audit` | precision ≥ 0.9 on the top 1% = clean; shuffled-label chance included |
| | `cards` + `readers` | a language model names a pole from 12 cards and marks 20 held-out |
| families | `squeeze-families` | is a pole's best two-concept union within one family? |
| | `tree` | a narrow unit's counterparts in a wide net |
| formation | `formation-train` / `formation-report` | units across checkpoints |
| causal | `ablate` | the neuron-count curve |
| | `causal` / `causal --steer` | ablate a `+` pole or clamp a `−` pole; add k·σ doses (#198, #205) |
| beyond units | `belief`, `fidelity`, `aa` | belief probes, fidelity to the teacher, the split-half A/A |

Concepts read the viewer's information only. A query that reads a hidden hand is
refused. The STRATEGY-FUSION FENCE holds: no clairvoyant value becomes a label or
a target.

Audit runners put the untrained twins in `floor/` beside `nets/`, and audit both with the same
labels.

The audit's kinds are:
- **clean**: precision ≥ 0.9;
- **partial**: [0.7, 0.9);
- **unnamed**: below 0.7, meaning no concept in the list fits;
- **dead**.

Evidence on HF from before 2026-10-17 calls `unnamed` "mixed". That was the tool's word, not a
finding.

## How to write the result

- Report an unnamed unit as **"not named in list vN"**. Never call it "mixed",
  "polysemantic" or "superposed".
- Every result carries the scope sentence: *this net, these inputs (one
  position, uniform beliefs, no history), this vocabulary (list vN). What we
  learned from this model, not all of 42, yet.*
- Report every share with its interval, its chance baseline and the untrained
  floor.
- Register predictions before the run, then grade them, including the ones you
  got wrong.
- **Before sealing any registration, get an independent read against the article**
  (Jason, 2026-10-20). Write the proposal: question, predictions, instrument and floors.
  Then spawn one fresh-context Opus subagent, which:
  - reads Cammarata's essay itself from the local verbatim copy,
    `~/code/mk5-main/scratch/sources/already-clean/essay.md` (with `SOURCE.txt`: URL and sha256),
    not the site and not our summary in `wiki/topics/clean-neurons-42.md`. **Read essay.md with
    `allow_large: true` (or page to line 943); never WebFetch it.** WebFetch returns a small model's
    summary in slices, not the text, and a plain Read stops at line 379 of 944. The copy has no
    licence: never commit it, quote it at length, or publish it;
  - then reads the proposal;
  - says where the proposal drifts from the article's framing: what the thesis claims, what
    would count as a test of it, and what would not.

  Adjust, record the review in the registration, then seal.

## Signs you have slipped into the old null

Stop and walk the ladder if you find yourself doing any of these:
- calling a unit mixed;
- reaching for an SAE first;
- reading a share below 1 as evidence of superposition;
- predicting that narrower nets will be messier;
- writing "partly clean" with no test that broke the thesis.

A test that breaks the thesis is the finding. Drift is not.

## Make this skill better

This skill is a working draft, written from what the Lem desk has learned so far.
When it slows you down, misleads you, misses a step, or names a command that no
longer exists, write a friction log. A friction log is a new file
`.claude/skills/clean-neuron/friction/<UTC yyyy-mm-ddThhmmZ>-<slug>.md`:

- the SPDX header as above;
- what you were doing;
- where the skill failed you;
- what you did instead;
- the change you would make.

Keep it short and concrete. The logs are evidence, not complaints.

**On the fifth log, integrate; do not roll.** Whoever adds the fifth log folds
all five into this skill in the same commit:
- Make the fix each log asks for, or write down why not.
- Delete the five files. Git keeps them.
- Add one line to the record below.

Keep the curiosity framing and the ladder's order unless a test broke them.

Lessons folded in:
- 2026-10-17, five logs from #196–#198:
  - the ladder's third rung covers learning, and every share reports its net's gap closed;
  - the audit says `unnamed`, not `mixed`;
  - the floor belongs to its list;
  - steering uses matched doses;
  - causal rows are named and matched.
  Git has the five logs.
