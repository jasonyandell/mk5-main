---
name: book-interp
description: Read Winning 42 and a 42 net (Lem, Zeb's field net) as one discipline, the already-clean way. Use whenever a task reads a book rule, encodes one in Silas, measures a gap between the book and Zeb, matches book concepts against net units, or writes a book or Lem result; and whenever you feel the pull to write "the book is wrong" or "this unit is mixed".
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Reading the book and the net together

*This is the best we know so far. It improves itself: see "Make this skill better" at the end.*

## What we are up to (say it to yourself first)

We do **not** claim interpretability is about to be solved. We are in no position to tell the
field what to do. Our claim is narrow, and it is Jason's: **the lessons of Cammarata's "Neural
Networks Are Already Clean" (https://already-clean.goodfire.com/) fit our particular problem
unusually well.** For Texas 42, here, superposition is not the right null, and reading the net is
not hopeless. That cuts against most of what you have read about interpretability, so you will
feel pulled back toward the conventional view. Don't argue the general case. Do the local work,
because it teaches us more about 42.

The book desk was already doing this before anyone read the essay. The two disciplines rhyme:

| The essay, about a net | Our charter, about the book ([[book-authority]]) |
|---|---|
| "Polysemantic" is more a claim about the observer than the neuron | A gap between the book and Zeb is a finding about our instruments |
| Superposition is the last hypothesis, not the null | "The book is wrong" is the last hypothesis; never the headline |
| A history of microscope errors | The rule-fidelity census: 44% of the first loss was our encoding, not the book ([[book-investigation-2026-10-14]]) |
| Short contexts, narrow data hid neurons' jobs | The arena violates the book's assumptions A1–A10 ([[book-assumptions]]) |
| A tool that looked only backward misread the Titanic neuron | Rule 6: calibrate before you chase an A/A miss ([[estimator-calibration-2026-10-16]]) |
| The network is the answer; interpretability is the search for the question | The book is the answer; the assumption list is our search for the question it answers |
| Assume clean: a messy neuron is then a lead | Assume the book is right: a gap is then a lead |

Both objects are learners' output. Roberson learned 42 at real tables and wrote down the
distinctions; a net learns 42 from data and grows units. Read each as its own reader would.

## The ladder for any gap

This covers a gap between the book and Zeb, or a unit no list names. Walk the rungs in order and
write down the rung you stopped at.

1. **Tools.** Is the measurement sound? Paired estimator, interval, chance baseline; is the A/A
   inside its band (#191); is the estimator calibrated for this pairing (unequal players: #192)?
2. **Explanations (the encoding).** Does our Silas encoding say what the book says, inside the
   rule's stated scope (A8)? A literal encoding fractures a rule the way an SAE fractures the V
   neuron into "car" and "cat". Presume the encoding wrong until a reason is found.
3. **Data and inputs.** Which of A1–A10 does this arm violate? The arena seats no one who models
   you across hands, no partner who reads the book's conventions. For a net: one position,
   uniform beliefs, no history (#202).
4. **Recursive generalization.** Is the rule (or unit) one idea grown along its facets past where
   we looked? Book rules have cores and offshoots ("protect the count" reaching into the bid, the
   lead and the discard). Find the core before calling an offshoot a contradiction.
5. **The radical reading** ("the book is wrong here", "this unit packs unrelated things"). Reach
   it only after 1–4 are written down, and hold it to the same burden of proof as the clean reading.

## Read a rule the way the essay reads a neuron

- **A rule is a gate, not a list of nouns.** Ask what it lets through and what it keeps out.
  "Protect the count" separates two kinds of trick; it is not a list of the five count tiles.
  Describe the rule without naming its tiles and the shared pattern pops out.
- **Check the negative pole.** For every rule, find the hands where it says *don't*, and test
  those too. Many book rules are hold-back rules, like the essay's restraint neurons (the book's
  case against a 32 or 33 bid, [[w42-book-second-pass]]). In the found-wallet graph the biggest
  push came from units holding back.
- **Strong activations carry the work.** Measure a rule where it bites: on the deals where it
  changes the play (#189's `changed` filter does exactly this). Averaging over deals where the
  rule is silent dilutes it toward a tie, and a tie is not agreement.
- **Worked hands are attribution graphs.** Roberson's reasons for a play are a circuit read aloud.
  Encode the chain, not just the move.
- **Explanations that can lose.** From a rule's text alone, write the hand that most demands it
  (the blind maximum), five near misses that shouldn't trigger it, and one far-away hand. If the
  encoding fires on the far one, the encoding is wrong.
- **Motifs over lists.** The S-row catalog is the enumerative list. The book's recurring patterns
  matter more, including the ones we have not named yet. Edge motifs, one distinction feeding the
  next, are the net-side form of a chain of book reasons.

## Matching the book against the net: the Tokyo guardrail

A visitor who knows New York may ask "where is the Brooklyn of Tokyo?", which is a fine opening
question. Rebuilding Brooklyn from seventy pieces of Tokyo is not; it makes Tokyo look messy.
- The book's vocabulary is a **starting question** for Lem's units, never the answer key. Lem's
  units are not the answer key for the book either.
- Where a unit and a rule coincide, the net holds the book's idea: say so, with the scope sentence.
- Where the book names a concept no unit carries: the net never needed it, the inputs can't show it
  (rung 3), or the arena never asked (A1–A10).
- Where a unit carries something the book doesn't name: a new word may be waiting, the way
  high-low frequency detectors were. The net's ideas are often better than ours, and so are
  Roberson's. Bring them to Jason as questions.
- Unit-level work (audits, squeeze, formation, ablation) goes through the `clean-neuron` skill.

## How to write it

- Never headline "the book loses" or "the book is wrong". Never call a unit "mixed",
  "polysemantic" or "superposed"; write "not named in list vN".
- Every number carries its interval and chance baseline. List the assumptions held and violated,
  and name the measurement that would restore them ([[book-authority]] "How to report").
- The scope sentence on every Lem-side result: *this net, these inputs, this vocabulary. What we
  learned from this model, not all of 42, yet.* On the book side: *what our arena can see under
  assumptions X.*
- Never claim more than the local bet. "Fits 42 unusually well", not "interpretability is solved".
- The book's text stays private: cite catalog rows by id and paraphrase.

## Before you seal a registration (Jason, 2026-10-20; every interp experiment)

1. Write the proposal: the question, the predictions, the instrument and its floors.
2. Before you seal it, spawn one Opus subagent with fresh context (cleared even on the usage
   diet, for one or two turns of read and review). It reads the framing document itself, not our
   summary:
   - for a test of the clean-neurons thesis, Cammarata's essay
     (https://already-clean.goodfire.com/), read from the local verbatim copy
     `scratch/sources/already-clean/essay.md` (unlicensed: local only, never committed), not
     `wiki/topics/clean-neurons-42.md`;
   - for an experiment about the book, the book itself, from the private OCR in
     `scratch/winning42/text/` (never published; the reviewer paraphrases and cites pages).
   Then it reads the proposal and says where it drifts from that framing: what the source
   claims, what would count as a test of it, and what would not.
3. Adjust the proposal, note the review in the registration (who read what, what changed), then
   seal it by committing before the first measured batch.

## Signs you have drifted

Stop and walk the ladder if you are about to:
- write "the book is wrong" or "the book loses";
- write "mixed" or "polysemantic";
- reach first for an SAE, or for a probe that can only answer a question we already had;
- read a tie as agreement, or an A/A miss as a finding before calibrating;
- use the book's words as Lem's labels, or the reverse, without the Tokyo check;
- generalize from 42 to all of interpretability.

## Make this skill better

This skill is a working draft. Use it, and fix it when it fails.

**Fix major problems now.** If the skill is wrong (it names a file, command or issue that no
longer exists, contradicts a ruling or the charter, or would steer someone into a bad result),
fix this file in the same commit as your work. Add a line to "Lessons folded in" below. Don't
leave a major problem for a log.

**Log the rest.** For friction, a missing step, a better example or a learning, add a file
`.claude/skills/book-interp/learnings/<UTC yyyy-mm-ddThhmmZ>-<slug>.md` containing:
- the SPDX header (as at the top of this file);
- what you were doing;
- where the skill helped or failed you;
- what you did instead;
- the change you would make.

Short and concrete; the logs are evidence.

**Log rolling.** Before you add a log, count the log files in `learnings/` (not its README). If there are already three,
roll them in the same commit as your own lesson:
- rewrite this skill fully, folding in all of them plus yours (make each change asked for, or
  write down why not);
- delete the log files (git keeps them);
- add one line to the record below.

Keep the position statement, the ladder's order and the Tokyo guardrail unless a test broke them.

Lessons folded in:
- 2026-10-20: the fresh-context framing review before sealing (Jason's ruling, via the Mayor).
