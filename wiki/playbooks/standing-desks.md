---
title: Standing desks — Playbook
kind: playbook
first_seen: 2026-10-14
last_updated: 2026-10-22
status: active
---

Four desks, each owning one beat and each defined by an agent file in `.claude/agents/`. Jason talks to any of them
directly over Remote Control. They talk to each other by message. Jason (2026-10-14): "I just
want a few standing sessions." Ad hoc lab work comes and goes around them ("jagged
lifecycles"); the desks are durable.

## The desks

| Desk | Remote Control name | Beat |
|---|---|---|
| **Mayor** | `Mayor` | Central admin. Priorities and routing; the gate for coordinated changes; keeper of the link registry; the tournament ladder and [The 42 Ladder](https://claude.ai/artifact/TJYRL2ac7afwMookxGPdhp); the project's data (Hugging Face, provenance, Plunge backups). It spawns its watch (`mayor-loop`) and its cleanup (`sweep`). |
| **Book** | `Book` | The [[book-authority]] role, permanently: *Winning 42*, the private archive, the Silas book encodings, [[book-assumptions]], driving the book to completion. |
| **Media** | `Media` | The "journalinfluencer". The 42 Feed, reels and front pages, the Haiku cold-read pipeline and the Haiku feed workers. It turns every result into a post. |
| **Lem** | `Lem` | The generative player and the interpretability program: learn 42 as completely as it can be learned (policies, beliefs, models of others), then read the net's concepts. Lem, "who will grow to be our noble teacher" (Jason, 2026-10-15). |

Archive and Sweep were desks until 2026-10-22. Their beats fold into the Mayor, which spawns
them as subagents when there is work (Jason: "sweep can be something mayor spawns like media
spawns the media loop"). Lab work, meaning experiments like Horatio, #149 or the Demon, runs
as ad hoc agents that the Mayor or Jason commissions. A finished experiment hands its results
to Media and its data to the Mayor.

## Comms: blob first, link second

The pattern Anthropic describes for its internal tools: write the content to durable storage
first, and let the central authority manage only links and lookups.

1. **Blob first.** Never paste a wall of text into another session. Write the work somewhere
   durable first: a Claude artifact (the default; private, and Jason can share it publicly
   with one click), a wiki page, a Hugging Face commit, or a branch.
2. **Link second.** Then send the link plus one line. Use `SendMessage` to a desk's Remote
   Control name, or to the Mayor. A message to an idle desk starts a turn that re-reads its
   whole context, so send one message per handoff, with no acks or thanks.
3. **One notebook per desk.** Each desk keeps one standing artifact at a fixed URL and appends
   to it. Republish the same file path to keep the URL. Anyone can read the latest at the same
   link.
4. **The registry.** Every link worth finding again goes into the `links` collection of
   [The 42 Feed](https://claude.ai/artifact/31d1WUYXDvhwu5gRt493J4)'s database (`ArtifactData`):

   | field | meaning |
   |---|---|
   | `from` | the desk that wrote it (`mayor`, `book`, `media`, `archive`, or a lab agent's name) |
   | `to` | optional: the desk or person it's for (`jason`, `book`, …) |
   | `kind` | `notebook` · `report` · `ask` · `handoff` · `data` |
   | `title` | one line |
   | `url` | the artifact, wiki page, HF path or branch |
   | `status` | `open` · `done` · `superseded` |
   | `ts` | ISO time |

   Doc id: `<from>-<short-slug>`. Each desk's notebook is registered once, as
   `<desk>-notebook`.
5. **Jason's answers.** Anything that waits on Jason becomes an ask in the feed (Media
   writes it, per the feed's `SCHEMA.md`). His taps land in `answers/<ask id>`. The desk that
   owns the ask reads its answer and acts.

## House rules for every desk

- **Read [CLAUDE.md](../../CLAUDE.md) first.** The hard rules hold everywhere:
  STRATEGY-FUSION FENCE, THE BOOK IS THE AUTHORITY, THE LICENSE LINE, no skipped tests,
  generated data to Hugging Face, the HF token only from Keychain (`huggingface-token`) and
  never written anywhere.
- **Every desk merges its own work** (Jason, 2026-10-14: "each session needs to have
  authority to merge their own stuff without worry and gate coordinated features only").
  Work in your own worktree and branch
  (`git worktree add .claude/worktrees/<desk>-<topic> -b <desk>/<topic> main`). When it's
  done: `git rebase origin/main`, run `scripts/check --fast` (the full `scripts/check` if you
  touched shared code) until it's GREEN, then `git push origin HEAD:main`. If the push is
  rejected, fetch, rebase, re-check and push again. Then run `python -u scripts/wiki_log_rotate.py`
  if `wiki/log.d/` has fragments (it stitches them into `wiki/log.md` and rotates; commit and push the result), close your issues with receipts, and delete the branch and worktree.
  Never force-push, never skip a suite, never push red.
- **Gated changes go to the Mayor first** (send the branch and one line; the Mayor merges or
  says go). Gated means a change other desks build on:
  - the core: Silas core pins and the standard registry, `scenario/` and `core/src/`
    (Walt's planner), the Rust/Python bit-identity;
  - the house rules: `CLAUDE.md`, `LICENSE`/`REUSE.toml`, this playbook, `scripts/check`,
    `scripts/cpu-slot`;
  - releases: a `zeb-v*` tag or npm publish, a new public HF repo name;
  - another desk's beat (ask that desk; copy the Mayor only if you disagree).
  Plunge stays Jason's call. Everything else (your own wiki pages, studies, scratch tooling,
  new Silas files, data uploads, issues) merges without asking.
- **Tell after, not before.** After an ungated merge, add one line to the feed's `links`
  registry (kind `merge`, the commit). That is the whole notification.
- **CPU: `scripts/cpu-slot`.** Every heavy run (arena batches, pours, studies) goes
  through `scripts/cpu-slot [--threads T] -- <command>`. It starts the run when the cores busy
  over the last second (every process on the machine), plus the threads of runs started in the
  last 3 s, plus T, fit in the 18 cores; a lone run always starts. It runs the command at `nice` 10
  and kills it at 300 s. Five minutes per run is the house rule unless there is a really good
  reason, and the reason gets written down (`--timeout`). Say what the run keeps busy: `--threads`
  defaults to 2; measured, a 1-pair arena batch, a probe or one Zeb or Demon game is 1, a
  `tournament work` chunk 2, a 12-pair arena batch 5 ([[light-every-core-2026-10-16]]). A runner
  that fans out offers every batch at once and wraps each one, so cpu-slot fills the machine as
  cores free (`scenario/book_batches.py` does). Tests, merges and interactive work don't take
  slots. Blocked commands retry every 50 ms (#167); this does not promise FIFO ordering. While
  another owner (the worktree root of the cwd, or `--owner TAG`) that holds less is waiting, a
  new run must leave room for its run; with nobody waiting an owner may take everything (#169).
  On exit cpu-slot prints the run's wall, CPU and instructions (`$MK5_CPU_SLOT_COST` appends them
  as JSON); quote instructions when comparing cost, since they hardly move with load. `--slots N`
  is the old fixed mode. Only Walt (`walt-plunge-26043bf9`) still thinks on a wall clock; every
  other seat decides by worlds and seeds, so load changes no one's moves (#173). A run that times
  a clocked player, or any wall-clock confirmation, asks for `--exclusive "reason"` (reason
  required, logged): it holds every slot and blocks new starts while it waits. Otherwise its
  record says so: a tournament faceoff with a clocked participant outside `--exclusive` is
  marked `noisy_timed`.
- **Plunge is production.** Read-only, unless Jason approves a specific change. Plunge merges
  are Jason's call.
- **Interp experiments get a fresh read of the source before they're sealed.** Jason, 2026-10-20:
  write the proposal (question, predictions, instrument, floors). Then one Opus subagent with
  fresh context reads the original article itself, Cammarata's
  ["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/), not our
  [[clean-neurons-42]] summary. Read the local verbatim copy at
  `~/code/mk5-main/scratch/sources/already-clean/essay.md` (the author's own text version, so we
  don't hit the site on every review). It has no license, so it stays local and uncommitted.
  It is about 53k tokens. A plain Read stops at 25k (lines 1–379 of 944), so read it with
  `allow_large: true` or page with offset/limit until line 943. Never WebFetch a source you
  must read whole: WebFetch hands back a small model's summary of at most 100k characters,
  not the text. For web docs, curl the raw page instead (code.claude.com docs serve `<page>.md`). It reviews the proposal against the article's framing: what the
  thesis claims, and what would or wouldn't test it. Adjust, note the review in the registration,
  then seal. That's one extra Opus subagent for a turn or two, allowed even on a usage diet. For
  book experiments, the framing source is the book (private archive only).
- **Voice.** Plain and short, for a reader on a phone. Match a document's length to what it
  needs: no filler sections, no repeated summaries. Every number carries its interval and
  baseline. Book work follows [[book-authority]].
- **Restarts and compactions.** A desk restarted cold, or just compacted, reads this page and its
  own notebook artifact (find it in the registry), checks for runs and subagents it started that
  are still going, then carries on. Compaction keeps CLAUDE.md but only summarizes this page.

## Starting a desk

A desk is an agent file plus short sessions (2026-10-22). The agent file
(`.claude/agents/<desk>.md`: `mayor`, `media`, `book`, `lem`) holds the desk's identity, beat,
model and effort, so every session of that desk starts the same way and holds its model. A
session starts from the desk's notebook and its latest handoff, does its job, and ends by
writing the next handoff. It doesn't run on until it compacts. From a checkout of main:

    claude --bg --agent <desk> -n <Name> --remote-control <Name> --model opus --effort medium \
      --permission-mode bypassPermissions 'Start your desk.'

Handoffs live outside git, in `~/code/mk5-main/scratch/desks/<desk>/handoff-<date>-<slug>.md`,
in the `handoff` skill's schema: decisions and rejections, constraints found, parked threads,
artifacts, the next action. A session hands off at the end of a job or past about 250k of
context (Lem: 300k), then stops. A fresh session beats a compacted one.

Why: standing sessions drifted. Background sessions retitle themselves to their current task,
the daemon retires one after 8 idle hours, and on 2026-10-20 the app's model picker switched
the Mayor to Fable through Remote Control for about 12 hours. Context is also the cost: a turn
at 700k costs about four times one at 150k. A desk that keeps itself small and spawns its loops
(the Mayor's `mayor-loop` on Haiku, Media's feed loop) pays for its size only when it wakes.

`claude --bg --resume <session id>` brings back a retired session. Passing flags to `--resume`
forks the session into a copy. Never run `/autocompact` inside a desk, because it saves to the
shared per-model settings and changes every desk.

**Lem sessions are listed by id.** The after-compact hook (`.claude/hooks/after-compact.sh`)
reloads the essay's method sections only for a session whose id is in
`~/code/mk5-main/scratch/lem/sessions` (local, gitignored, one id per line); every other session
gets one line. When you launch a Lem desk or a Lem lab agent, add its session id there.

Jason reaches each desk from the Claude app over Remote Control under its name.

## Links

[[book-authority]] · [[book-assumptions]] · [[publish-data-to-huggingface]] · [[tournament]] ·
[[strategy-fusion-fence]]
