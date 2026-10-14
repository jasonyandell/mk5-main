---
title: Standing desks — Playbook
kind: playbook
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

A few long-lived Claude Code sessions, each owning one beat. Jason talks to any of them
directly over Remote Control. They talk to each other by message. Jason (2026-10-14): "I just
want a few standing sessions." Ad hoc lab work comes and goes around them ("jagged
lifecycles"); the desks are durable.

## The desks

| Desk | Remote Control name | Beat |
|---|---|---|
| **Mayor** | the lead session | Central admin. Priorities and routing; the only writer to `main`; the CPU budget; keeper of the link registry. |
| **Book** | `Book` | The [[book-authority]] role, permanently: *Winning 42*, the private archive, the Silas book encodings, [[book-assumptions]], driving the book to completion. |
| **Media** | `Media` | The "journalinfluencer". The 42 Feed, reels and front pages, the Haiku cold-read pipeline and the Haiku feed workers. It turns every result into a post. |
| **Archive** | `Archive` | Data. Hugging Face datasets, the Plunge backups, the tournament store, "recover everything to HF", data hygiene and provenance. |

Lab work, meaning experiments like Horatio, #149 or the Demon, runs as ad hoc agents that the
Mayor or Jason commissions. A finished experiment hands its results to Media and its data to
Archive.

## Comms: blob first, link second

The pattern Anthropic describes for its internal tools: write the content to durable storage
first, and let the central authority manage only links and lookups.

1. **Blob first.** Never paste a wall of text into another session. Write the work somewhere
   durable first: a Claude artifact (the default; private, and Jason can share it publicly
   with one click), a wiki page, a Hugging Face commit, or a branch.
2. **Link second.** Then send the link plus one line. Use `SendMessage` to a desk's Remote
   Control name, or to the Mayor.
3. **One notebook per desk.** Each desk keeps one standing artifact at a fixed URL and appends
   to it. Republish the same file path to keep the URL. Anyone can read the latest at the same
   link.
4. **The registry.** Every link worth finding again goes into the `links` collection of
   [The 42 Feed](https://claude.ai/artifact/TQVJu2Wu78adYzPwTJLEdR)'s database (`ArtifactData`):

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
- **Main has one writer: the Mayor.** A desk works in its own git worktree on its own branch
  (`git worktree add .claude/worktrees/<desk>-<topic> -b <desk>/<topic> main`), commits, and
  sends the Mayor the branch name. It never pushes `main`, never merges, never force-pushes.
- **CPU budget.** Heavy runs (arena batches, pours, long test suites) need the Mayor's OK.
  At most about two heavy experiments run at once on this Mac. Every run sits under
  `timeout 300`. Check `uptime` before starting one.
- **Plunge is production.** Read-only, unless Jason approves a specific change. Plunge merges
  are Jason's call.
- **Voice.** Plain and short, for a reader on a phone. Every number carries its interval and
  baseline. Book work follows [[book-authority]].
- **Restarts.** A desk restarted cold reads this page and its own notebook artifact (find it
  in the registry), then carries on.

## Starting a desk

From `/Users/jason/code/mk5-main`, the Mayor runs one detached tmux session per desk:

    tmux new-session -d -s desk-<name> -c /Users/jason/code/mk5-main \
      "claude --remote-control <Name> -n <Name> '<the desk prompt>'"

The desk prompt names the desk and says: read `wiki/playbooks/standing-desks.md`, create or
find your notebook, register it, then take your first job. Jason reaches each desk from the
Claude app over Remote Control under its name.

## Links

[[book-authority]] · [[book-assumptions]] · [[publish-data-to-huggingface]] · [[tournament]] ·
[[strategy-fusion-fence]]
