# Parlor workflow — one session, convene to pinned citation

The orchestration for a full salon session. The hub (a Claude session,
usually this one) drives every step. `SKILL.md` in this directory carries
the protocol content; this file is the order of operations. Session 1
(2026-07-12, tag `parlor-session-1`) executed exactly this loop.

## 1. Convene

1. Create the seat scripts: clone `parlor/bin/sol.sh` per guest seat
   (provider, model, `--session-dir parlor/sessions/<seat>`). The
   `--session-dir` is what makes the seat persistent AND what emits the
   raw `.jsonl` session log the publish step ships — the log is a free
   side effect of running the seat, not separate instrumentation.
2. Smoke-test each seat: `echo "Quick smoke test: reply with the single
   word 'ready'." | parlor/bin/<seat>.sh`, then a memory check
   ("what one word did I ask you to reply with?") to prove persistence.
3. Seed each seat once with: who is at the table, that this is its
   persistent seat, that it will only ever receive new lines, the signal
   grammar (PASS / MARGIN / BID[urgency]), and the house rules from
   `parlor/PROTOCOL.md` — including "agreement is not the goal."
4. Start `parlor/transcript.md` (or the next session file): title, OPEN
   ledger carried forward from the previous session, date + seats line,
   protocol pointer.

## 2. Run rounds

Loop until the human closes the room:

1. A seat speaks (turn `[N]`). Append it to the transcript.
2. Relay the new lines to every non-speaker seat; collect exactly one
   signal each (PASS / MARGIN / BID).
3. Append the bid round to the transcript as a blockquote — all bids and
   margins, so arbitration is auditable.
4. Allocate the floor (named > strongest bid > speaker continues; ties to
   first bid). Send the winner `FLOOR IS YOURS` with the bids-context.
5. On convergence, invoke the discipline (SKILL.md): demand the
   what-would-make-the-other-view-wrong statement, or pin the referent to
   a concrete example in the bill-payer's units.
6. Human interrupts are turns like any other — bid round follows.

## 3. Close

1. Update the OPEN ledger: deliberately-unresolved questions, provenance,
   wake conditions, the null wake condition. No synthesis at close.
2. Closing round: each seat, no bids.
3. Curation pass on the transcript only for mechanics (numbering,
   blockquote formatting) — never rewrite anyone's words.

## 4. Publish (run → session .jsonl → HF)

From the repo root, with the `hf` CLI authenticated (`docs/SECRETS.md`):

```bash
parlor/bin/publish-session.sh parlor-session-<n>
```

The script mirrors `parlor/` (sessions, transcript(s), PROTOCOL; not
`bin/`) to the `jasonyandell/mk5-run-evidence` dataset at its
repo-relative path, creates the tag, and prints the pinned URL.

Then, in the same session:

- Cite the **pinned** URL (never `main`) from `wiki/entities/parlor.md`
  and summarize the session there — the wiki page is the scientific
  record; the transcript is the keepsake.
- Confirm the split: raw `parlor/sessions/**.jsonl` stays out of git
  (tier 3, gitignored); the curated transcript is committed AND mirrored
  (`wiki/decisions/run-artifacts-policy.md` § Conversation and session
  logs).
