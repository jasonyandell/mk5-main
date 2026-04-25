# AGENTS.md — Wiki Schema for `wiki/`

This file tells any LLM agent how to read, write, and extend this wiki.

## What this wiki is

This is Karpathy's LLM-Wiki pattern ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) applied to this project. Instead of ingesting sources as they arrive in real time, we **replay the project's git history** commit-by-commit and build the wiki as if we'd been doing this since the first commit.

The raw sources are the project's own docs and code, read at specific commits via `git show <sha>:<path>`.

The wiki lives in `wiki/` and is an LLM-owned, compounding artifact. Humans curate; the wiki compiles.

## Three voices — and the one we use

We track **the frontier**: what the project understood to be true *at this commit*. We do not distinguish "what we thought then" vs. "what we later learned." We just update pages as the frontier moves.

- If the project at commit A believed STaR was the path to reasoning transfer, the [[star]] page at ingest-A reads like STaR is the plan.
- If at commit G the project redirects away from STaR, we update [[star]] then. We do not preemptively hedge early pages.
- `log.md` provides the timeline. Individual pages evolve in place.

## Directory layout

```
wiki/
├── AGENTS.md          ← this file
├── index.md           ← catalog of every page, updated every ingest
├── log.md             ← chronological append-only record of ingests
├── entities/          ← named things: projects, people, models, systems, named artifacts
├── topics/            ← concepts and methods: STaR, backwards curriculum, K1 grading
├── experiments/       ← a specific training run, eval, or named experiment
├── decisions/         ← explicit design choices worth their own page
├── sources/           ← per-commit or per-doc digest pages
├── trails/            ← thematic walkthroughs stitching pages together
└── questions/open.md  ← open questions raised but not yet answered
```

Do not create directories or pages speculatively. Create a page the first time an ingest needs it.

## Page conventions

### Frontmatter

Every page starts with YAML frontmatter:

```yaml
---
title: Human Readable Title
kind: entity | topic | experiment | decision | source | trail
first_seen: <commit-shortsha>
last_updated: <commit-shortsha>
status: active | retired | superseded
---
```

`status: active` means the frontier still endorses this. `retired` means the project moved on. `superseded` means another page replaced it; link forward.

### Body

- 3rd person, declarative. No "we did X." No "I believe Y."
- Terse. Wiki pages are reference, not essays.
- Every named entity or concept is a `[[backlink]]` on first mention in a section.
- **Backlink style: prefer bare** (`[[star]]`, `[[backwards-curriculum]]`, `[[gemma-4-e2b]]`). Obsidian's fuzzy resolution finds the file regardless of directory. This keeps page bodies readable.
- Use the fully-qualified form `[[path/name|alias]]` only in `index.md`, in log entries that need to disambiguate cross-kind pages, and when two pages share the same bare name (in which case rename one).
- Cite sources inline: `([lem/OVERVIEW.md @ a8bccfa](../sources/a8bccfa.md))`.
- Headings: `##` for major sections, `###` for subsections. No `#` — the title is in frontmatter.

### Filenames

- Lowercase, hyphen-separated, no spaces, no underscores.
- `entities/lem.md`, `topics/backwards-curriculum.md`, `experiments/stage-0-v1.md`.

## Operations

### When to update the wiki

If you ship a commit, add a doc section, close an experiment, or retire a decision — **update the wiki in the same session**. Don't batch. The wiki stays useful only if it tracks the current frontier; a week-late update is usually a rewrite.

Concrete triggers:
- A new commit lands that introduces/retires an entity, topic, experiment, or decision.
- An experiment produces a result that falsifies or confirms a hypothesis already on a page.
- A doc under `lem/`, `burl/`, `gus/`, or `forge/` gets a new section worth citing.
- A page on disk is wrong about the current frontier. Fix it in place — don't leave a "then vs. now" note.

### How to update: single-commit path

For one commit you just made or are ingesting:

1. **Read the commit.** `git show <sha>` for the message + diff. Read any new/changed docs it references.
2. **Decide which pages move.** Usually 3–8 pages: one entity + 1–2 topics + 1 experiment + the commit's `sources/<sha>.md` digest.
3. **For each touched page:**
   - Update the body so it reads as the current frontier (no hedging, no "previously we thought"). Bump `last_updated` to the commit sha.
   - If the page is being retired/superseded, set `status:` accordingly and add a forward link.
4. **Create `sources/<sha>.md`** — a compact digest: frontmatter + commit message quote + files-changed table + short "What this commit establishes" narrative + bare `[[backlinks]]` to every touched page.
5. **Update `index.md`** — add new pages to the catalog, update hooks on pages whose gist changed.
6. **Append to `log.md`** — one entry in the format below.
7. **Run the lint pass** (see Lint section). Fix orphans and broken backlinks before you stop.

### How to update: multi-commit replay path

For a batch of commits (e.g. catching up after a quiet period, or replaying a trail from scratch):

1. **Slice the range into ingests** along natural finding/document boundaries, not one-commit-one-ingest. A good ingest has a coherent theme and 10-15 touched pages (Karpathy's rule of thumb). Fewer than 3 touched pages → the unit is too small; more than 20 → split it.
2. **Run ingests chronologically** — earlier commits first, so later ones can revise pages instead of having to preemptively hedge.
3. **Per ingest, run three roles** (in one session or via a scribe team):
   - **Archivist** reads the raw sources (docs, code diffs, commit message) at the target sha(s) and produces a compact bundle: entities introduced, entities updated, concepts introduced, concepts updated, experiments performed, decisions made, questions raised.
   - **Scribes** update page sets in parallel on disjoint directories (one per `entities/` / `topics/` / `experiments/`). Each receives the bundle + current page contents and rewrites so the page reflects the new frontier.
   - **Indexer** updates `index.md` (page catalog) and `log.md` (one entry per ingest) and sweeps for dead links + orphans.
4. **Don't preemptively hedge.** If commit A believed X and commit G reverses it, write A's pages as if X is true; revise at G.

### Scribe dispatch pattern (for team-based ingests)

When orchestrating scribes via a team tool:
- One scribe per top-level dir (`entities/`, `topics/`, `experiments/`) to keep edits disjoint.
- Pass each scribe: commit range, theme, which pages to create vs. update, explicit instruction to check `topics/` and `entities/` before creating to avoid duplicates, and any backlink conventions to avoid (e.g. stale names from a prior rename).
- Run the indexer after scribes finish — never in parallel with them — so it sees a stable state.
- Scribes signal completion by file-on-disk check, not just confirmation messages. If a page's timestamp hasn't moved, the work hasn't landed.

### Query

Ask questions against the wiki. Start at `index.md`, drill into relevant pages, follow backlinks.

### Lint

Periodic health check:
- Contradictions between pages at the same frontier
- Orphan pages (no inbound links)
- Concepts mentioned without their own page
- Pages that say "TBD" or `?` and never got filled
- Stale `status: active` claims that a later ingest should have flipped to `retired`

## Log format

`log.md` entries use this prefix so `grep "^## \[" log.md` gives a clean timeline:

```
## [YYYY-MM-DD | <shortsha> | <subject>]

**Touched pages:** [[page-a]] [[page-b]] ...
**Added:** ...
**Updated:** ...
**Retired:** ...
**Questions opened:** ...
```

## Question handling

If an ingest raises a question the current sources don't answer:

1. Log it in `questions/open.md` with the sha that raised it.
2. Mark the relevant page's claim with `?` and a short note.
3. Keep ingesting. Do not stall to resolve.

When a later ingest resolves the question, remove it from `questions/open.md` and append to `questions/resolved.md` with the resolving sha.

## What goes in the wiki vs. what stays in sources

- **Wiki** = synthesis, cross-reference, current frontier. Terse.
- **Sources** = verbatim doc snapshots at a specific sha, for citation. Long-form raw extracts belong here.
- When a page needs to quote a source, quote sparingly and link to `sources/<sha>.md` for the full text.

## Backlinks, not categories

Organization emerges from backlinks, not from rigid category taxonomy. The Obsidian graph view should show clustering; if it doesn't, the wiki is under-linked.
