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

### Ingest

Triggered by the orchestrator when advancing to a new commit or commit cluster.

1. **Archivist** reads the raw sources (docs, code diffs, commit message) at the target sha and produces a compact bundle: entities introduced, entities updated, concepts introduced, concepts updated, experiments performed, decisions made, questions raised.
2. **Scribes** run in parallel on disjoint page sets. Each scribe receives the bundle + current contents of the pages it must touch. Each scribe's job is to update the page so it reflects the new frontier.
3. **Indexer** updates `index.md` (adds/updates page catalog entries) and `log.md` (appends a timestamped entry for this ingest).

A single ingest typically touches 10-15 pages (Karpathy's rule of thumb). If an ingest touches fewer than 3, the unit was probably too small; if it touches more than 20, split it.

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
