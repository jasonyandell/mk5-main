# AGENTS.md — Wiki Schema for `wiki/`

This file tells any LLM agent how to read, write, and extend this wiki.

## What this wiki is

This is Karpathy's LLM-Wiki pattern ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) applied to this project. The wiki was bootstrapped by **replaying the project's git history** commit-by-commit (LEM, Burl, and Gus trails). That bootstrap is done — the wiki now lives in steady-state use.

Three layers, per Karpathy:

- **Raw sources** — the project's own docs and code, plus historical snapshots in `sources/<sha>.md`. Read-only.
- **The wiki** — the markdown under `wiki/` minus `sources/`. LLM-owned, mutable, compiles to reflect the current frontier.
- **The schema** — this file. Tells you how to read, write, and extend the wiki.

The default mode for any agent in this repo is: **consult the wiki first, then read code.** Update the wiki as a side effect of any work that changes what's true. Don't batch updates.

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
├── playbooks/         ← repeatable how-to procedures (e.g. perf-sprint kickoff)
└── questions/open.md  ← open questions raised but not yet answered
```

Do not create directories or pages speculatively. Create a page the first time an ingest needs it.

## Navigation roles

The wiki should be usable without loading the whole catalog into context. Treat
pages as one of three navigation roles:

- **Frontier hubs** — short entity pages that summarize the current state of a
  major workstream and route to trails. Examples: [[lem]], [[burl]], [[gus]],
  [[w42]], [[forge]], [[engine]].
- **Trails** — curated walkthroughs through a cluster of pages. Trails are the
  preferred way to traverse large evidence piles without promoting every leaf
  page into a headline topic.
- **Leaf pages** — experiments, decisions, chapter inventories, receipts,
  source digests, and detailed reports. Leaf pages stay source-backed and
  discoverable, but they are not default orientation material.

This preserves the Karpathy/Obsidian shape: organization still emerges from
backlinks and curated trails, not from a rigid folder taxonomy.

## Page conventions

### Frontmatter

Every page starts with YAML frontmatter:

```yaml
---
title: Human Readable Title
kind: entity | topic | experiment | decision | source | trail | playbook
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
- **Backlink style: prefer bare in body text.** Obsidian's fuzzy resolution finds the file regardless of directory, and bare links keep pages readable.
  - ✅ Body: `See [[burl-2000-harvest]]; tuned per [[max-tokens-2048-floor]]; resilient via [[batched-harvest-resilience]].`
  - ❌ Body: `See [[experiments/burl-2000-harvest]]; tuned per [[decisions/max-tokens-2048-floor]]; resilient via [[topics/batched-harvest-resilience]].`
  - Use the fully-qualified form `[[path/name|alias]]` **only** in `index.md`, in log entries that need to disambiguate cross-kind pages, and when two pages share the same bare name (in which case rename one).
  - When you create a new page, link to it bare from every page that mentions it — and audit your existing pages for stale qualified links to it.
- Cite sources inline: `([lem/OVERVIEW.md @ a8bccfa](../sources/a8bccfa.md))`.
- Headings: `##` for major sections, `###` for subsections. No `#` — the title is in frontmatter.

### Filenames

- Lowercase, hyphen-separated, no spaces, no underscores.
- `entities/lem.md`, `topics/backwards-curriculum.md`, `experiments/stage-0-v1.md`.

## Operations

The three operations, in order of how often you'll do them: **Query** (every session), **Update** (when something changes), **Lint** (occasionally).

### Query — the default operation

Most sessions start here. Before reading code, check whether the wiki already knows.

**When to query:**
- The user asks "what is X / why did we Y / how does Z work" about LEM, Burl, Gus, forge, or any historical artifact.
- You're about to read a doc under `lem/`, `burl/`, `gus/`, or `forge/` for orientation — check the corresponding `entities/<project>.md` first; it's the synthesized view.
- You're picking up a thread from a prior session ("we were working on the consistency regularizer…") — `wiki/log.md` and the relevant entity page are the fastest catch-up.
- You hit a concept you don't recognize (LAMIR-1, qMAE plateau, q-bootstrap-belief, K1 grading, …). Bare `[[backlink]]` names usually map to a page; check there.
- You're about to design something the project has tried before. Check `experiments/` and `decisions/` for the prior art.

**How to query:**

1. If the user names a major family, start at the frontier hub or trail for that
   family: [[lem]], [[burl]], [[gus]], [[w42]], [[forge]], [[engine]], or an
   appropriate page under `trails/`.
2. If the user names a specific concept, jump straight to the likely bare page
   name or use `rg` over `wiki/` to find it. Prefer this to loading the full
   catalog when the slug or phrase is already known.
3. Use `wiki/index.md` as the catalog fallback and route map, not as mandatory
   first context for every query.
4. Read the relevant pages. Follow backlinks across roles: frontier hub → trail
   → leaf pages → decisions → source digests.
5. **Cite back to the wiki** when answering the user, with the page path. If the wiki doesn't have the answer, say so explicitly — don't invent.
6. **File good answers back.** If your synthesis was non-trivial — multiple pages stitched together, a question you had to dig for — capture it as a wiki update before ending the session. Either: extend an existing page with the new framing, create a new `topics/` page if the synthesis is reusable, create or extend a trail if the answer is mainly navigational, or open a `questions/open.md` entry if the answer revealed a gap.

The query → file-back loop is what makes the wiki compound. A query that doesn't leave the wiki better than it found it is a missed update.

### Update — when something changes

If you ship a commit, add a doc section, close an experiment, or retire a decision — **update the wiki in the same session**. Don't batch. The wiki stays useful only if it tracks the current frontier; a week-late update is usually a rewrite.

Concrete triggers:
- A new commit lands that introduces/retires an entity, topic, experiment, or decision.
- An experiment produces a result that falsifies or confirms a hypothesis already on a page.
- A doc under `lem/`, `burl/`, `gus/`, or `forge/` gets a new section worth citing.
- A page on disk is wrong about the current frontier. Fix it in place — don't leave a "then vs. now" note.
- A query revealed a gap, contradiction, or unsynthesized concept (see Query, step 4).

#### Single-commit path

For one commit you just made or are ingesting:

1. **Read the commit.** `git show <sha>` for the message + diff. Read any new/changed docs it references.
2. **Decide which pages move.** Usually 3–8 pages: one entity + 1–2 topics + 1 experiment + the commit's `sources/<sha>.md` digest.
3. **For each touched page:**
   - Update the body so it reads as the current frontier (no hedging, no "previously we thought"). Bump `last_updated` to the commit sha.
   - If the page is being retired/superseded, set `status:` accordingly and add a forward link.
4. **Create `sources/<sha>.md`** — a compact digest: frontmatter + commit message quote + files-changed table + short "What this commit establishes" narrative + bare `[[backlinks]]` to every touched page.
5. **Update `index.md`** — add new pages to the catalog, update hooks on pages whose gist changed.
6. **Append to `log.md`** — one entry in the format below.
7. **Run the lint pass** (see Lint section). Fix orphans and broken backlinks before you stop. Skim the touched pages once for `[[kind/name]]` patterns in body text — those should almost always be bare `[[name]]` (see Backlink style above).

#### Multi-commit replay path

For a batch of commits (e.g. catching up after a quiet period, or replaying a trail from scratch):

1. **Slice the range into ingests** along natural finding/document boundaries, not one-commit-one-ingest. A good ingest has a coherent theme and 10-15 touched pages (Karpathy's rule of thumb). Fewer than 3 touched pages → the unit is too small; more than 20 → split it.
2. **Run ingests chronologically** — earlier commits first, so later ones can revise pages instead of having to preemptively hedge.
3. **Per ingest, run three roles** (in one session or via a scribe team):
   - **Archivist** reads the raw sources (docs, code diffs, commit message) at the target sha(s) and produces a compact bundle: entities introduced, entities updated, concepts introduced, concepts updated, experiments performed, decisions made, questions raised.
   - **Scribes** update page sets in parallel on disjoint directories (one per `entities/` / `topics/` / `experiments/`). Each receives the bundle + current page contents and rewrites so the page reflects the new frontier.
   - **Indexer** updates `index.md` (page catalog) and `log.md` (one entry per ingest) and sweeps for dead links + orphans.
4. **Don't preemptively hedge.** If commit A believed X and commit G reverses it, write A's pages as if X is true; revise at G.

#### Scribe dispatch pattern (for team-based ingests)

When orchestrating scribes via a team tool:
- One scribe per top-level dir (`entities/`, `topics/`, `experiments/`) to keep edits disjoint.
- Pass each scribe: commit range, theme, which pages to create vs. update, explicit instruction to check `topics/` and `entities/` before creating to avoid duplicates, and any backlink conventions to avoid (e.g. stale names from a prior rename).
- Run the indexer after scribes finish — never in parallel with them — so it sees a stable state.
- Scribes signal completion by file-on-disk check, not just confirmation messages. If a page's timestamp hasn't moved, the work hasn't landed.

### Lint — occasional health check

Run when the wiki feels off, or after a large ingest:

- Contradictions between pages at the same frontier
- Orphan pages (no inbound links)
- Concepts mentioned without their own page
- Pages that say "TBD" or `?` and never got filled
- Stale `status: active` claims that a later ingest should have flipped to `retired`
- Backlinks to non-existent pages (dead links)
- Open questions in `questions/open.md` that a later commit silently resolved
- Plan or wiki claims of pre-existing artifacts (corpora, adapters, eval sets) that a manifest/config check could verify — flag for verification before downstream consumption.

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
