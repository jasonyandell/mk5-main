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
├── index.md           ← thin router; catalogs split per kind in index-<kind>.md
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

Routing rules:

- Every `experiments/` page is linked (bare) from its family hub, a trail, or a
  campaign rollup that is itself so linked. The linter's RT01 check enforces
  reachability; "linked from the index" does not count as routed.
- Each campaign has exactly **one** current rollup; all others carry
  `complete`/`superseded` and a forward link to it. The rollup enumerates every
  member as a real backlink — never a prose range ("ch01 through ch16" once left
  fourteen pages with no inbound link from their own trail).
- Hubs have a size budget: ~150 lines. When a hub exceeds it, extract a trail
  and thin the hub to summary + routes.
- When two families share a numbering vocabulary (`phase*`, `wave*`, `v*`,
  `iter*`, rung `#N`), each rollup opens with a one-line "which clock is this"
  note. When a body cites a wave/rung with no page, link the page that absorbed
  it.

## Page conventions

### Frontmatter

Every page starts with YAML frontmatter:

```yaml
---
title: Human Readable Title
kind: entity | topic | experiment | decision | source | trail | playbook
first_seen: YYYY-MM-DD
last_updated: YYYY-MM-DD
status: active | complete | retired | superseded
---
```

- `active` — a live workstream: someone would add to this page this month.
- `complete` — ran and concluded; the finding stands; nothing replaced it. The
  terminal state for experiments, campaigns, and era chronicles.
- `retired` — the frontier abandoned this line; the page is history.
- `superseded` — a named replacement exists; a forward link is **required**.

Lifecycle rules: an experiment page leaves `active` in the same session its
result lands — `complete`, `retired`, or `superseded`, never "active by
default." Verdicts (`contradicted`, `underpowered`, …) are body content, not
status values. A page may not stay `active` across an era boundary without a
`last_updated` bump.

`first_seen` and `last_updated` are dates (`YYYY-MM-DD` — the commit date, not
the sha). Provenance shas live in body citations
(`([… @ a8bccfa](../sources/a8bccfa.md))`), which pages already carry.

Optional, hubs only: `phase:` is a one-paragraph live tracker of the current
frontier, and must carry a date. If that date is older than the page body's
newest claim, the field is stale — fix it in the same session.

### Body

- 3rd person, declarative. No "we did X." No "I believe Y."
- Terse. Wiki pages are reference, not essays.
- Every named entity or concept is a `[[backlink]]` on first mention in a section.
- **Backlink style: prefer bare in body text.** Obsidian's fuzzy resolution finds the file regardless of directory, and bare links keep pages readable.
  - ✅ Body: `See [[burl-2000-harvest]]; tuned per [[max-tokens-2048-floor]]; resilient via [[batched-harvest-resilience]].`
  - ❌ Body: `See [[experiments/burl-2000-harvest]]; tuned per [[decisions/max-tokens-2048-floor]]; resilient via [[topics/batched-harvest-resilience]].`
  - Use the fully-qualified form `[[path/name|alias]]` **only** in the index files (`index.md`, `index-*.md`), in log entries that need to disambiguate cross-kind pages, and when two pages share the same bare name (in which case rename one).
  - When you create a new page, link to it bare from every page that mentions it — and audit your existing pages for stale qualified links to it.
- Cite sources inline: `([lem/OVERVIEW.md @ a8bccfa](../sources/a8bccfa.md))`.
- Headings: `##` for major sections, `###` for subsections. No `#` — the title is in frontmatter.

### Choosing a kind

Filing a new page, in order:

1. Is it a walkthrough of other pages (a narrative, an era history, a curated
   reading order)? → `trails/`.
2. Is it a digest of an external or historical artifact (a commit, a doc, a
   book chapter, a conversation)? → `sources/`.
3. Did someone run something and get a result? → `experiments/`. **The finding
   lives on the experiment page.** A `topics/` page for the underlying concept
   exists only when ≥2 experiments cite it or the concept outgrew its origin —
   and then it routes to the experiments rather than restating them.
4. Is it an explicit choice with alternatives that were rejected? → `decisions/`.
5. Is it a named, durable system or artifact (repo subproject, model, adapter
   lineage, external tool)? → `entities/`. A training-run artifact is a receipt
   (experiment), not an entity; the *lineage* is the entity.
6. Otherwise — a concept, method, or synthesized piece of knowledge → `topics/`.

### One home per fact

A frontier fact lives on exactly one page; everywhere else links to it. Pages
that restate the same fact must move in lockstep when the frontier moves — and
they won't.

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
3. Use the catalogs (`wiki/index.md` router → `index-<kind>.md`) as the fallback route map, not as mandatory
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
5. **Update the catalog** — add new pages to the right `index-<kind>.md` (entry shape: `- [[path|name]] — hook (status)`), update hooks on pages whose gist changed. The linter's IX checks enforce bidirectional coverage.
6. **Append to `log.md`** — one entry in the format below.
7. **Run the lint pass** (see Lint section). Fix orphans and broken backlinks before you stop. Skim the touched pages once for `[[kind/name]]` patterns in body text — those should almost always be bare `[[name]]` (see Backlink style above).

#### Multi-commit replay path

For a batch of commits (e.g. catching up after a quiet period, or replaying a trail from scratch):

1. **Slice the range into ingests** along natural finding/document boundaries, not one-commit-one-ingest. A good ingest has a coherent theme and 10-15 touched pages (Karpathy's rule of thumb). Fewer than 3 touched pages → the unit is too small; more than 20 → split it.
2. **Run ingests chronologically** — earlier commits first, so later ones can revise pages instead of having to preemptively hedge.
3. **Per ingest, run three roles** (in one session or via a scribe team):
   - **Archivist** reads the raw sources (docs, code diffs, commit message) at the target sha(s) and produces a compact bundle: entities introduced, entities updated, concepts introduced, concepts updated, experiments performed, decisions made, questions raised.
   - **Scribes** update page sets in parallel on disjoint directories (one per `entities/` / `topics/` / `experiments/`). Each receives the bundle + current page contents and rewrites so the page reflects the new frontier.
   - **Indexer** updates the `index-<kind>.md` catalogs and `log.md` (one entry per ingest), then runs `scripts/wiki_lint.py`.
4. **Don't preemptively hedge.** If commit A believed X and commit G reverses it, write A's pages as if X is true; revise at G.

#### Scribe dispatch pattern (for team-based ingests)

When orchestrating scribes via a team tool:
- One scribe per top-level dir (`entities/`, `topics/`, `experiments/`) to keep edits disjoint.
- Pass each scribe: commit range, theme, which pages to create vs. update, explicit instruction to check `topics/` and `entities/` before creating to avoid duplicates, and any backlink conventions to avoid (e.g. stale names from a prior rename).
- Run the indexer after scribes finish — never in parallel with them — so it sees a stable state.
- Scribes signal completion by file-on-disk check, not just confirmation messages. If a page's timestamp hasn't moved, the work hasn't landed.

### Lint — mechanical first, judgment second

Run `python -u scripts/wiki_lint.py` (from the repo root) after any session
that touched `wiki/` — it checks frontmatter validity, dead links, ambiguous
names, orphanhood, experiment routing, index coverage, and log rotation.
`--strict` must exit clean before pushing.

Then the judgment checks the linter can't do — run these when the wiki feels
off, or after a large ingest:

- Contradictions between pages at the same frontier
- Orphan pages (no inbound links)
- Concepts mentioned without their own page
- Pages that say "TBD" or `?` and never got filled
- Stale `status: active` claims that a later ingest should have flipped to `retired`
- Backlinks to non-existent pages (dead links)
- Open questions in `questions/open.md` that a later commit silently resolved
- Plan or wiki claims of pre-existing artifacts (corpora, adapters, eval sets) that a manifest/config check could verify — flag for verification before downstream consumption.

## Log format

**The log is a changelog, not a chronicle.** An entry is the header line plus at most ~5 lines of pointers. If an entry wants a paragraph of synthesis, that content belongs on a page — write or extend the page and link it from the entry. The log carries no claim that isn't reachable through a link. (This rule exists because the log once grew to 3,600 lines of narrative nobody — human or LLM — could read; the history is preserved in `log-archive.md`.)

Entries use this prefix so `grep "^## \[" log.md log-archive.md` gives the full timeline:

```
## [YYYY-MM-DD | <shortsha> | <subject>]

**Touched pages:** [[page-a]] [[page-b]] ...
**Added:** ...
**Updated:** ...
**Retired:** ...
**Questions opened:** ...
```

**Rotation (mechanical, no judgment):** `log.md` holds a phase-by-phase digest at the top plus the most recent ~10 entries. When appending pushes it past ~15 entries, in the same session: move the oldest entries verbatim to `log-archive.md` (append at its end), and fold their gist into the digest — one line per phase, links only, no new claims. The entry count is the trigger; do not wait for the file to "feel" long.

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

Sources are two shapes: per-commit digests `sources/<sha>.md`, and named
per-document snapshots `sources/<slug>.md` (postmortems, research beads,
conversation-era digests, reader reports, book-chapter digests) for artifacts
with no single commit. Both use `kind: source` and are cataloged in the index.
Multi-file report bundles live in a dated subdirectory
(`sources/book-second-pass-2026-07-07/`).

## Backlinks, not categories

Organization emerges from backlinks, not from rigid category taxonomy. The Obsidian graph view should show clustering; if it doesn't, the wiki is under-linked.

## Anti-rot rules (added by the 2026-07 overhaul)

The July 2026 archaeology audited every page against the repo and found 211+ discrepancies with a consistent shape. These rules exist so the same rot cannot recur:

1. **Status must be falsifiable.** A `status:` field is a claim; it needs evidence reachable from the page. When an experiment's "next steps" never run, the page gets the one-line frontier truth ("the planned X never ran; the project pivoted to Y at `<sha>`") — past tense, forward-linked. A status field that is never flipped is decoration.
2. **Corrections must be reachable from the error.** When a result is retracted, reversed, or re-measured (Wave 3→4, the perf-batch retraction), the page that carries the stale number gets the correction or a forward link — a correction a cold reader can't reach from the error did not happen.
3. **Questions, not retrofitted goals.** Experiment pages state the question asked, in question form, subjunctive preserved ("can X…?"). Never "the goal was X" unless traceable to Jason's words. Goals vs instruments per [[the-wall]] — do not promote instruments to goals.
4. **Names doctrine.** Every named thing is classified BUILT (repo artifacts), IDEATED (conversation-only), or RENAMED (mapping stated). Conversation-space names (Harl, LLem, LEWM, walker) are real project knowledge but are never claimed as artifacts.
5. **Dates trace to artifacts.** Every date traces to a git timestamp, bead timestamp, or conversation `created_at` — never to a narrative's memory of a date (known failure: summaries shift dates by a year and migrate facts between eras).
6. **Privacy firewall for conversation-sourced content.** This repo is public domain; Jason's claude.ai conversations are not. `sources/claude/*` digests carry only Texas-42-project content — nothing about work, family, or any non-42 personal thread, not even in paraphrase. When in doubt, omit.
7. **"Worth a bead" is not a resting state.** A flagged risk either becomes a tracked issue in the same session or the flag is removed; parked risks evaporate (documented instance: the WorldSamplerMRV sampler bias).
8. **Audits leave no residue on pages.** The wiki is compressed context, not a maintenance record: a reader wants what's true, not the story of how the wiki got it right. An audit's corrections land in place; a genuinely unresolved source discrepancy gets a one-line note next to the number it qualifies; live questions go to `questions/open.md`; the audit event is one `log.md` entry, and the correction trail is git history. Pages never carry "reviewed on <date>" sections or verdict stamps.
