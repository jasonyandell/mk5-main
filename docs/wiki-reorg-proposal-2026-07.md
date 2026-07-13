# Wiki Reorganization Proposal — 2026-07-13

**Status:** adopted and executed 2026-07-13 (same branch). All five waves ran;
`scripts/wiki_lint.py --strict` exits clean. The appendix preserves the pre-adoption
audit numbers. Open choices resolved as recommended, plus: canonical w42 rollup =
`w42-book-claim-synthesis-and-ai-directions`.

**Method:** full crawl of all 507 pages (~498k words) by four parallel surveyors
(navigation layer; entities/topics taxonomy; experiments; decisions/playbooks/sources/log),
cross-checked by an independent mechanical pass over the link graph and frontmatter.
Every load-bearing claim below was verified against the file it cites.

---

## TL;DR

The wiki's *content* is in good shape — link integrity is near-perfect, the log
rotation works, the best pages (jud, partnership-wall-research, the engine cluster)
are excellent. What's failing is the **schema layer**: the conventions in AGENTS.md
no longer describe the corpus, the boundaries between kinds are undefined exactly
where agents actually file pages, routing hasn't kept up with growth, and nothing
is checked mechanically — so every rule drifts until an audit notices.

Five findings, five fixes:

| # | Finding | Fix |
|---|---|---|
| 1 | Schema–corpus divergence: 84 pages use statuses AGENTS.md never defined; 43 pages have malformed `last_updated`; a whole source type is uncodified | Amend AGENTS.md to match validated practice (§A1–A3) |
| 2 | `status: active` is decoration: ~80 pages marked active were last touched in April–May; finished work never gets flipped | Lifecycle rules + one-time normalization (§A1, Wave 1) |
| 3 | Routing debt: index.md misses 31 pages; `gus.md` is a 726-line "hub" routing 1 of 22 leaves; 8 competing w42 rollups, 4 routers calling a superseded page "live" | Routing rules: mandatory leaf coverage, one-current-rollup, hub size budget (§A5, Wave 3) |
| 4 | Kind boundaries undefined at the edges: topic-vs-experiment double-filing produced **all 3 orphans in the wiki**; 7 era chronicles are trails filed as topics; 17 book digests are sources filed as experiments | Kind decision rules (§A4) + re-kind moves (Wave 2) |
| 5 | No enforcement: every convention is a hope | `scripts/wiki_lint.py`, report mode now, `--strict` as a gate after Waves 1–2 (§B) |

---

## What's healthy — protect it during cleanup

The surveyors were asked to find problems and explicitly cleared these. A reorg
that "fixes" them would be a regression:

- **Link integrity.** Zero ambiguous bare names across 507 files, only 12 dead
  links in live pages, and only 3 orphan pages in the entire wiki. Obsidian-style
  bare-link resolution is safe — which also means **directory moves are cheap**
  (links resolve by basename; only `index.md` and stale path-qualified links need
  updating, and the linter catches both).
- **The log system.** `log.md` is a genuine changelog: 13 entries + a digest that
  is the single best orientation object in the wiki. The `grep "^## \["` timeline
  works across all 169 entries in both files. Only two stale count labels (§Wave 1).
- **decisions/.** No orphans, median ~9 inbound links, and the deliberately
  content-shaped pages (e.g. `decisions/research-lane-selection.md`) read better
  than any ADR template would. **Do not impose a template.**
- **The playbooks split.** perf-sprint's 6-page hub/template/appendix structure is
  right: templates paste clean, appendices don't bloat the hub.
- **Intentional story/spec splits.** `suit-algebra` vs `suit-algebra-spec`,
  `multiplayer-lineage` vs `multiplayer-pattern`, and the engine hub-and-cluster
  are the wiki at its best — history and current-reference deliberately separated,
  each page naming the split. This is a *pattern to codify*, not duplication.
- **The strongest recent pages.** `entities/jud.md` (the one discoverability trace
  that worked end-to-end), `trails/partnership-wall-research.md` (the model trail),
  and the champion-era "registered predictions, graded" experiment format — the
  best writing in `experiments/`.

---

## The five findings, with evidence

### 1. The schema and the corpus disagree

AGENTS.md declares `status: active | retired | superseded`. The corpus contains
**nine** status values: `active` (272), `complete` (78), `superseded` (69),
`retired` (25), plus one-offs `closed-on-completion` (2), `contradicted`,
`reframed`, `plan`, `retired-as-lem-base`. `complete` is not noise — it is a
de-facto fourth value with real semantics ("ran and concluded; the finding
stands; nothing replaced it") that the schema lacks, used consistently on 78
pages. The corpus is right and the schema is wrong.

Same shape elsewhere:

- `last_updated` format drift on 43 pages: shortsha (most), bare dates,
  `local-2026-05-01`, and one literal `era4`.
- `sources/` contains a second, legitimate, uncodified type: named per-document
  snapshots (`mccfr-exploration.md`, `research-night-2026-07-13.md`,
  `claude/era*.md`, `book-second-pass-2026-07-07/reader-*.md`) alongside the
  per-commit sha digests AGENTS.md describes. Well-formed, load-bearing, invisible
  to the schema.
- Extra frontmatter keys (`phase:`, `supersedes:`, `superseded_by:`) on ~10 pages,
  undocumented. `phase:` on `entities/champion.md` is the best frontier signal in
  the wiki — and it's not in the schema.

### 2. Status is not a frontier signal

Mapping every `status: active` content page's `last_updated` to a real date:
**74 pages marked active were last touched in April 2026** (the gus/burl era),
plus ~15 more from May. Pages whose own body contradicts the frontmatter include
`entities/iter3-rules-adapter.md` (active; body: "the chain stopped because the
mechanism it fed was abandoned"), `entities/gen-fleet.md` (`plan`; body: "never
launched... superseded"), `topics/rules-as-tools.md` (active; body has a
"Status (dormant since 2026-05)" section). This is exactly anti-rot rule #1's
"a status field that is never flipped is decoration."

There's a deeper problem than staleness: `active` conflates **live workstream**
(jud, champion) with **settled durable truth** (`suit-algebra`, `belief-bayes-ceiling`,
the engine findings). Both are "endorsed by the frontier," so both say `active`,
and a reader cannot tell current work from finished knowledge without opening
each page. That's the reason `complete` was invented in the field.

The most-consulted current-frontier page has the same disease at higher stakes:
`entities/champion.md`'s `phase:` field still says "#26 = live frontier ...
closed 2026-06-14" — a month behind its own body, which runs through jud v0/v1
and stage-0 closure.

### 3. Routing debt: leaves grow faster than routes

- **index.md** claims to catalog every page and misses 31, including the entire
  `w42-bookval-v1-wave2-*` cluster and the two most recent source digests
  (`sources/b28fb55a.md`, `sources/research-night-2026-07-13.md` — the newest
  ingest never got catalogued). At 93 KB it exceeds a single Read call, its
  Experiments section is a flat 146-entry list with no sub-headers, and it's
  organized by page-kind while agents query by workstream.
- **Hubs bloat where trails are missing.** AGENTS.md defines hubs as "*short*
  entity pages that summarize and route." Measured: engine 68 lines, forge 127
  (on-spec) … champion 341, jud 386 (bloated) … burl 586, gus 726 (**the entire
  ingest history pasted inline**). The pattern is exact: gus and burl are the two
  biggest un-trailed clusters, so the hub became the evidence pile. `gus.md`
  routes 1 of its 22 experiment leaves; the linter finds 13 gus pages unreachable
  from any hub or trail even allowing a rollup hop.
- **Eight competing rollups** for the w42 book-validation campaign (ledger schema,
  populated ledger, phase-2 synthesis, phase-4 board, phase-4 audit, two superseded
  reports, and a 48 KB post-phase-4 synthesis) — none authoritative. Sharpest
  symptom: `experiments/w42-book-validation-campaign.md` is `status: superseded`
  ("Dormant since Wave 5") yet four live routers still advertise it —
  `trails/w42-book-validation.md:105` ("the live campaign surface"),
  `entities/w42.md:77` and `:159`, `experiments/winning42-strategy-measurement.md:31`.
  A cold agent following any hub lands on a dormant page labeled live.
- **Discoverability traces** (cold agent, starting from the hub): "what was jud
  v1's result?" ✅ works. "Did the 84-bid claim validate?" ❌ smeared across ≥5
  pages with no reconciling verdict — `w42-84-claim-validation.md` itself admits
  "no page reconciled this page's rows against that later work." "What did the
  reentry test conclude?" ❌ two pages, two verdicts, and the v1 page isn't linked
  from the trail at all.

### 4. Kind boundaries are undefined exactly where filing happens

- **topic vs experiment** — the costliest one. Gus findings were systematically
  double-filed: a `topics/` concept page *and* an `experiments/gus-*` page for the
  same result (shine-analysis, probe-analysis, blunder-detector,
  consistency-regularizer, pi-opp-head, belief-co-train, the lamir1 pages). The
  experiment twin got all the inbound links; the topic twin rotted. **The only
  three orphans in the wiki — `topics/shine-analysis`, `topics/probe-analysis`,
  `decisions/engine-adrs` — are two of these shadow twins plus one leaf.** An
  agent with a new finding has no rule for which page to update.
- **Era chronicles filed as topics.** Seven narrative walkthroughs with date-range
  titles, several self-describing as "the narrative hub" (`the-gestation`,
  `breakthrough-and-oracle`, `the-analysis-epic`, `eq-genesis`,
  `alphazero-under-imperfect-information`, `the-book-enters`,
  `pre-ml-ai-attempts`) sit in `topics/` — but they are exactly AGENTS.md's
  definition of a trail, and `trails/` already holds this shape.
- **Book digests filed as experiments.** All 17 `winning42-*` chapter pages are
  chapter → measurable-concepts harvests: no question, no method, no result, no
  verdict. They are per-document source digests (the uncodified named-source type
  from finding 1), inflating the experiment count by 10% and mixing genres.
- **Receipts filed as entities.** 10 adapter pages track two linear supersession
  chains (stage-0→kerry→v3…v10; burl-iter0→iter1→iter3), nine of ten
  superseded/complete. Each is a legitimate receipt, but `kind: entity` overstates
  a training-run artifact, and ten flat siblings carry the lineage story worse
  than one lineage page would.
- **Naming collision hazard:** `entities/arena.md` (active champion harness) vs
  `entities/selfplay-arena.md` (retired Burl orchestrator) — two systems,
  near-identical names, bare `[[arena]]` links now ambiguous to humans (the file
  resolver is fine; readers aren't).

### 5. Nothing is enforced mechanically

None of the above is anyone failing to care — it's drift with no detector. The
lint pass in AGENTS.md is judgment-only ("run when the wiki feels off"). All five
findings were sitting in plain sight, mechanically checkable, invisible until an
audit. The July archaeology's 211-discrepancy cleanup added anti-rot *rules*; it
didn't add a *tool*, so the same classes of rot restarted immediately (the wave-2
pages that landed after it are the ones missing frontmatter and index entries).

---

## The proposal

### A. Amend AGENTS.md — make the schema describe validated practice

Paste-ready language; each block replaces or extends the named section.

**A1. Status enum** (replace the three-value line):

> ```yaml
> status: active | complete | retired | superseded
> ```
> - `active` — a live workstream: someone would add to this page this month.
> - `complete` — ran and concluded; the finding stands; nothing replaced it.
>   The terminal state for experiments, campaigns, and era chronicles.
> - `retired` — the frontier abandoned this line; the page is history.
> - `superseded` — a named replacement exists; a forward link is **required**.
>
> Lifecycle rules: an experiment page leaves `active` in the same session its
> result lands — `complete`, `retired`, or `superseded`, never "active by
> default." Verdicts (`contradicted`, `underpowered`, …) are body content, not
> status values. A page may not stay `active` across an era boundary without a
> `last_updated` bump.

**A2. Timestamps** (replace the `first_seen`/`last_updated` sha convention):

> `first_seen` and `last_updated` are dates, `YYYY-MM-DD` — the commit date, not
> the sha. Provenance shas stay in body citations (`([… @ a8bccfa](../sources/a8bccfa.md))`),
> which pages already carry. Rationale: shas proved unorderable by readers and
> drifted into `local-*` forms; dates make staleness visible and mechanically
> checkable. *(Judgment call — see Open Choices.)*

**A3. Named sources** (add under "What goes in the wiki vs. what stays in sources"):

> Sources are two shapes: per-commit digests `sources/<sha>.md`, and named
> per-document snapshots `sources/<slug>.md` (postmortems, research beads,
> conversation-era digests, reader reports, book-chapter digests) for artifacts
> with no single commit. Both use `kind: source` and are cataloged in `index.md`.
> Multi-file report bundles live in a dated subdirectory
> (`sources/book-second-pass-2026-07-07/`).

**A4. Kind decision rules** (add to Page conventions):

> Filing a new page, in order:
> 1. Is it a walkthrough of other pages (a narrative, an era history, a curated
>    reading order)? → `trails/`.
> 2. Is it a digest of an external or historical artifact (a commit, a doc, a
>    book chapter, a conversation)? → `sources/`.
> 3. Did someone run something and get a result? → `experiments/`. **The finding
>    lives on the experiment page.** A `topics/` page for the underlying concept
>    exists only when ≥2 experiments cite it or the concept outgrew its origin —
>    and then it routes to the experiments rather than restating them.
> 4. Is it an explicit choice with alternatives that were rejected? → `decisions/`.
> 5. Is it a named, durable system or artifact (repo subproject, model, adapter
>    lineage, external tool)? → `entities/`. A training-run artifact is a receipt
>    (experiment), not an entity; the *lineage* is the entity.
> 6. Otherwise — a concept, method, or synthesized piece of knowledge → `topics/`.

**A5. Routing rules** (add to Navigation roles):

> - Every `experiments/` page is linked (bare) from its family hub, a trail, or a
>   campaign rollup that is itself so linked. The linter's RT01 check enforces
>   reachability; "linked from index.md" does not count as routed.
> - Each campaign has exactly **one** `status: active` rollup; all others are
>   `complete`/`superseded` with a forward link to it. The rollup enumerates every
>   member as a real backlink — never a prose range ("ch01 through ch16" left
>   ch02–ch15 with no inbound link from their own trail).
> - Hubs have a size budget: ~150 lines. When a hub exceeds it, extract a trail
>   and thin the hub to summary + routes. (gus.md at 726 lines is the cautionary
>   tale.)
> - When two families share a numbering vocabulary (`phase*`, `wave*`, `v*`,
>   `iter*`, rung `#N`), each rollup opens with a one-line "which clock is this"
>   note. When a body cites a wave/rung with no page, link the page that absorbed it.

**A6. One home per fact** (add to Page conventions):

> A frontier fact lives on exactly one page; everywhere else links to it. ("The
> current best player is `margin:wp`(head_8)+`lens:ev`" is today restated verbatim
> on champion, jud, the-wall, and partnership-wall-research — four pages that must
> move in lockstep, and the lockstep has already slipped once.)

**A7. The `phase:` field** (add to Frontmatter):

> Optional, hubs only: `phase:` is a one-paragraph live tracker of the current
> frontier, and must carry a date. If the date is older than the page body's
> newest claim, the field is stale — fix it in the same session.

### B. Enforcement — `scripts/wiki_lint.py`

Ships in this PR. Checks (codes in the file's docstring): frontmatter presence
and required fields, kind-vs-directory, status enum, timestamp format, dead
links, ambiguous bare names, path-qualified body links, orphans, experiment
routing (RT01), bidirectional index coverage, log rotation count, unlinked
sources. Report mode today: **112 errors, 1009 warnings**. Wire-up path:

1. Now: replace the judgment-only lint list in AGENTS.md with "run
   `python -u scripts/wiki_lint.py`, then the judgment checks" (contradictions,
   silently-resolved questions — those stay human).
2. After Waves 1–2 zero the errors: add `--strict` to the session-completion
   checklist, and optionally a CI step. WARN-level checks (qualified links,
   routing) stay advisory until Wave 3/5 clears them.

### C. Cleanup waves — each sized to one session, in leverage order

**Wave 1 — metadata truth (mechanical, low risk).**
Normalize the 6 one-off statuses (`closed-on-completion`→`complete` ×2,
`contradicted`→`complete` + verdict to body, `reframed`→`superseded` + link,
`plan`→`superseded` for gen-fleet — its body says "never launched",
`retired-as-lem-base`→`retired`). Flip the stale actives (~40 experiment leaves
+ the 6 body-contradicts-frontmatter pages the taxonomy survey names). Fix the
43 malformed timestamps. Refresh `entities/champion.md`'s `phase:` and
`last_updated`. Add the 31 missing index entries, remove the `[[page]]` phantom.
Fix the 12 dead links (incl. the `[[user_role_and_north_star]]` memory-slug leak
in `topics/at-risk-points.md`). Correct the log digest labels ("145 archived" →
156, range → 2026-07-10). Rename `selfplay-arena` → `burl-selfplay-arena`.
Create `decisions/beads-to-gh-issues.md` (the one genuinely undocumented major
decision). Add the one-line staleness note to `playbooks/perf-sprint.md`
(its TeamCreate/team_name plumbing predates the current Agent surface).
Sweep `questions/open.md`: ~17 of 25 open questions belong to dead lineages
(LEM Stage-0/1, Burl) — move to `resolved.md` as "closed by pivot, not answered,"
which `entities/lem.md` already states in prose.

**Wave 2 — re-kind moves (bare links make these cheap).**
Move the 7 era chronicles `topics/` → `trails/` (`kind: trail`). Move the 17
`winning42-*` pages → `sources/` (`kind: source`). Move the 3 single-experiment
topics (`belief-co-train`, `belief-propagation-gap`, `belief-bayes-ceiling`) →
`experiments/`. Re-home `topics/book-strategy-player-phase-1-build.md` (a build
contract for never-built work) into `decisions/` or merge into its entity page;
same review for `-recording`/`-extension-points`. Update `index.md` sections and
grep `CLAUDE.md` + skills for any path-qualified references to moved pages. Run
the linter — IX/LN checks catch every missed reference.

**Wave 3 — routing (the biggest usability win).**
Write `trails/gus-line.md` (routes all 22 gus leaves; fixes the 13 RT01 warnings)
and `trails/champion-ladder.md` (rungs #20→#33, currently reconstructable only
from two bloated hubs). Thin `gus.md` and `burl.md` to hub size against the A5
budget. Fix the four "live campaign" pointers to say what `w42-book-validation-campaign.md`
already knows about itself. Designate **one** current w42 rollup (recommend: the
phase-4 final audit as `complete` baseline + `w42-book-claim-synthesis-and-ai-directions`
as the routing synthesis — or collapse; Jason's call), demote the rest with
forward links. Write the two missing verdict reconciliations: one paragraph on
`w42-84-claim-validation.md` pointing through phase-3/4 work, one on the reentry
pair saying which verdict stands. Merge or forward-link the orphaned shadow
topics (`shine-analysis`, `probe-analysis`) into their experiment twins.

**Wave 4 — index restructure.**
Split `index.md` into per-kind files (`index.md` becomes a thin router +
"start here"; `index-experiments.md` etc.), or keep one file but give the
Experiments section workstream sub-headers (LEM / Burl / Gus / W42-claims /
W42-bookval / champion-jud) — the axis agents actually query. Either way the
linter's bidirectional coverage check keeps it honest from then on. Add the
missing sources to the catalog with hooks.

**Wave 5 — optional/judgment (each independently skippable).**
(a) Adapter lineage consolidation: 10 entity pages → 2 lineage pages
(`stage-0-adapter-line`, `burl-adapter-line`) with comparative eval tables,
per-adapter receipts re-kinded to experiments. (b) The 984 path-qualified body
links: one mechanical sweep to bare form. (c) Champion-quartet consolidation:
apply A6 so champion/jud/the-wall/partnership-wall each own distinct facts.
(d) `playbooks/research-night.md` if that operating mode recurs (currently
trapped in `sources/research-night-2026-07-13.md`).

### What NOT to do

- **No ADR template for decisions/** — the content-shaped pages are better.
- **Don't collapse the playbook split** or the story/spec page pairs (codify the
  pattern instead — it's now implicit in A4 rule 3's "routes rather than restates").
- **Don't mass-rename the w42 files** (`bookval-v1/v2/v3`, `phase2/3/4`). The
  numbering is genuinely confusing (three unrelated clocks), but renames churn
  hundreds of links for cosmetic gain; the A5 "which clock" notes + one current
  rollup solve the actual confusion at 1% of the cost.
- **Don't auto-generate index.md from frontmatter.** Considered and rejected: the
  hand-written hooks are synthesis (the index's real value), and lint-enforced
  bidirectional coverage already gives the no-page-lost guarantee generation would.
- **Don't add a `workstream:` field to 500 pages.** Slug prefixes + index
  grouping + hubs already encode it; the churn isn't worth a redundant axis.

### Open choices

1. **Timestamps (A2):** dates everywhere (recommended) vs keeping shas and just
   banning the `local-*`/`era4` forms. Dates win on readability + lintability;
   shas win on provenance-purity (anti-rot rule 5 is satisfied either way since
   body citations keep the shas).
2. **`winning42-*` destination:** `sources/` (recommended — they're per-document
   digests, the type A3 codifies) vs a new `references/` dir (rejected by default:
   new top-level dirs need to earn their existence).
3. **`complete` vs folding into `retired`:** recommend keeping `complete` — 78
   pages already use it and "finished, still true" vs "abandoned" is a real
   distinction the frontier voice needs.
4. **Which w42 rollup is *the* one** (Wave 3) — needs your read on whether the
   48 KB synthesis or the phase-4 audit is the page you'd actually send someone to.
5. **How hard to gate:** `--strict` errors-only in the session checklist
   (recommended), vs full CI, vs report-only forever.

---

## Appendix: linter output at proposal time (2026-07-13)

```
FM02  19  missing required frontmatter fields (wave-1/2 bookval pages)
FM04   6  one-off statuses (after admitting `complete` to the enum)
FM05  43  malformed last_updated (local-*, era4, bare dates under sha regime)
IX01  31  pages on disk missing from index.md
IX02   1  index entry with no file ([[page]])
LN01  12  dead links in live pages (+16 informational in frozen archive)
LN02   0  ambiguous bare names
LN03 984  path-qualified body links (should be bare)
OR01   3  orphans: topics/shine-analysis, topics/probe-analysis, decisions/engine-adrs
RT01  18  unrouted experiment pages (13 of them gus-*)
SC01   4  never-linked sources (the four reader-*-report files)
LG01   0  log rotation compliant (13 entries)
```

Verified quotes anchoring the routing finding: `trails/w42-book-validation.md:105`
("[[w42-book-validation-campaign]] is the live campaign surface") and
`entities/w42.md:77,159` ("the live multi-wave campaign") vs that page's own
frontmatter (`status: superseded`) and opening ("Dormant since Wave 5").
