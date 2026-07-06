# Wiki Structural Audit

Scope: `/Users/jason/code/mk5-main/wiki/` (read-only). Built the link graph with grep + a small Python pass (`scratchpad/linkgraph.py`, `cluster.py`, `bfs.py`). Goal: find the structural failures that let a fresh strong model miss three in-wiki facts last night — the K&L look-ahead warning, candlewax bimodality, and the distillation-lineage prior art.

**Headline:** the graph is *connected* but not *signposted*. Every missed fact is reachable in 1–3 hops, so this is not a broken-link problem. It is a seam problem: the frontier hub (champion/jud) links to the old hubs (gus, burl) only generically, the load-bearing warnings are buried mid-page with no anchor, the same object has three names across three eras with no concordance, and there are only 3 trails for 431 pages. A reader has no *thematic reason* to follow the one edge that matters.

---

## 1. Census

| Kind (by dir) | Count |
|---|---|
| experiments | 160 |
| sources | 133 |
| topics | 66 |
| entities | 40 |
| decisions | 18 |
| playbooks | 6 |
| trails | **3** |
| questions | 2 |
| index / log | 2 |
| **Total** | **431** |

- **Frontmatter `kind`:** 11 pages carry no `kind` field (index, log, both `questions/`, and 7 others) — minor.
- **Status-vocabulary drift (lint):** the schema (`AGENTS.md`) defines only `active | retired | superseded`. On disk: 341 `active`, 55 *missing*, and a long tail of off-schema values — `complete` (17), `closed-on-completion` (2), `contradicted` (1), plus one-offs `mixed`, `exploratory`, `reframed`, `confirmed`, `plan`, `retired-as-lem-base`. Nine distinct statuses beyond the three allowed. This is harmless to reading but means `status:` cannot be trusted as a filter.
- **Date coverage:** older pages stamp git shas (`be7efc4`, `8c1bb14`, …); the Burl-lab-onward pages stamp `local-2026-05-0X`. The newest `last_updated` clusters are `local-2026-05-02` (36 pages) and `local-2026-05-03` (32 pages) — i.e. the last big *indexing* sweep predates the entire champion/jud campaign (jud v0/v1, plateau-probe, sources `0d82a97`), which landed 2026-07-06 and was appended without a re-lint. **The index's own "Start here" was last curated before the current frontier existed.**

### Orphans (no inbound content link; index/log excluded)
61 total, but the breakdown matters:

- **53 of 133 source digests (40%) are orphaned** — reachable only through index.md/log.md, never from the pages whose commit they document. `AGENTS.md` §Single-commit-path step 4 says each `sources/<sha>.md` should be back-linked from every touched page; that back-link is missing for 40% of them. Examples: `2a09050` (introduces the regret metric), `cf8ff79` (co-train falsified), `b42669a` (the 8-mode LAMIR ceiling). The evidence exists but is unreachable by topic.
- **5 topic orphans** — `lazy-iterable-dataset`, `past-belief-future-direction`, `probe-analysis`, `q-head-augmentation`, `shine-analysis`. All are Gus-cluster leaves that link *outward* but nothing links *in*. `probe-analysis` and `shine-analysis` in particular are interpretability receipts a fresh Gus reader would want and cannot find by traversal.
- **1 trail orphan** — `wiki-entrypoints` itself is only referenced from index.md, so the "route map for agents" is not reachable by following links from any content page.
- 2 `questions/` pages (expected).

### Dead links (target has no page): 17
Mostly template noise from AGENTS.md/log.md (`page-a`, `backlink`, `<sha>`, `...4 lists of 0..7…`). The **real** dead links:
- `burl-perf-phase1` — phases 0/2/3 have pages; phase 1 is referenced but absent.
- `plunge` — champion.md/jud say the champion "was exported to plunge as the `onyx` player"; `plunge` has no page.
- `explore-game-cache-bug`, `napkin-formula`, `spec-decode-acceptance` — referenced, never created.
- `project_gus_as_burl_belief_brain`, `user_role_and_north_star` — **memory-file slugs leaking into wiki body text** as `[[links]]`. These are `~/.claude/.../memory/` names, not wiki pages; they will never resolve.

---

## 2. Seam map

Clustered the 298 non-source content pages by era/subsystem (script assigns explicit membership + prefix rules; `decisions`/`experiments` buckets below are unassigned fallbacks and noisy — ignore them as clusters).

**Cluster sizes:** w42 106 · burl 55 · gus 46 · lem 33 · champion/jud 12 · forge/core 8 · perf-infra 6.

**Within-cluster link density is healthy** (w42 691, burl 295, lem 245, gus 221, champion/jud 61). The clusters are internally well-linked. The problem is *between* them.

### Cross-cluster crossings (both directions summed)

| pair | crossings | read |
|---|---|---|
| **champion/jud ↔ burl** | **3** | **the worst load-bearing seam** (see §Final) |
| champion/jud ↔ lem | 1 | genuinely unrelated eras — benign |
| champion/jud ↔ perf-infra | 0 | benign |
| gus ↔ lem | 14 | thin but eras are adjacent |
| lem ↔ w42 | 0 | unrelated eras — benign |
| perf-infra ↔ {gus, w42, lem, champion} | 0 each | perf-infra is an island touching only burl (8) |
| champion/jud ↔ gus | 35 | the one healthy cross-era seam |
| champion/jud ↔ w42 | 12 | ok (jud pages live under w42-* names) |
| gus ↔ w42 | 82 | strong (strategy-tag probes bridge them) |
| forge/core ↔ w42 | 51 | strong |
| burl ↔ lem | 95 | strong (handoff trail does its job) |

**Named near-zero seams that hide knowledge:**
1. **champion/jud ↔ burl = 3.** The entire tool-using-play body of work — candlewax, reasoning-coherence-verification, distribution-legibility, the belief-trajectory tooling — lives in `burl` (55 pages). The current frontier (`champion/jud`, 12 pages) touches it 3 times. This is the seam that hid candlewax.
2. **perf-infra is an island** (0 crossings to everything except burl). `batched-harvest-resilience`, `continuous-batching-dispatcher-design`, and the 6 perf-sprint playbooks are unreachable from any model-work cluster.
3. **champion/jud → gus is 35 but generic.** The link is hub-to-hub (`jud → gus`), not warning-to-consumer. jud reaches gus.md in 1 hop, but gus.md is 637 lines and the K&L warning is an unlabeled subsection at line 417 (see §4).

---

## 3. Vocabulary concordance candidates

The same object is named differently per era, and the names never co-occur — this is the mechanism of the misses. Grep evidence:

| Concept | Names in use | Where each lives | Do they co-occur? |
|---|---|---|---|
| **Multi-peaked outcome PDF** | "candlewax", "bimodal / multimodal", "melted blob" (jud's "un-melt the eq blob"), "mixed-mode geometry", "spike / disaster tail" | candlewax+bimodal → **burl cluster** (`topics/candlewax`, `tool-orchestration`, `reasoning-coherence-verification`, `iter5-e2`); "melted/blob" → **champion/jud cluster** (`jud`, `belief-conditioned-self-play`, `rank-vs-price`, `burl-lab`) | **NO.** `grep melt` and `grep candlewax` share zero pages. `w42-jud-v0` independently writes "bimodal realized-outcome support" without ever linking `[[candlewax]]`. |
| **Oracle value signal** | "E[Q]", "expected-q-value", "eq blob", "oracle E[Q]", "Q-value", "V_realized", "double-dummy" | 156 pages mention some form; the entity is `expected-q-value` | partial — the newer `V_realized`/`margin_net` (jud) is never tied back to `expected-q-value` |
| **Distilling the oracle into a small net** | "student-distillation", "dense-q-supervision", "distilled V/Q", "E[Q] distilled as bootstrap" (champion.md:176), "bid_net distillation", "consistency-regularizer" | distillation *lineage* topics → **gus cluster island**; "distilled as bootstrap" → **champion.md**, unlinked | **NO.** champion.md:176 proposes "E[Q] distilled as bootstrap" and never links `student-distillation`/`dense-q-supervision`/`lamir1-ceiling`. |
| **Look-ahead search over sampled worlds** | "PIMC", "LAMIR", "belief-weighted worlds", "judsearch", "world-sampling", "q-bootstrap-belief" | pimc/lamir1 → gus; judsearch → jud; belief-weighted worlds → champion | partial; `pimc` is the one term that does bridge (jud→pimc 1 hop) |

**Highest-value concordance to build:** a single page (or `topics/candlewax` alias-block) that declares **candlewax ≡ bimodal ≡ multimodal ≡ "melted eq blob" ≡ mixed-mode**, linked from both `jud` and `candlewax`. This one alias would have caught last night's collapse-by-mean.

---

## 4. Contradiction / blind-spot scan

Pages at the same frontier making claims that are incompatible *or* that one warns against the other with no cross-link:

1. **champion.md:176 "E[Q] distilled as bootstrap" (+ opponents-in-rollout) vs gus.md:417 K&L warning.** gus.md states verbatim: *"Kubíček & Lisý explicitly warn that a value function trained by distillation cannot be used for look-ahead reasoning: scalar noise in the distilled V/Q is enough to flip argmax at decision boundaries."* champion.md and jud both propose exactly distilled-value-for-look-ahead (`judsearch`, "E[Q] distilled as bootstrap"). **Neither champion.md nor jud.md nor `belief-conditioned-self-play` links the warning.** The K&L warning *pre-explains* the JS3 null in `w42-jud-v1` ("best test CE, zero play gain"), but the null page never cites it. This is miss #1 made structural.

2. **`w42-lens-v1-utility-head-to-head` "EV wins decisively" vs `topics/candlewax` "the mean is a poor summary."** lens-v1:135 declares *"Lens(ev) beats Lens(p_make) by +5.42 pts/hand … EV-greedy is the strongest of the four lenses."* candlewax:11 declares the mean *"misleadingly suggest[s] the decision is neutral."* **lens-v1 contains zero references to candlewax/bimodal; candlewax contains zero references to lens.** A reader who takes lens-v1 at face value (as the fresh model did) builds a mean-collapsing play policy — precisely miss #2. The reconciliation (EV wins *on aggregate* but hides risk *per-decision* on bimodal states) exists nowhere.

3. **`student-distillation` lineage is a sealed island.** `student-distillation`, `dense-q-supervision`, `lamir1-ceiling`, `pi-opp-head`, `belief-co-train`, `probe-analysis` are inbound-linked **only from each other** (all inbound sources are other gus topics). Nothing in champion/jud/w42-jud reaches them by traversal. A frontier reader re-proposes distillation as novel because the prior-art cluster has no edge into the frontier — miss #3 made structural.

---

## 5. Entry-path test (fresh reader simulation)

BFS over the real link graph (`bfs.py`):

**From the full "Start here" hub set** (wiki-entrypoints, lem, burl, gus, w42, w42-book-validation, book-strategy-player, champion): candlewax **2 hops**, student-distillation **2**, lamir1-ceiling **2**, dense-q-supervision **2**, gus-probe **2**. Everything is close — *if you enter through the old hubs.*

**From `champion` alone** (the frontier reader's real entry): candlewax **3 hops** (champion→burl→tool-orchestration→candlewax), student-distillation 2, gus 1, burl 1.
**From `jud` alone:** candlewax **3 hops**, gus 1, pimc 1, burl 2.

**What "Start here" actually foregrounds:** 8 bullets — wiki-entrypoints, lem, burl, gus, w42, w42-book-validation, book-strategy-player, champion. **`jud` is not in "Start here"** (it's buried inside the champion bullet and the Entities list). The champion bullet is a ~400-word wall of rung numbers. Candlewax, K&L, and "what's been tried and failed" appear nowhere in the top of index.md. The reader's first screen is a flat catalog header, not an entry trail.

**The real failure is not hop-count, it's the edge a reader would follow.** candlewax is 3 hops from jud but only via `champion→burl→tool-orchestration→candlewax` — a path with no thematic signpost telling a reader working on *bimodal play policy* to take it. The K&L warning is 1 hop (inside gus.md) but sits at line 417 of a 637-line page with no `###` anchor a reader would grep to. Reachability ✓, discoverability ✗.

**"What's been tried and failed" has no home.** There is no page answering "what value/look-ahead approaches has this project already falsified?" The answers are scattered across `lamir1-ceiling`, `belief-co-train` (co-train falsified), `q-head-augmentation` (path closed), `router-reality-check` (every non-oracle fallback hurts), `gus.md`:417 (K&L). A fresh reader has no single failure-ledger to consult before re-proposing.

---

## 6. Trail inventory

**Present (3):** `wiki-entrypoints` (route map, itself orphaned), `lem-to-burl-handoff` (good — this seam has 95 crossings *because* the trail exists), `w42-book-validation` (good — routes the 106-page w42 pile).

**The two eras with working trails (lem↔burl, within-w42) are exactly the two with healthy crossings. The eras without trails are exactly the near-zero seams.** Trails demonstrably drive the link density. Missing mechanism-trails, in priority order:

1. **`value-for-play` / distillation lineage** — thread `student-distillation → dense-q-supervision → lamir1-ceiling → gus.md K&L warning → q-head-augmentation → belief-co-train → jud judsearch/JS3`. Directly closes misses #1 and #3. This is the single highest-value trail.
2. **`distribution-collapse`** — thread `candlewax ≡ bimodal ≡ melted-blob → w42-phase2-distribution-aware-ev-report → distribution-lens-reranker → lens-v1 "EV won (aggregate)" → jud play policy`. Closes miss #2 and reconciles contradiction #2.
3. **`gus-to-champion handoff`** (mirror of lem-to-burl) — the champion/jud↔gus 35-crossing seam is generic hub links; a curated trail would signpost *which* gus findings the champion consumes and which it must not (the K&L caveat).
4. **`world-sampling / belief`** — `pimc → lamir1 → belief-conditioned-self-play → joint-world-tensor → judsearch belief-lift worlds`.

---

## Overhaul recommendations (ranked by fresh-reader impact ÷ effort)

1. **[High impact / low effort] Add a candlewax concordance alias-block + two backlinks.** In `topics/candlewax`, add a bold line: *"Also called: bimodal / multimodal outcome PDF; the 'melted eq blob' in jud vocabulary; 'mixed-mode geometry'."* Add `[[candlewax]]` to `jud.md` (near "un-melt the eq blob") and to `w42-jud-v0` (at "bimodal realized-outcome support") and to `w42-lens-v1` (at the EV-wins claim). ~5 edits; directly prevents miss #2.

2. **[High / low] Anchor + backlink the K&L warning.** Give gus.md:417 a `### Distilled value cannot drive argmax look-ahead (Kubíček & Lisý)` heading, and link it *bare* from `jud`, `champion`, `belief-conditioned-self-play`, and `w42-jud-v1` (at the JS3 null). Turns a buried paragraph into a discoverable warning. Prevents miss #1.

3. **[High / medium] Create `trails/value-for-play.md`** (the distillation lineage trail, §6 item 1) and `trails/distribution-collapse.md` (§6 item 2). Register both in `wiki-entrypoints` and in index.md "Start here." This is the durable fix — trails are what drive crossings (proven by lem-to-burl). Closes misses #1 and #3 by traversal, not just by one link.

4. **[High / low] Add a "What's been falsified" section to `index.md` top** (or a `topics/negative-results-ledger` page) listing: co-train (falsified), q-head-augmentation (path closed), every-non-oracle-fallback (router-reality-check), distilled-value-for-look-ahead (K&L + JS3), greedy-value-play (JP3). A fresh reader consults it before proposing. Cheap; high leverage against re-proposal.

5. **[Medium / low] Rewrite index.md "Start here."** It was last curated pre-champion (2026-05-03). Promote `jud` to its own bullet (it's the frontier); cut the 400-word champion wall to one line + link; add `value-for-play` and `distribution-collapse` trails. Foreground *entry trails*, not the catalog.

6. **[Medium / medium] Back-link the 53 orphaned source digests and the 5 orphan topics.** Per AGENTS.md the back-link is already required; 40% of sources violate it. Sweep: for each `sources/<sha>.md`, add its bare link to the pages it touches. Un-orphan `probe-analysis`, `shine-analysis`, `lamir1-ceiling`-inbound, etc.

7. **[Low / low] Fix the 5 real dead links + memory-slug leaks.** Create or repoint `burl-perf-phase1`, `plunge`; delete the `[[project_gus_as_burl_belief_brain]]` / `[[user_role_and_north_star]]` memory-slug links from wiki body text.

8. **[Low / low] Reconcile `status:` vocabulary** to the schema's `active|retired|superseded` (or amend AGENTS.md to bless `complete`/`closed-on-completion`). Nine statuses in use, three allowed.

---

### 5-line summary
The wiki is well-linked *within* each project era and starved of links *between* them: 3 trails cover 431 pages, and the two working trails (lem→burl, w42) are precisely the two seams that are healthy. Every fact the fresh model missed is 1–3 hops reachable, so the failure is discoverability, not connectivity — the frontier hub (champion/jud) links the old hubs (gus, burl) only generically, the K&L warning is buried at gus.md line 417 with no anchor, and the same object is named "candlewax" in burl and "melted eq blob" in jud with zero co-occurrence. Two cheap alias/anchor edits plus two mechanism-trails (`value-for-play`, `distribution-collapse`) would have caught all three misses. Secondary rot: 40% of source digests are orphaned, index's "Start here" predates the entire current campaign, and `status:` has drifted to 9 values against a 3-value schema.

**Single worst seam:** `champion/jud ↔ burl` = **3 cross-links** — the 12-page live frontier is near-severed from the 55-page tool-using-play cluster that holds candlewax, reasoning-coherence-verification, and the whole distribution-legibility body of work. `topics/candlewax` has 6 inbound links and **0** of them come from the gus/champion/jud eras.
