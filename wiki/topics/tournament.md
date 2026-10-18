---
title: "The tournament: an append-only, Bradley–Terry-rated ladder of Texas 42 players"
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-18
status: active
---

## What it is

A standing ladder of the project's players, kept by a worker loop that any machine can run
and kill at any moment. Code: `tournament/` (`python -u -m tournament --help`). Records:
the Hugging Face dataset
[jasonyandell/texas-42-tournament](https://huggingface.co/datasets/jasonyandell/texas-42-tournament)
(public, CC0). Successor in spirit to the February ranking layer ([[eval-matrix-bradley-terry]]),
which fitted the same model to one in-memory win matrix and was never run at size.

```bash
python -u -m tournament table                       # the derived rating table
timeout 300 python -u -m tournament work --budget-s 200   # one chunk of the worker loop
python -u -m tournament add-zeb 0.3.0                # a new npm release joins (no code change)
```

## Fixed-contract Gus ladder

`marks7-bid30` uses this same append-only store, duplicate games and fitted
Bradley–Terry model. Both partnerships use the pinned arena `Bid30Bidder`:
open 30, otherwise pass, declare the best pip trump. First to seven marks.
Its separate anchor is `zeb-0.4.1-contested-bid30` at 1500: the immutable
Zeb 0.4.1 native play pin with the common bidder replacing its auction.
These ratings cannot be compared numerically with the ordinary `marks7` scale.
[[gus-tournament-pilot-2026-10-18]] records the first 30 games and the canonical
Hub synchronization still pending under its repository commit quota.

`python -u -m tournament add-gus g020 gus-g020` resolves a Gus release tag to
an immutable Hub commit, hashes `gus.pt`, and appends a complete participant
record. Each record pins the model, encoder, forward pass, adapter and bidder
sources. Source or cached-weight mismatches fail closed. Gus uses CPU float32,
one thread, one forward pass and legal argmax; exact ties use lowest tile id.
Only the original own hand and public bid-30/pip/play prefix reach the net.
No learned bidder is claimed. Contract-scoped participants cannot enter the
ordinary-auction ladder; ordinary participants cannot silently enter this one.

```bash
python -u -m tournament table --ladder marks7-bid30
scripts/cpu-slot --threads 2 --timeout 60 -- scratch/venv/bin/python -u -m tournament work \
  --ladder marks7-bid30 --max-faceoffs 1
```

The external timeout caps a run, not thinking. A timeout preserves its claim
and completed hands; it is unfinished evidence, never a scored loss or a
reason to reroll a seed. `work --budget-s` alone does not cap a faceoff.

## Participants, pinned

Joint tile/splat checkpoints use a separate `gus-splat` adapter, preserving the
original tile-only participant pins. `python -u -m tournament add-gus-splat NAME
REVISION --head tile` resolves an immutable model release and pins both output
implementations; `--head splat` is a distinct deployment policy. `add-demon-bid30`
registers the existing 100-world root-reuse Demon cardplay in this same restricted
ladder. The common auction replaces its native bidder; ordinary-auction ratings
and early-40 bidding strength do not transfer to this comparison. Neither command
changes an existing participant or the default roster. Its `paired-deal-v1`
seed protocol derives draws from faceoff seed, pair, side, hand and decision;
candidate names and claim timestamps cannot change the fixed opponent's draws.
The original Demon participant keeps its original seed protocol. The Demon remains a
fenced competitor: no internal estimates become Gus labels.

Every participant is a record, `participants/<id>.json`, added once and never edited; its bytes
are a function of the participant alone, so two writers adding it write the same file.

| id | what | pinned by |
|---|---|---|
| `walt-plunge-26043bf9` (withdrawn from scheduling, records kept) | Plunge's production Walt, difficulty `native-partner`; thinks on a wall clock (`"clock": "real"`) | Plunge commit `26043bf9` (what plunge.texas42.workers.dev/version.json served 2026-10-14); `walt-player.wasm` sha256 `40d7a1ee…`, `book/played.json` `3e2bf7ae…`, `manifest.json` `e15bc53e…`, each fetched from that commit and checked on load; built from texas-42 `cb1ef3b2` |
| `walt-plunge-26043bf9-fair` (anchor, 1500) | the fair Walt ([[fair-walt-2026-10-16]]): production Walt's identical wasm, book and manifest (the same pin, the same three sha256s) driven by a host whose `now_us` is a counter, `"clock": "virtual"`, `clock_step_us` 0.839 | the same Plunge commit `26043bf9` and checks as the row above; the step is 1e6 over production's pooled `now_us` reads per second on the registered panel |
| `zeb-0.1.0` | `@jasonyandell/zeb` 0.1.0, 160 worlds, cutoff 0.8 | npm integrity `sha512-GyEDG0FE…`, shasum `30aee968…` |
| `zeb-0.2.0` | `@jasonyandell/zeb` 0.2.0, same settings | npm integrity `sha512-YnlKLquB…`, shasum `12763486…` |
| `eq-n10-fenced` | E[Q] n=10 (`heuristic+lens:ev`), the oracle era's champion | checkpoint sha256 `5f3f00de…`; the faceoff records the mk5 commit that ran it |
| `demon-pimc-n100-b88c5432` | the Demon ([[the-demon]]): CPU PIMC, 100 worlds, Silas's double-dummy solve per world, margin tiebreak, `heuristic` bidder | mk5 commit `b88c5432` and the git trees of `silas/` `75b67c11…`, `scenario/core/` `11735eb0…`, `tournament/demon/` `40738d38…`; built from those trees on the worker |
| `demon-pimc-n100-18ac096d` | the Demon with compact per-world search tables, 100 worlds, the same values and tie rules | mk5 `18ac096d`, pinned source trees; [[demon-search-tables-2026-10-14]] |
| `demon-pimc-n400-18ac096d` | the compact Demon at 400 worlds; experimental challenger, strength not established | the same source pin, distinct world-count config and id; [[demon-search-tables-2026-10-14]] |
| `demon-pimc-n100-228812ff`, `demon-pimc-n1600-228812ff` | exact compact search and distinct uniform worlds; the larger budget remains experimental | mk5 `228812ff`, immutable source trees; [[demon-power-2026-10-14]] |
| `demon-pimc-n100-bid7ff83929-4454f603` (withdrawn) | historical ten-declaration PIMC bidder; excluded from straight-game scheduling and ratings | immutable record retained; [[demon-vs-zeb-2026-10-15]]'s declaration-domain audit |
| `demon-pimc-n100-bidc102a137-straight-2e233d29` | 100 play worlds, 64 bidding worlds, opponent-held point-bid cutoff 0.5, opening cutoff 0.875, nine straight declarations | mk5 `2e233d29`, immutable source trees and registered record; [[demon-overcalling-2026-10-15]] |
| `demon-pimc-n100-bid9dee8eeb-straight-bounds-support-bait40-a561552d` | 100 play worlds, 64 bidding worlds, early bid 40, opening/opponent cutoffs 0.875; targets Zeb 0.4.1's overcall rule | mk5 `a561552d`, exact registered record and independent win/CPU confirmation; [[demon-ascent-2026-10-16]] |

Seating. The referee is the arena (`arena/engine.py`: deal, auction, play, marks to 7). A
JavaScript participant runs in `tournament/seat/host.mjs`, one process per team per game, sent
its own dealt tiles and the public record only.

- **Walt** is driven as Plunge drives it, ported line for line with the source cited in
  `host.mjs`: partner pass; the frozen bid book on a catalogue hand, else the shipped live survey
  (rounds of 4/12/40/160 worlds over the nine declarations on a 4-worker pool, one 20 s
  deadline, bid the minimum raise if the best declaration prices at 3/4); plays per
  `livePlayerCall` (160 worlds on the bidder's opening lead, else 40 with the partner check), seeded
  `sunshine/<game>/<hand>`; every reply cross-checked against the table's leader and points.
  Two honest differences from the phone: tournament deals are random, so the live survey bids
  almost every hand where Plunge (which deals only its 130 catalogue deals) uses the book; and
  production's budgets are wall-clock, so a reply depends a little on the machine. That record
  says `"clock": "real"`, is **withdrawn from scheduling** (its faceoffs stay in the fit), and
  a faceoff it plays is marked `noisy_timed` unless it ran under `cpu-slot --exclusive` (#173).
  The seat that plays is the **fair Walt** (`walt-plunge-26043bf9-fair`, #181): the same wasm
  behind a host whose clock is virtual. The n-th `now_us` read of a call returns
  `floor(n × clock_step_us)` microseconds, so the 14 s, 20 s and 500 ms budgets are counts of the
  solver's own work. The survey's four-worker race becomes a schedule in virtual time (the nine
  jobs in declaration order, each to the worker free first, each with the budget the 20 s
  deadline leaves it; the real pool still runs a round at once and a job that overran its true
  budget is rerun with it), so a game replays bit for bit on any machine at any load.

- **FAIR CLOCKS** (CLAUDE.md, Jason 2026-10-16: "all players in the arena should be made fair not
  wall"). Every seated player thinks on worlds, nodes, work units or a virtual clock, never wall
  time; wall time, CPU time and instructions are reported, not budgeted. Enforced three ways:
  `deterministic_budget(record)` (no `"clock": "real"`; a Walt seat must declare the virtual
  clock and its step) gates `schedulable()`, which `runnable()` and resumed claims use; `work()`
  refuses to claim a pair that fails it; and `seats.seat()` refuses to build such a seat unless
  asked for a `historical` research seat (the paired check against withdrawn production Walt).
  `tournament/tests/test_fair_clocks.py` pins all of it. Audit of `arena/` (#181): every
  `time.time()`/`monotonic` is a progress log or a reported `elapsed_s`; no arena seat reads a
  clock to decide. Zeb (npm and native), the Demon (`ms` reported only) and E[Q] n=10 decide by
  worlds and seeds, and the study exploiter takes node budgets (#173).
- **Zeb** bids with its own rule (`forecast` + `chooseBid`) behind Plunge's partner-pass
  convention, never past one mark, declares the likeliest declaration at the contract, plays with
  `play`, all at 160 worlds, seed 0 (the package defaults).
- **E[Q] n=10** is a fenced measuring stick (STRATEGY-FUSION FENCE, [[strategy-fusion-fence]]
  § Sanctioned uses, "Baselines"): seated only by `tournament/eq_fenced_seat.py`, only on a machine
  with MPS or CUDA (forge is GPU-only), never read by anything else.
- **The Demon** is the CPU fusion stick (STRATEGY-FUSION FENCE, [[the-demon]]): seated only by
  `tournament/demon_fenced_seat.py`, a Rust binary per team per game, sent its own tiles and the
  public record and checked against the table's leader and points. The original baseline pins
  use the arena's `heuristic` bidder. The confirmed overcalling form uses common-world PIMC
  forecasts and a lower cutoff over an opponent's point bid ([[demon-overcalling-2026-10-15]]).
  PIMC bidding candidates must explicitly restrict their
  declarations to the nine straight ones (0–7 and 9); the first ten-declaration pin is
  incompatible and withdrawn. It needs
  `cargo` and `git`: the worker builds it from the pinned trees (`git archive <tree>`, then
  `cargo build --release --locked`, about 40 s once per machine), never from the checkout.

- **A configured Zeb is a seat** (`zeb-native`: [[rematch-zeb-041-vs-demon-2026-10-15]]). A config is
  a participant, two configs two seats, so the ladder measures the strategic tradeoffs between them:
  `zeb-0.4.0-match` (match-aware, pip declarations as measured) and `zeb-0.4.1-contested`
  (`overcall: true`). Each pins its release commit, the git tree of each Rust source directory, the
  sha256 of the shipped `index.js` and `match.js`, and the wasm; the player is the identical Rust
  built natively from the pinned trees plus the JSON-lines wrapper in `tournament/zeb_native/`
  (`tournament/zeb_native_seat.py`; needs `node`, `cargo`, `git`, advertised as capability
  `zeb-native`). A rebuild embeds its own paths, so its binary differs byte-wise from the one the
  trials measured (`c76303d0`); every input is checked, and the rebuilt binary's own hash is recorded
  and checked on load. Their ids and records are the rematch's, unchanged. The confirmed overcalling
  Demon (`demon-pimc-n100-bidc102a137-straight-2e233d29`) is on the roster as a fenced seat.

Old programs can join the same way: [[legacy-42-bots]] surveys them and designs a `legacy` kind,
pinned by the sha256 of what was run and never redistributed. The GPL `texas42` 0.46 AI (2003)
already plays at this referee in a pilot; Win42 is the first target (#157).

Zeb 0.3.0 joins when it is on npm: `add-zeb 0.3.0` reads the registry's integrity and appends
the record; the scheduler then plays it first (an unplayed participant has the most to teach).

## The math

**Game and faceoff.** A game is the tournament game, first to 7 marks (ladder `marks7`; the
book's game to 250 is defined as `points250` and not yet played). A faceoff is `pairs`
duplicate pairs: two games on the same deals with the seats swapped (hand seeds depend on the
pair's seed and the hand number, never on who sits where), so card luck cancels inside a pair.

**Rating model: Bradley–Terry, fitted, not online Elo.** P(i beats j) = σ(θ_i − θ_j), fitted by
maximum a posteriori over every game ever recorded, with a weak prior θ ~ N(0, 2²) (347 Elo; it
keeps an unbeaten player finite and is overruled by a few games). Reported as
1500 + (400 / ln 10)(θ_i − θ_anchor), anchored on the fair Walt (production Walt until #181). Online Elo would depend on
the order records arrive in, which concurrent workers do not choose; the fit does not (tested).

**Intervals that know about the pairing.** The two games of a duplicate pair share their cards,
so they are not independent. The point estimate is unaffected (a pair's games enter the
likelihood exactly as two games would), the variance is not: it is the cluster-robust sandwich
V = H⁻¹(B + P)H⁻¹, one cluster per pair. A pair that splits 1–1 carries less doubt than two
independent games, one that always agrees carries more (tested both ways). Measured coverage of
the 95% intervals on simulated duplicate pairs: 94.8% without deal luck, 93.8% with strong deal
luck (800 intervals each). Intervals are for θ_i − θ_anchor.

**Re-anchoring (#181).** The anchor moved from production Walt to the fair Walt. The fit is
unchanged by that: P(i beats j) = σ(θ_i − θ_j) depends only on differences, so the anchor chooses
which player reads 1500 and nothing else. Every rating shifts by one constant, every gap between
two players is the same, and the games stay what they were (`tournament/tests/test_fair_clocks.py`
fits one set of games on both anchors and compares). Intervals are for θ_i − θ_anchor, so they
are re-expressed against the new anchor and the fair Walt's own interval is a point. Until
the fair Walt has played, its rating is the prior's and the table is still linked through production
Walt's kept records. Rows in "The first round" below are the old scale (production Walt = 1500).

**Scheduling for information.** One faceoff between i and j adds about w = games·p(1−p) of Fisher
information along e_i − e_j; the trace of the (centred) rating covariance then falls by
Δ = w·xᵀV²x / (1 + w·xᵀVx). The worker plays a pair whose Δ is within 80% of the best, chosen
at random among those so concurrent workers spread out. This is "the pair whose intervals
overlap most", made exact: large when the two are close and their difference is uncertain.

The table also reports, per pair, marks/game with a 95% interval over duplicate pairs (the house
metric of [[scenario-player]] and [[champion-ladder]]).

## Append-only, conflict-free by construction

The whole state is a directory of immutable files. A faceoff is one file,
`faceoffs/<ladder>/<claimed UTC>-<a>--<b>-<64-bit random seed>.json`, holding every hand (deal,
auction, plays, points, marks), the participants' full records and the harness (mk5 commit, Node,
Python, the cpu-slot mode and load). Since #173 each game also records what it cost beside its wall
seconds: the referee's CPU and instructions and each seat process's CPU and own instructions. It is
written once, in its own commit, at a path no other writer can choose. There is no
index, manifest or stored rating: every table is derived by reading the directory. Two commits that
add different new paths touch disjoint trees, so every interleaving rebases cleanly; an identical
re-append is a no-op, and different bytes at an existing path are refused.

On the Hub (`HfStore`) each append is one `create_commit` with no `parent_commit`, which the Hub
applies on top of its current head. Reads are incremental (2026-10-16): records are immutable, so
`HfStore.read_all` lists the tree once per prefix and resolves only the files its local cache lacks.
The earlier `snapshot_download` re-checked every file (~694 resolver requests) whenever the Hub head
had moved, which with two workers hit the 5000-resolvers-per-5-minutes quota and killed a worker;
now a read is 2 listing calls and 0 resolves when nothing is new (`test_hf_*` in `test_store.py`).
A plain git remote (`GitStore`) is commit, push, and on a
rejected push `pull --rebase` and push again.

**Tested deliberately (2026-10-14).**

- `tournament/tests/test_store.py`, real git: 5 writers in separate clones, 8 appends each, racing
  pushes to one bare remote. 40/40 records, 40 commits of one file each, no merge commits, and the
  race really happened (52 rejected pushes in the recorded run, each rebased and replayed). Controls: a deterministic
  interleave (B pushes first, A's push is rejected and rebases); two writers adding the same
  participant (one add); and the design avoided, two writers editing one shared `index.json`,
  which does conflict.
- `python -u -m tournament.hf_race --repo jasonyandell/texas-42-tournament-selftest`, the Hub
  itself: 6 processes released by a barrier, 4 appends each. 24/24 files, 24 new commits, 0
  retries, each re-append a no-op, each edit refused. PASS, in 5.4 s.

## Interruptible and resumable

The worker loop: finish any claim left behind → read records → derive ratings → choose a pair →
claim (a local directory named by the faceoff id: spec and random seed) → play its games in
parallel → assemble the record once, save it in the claim → append → delete the claim.

- A game rewrites its progress file after every hand; a restarted game restores marks, points,
  dealer and hand number and deals the next hand from the same seed. A kill loses at most the hand
  in flight.
- Killed between saving the record and the append: the resume appends the saved bytes. Killed
  between the append and the cleanup: the resume re-appends identical bytes, which is a no-op. No
  path can count a faceoff twice.
- A worker holds an `flock` on the claim it runs; a second worker sharing the directory leaves it
  alone, and the kernel drops the lock when the holder dies.
- A claim lost with its machine is a faceoff never recorded, which biases nothing: whether it
  finishes does not depend on how it was going.

Tested in `tournament/tests/test_worker.py` (a hard `os._exit` after the third hand, then a resume
that keeps those hands as played and deals the rest exactly as an uninterrupted game; kills around
the append; the held claim) and live on the Hub: `timeout 45` killed the first real faceoff
(E[Q] vs Walt) with 6 hands saved, and the next run finished that same faceoff.

## The first round (2026-10-14)

Run on this Mac at mk5 `0b1a1f21`…`1cbb8417` (each record names its commit), two workers at once for
most of it, one duplicate pair per faceoff, about 1–3 minutes a faceoff. `python -u -m tournament
table` reproduces it from the records.

Ladder `marks7`: 9 faceoffs, 18 games. Bradley–Terry, anchor `walt-plunge-26043bf9` = 1500, 95% cluster-robust intervals.

| participant | Elo | 95% CI | games | wins |
|---|---|---|---|---|
| `zeb-0.2.0` | 1812 | [1525, 2099] | 12 | 11 |
| `walt-plunge-26043bf9` | 1500 | anchor | 8 | 2 |
| `eq-n10-fenced` | 1430 | [1217, 1642] | 8 | 3 |
| `zeb-0.1.0` | 1403 | [1149, 1657] | 8 | 2 |

| pair (first − second) | duplicate pairs | games | wins | marks/game | 95% CI |
|---|---|---|---|---|---|
| `eq-n10-fenced` vs `walt-plunge-26043bf9` | 1 | 2 | 1–1 | +0.00 | too few pairs |
| `eq-n10-fenced` vs `zeb-0.1.0` | 2 | 4 | 2–2 | +0.25 | too few pairs |
| `eq-n10-fenced` vs `zeb-0.2.0` | 1 | 2 | 0–2 | -4.00 | too few pairs |
| `walt-plunge-26043bf9` vs `zeb-0.2.0` | 3 | 6 | 1–5 | -1.67 | too few pairs |
| `zeb-0.1.0` vs `zeb-0.2.0` | 2 | 4 | 0–4 | -2.25 | too few pairs |

How to read it. Nine faceoffs is a smoke test of the loop, not a ranking; the chance baseline for
any pair is 50%. Zeb 0.2.0's 11 of 12 is the only interval that clears the anchor, and a sandwich
interval from this few clusters runs narrow, so it is a lead to watch, not a finding. Zeb 0.1.0
losing 0–4 to 0.2.0 (two duplicate pairs) sits beside the arena's null between the two
(−0.02 [−0.35, +0.29] marks/game, [[scenario-player]] § Foundation repair) and does not contradict
it yet: 4–0 at even odds is 1 in 16. Marks intervals are printed from five duplicate pairs up.

## The cloud worker

**Cost (checked 2026-10-14).** No extra charge on Max 20x unless usage credits are turned on:
"It counts toward your plan's usage limits alongside the rest of your Claude and Claude Code
usage, and there's no separate charge for the cloud VM"
([Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web));
"Without usage credits, additional runs are rejected until your usage window resets"
([routines](https://code.claude.com/docs/en/routines)); usage credits are opt-in and "billed at
standard API rates"
([support 12429409](https://support.claude.com/en/articles/12429409-manage-extra-usage-for-paid-claude-plans));
nothing in the docs bills Haiku to usage credits (the features that do are fast mode, Fable on some
plans and Sonnet 1M context; [model config](https://code.claude.com/docs/en/model-config)).
Check claude.ai/settings/usage that usage credits are off.

**What it needs before it can run.**

1. This code on GitHub (`claude --cloud` clones the remote at the current branch).
2. The HF token as a **network secret** on the cloud environment (claude.ai/code → edit
   environment → Network secrets → Add secret: Bearer, allowed websites `huggingface.co`,
   `*.huggingface.co`, `*.hf.co`, header `Authorization`, prefix `Bearer`). The agent proxy attaches
   it outside the VM, so the token never enters the session, a file or a log
   ([cloud environments](https://code.claude.com/docs/en/cloud-environments#add-network-secrets)).
   Never an environment variable: every session of the environment can read those.
3. Network access Trusted, plus `download.pytorch.org` (the arena imports torch; the CPU wheel is
   small, PyPI's Linux wheel pulls CUDA), and `static.crates.io`/`index.crates.io` for the Demon's
   build. The worker seats Walt, Zeb and, where `cargo` is on PATH, the Demon and the configured native Zebs; E[Q] needs a GPU and
   runs on the Mac.

**Start it** (from a checkout of the merged main): `claude --cloud --model haiku "Read
wiki/topics/tournament.md, section 'The worker prompt', and do exactly what it says."` If the
session does not come up on Haiku, send `/model haiku` as its first message.

### The worker prompt

> You are the tournament worker for the Texas 42 repo you are in. Your only job is to run the
> worker loop below and report. Do not edit, create or delete any file in the repository, do not
> commit or push, do not print environment variables, and never write a token anywhere.
>
> 1. Setup, once: `python3 -m venv /tmp/tv && /tmp/tv/bin/pip install -q numpy huggingface_hub &&
>    /tmp/tv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu`. Check
>    `node --version` prints v22 or later.
> 2. Check the store: `/tmp/tv/bin/python -u -m tournament table`. It must print a table. If it
>    fails with 401 or 403, stop and report "the HF network secret is missing".
> 3. Loop: run `TOURNAMENT_WORKER=cloud-haiku timeout 300 /tmp/tv/bin/python -u -m tournament work
>    --budget-s 200` (give the Bash tool a timeout of 320000 ms). Each run appends faceoffs and exits;
>    a run that times out is fine, the next run resumes it. Repeat.
> 4. After every 10 runs, run step 2 again and post the table.
> 5. Stop and report if the same error appears in three runs in a row, if a usage-limit message
>    appears, or after 100 runs. Report the last table and any errors verbatim.

## Links

[[scenario-player]] (hub) · [[eval-matrix-bradley-terry]] (the February predecessor) ·
[[champion-ladder]] (the paired-halves arena it reuses) · [[strategy-fusion-fence]] (E[Q]'s
seat) · [[the-demon]] (the CPU PIMC stick) · [[publish-data-to-huggingface]] (where records live) · [[legacy-42-bots]] (old
programs as opponents)
