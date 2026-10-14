---
title: "Legacy 42 bots: old Texas 42 programs as ladder opponents and as a source of ideas"
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## What it is

A survey of old Texas 42 computer players (shareware, palmtop, open-source, early mobile)
taken 2026-10-14. It asks two things of each: can its AI be brought back to sit at the
[[tournament]] ladder as a pinned participant, and does its bidding and play hold lessons the
[[book-authority]] can test? It includes the legal line, a ranking, the first target (Win42, #157),
a working pilot of the one bot that could be revived in minutes (the GPL `texas42` 0.46 AI),
and the adapter that seats a legacy bot.

## The legal line

The repo and the Hugging Face datasets are public, so these rules hold for every candidate.

- **Never published:** a third-party binary, installer, ROM, disassembly, decompiled source,
  help text, or asset. That covers git, HF, issues and wiki pages. Local copies live in
  `scratch/` (gitignored) or a cache outside the repo.
- **Published:** our own game records (deals, bids, plays, outcomes), the sha256 of what was
  run, and adapter code we wrote.
- **Allowed:** running a lawfully obtained copy locally to interoperate with it and to record
  outcomes. Also allowed: clean-room descriptions of a heuristic, from documentation or from
  observed behaviour.
- **Not allowed:** copying a program's code or text.
- **Preferred paths, in order:** open source first, then author-documented interfaces, then
  the author's permission. Disassembly (Ghidra) is the last resort. It is never needed for
  the top two targets.
- **GPL code** (the `texas42` 0.46 AI) may be run, modified and even redistributed under
  GPL-2. Linking it into a seat makes that seat binary GPL. mk5 is MIT, so the adapter stays
  out of the tree until Jason picks a home: a GPL-2 subdirectory, a separate repo, or an
  on-demand build from the SourceForge tarball.

## Candidates

Checked 2026-10-14. "Runs today" means on this Mac (arm64, macOS 26) without extra software.

| # | Program | Era / platform | Author | Available today | License / terms | Source? | AI, as known | How it fails today |
|---|---|---|---|---|---|---|---|---|
| 1 | **Win42** | 1996 to ~2021 (v3.91; site last modified 2021-05); Win 3.x (VB3, `threed.vbx`), later 32-bit Windows | Curtis Cameron, Plano TX | Demo `WinDom391.exe` at [ccdominoes.com/download](https://www.ccdominoes.com/download.html); 1997 build on [archive.org/details/WINDOMIN](https://archive.org/details/WINDOMIN); 2002–2019 installers on the Wayback Machine | Shareware: evaluate, then register ($17, [terms](https://www.ccdominoes.com/terms.html)). Free redistribution of the eval copy is encouraged. No reverse-engineering clause seen in the 1997 help or the purchase terms. | No | Plays 3 seats. Suggests bids and plays **and explains why**. Nel-O, plunge, sevens. Keeps per-player bid statistics. Saved games are plain text with the full deal (registered users only). The community standard for decades ([pagat](https://www.pagat.com/domino/trick/42.html): "a very good shareware computer version") | Windows only: needs Wine, CrossOver or a VM on the Mac. The 16-bit builds need 32-bit Windows or an emulator. |
| 2 | **texas42 0.46** | 2003; Linux/Qt 3; C++ | Jeremy Ellington (`zoltar74`), Roger Sunshine | [sourceforge.net/projects/texas42](https://sourceforge.net/projects/texas42/) (tarball 2003-09-18, sha256 `c3f50f4a…a7b57`) | **GPL-2** (with a Qt linking exception) | **Yes** | `AiPlayer.cpp`, 1,221 lines of rules. Bids 42 minus estimated loss. Leads walkers, then trump. Dumps count on partner's trick. Never declares doubles or no-trump. Also a Win42 network guest client. | Qt 3 is gone and the project never had a headless mode. **Revived in the pilot below.** |
| 3 | **42-LX** v1.2 | 1995–96; DOS on the HP 100LX/200LX palmtop | Curtis Cameron | Wayback: `ccdominoes.com/hplx/42-lx.zip` (2003 capture, zip sha256 `a6597c7c…aad8e2`) | **Freeware** ("free of charge… share it with others") | No (packed DOS exe, 29 KB) | "Plays the same game as the Windows version". Suggests and explains moves ("S" key). Logs every hand to `42-LX.RES`. Keyboard-only text input for bids and trump. | Needs the HP LX's 640×200 display. DOSBox is a candidate (not installed here). |
| 4 | **Palm42** | ~2003–08; Palm OS 3.0+ | Curtis Cameron | Wayback: `ccdominoes.com/download/Palm42.prc` (2006 capture) | Shareware, 15-day trial | No | "Plays very much like the Win42 game". Suggest and explain buttons. | Needs a Palm emulator (CloudpilotEmu) and a Palm OS ROM. ROM licensing is murky, so this is the least clean path. |
| 5 | **Texas 42 / Texas 42 HD** | 2012–2020 (v5.3); iOS 10+ | MaroonEntertainment | Still listed on the App Store ([iPhone](https://apps.apple.com/us/app/texas-42/id537029259), [iPad](https://apps.apple.com/app/id543049466)). Not updated since April 2020. | Apple standard EULA (prohibits reverse engineering beyond what law permits) | No | Three styles: conservative, normal, aggressive. The listing says its main goal was an AI that doesn't frustrate players. A TexAgs thread reports crashes on iOS 6–7. | Stale. It may run on an Apple Silicon Mac as an iPad app; the only control path is UI automation. |
| 6 | **Tk42** | ~1999; Tcl/Tk on Unix | Christopher Mobley (TI) | Not found online | Unknown | Unknown | A Win42 network client, probably with no AI of its own | Lost. Its contribution survives in `texas42`'s `WIN42_SPEC.txt`: Cameron's 1999 email describing the protocol. |
| 7 | GitHub dead projects | 2013–2021 | various | [vsrz/Tex42](https://github.com/vsrz/Tex42) (C++ 2013, BSD-2); [DaseinUXD/flash42](https://github.com/DaseinUXD/flash42) (ActionScript 2013, no license); [mathman93/FortyTwo_AI](https://github.com/mathman93/FortyTwo_AI) (Julia 2021, no license); [stephengolub/texas42](https://github.com/stephengolub/texas42) (web 2013) | as listed | Yes | Tex42's `PlayerAI` bids by trump length (≥6 → 84, 5 → 35, 4 → 31, else 30) and plays at random. flash42 has no AI. FortyTwo_AI is an unfinished ML scaffold. | Little to revive. Tex42's bidder is a two-minute reimplementation if a "trump-count" rung is wanted. |

**Alive, so out of scope** (they still run, so there is nothing to revive): Eggers 42
([itch.io](https://eggers-studio.itch.io/eggers-42), Win/Mac, bots "really good"), aziruar's
[42](https://aziruar.itch.io/42) (browser, casual/expert), [Follow Me 42](https://followme42.com/)
(web, beginner to expert), 42 Dominoes (Sergey Skosyrev, iOS, updated 2026-09), Domino Texas 42
(Web2mil, iOS 2023), [Board Game Arena](https://en.boardgamearena.com/gamepanel?game=fortytwo),
[Plunge](https://github.com/jasonyandell/plunge) (Walt, already the anchor). Seating a live
online service's bots would need that service's terms. Not pursued.

## Revival paths, cheapest first

1. **Open source, port or run.** `texas42` 0.46: a 70-line Qt shim plus a stateless line-protocol
   driver compiles the original `AiPlayer.cpp` unchanged. Done (pilot below).
2. **Author-documented interface.** Win42's network game. In Cameron's design the *host* is the
   network client and each *guest* is a server. A harness can listen as a guest, and Win42 then
   sends everything in plain text. Per the 1999 spec that includes the `shuffle` command, which
   carries the guest's own seven tiles; `bid?` with the auction so far; `bidwin`; `play`;
   `take`; and `hand` with the score. The harness answers `bid N;` and `play 0, a, b;`. The
   `netdebug` chat command shows all traffic. **No disassembly, no screen reading.** Deal
   control: registered Win42 opens plain-text `.42` saved games that hold the full deal. The
   alternative is to test whether a guest who deals supplies the shuffle.
3. **Emulator plus input automation.** 42-LX under DOSBox: type bids and trump as text, press F2–F8
   to play, read outcomes from `42-LX.RES`, and read the screen for the AI's plays. Palm42
   under CloudpilotEmu (needs a ROM). The iOS app through Mac Catalyst UI automation.
4. **Reverse engineering (last resort).** Ghidra on 42-LX's 29 KB exe (packed; the AI is a
   small routine) or on Win42 (the 1997 build is Visual Basic 3, which decompiles well). Only with the author's written
   permission, only to describe the heuristic clean-room, and never committed.

## Ranking

Value is ladder interest plus idea yield. Feasibility is the path above. Cleanliness is legal
standing. Each is scored 1–3 and the product decides.

| Rank | Candidate | Value | Feasibility | Clean | Product | Note |
|---|---|---|---|---|---|---|
| 1 | Win42 | 3 | 2 | 2 (3 with permission) | 12 (18) | The strongest legacy AI, with explanations. Protocol harness already documented. |
| 2 | texas42 0.46 | 1 | 3 | 3 | 9 | Already revived. A floor rung and a protocol reference for Win42. |
| 3 | 42-LX | 2 | 2 | 3 | 12 | Freeware Cameron lineage. Screen reading is the cost. A fallback if Win42 stalls. |
| 4 | Texas 42 (iOS) | 2 | 1 | 2 | 4 | Three styles would be three rungs. UI automation only. |
| 5 | Palm42 | 2 | 1 | 1 | 2 | Same lineage as 42-LX. ROM licensing problem. |
| 6 | Tex42 / flash42 / FortyTwo_AI / Tk42 | ≤1 | — | — | — | Nothing to seat. |

**First target: Win42 (#157).** 42-LX ties it on the product, but Win42 wins on value at equal
cleanliness once permission is in hand. It is the program 42 players actually used for nearly
30 years, it explains its own reasons, and its interface was documented by its author for
exactly this kind of interoperation. Cameron wrote in 1999, "I think that would be pretty
cool," to a Linux client that would speak his protocol. It was not piloted here: Wine is not
installed on this Mac, and the registered features that give deal control cost $17. Both are
Jason's call. The issue lists the steps.

## Pilot: texas42 0.46 at the arena's table (2026-10-14)

Built in `scratch/legacy-42/` (gitignored; the pilot's script and records are `zeb/legacy-42/` in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10), the GPL sources and the shim left out until their home is chosen). The SourceForge tarball, the original `AiPlayer.cpp`,
`Domino.cpp` and `DominoList.cpp` unchanged, a Qt 3 shim (QString, QValueList, QPair, no-op
QTimer), a stub `Engine` that records the decision, and `seat.cpp`. Each request rebuilds the
AI's state from the public record (hand, history, current trick by relative seat, trump) and
calls the AI's own `suggestBid`, `suggestTrump` and `playDomino_II`. Built with Apple clang 21;
seat binary sha256 `3329b816…07179d` (local build, not distributed). `pilot.py` seats it through
the arena referee (`_LiveGame` + `_run_lockstep`, marks7, duplicate pairs: same seed, seats
swapped).

| Opponent | Duplicate pairs | Games (legacy won) | Marks per game, legacy–opp | Plays | Illegal |
|---|---|---|---|---|---|
| `heuristic+random` (uniform-random play) | 2 | 4 (4) | 7–3, 7–3, 7–2, 7–6 | 588 | 0 |
| `zeb-0.2.0` (tournament seat) | 1 | 2 (1) | 2–7, 7–3 | 266 | 0 |

How to read it: this is a smoke test, not a rating. The chance baseline is 50%, and one
duplicate pair against Zeb splitting 1–1 says nothing about strength. What the pilot shows is
that the adapter is faithful: every play the AI chose was legal under the arena's rules
(854/854), and its bidding behaves as the source reads.

- **Opening and raising.** It opens at 30 whenever its estimate reaches 30 and it speaks first
  (31 of 31 times). Otherwise it bids the minimum raise, and never higher than 32 in 138 bids.
  It never jumps.
- **Declarations.** It declares only suits 0–6, never doubles or no-trump.
- **Two bugs, kept as written.** The 5-5's loss is unreachable because a `break` comes before
  it in `pointLossForVal`. The trump-loss term is an empty TODO that returns 0.

Seated on the ladder it would be a low rung, roughly "a 2003 hobbyist's rules".

## Ideas worth testing (clean-room, for the Book Authority)

1. **"42 minus what you might lose."** Three independent sources bid this way: Win42's help
   ("think about how many points you might lose, then subtract that from 42"), `texas42`'s
   `possiblePointLoss` (one point per off domino plus the count each off suit exposes), and
   [[silas]]'s computed call `(bid (- 42 (risk t)))`. This convergent evidence is
   [[book-authority]]-shaped and worth a line on [[book-assumptions]] when the risk term is
   graded.
2. **The 32–34 plateau.** Win42's help argues that the losable budget falls in coarse steps (a
   ten-count, then five-counts plus tricks). So once a hand can make 32 it can usually make 33 or
   34, and the higher bid costs little. Testable on the pour: P(make 34 | make 32) and its cost in
   marks against the minimum raise, compared with Zeb's bidder.
3. **Make-rate calibration.** Win42 advises that a competitive bidder makes 60–65% of contracts
   and wins about a quarter of auctions. That is a cheap external check on Zeb's bidder in the
   arena.
4. **Public state Win42 tracks.** These come from the protocol, so they are documented and not
   disassembled. Each hand carries flags per player for being known void in each suit, out of
   doubles, out of trumps, and **out of count**. The table carries "walker" (certain trick
   winner), "likely walker" (wins unless trumped), and "ignore count" (a team within one point
   of deciding the hand). "Known out of count" is a public belief bit that the
   [[count-fate-ledger]] could use.
5. **`texas42`'s play rules.** These are simple and book-like. On partner's winning trick, give
   the highest count. On an opponent's trick, give count only with a winner, otherwise trash
   that avoids doubles and trump. Lead walkers first, the 5-5 before the others. With no walker,
   lead trump while keeping one back, then doubles. They make a floor to measure the book's
   richer rules against.

## Adapter design: a legacy participant

A legacy bot joins the ladder as any participant does, through an immutable record
(`tournament/participants.py`) and a seat (`tournament/seats.py`). Nothing changes in the
referee or the store.

- **Kind `legacy`.** The pin names what was run without carrying it:
  `{"program": "texas42-sf", "version": "0.46", "source_sha256": "c3f50f4a…", "adapter": "<mk5 path or repo@commit>", "runtime": "native"}`.
  For Win42: `{"program": "win42", "version": "3.91", "installer_sha256": "cf27f29f…", "exe_sha256": "<WIN42.EXE as run>", "runtime": "wine-<version>", "settings": {…rules dialog…}}`.
  The id carries the version and a short hash: `win42-3.91-<exe8>`, `texas42sf-0.46-<src8>`.
- **`needs`** names what a worker must hold locally: `["legacy:win42", "wine"]` or
  `["cxx"]`. A worker without them skips the participant, as a CPU worker skips
  E[Q]'s `gpu`.
- **Materialize, never fetch-and-publish.** `materialize()` looks for the bytes in a local
  cache outside git (`~/.cache/texas42-legacy/<id>/`). It refuses unless the sha256 values match
  the pin. For GPL `texas42` it may instead download the SourceForge tarball, check
  `source_sha256`, and build. The built binary's sha256 depends on the compiler, so it goes into
  each faceoff's `harness` block, not the pin.
- **Seat.** The same JSON-line contract `host.mjs` serves (`bid`, `declare`, `play`; the
  seat's own tiles and the public record only), so `HostSeat` is reused unchanged. The
  per-program translator is the only new code. For `texas42` it maps tiles to its "hl" strings
  and the trick to relative seats, as `pilot.py` does.
- **Win42 inverts who referees.** Win42 only plays as host, its AI sitting at the seats it
  simulates; guests are servers it connects to. So the Win42 bridge seats the *opponent*:
  the opponent's two seats are guests the harness serves, and each `bid?` or play turn is
  forwarded to the opponent's own seat process. Win42 deals and keeps score. The bridge
  re-checks every trick and hand result against the arena's rules (`forge/zeb/game.py`) and
  writes the faceoff record in the tournament's format, deals included. Duplicate pairs need
  deal control: a saved `.42` file opened by the registered version, or a guest shuffle if
  Win42 accepts one. Open question: can the host's own seat be computer-simulated, or must
  one seat be driven through the UI? (#157)
- **Illegal replies.** Every reply is checked against `legal_actions`. An illegal reply is
  recorded in the faceoff with a deterministic fallback (first legal slot). It is never silent.
- **Fidelity.** Choices that change behaviour without being part of the bot are recorded in
  the participant's `config`: hand order (the AI breaks ties by list position; the pilot passes
  the arena's tile-id order), rules dialog settings, and Win42's play speed.

## Links

[[tournament]] (where a revived bot sits) · [[book-authority]] (what the ideas are for) ·
[[silas]] (the computed call the loss-count bid echoes) · [[count-fate-ledger]] ("known out of
count") · [[eval-matrix-bradley-terry]] (the older ladder) · #157 (Win42 revival)
