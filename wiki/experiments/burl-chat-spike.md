---
title: burl-chat spike — first interactive sessions and findings
kind: experiment
first_seen: cba521d
last_updated: local-2026-05-01
status: complete
---

## What

First end-to-end use of the [[burl-chat]] workbench. Goal: prove the loop "harvested decision → loaded as conversation prefix → user asks a question → model replies in prose" can produce useful output, before paying the cost of generating a Q&A training corpus.

## Setup

- Model: `mlx-community/gemma-4-e2b-it-bf16` ([[gemma-4-e2b]]) on M5 Max via [[mlx-lm]]. Tested both base and with `e1-rank16` adapter ([[experiments/iter5-e1-rank-sweep]]).
- Harvest: `scratch/belief_trajectory_rollout/harvest_batched_20260426_031338/` (2800 corpus_index rows; later batched harvest than the canonical 2000-decision [[burl-2000-harvest]]).
- Decision: `global_idx=0`, seed 0, bucket `BURL_BREAKS_CONSENSUS`, burl_play=14 (4-4), oracle_play=20 (5-5), regret 12.12.
- Conversation prefix: 18 typed segments — system (5310 chars) + user state (295 chars) + 1 thought + 5 tool_calls + 5 tool_results + 4 assistant_texts + commit_play(14). Plus the [[chat-mode-primer]] injected after the commit.

## Findings

### 1. Adapter lock-in is real and total ([[play-adapter-lock-in]])

With `BURL_CHAT_ADAPTER_PATH=burl/adapters/e1-rank16`, the model **cannot be talked to**. Even with:
- `enable_thinking=True` to encourage prose
- The chat-mode primer establishing a conversational cadence
- An explicit user instruction "**do not output a tool call**"
- A direct question about tool design ("how could the tools have been more clear")

…the model emits a partial answer (~3 sentences) then snaps back to "I am Burl, I have committed play 14, I must execute the task" mode and outputs `commit_play({"domino_id":14})` again — re-committing a play it already committed. The training distribution is loud enough to override the in-context recency signal of the primer plus an explicit prohibition.

With the same prefix and **no adapter** (base Gemma 4 E2B), the model engages immediately. First clean session produced a structured three-section critique: *what made it challenging, what would have made it easier, in short* — citing actual numbers from the harvest's tool outputs (+11.7 Q lift, -5.8 Q drop, 6-0 vs 2-2 catalyst dominoes) and proposing three concrete tool-design improvements:

1. Structured summaries before raw data ("Play 14 shows high variance. Best case +11.7 Q, worst case -5.8 Q." before the histogram).
2. Strategic labels ("Key Insight: Play 14 is extremely volatile.").
3. "Why" over "how" ("This play maximizes your chance of hitting the 30-count bid.").

That third suggestion is exactly the [[topics/rules-as-tools]] insight applied to evaluative tools, and exactly the [[topics/at-risk-points]] frame Roberson uses. The base model invented it from a 9-word user prompt.

### 2. The chat-mode primer is doing real work

Without the primer turn after `commit_play`, even the base model produces "I am Burl, my next action is to call commit_play" responses. With the primer ("Yeah, I committed 14. The decision is done — ask me anything about it..."), the same base model produces fluent Q&A. Recency-weighted in-context learning over a strong distribution. See [[chat-mode-primer]].

### 3. Tool vocabulary gap

The base model's spontaneous critique used Gus/forge vocabulary (Q axis, mean shifts, catalyst dominoes), not Roberson vocabulary (offs, walkers, double ahead of your off, at-risk-points). That's because the harvest's *tool outputs* speak Gus, not Roberson. To get the family-reunion voice (the project's voice standard, since Roberson's *Winning 42* names the user's actual cousins), the tools themselves need to surface Roberson framing, OR a Roberson primer needs to ride the system prompt. Probably both.

### 4. Bucket signal is plausibly real through prose

Decision #0 is `BURL_BREAKS_CONSENSUS` with regret 12.12 — a sharp loss. The model's self-critique ("the previous turn involved calling `trick_winner` (which returned 21)") shows it remembers the trace and can speak to specific tool calls. Whether bucket category (agreement vs disagreement vs forced) maps to qualitatively different self-critiques is the next thing to sample. Three or four decisions from each bucket would tell us.

## Plumbing bugs found and fixed

### MLX thread affinity

MLX's default GPU stream is bound to the thread that creates it. FastAPI's lifespan loaded the model on the asyncio main thread; the first generate call ran in `loop.run_in_executor(None, ...)` which uses a default-pool thread. Result: `There is no Stream(gpu, 0) in current thread.`

Fix: dedicated `ThreadPoolExecutor(max_workers=1)` that loads the model and runs all generate calls. Single-thread by construction.

### CRLF SSE frame separator

sse-starlette emits frames as `data: {...}\r\n\r\n` per the SSE spec. Browser `TextDecoder` returns literal bytes including `\r`. A naive `buf.split("\n\n")` finds no frame boundary, accumulates indefinitely, yields nothing. The Network tab still shows 41 kB delivered — bytes arrive, just unparsed.

The bug was invisible to every prior `curl` test because terminals strip CR. **Methodological lesson: SSE consumers must be tested with `repr()` of raw bytes, not eyeballed terminal output.**

Fix: strip `\r` before splitting in the streaming consumer.

### Svelte 5 reactivity on parser-mutated objects

The streaming parser appended characters via `last.content += text`. Once the segment object is in `$state`, Svelte wraps it in a proxy; the parser still holds a reference to the unproxied original. Mutations through the unproxied reference don't trigger reactivity, so the UI shows a stable empty segment while content silently grows in memory.

Fix: replace the segment at its index with a fresh `{...last, content: ...}` object on every append. New reference, Svelte sees the change, re-renders.

## Diagnostic protocol that landed the SSE bug

Three quick checks, in this order, distinguished six failure modes in ~30 seconds:

1. **Network tab → /api/chat row:** does Size grow during the request, or only at the end? (Catches transport/buffering bugs.)
2. **Console:** any red errors during streaming? (Catches JS exceptions in the consumer.)
3. **`document.querySelectorAll('.seg').length`** before and after the request. (Distinguishes "segments not added" from "segments added but not visible.")

For this bug: Network said 41 kB delivered fine. Console was clean. `.seg` count stayed flat. That triangulated cleanly to "consumer not yielding events" — which is the SSE framing bug, not transport, not rendering.

## What's next

- **Capture corpus rows.** Promote good conversations (like the tool-critique one) to a JSONL corpus under `burl/chat/corpus/`. First step toward [[post-commit-q-and-a]].
- **Roberson primer.** Try injecting Flemmons' foreword + selected paragraphs from chapter 2 of *Winning 42* as a system-prompt prefix. Does the model's voice shift toward family-reunion register?
- **Sample by bucket.** Three or four decisions each from `ALL_AGREE_CORRECT`, `BURL_BREAKS_CONSENSUS`, `BURL_INDEPENDENT_RIGHT`. Does the self-critique vary meaningfully?
- **Tool-output redesign.** The model's own three-point critique is a backlog. The eq_outcome_distribution and probe tools could grow a "headline" field rendered before the histogram.

## Session 2 (2026-05-01) — improvised-tools loop and the tool wishlist

Second working session, this time with [[improvised-tools]] wired up: an MCP bridge lets Claude register hot tools mid-conversation, the workbench grew a per-tool selection popover with delete buttons, and tools persist to `burl/chat/server/tools_library/` so they survive restarts.

### Three meta-asks across two decisions

Burl produced three different tool requests across decisions #0 and #7. Each was answered with a hot-registered tool ([[improvised-tools]]). Pattern documented in [[burl-tool-wishlist]]:

1. **Decision #0, "comprehensive game state visualizer"** → built `board_snapshot` (wraps the existing `render_full_board_snapshot` in `burl/wax_museum/snapshot.py`).
2. **Decision #7, "strategic synthesis engine that picks the best play"** → built `legal_plays` instead. The literal request would have violated the wax_museum doctrine *"state tools answer WHAT IS the state — they never tell you WHAT TO DO."* The underlying need was led-suit comprehension: Burl had just attempted to commit 27(6-6) when must-follow on lead 2(1-1) had set the led suit to ones and the only legal plays were 4(2-1) / 16(5-1). It misread the rejection "must follow suit 1 (led by 2)" as referring to trump-id 1.
3. **Decision #0 (second pass), "the input was slightly confusing"** with a sketched `[GAME STATE]` / `[CONTEXT & GOAL]` / `[PROTOCOL]` format → built `state_brief` to Burl's exact spec.

### Findings reinforced

- **[[play-adapter-lock-in]] manifests at the meta layer.** Burl cannot drop into open prose Q&A even *about* itself; what it produces in chat is structured tool plans. But the *content* of those plans names real failure modes correctly — the model is more competent talking about its tools than about the game.
- **Doctrine ("state, not strategy") matters even at the improvised-tool layer.** Burl will ask for a play-picker. The right move is to interpret the underlying state-comprehension need and build a state tool, since strategy-pickers would foul the eventual training corpus by encoding decisions the model is supposed to make for itself.
- **The wishlist is a corpus signal.** Each (Burl-asks, Claude-implements, demonstrates-improvement) triple is a candidate row for the future [[post-commit-q-and-a]] training set.

### Plumbing bugs found and fixed (session 2)

- **`advertise tools` button was inert.** Only mutated the local `segments` array; never re-streamed the model. Fix: `await send("")` after appending the synthetic user turn (the existing continuation path that re-streams against current segments without appending a new user turn). Doctrine: any UI button that appends a segment intended to elicit a response must explicitly trigger the continuation.
- **`$effect` clobbered manual unchecks.** The auto-select logic re-added every known tool to the selection set on every render, so the user couldn't drop a tool from the advertise list. Fix: a separate `seenToolNames` set so tools auto-select only on first appearance; subsequent unchecks stick.
- **Disk persistence layer landed.** Each `register()` writes `tools_library/<name>.py` with a `DESCRIPTION` constant + the source. On module import, `_hydrate_from_disk()` rehydrates the registry. The library is intentionally human-readable and check-in-able; promoting a tool to `burl/wax_museum/tools.py` is the on-ramp from improvised → permanent.

### Second wave (later same day) — `play_brief`, rerun-fresh, and the first measurable lift

Three follow-on changes once the loop was established:

1. **`play_brief` tool** — Burl asked for `explore_game` output reshaped with a HEADLINE (variance + p_make), modes sorted by mass with `[BIG WIN]`/`[WIN]`/`[NEAR-BREAKEVEN]`/`[LOSS]`/`[DISASTER]` labels + catalysts, and a risk-profile line. Built on top of `WaxContext.get_or_build()` so it shares the cache with `explore_game` — calling both costs one set of oracle samples, not two. Doctrine intact — labels describe outcomes, not picks.

2. **Rerun-fresh in the workbench.** New header button: pick a decision, check the desired improvised-tool subset, click **rerun fresh**. The workbench replaces `segments` with `[harvested_system + appended_tool_declarations, harvested_first_user_message]` and re-streams from turn 1. Burl plays the decision again with the new tools available from the system prompt onward — no chat-mode primer, no harvested trace. This is the surface where wishlist tools earn or lose their keep.

3. **First measurable lift.** Reran harvest_batched_20260425_072910 decision #1 (`BURL_BREAKS_CONSENSUS`, defense, position 2/4 in trick 1, lead 14(4-4) → led suit = 4s). Burl's hand: 2(1-1), 4(2-1), 18(5-3), 19(5-4), 21(6-0), 23(6-2), 25(6-4); legal 4-x options: 4(2-1), 19(5-4), 25(6-4).

   - **Original harvest:** `burl_play = 25(6-4)`, regret 3.53. Threw a 10-point count domino on trick 1, on defense.
   - **Rerun fresh (with `state_brief`, `legal_plays`, `play_brief`, `board_snapshot` all available):** Burl called `state_brief` first, immediately enumerated the legal subset (`I have 4(2-1) and 19(5-4) and 25(6-4). I can follow suit.`) before any sampling, then committed `4(2-1)` — a 0-count blank that defends correctly. The original harvest never reached that enumeration step.

   Different choice than the oracle (which wanted 19(5-4)) but defensively sound — preserved the 10-point 6-4. State-brief's upstream legality clarity is now confirmed across two seats and roles (defense seat-3 trick-2 follow-suit + this defense seat-1 trick-1 follow-suit case). Working hypothesis: across many seeds, state-brief lifts the rerun's mean regret on follow-suit decisions where the original Burl explored an illegal candidate before catching the constraint.

### Findings reinforced (second wave)

- **`play_brief` was registered but not used.** The protocol section of the system prompt mentions `explore_game(play=X)` literally; Burl follows that and never discovers `play_brief` even though its declaration is in scope. Lesson: adding a tool makes it *callable*; getting it *called* requires the protocol text to name it. Cheaper than adapter retraining; documented in [[improvised-tools]] under "Adoption asymmetry."
- **A separate reasoning bug surfaced**: [[count-vs-pip-sum-confusion]]. In the rerun-fresh decision-1 trace, Burl wrote *"4(2-1) is a low count domino (2 points). 19(5-4) is medium count (5 points). 25(6-4) is high count (10 points)"* — confusing pip-sum with count value. The actual count values are 0/0/10. He landed on 25 being expensive by accident (pip-sum 10 = count value 10 for that domino). This is a categorical-vs-continuous miss in the rules grounding, separate from anything `state_brief` or `legal_plays` covers. Patch path TBD; the [[burl-tool-wishlist]] doctrine says wait for Burl to ask for a count-ledger tool.
- **The lock-in is intact even with better tools.** Six tool calls (`state_brief` → `belief_trajectory` → `explore_game` → `probe_best_case` → `probe_worst_case` → `commit_play`) to land on a play that was deducible from `state_brief` alone. The new tools made the ritual better-informed; they did not break Burl out of the ritual. That's a structural finding, not a fixable one — and not a problem when the ritual produces a defensible play.

### Third wave (later same day) — autoFillFromHarvest silent-fallback bug, reflection-deafness

Two findings landed during the next round of rerun-fresh testing on harvest_batched_20260426_031338 decision #0:

**autoFillFromHarvest silent-fallback bug.** When Burl emits `explore_game(play=25)` in a rerun-fresh session, the workbench's stream-end handler runs `autoFillFromHarvest(tc)` to pre-populate the manual-review draft with a matching response from the harvest's recorded events. The original implementation's last line was `return matches[0].content` — i.e., if no exact arg-match was found, fall back to the *first* recorded `tool_result` for that tool name. For decision #0 the harvested explore_game was for `play=14`. So every rerun-fresh `explore_game(play=X)` for any X≠14 silently pre-populated with the harvest's play=14 data; the user clicked "feed" without re-reading; Burl reasoned over fabricated outcome distributions thinking he had explored multiple candidates. Three different play arguments (25, 20, 20) returned identical outputs in one shared trace.

Fix landed in `burl/chat/web/src/App.svelte`: `autoFillFromHarvest` now returns `null` on args mismatch (caller must fetch live or supply manually). Plus `explore_game`, `probe_best_case`, and `probe_worst_case` joined `AUTO_SERVE_BASE` so they live-dispatch to the chat server's `tools_runner` instead of going through the harvest-fallback path at all. Trade-off: the live samples are not byte-identical to what the harvest recorded (N=20 fresh resample), so join-at-end mode no longer reproduces the harvest's exact numbers — but it always returns correct numbers for the actual args. For "what did the model see at this point in the harvest?" the user reads the events panel directly; the chat path is now correct-by-args, not faithful-to-harvest.

Methodological lesson: the bug was invisible during normal use because the auto-filled wrong-args response is *structurally identical* to a real response (same prose schema, same numeric ranges, same catalysts format). It only became visible when one trace happened to call `explore_game` with three different play arguments and got identical output. Diagnostic protocol going forward: when comparing tool outputs across changed args in a rerun-fresh session, sanity-check that the headers (`PLAY: X(p-p)`) actually match the requested play before reading the rest of the prose. The improvised `play_brief` tool surfaces the play in its first line for exactly this reason.

**Reflection-deafness.** During the same trace the user injected, via the harness's free-text feedback channel routed back as a `commit_play` tool_result: *"that is not the best play. why?"* Burl's response was zero-engagement: re-ran `explore_game`, re-probed, re-committed `14`. Three identical commits in a row, no reflection. Documented as [[burl-reflection-deafness]] — third symptom of [[play-adapter-lock-in]] (alongside the wishlist's tool-spec response shape and the post-commit primer's load-bearing role). The implication for [[post-commit-q-and-a]]: the eventual training corpus has to include reflection turns explicitly, since the current model cannot produce them organically.

## Status

This spike is complete and superseded — [[burl-lab]] (dormant since 2026-05-07) was
built specifically on the findings named here (adoption asymmetry, hand-edited primer
drift, performance as second-pass derivation, reflection-deafness needing phase-scoped
retries). No further burl-chat sessions ran after this window.
