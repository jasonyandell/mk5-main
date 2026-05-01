---
title: burl-chat spike — first interactive sessions and findings
kind: experiment
first_seen: cba521d
last_updated: cba521d
status: active
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

The base model's spontaneous critique used Gus/forge vocabulary (Q axis, mean shifts, catalyst dominoes), not Roberson vocabulary (offs, walkers, double ahead of your off, at-risk-points). That's because the harvest's *tool outputs* speak Gus, not Roberson. To get the family-reunion voice ([[user_role_and_north_star]]'s standard, since Roberson's *Winning 42* names the user's actual cousins), the tools themselves need to surface Roberson framing, OR a Roberson primer needs to ride the system prompt. Probably both.

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
