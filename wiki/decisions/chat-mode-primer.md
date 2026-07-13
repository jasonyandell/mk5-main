---
title: Chat-mode primer — synthetic assistant turn after commit
kind: decision
first_seen: 2026-04-30
last_updated: 2026-07-13
status: complete
---

## Decision

When loading a wax_museum decision into [[burl-chat]] as a conversation prefix, inject a synthetic assistant turn after the `commit_play` segment. Default text:

> Yeah, I committed N. The decision is done — ask me anything about it and I'll talk it through with you. No more tool calls.

The user's first question follows this primer turn, not the raw `commit_play` envelope.

## Why

[[burl]]'s training distribution is `(system="you are Burl, pick a play", user=state, assistant=thought + tool_calls + commit_play)`. Adapters distilled on this shape (especially [[experiments/iter5-e1-rank-sweep|e1-rank16]]) have been pulled hard toward emitting `commit_play` as the next assistant turn. Even base [[gemma-4-e2b]], without an adapter, shows residual pattern-matching: the system prompt is 5,300 chars of "call tools, commit_play when ready," and the reconstructed reasoning trace ends with `[committed play 14]` — both signals telling the model "you are mid-decision, the next thing is to commit."

A user message at the end ("ask me anything") is one short string against ~14 K tokens of "do tool calls." The flea cannot beat the elephant on a single turn.

The primer flips the most-recent turn from "tool call" to "conversational handoff," exploiting models' strong recency bias for in-context pattern matching. The model now sees: …state…, …reasoning…, commit, **chat handoff**, user question — and continues the chat-handoff cadence instead of the play-decision cadence.

## Evidence

[[burl-chat-spike]] confirmed the primer is load-bearing:

- **Without primer + base Gemma:** model produces "I am Burl, my next action is to call commit_play" responses; refuses to engage as a conversation partner.
- **With primer + base Gemma:** model produces fluent prose Q&A, structured critiques, references actual numbers from the harvest's tool outputs.
- **With primer + e1-rank16 adapter:** model partially engages then snaps back to `commit_play` mid-response. Recency is not strong enough to override the adapter weights. See [[play-adapter-lock-in]].

## How to apply

Frontend (`burl/chat/web/src/App.svelte`) appends the primer segment in `pickDecision` after loading the harvest's typed segments. The primer becomes the last assistant turn before the user's message in `buildMessagesForModel`'s output, so the model sees it in context.

Default template uses the integer `final_play` from the commit. Future iterations may:
- Use Roberson vocabulary ("Yeah, I laid down the 4-4. The hand's done — ask me about the offs, the count, or the partner read.").
- Be user-editable per session.
- Vary by bucket (a `BURL_BREAKS_CONSENSUS` decision could prime with "I disagreed with the bot here. Ask me about it.").

## Limits

- **Does not unlock locked adapters.** The primer makes a steerable model steerable; it does not make a welded model talk. See [[play-adapter-lock-in]].
- **Voice not yet right.** The default template is generic LLM cadence, not family-game register. Open follow-up: Roberson-flavored primer per [[post-commit-q-and-a]].
- **No structural change to the prefix.** The primer is appended; the play-mode system prompt is still there. If even base Gemma starts drifting (as it occasionally does mid-response), stripping/replacing the system prompt at chat time is the next lever.

## Related

- [[burl-chat]] — the workbench
- [[burl-chat-spike]] — first sessions
- [[play-adapter-lock-in]] — what the primer can't fix
- [[post-commit-q-and-a]] — the research direction this enables
