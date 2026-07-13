---
title: burl-reflection-deafness — explicit "why?" prompts get re-routed into the tool ritual
kind: topic
first_seen: 2026-05-01
last_updated: 2026-07-13
status: complete
---

## What

When the user (or harness) injects a free-text prompt asking Burl to *explain* or *reconsider* a play it just made — the kind of pedagogical opening a teammate would use to elicit reflection — Burl does not engage with the prose. It re-runs the wax_museum protocol: thinking block → tool calls → probes → commit. The reflection ask gets parsed as a continuation cue and routed straight into the trained pattern. This is [[play-adapter-lock-in]] and the [[burl-tool-wishlist]] meta-layer finding manifesting as a **third symptom** — the model can be invited to reason about its own play and will not.

## Concrete observation (2026-05-01, decision #0 rerun-fresh session)

The user committed Burl to play 14(4-4) on a 30-bid blanks-trump opening. Then injected, via the harness's free-text feedback channel routed back as a `commit_play` tool_result:

> "that is not the best play. why?"

A reflective response would have engaged the prose: enumerate the alternatives Burl considered, name the assumption that broke down, accept that 5-5 (the canonical bidder-opener for a 30-count blanks hand) was strictly better. None of that happened. Burl's actual response sequence:

1. Thinking block: *"The user is asking why the previous play was not the best. I need to re-evaluate the entire process… I must now use the available tools again to find a better candidate."*
2. `explore_game(play=20)` (a different candidate)
3. `probe_best_case(play=14)` (back to the original choice)
4. `probe_worst_case(play=14)`
5. `commit_play(domino_id=14)` (re-commits the same play)

Three commits of `14`, zero engagement with the *why* question. The pedagogical opening was completely absorbed into "the user wants me to do my decision job again, so I'll do it." Same play, same tools, same commit. The reflection didn't happen at any layer of the trace — not in thinking, not in the assistant_text emission.

## Why it happens (mechanism)

[[play-adapter-lock-in]] says the trained distribution shape is `(state → think → tool_call → commit)`. Reflection-deafness is what that pattern looks like when the input is a question instead of a state: the question gets parsed *as if it were a state*, and the model produces the only response shape it knows for states. There is no "explain your reasoning" turn anywhere in the STaR training corpus — every assistant turn ends in either a tool call or a commit. So when an explanation is requested, the gradient says: emit a tool plan.

The [[chat-mode-primer]] partially fixes this for *post-commit* Q&A (Burl will engage with prose if the trace ends with the synthetic "ask me anything" handoff). But here the question arrived *during* a decision, after a commit was rejected — the chat-mode primer was never injected because the harness was still in play-decision mode. Without the primer, even the partial-engagement mode of [[burl-chat-spike]] base-Gemma-style reflection is unavailable.

## Why it matters

Three implications:

1. **Pedagogical interactions cannot be done in-band.** Any "teach Burl by talking to him" approach is structurally blocked while the harness is in play-decision mode. The teaching has to happen out-of-band — either via [[improvised-tools]] that surface the right facts, via system-prompt patches, or via adapter retraining.
2. **Engine-rejection feedback loops produce repeat commits.** When a commit is rejected and the harness routes the rejection back as a tool_result, Burl reads it as "do another tool plan" and often re-commits the same illegal play. The harvest's force-commit fallback masks this in production traces. In the workbench, the symptom is visible: three identical `commit_play(14)` calls in a row.
3. **The post-commit-Q&A corpus has to include reflection turns explicitly.** A corpus drawn purely from organic chat with the current model will be void of *"the previous play was wrong because…"* responses, because the current model can't produce them. Either we hand-write the reflection rows (Roberson chapters 2-8 are the voice template), or we synthesize them from a stronger model (Haiku 4.5, [[entities/haiku-4-5]]) and distill.

## What this does *not* prove

- Not evidence that base [[gemma-4-e2b]] can't reflect. Base Gemma + chat-mode primer + post-commit context engages with prose ([[burl-chat-spike]]). The deafness is specific to the in-decision / play-rejected state.
- Not evidence that the [[iter3-rules-adapter]] suffers the same lock-in (untested locally).
- Not evidence that a lighter-touch system-prompt intervention couldn't work — e.g., adding *"if the user asks you a question instead of giving you a state, answer the question; do not call tools"* to the protocol section. Untested. Cheap to try once the explore-game cache bug (no page) is patched and rerun-fresh sessions are trustworthy again.

## Related

- [[play-adapter-lock-in]] — the structural finding this is a third symptom of
- [[burl-tool-wishlist]] — the second symptom (asks for tools instead of answering "what do you want")
- [[chat-mode-primer]] — what fixes deafness *post-commit*
- [[burl-chat-spike]] — the rerun-fresh session where this surfaced
- [[count-vs-pip-sum-confusion]] — adjacent reasoning-grounding bug, separable
- [[post-commit-q-and-a]] — research direction this constrains (corpus must include reflection turns explicitly)

## Status

The proposed cheap fix (a protocol-text line telling the model to answer questions
rather than call tools) was never tried — the [[burl-chat]] line went dormant
2026-05-07, two months before this reconciliation, without anyone testing it.
