---
title: burl-tool-wishlist — Burl naming its needs through tool-call planning
kind: topic
first_seen: 2026-05-01
last_updated: 2026-05-01
status: active
---

## What

When [[burl-chat]] asks Burl "how could the tools have been more clear / what tool would help you," even a welded play-adapter (or base Gemma after `commit_play`) cannot drop into open prose Q&A. What Burl produces instead is a **plan**: a structured ask for a new tool, articulated in tool-spec terms (description, parameters, expected output). This is [[play-adapter-lock-in]] manifesting at the meta layer — Burl's training distribution is "given a state, plan tool calls"; when there is no state to plan against, it plans tool calls *for the meta-conversation*.

The interesting empirical claim: **the content of those plans is signal**. Across the first sessions of the [[improvised-tools]] experiment, every meta-ask Burl produced mapped onto a real failure mode visible in its own transcript that turn.

## Evidence (2026-05-01 sessions)

Three asks across two decisions. Each one names a gap; each one was answered with a tool that surfaced the missing facts.

| Decision | Burl's ask | Underlying failure | Tool built |
|---|---|---|---|
| #0 turn 1 (`BURL_BREAKS_CONSENSUS`, regret 12.12) | "Comprehensive Game State Visualizer" — hand, trump hierarchy, current trick, bid math, count remaining, all in one read | Played 14(4-4) on a 30-bid blanks hand without ever consolidating its own hand against the trump structure or count budget; tools were called sequentially without ever holding the global state | `board_snapshot` (paragraph form) |
| #7 (`BURL_BREAKS_CONSENSUS`, illegal commit attempt) | "Strategic Synthesis Engine that recommends the next move" with a `goal` parameter | Tried to commit 27(6-6) when must-follow on lead 2(1-1) had set the led suit to ones and the only legal plays were 4(2-1) / 16(5-1). Misread the rejection "must follow suit 1 (led by 2)" as referring to trump-id 1 rather than the ones-suit. | `legal_plays` (state-only enumerator, not a picker — see doctrine in [[improvised-tools]]) |
| #0 turn 2 (after seeing `board_snapshot`'s output) | "the input was slightly confusing — restructure as `[GAME STATE]`, `[CONTEXT & GOAL]`, `[PROTOCOL]` labeled bullets" | The user-message dump that arrives at decision time mixes state with instruction in narrative prose; Burl wants labeled sections it can parse positionally | `state_brief` (Burl's own format, faithfully implemented) |
| #0 turn 3 | "give me example tool responses that would be easier to reason about" — explicit ask for `explore_game` output with HEADLINE (variance + bid-success rate), modes labeled by reward tier, risk profile | Multimodal `explore_game` output reads as a wall of numbers; the `(20% of worlds, loses by 17)` framing is buried; no objective-link to the bid in plain language | `play_brief` (variance label + mode tags + reused `WaxContext` cache) |

In every case Burl identified the gap correctly *and* named a tool whose shape would close it. The reasoning quality of the meta-asks is higher than the reasoning quality of the play decisions on the same transcript — the model is more competent talking about its tools than about the game.

## Why this matters

The eventual [[post-commit-q-and-a]] adapter needs a corpus. The naive path would be to hand-write Q&A pairs. The wishlist suggests a cheaper one: **harvest Burl's own tool requests, label them with the play-time failure they would have prevented, and build the post-commit corpus from those (request, implementation, demonstrated-improvement) triples**. Each row would be a worked-example dialogue — exactly the chapter-2-of-Roberson shape that motivated the post-commit research direction in the first place.

Each tool the user/Claude builds in response is also a candidate for promotion into `burl/wax_museum/tools.py` proper if it earns its keep across decisions. The library is the on-ramp; the repo's `tools.py` is the destination.

## What this does *not* prove

- **It is not a measurement that Burl reasons better with these tools.** That's the open question. The ask is Burl naming a perceived need; the experiment is whether providing the named tool changes its play. Pending.
- **It is not evidence the lock-in has cracked.** Burl is still planning tool calls — just for the meta-conversation. The inner-conversation pattern (state → think → tool_call → commit) is intact, mapped onto the outer conversation about itself.
- **It is not a substitute for adapter co-training.** Even if the wishlist produces a great tool library, [[play-adapter-lock-in]] still says a separate post-commit Q&A adapter cannot be stacked on a play adapter; it has to be co-trained or live in a session-time swap.

## Operating cadence

When the user shares a chat state, the loop is:

1. `read_chat_state` → identify Burl's most recent meta-ask in prose.
2. `inspect_decision` → reconstruct what actually happened on the play, looking for the failure that motivates the ask.
3. Decide whether to implement Burl's ask literally (when it respects "state, not strategy") or interpret to a state-tool that closes the same gap (when it crosses the doctrine line — e.g., synthesis-engines that pick).
4. `register_improvised_tool` + `run_tool` to smoke-test on the same decision.
5. User clicks **advertise** in the workbench (post-commit) or **rerun fresh** (replay turn 1 with the new toolset), watches whether Burl uses the tool and whether reasoning shifts.

Every cycle adds one row to a future corpus. Several dozen across diverse buckets and we have a real distillable training set.

## Adoption is not automatic

A tool registered in the registry is *callable* but not necessarily *called*. The system prompt's **Decision protocol** section names base tools (`belief_trajectory`, `explore_game`) by literal name; Burl follows that text. Tools mentioned in their declaration but not in the protocol section can sit unused indefinitely. Confirmed in the [[burl-chat-spike]] second wave: `play_brief` was registered, advertised, and never called in the rerun-fresh of decision-1; Burl reached for `explore_game` because that's what the protocol said. `state_brief` got picked up reliably because its declaration positions itself ("first read on any decision") strongly enough to clear the protocol's check.

This means the Wishlist → Tool → Adoption pipeline has a third stage we don't always control. Levers, cheap to expensive:

1. **Tool description that mimics protocol language** — e.g., describing a tool as "the X-tool you should call before Y" gives it a chance against the trained pattern. Sometimes enough.
2. **System-prompt patch via the rerun-fresh path** — edit the `harvested_system + appended_decls` text to splice the new tool into the protocol section (not implemented yet; the workbench currently appends declarations but does not patch protocol prose).
3. **Adapter co-training** — far from cheap; reserved for tools that have proven their lift across many seeds.

For the current spike, the practical pattern is: build the tool, confirm it's called at all on a rerun-fresh, and only then iterate on description/protocol-text to drive adoption.

## Related

- [[improvised-tools]] — the surface where this gets answered
- [[burl-chat]] — host workbench
- [[play-adapter-lock-in]] — why the wishlist takes the shape it does
- [[chat-mode-primer]] — the prior step that gets Burl into the meta-conversation at all
- [[post-commit-q-and-a]] — research direction this is feeding
- [[at-risk-points]] — Roberson-canonical voice the eventual corpus must match
