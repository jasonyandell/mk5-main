---
title: Play-adapter lock-in — STaR-distilled adapters cannot be talked out of commit_play
kind: decision
first_seen: 2026-04-30
last_updated: 2026-05-01
status: active
---

## Finding

A [[burl]] adapter distilled via [[star]] on play-decision traces (specifically tested: [[experiments/iter5-e1-rank-sweep|e1-rank16]]) cannot be redirected to plain conversation through prompting alone, even with all of:

- The full chat-cadence prefix from [[burl-chat]] (system + user state + Burl's reasoning trace + commit_play(N) + [[chat-mode-primer]] saying "ask me anything").
- `enable_thinking=True` to encourage prose generation.
- An explicit user instruction `do not output a tool call`.
- A direct question that does not require any tool ("how could the tools have been more clear?").

The model produces a partial conversational response (~3 sentences acknowledging the question), then snaps back to "I am Burl, I have committed play 14, I must execute the task" mode and outputs `commit_play({"domino_id":14})` — re-committing a play it already committed in the prefix.

## Implication

**Stacking a post-commit Q&A adapter on top of a play-decision adapter does not work.** The play distribution is loud enough to override the in-context recency of a chat primer plus an explicit prohibition. Any attempt to teach Burl conversational follow-up by SFT-ing a chat corpus on top of an existing iter-N adapter will fight the existing weights.

Two viable paths:

1. **Multi-task co-training.** Train an adapter that sees both play decisions AND post-commit Q&A in its corpus, jointly. The model learns both cadences and selects by cue.
2. **Train from base for chat.** Treat the post-commit Q&A adapter as a separate adapter loaded specifically for chat sessions. Swap adapters at session boundary. Loses voice continuity unless the play and chat adapters are co-distilled.

## Evidence (counterfactual A/B)

Same [[burl-chat]] prefix, same primer, same user question. Tested on `BURL_BREAKS_CONSENSUS` decision #0 from `harvest_batched_20260426_031338` (seed 0, burl_play=14, oracle_play=20, regret 12.12).

**Adapter loaded (`e1-rank16`):**
- Locked into `commit_play(domino_id=14)` output. Mid-response self-correction: "Since I have already called commit_play, the system will now process that. If it rejects as illegal, I will get another turn..." — re-derives the play-decision protocol and exits the user's question.

**No adapter (base Gemma 4 E2B):**
- Engages cleanly. Produces a structured three-section critique of the tool surface (*what made it challenging, what would have made it easier, in short*), citing actual numbers from the harvest's tool outputs (+11.7 Q lift, -5.8 Q drop, 6-0 vs 2-2 catalyst dominoes), and proposing three concrete tool-design improvements aligned with [[topics/rules-as-tools]] and [[topics/at-risk-points]]. Produces real product feedback from a 9-word user prompt.

The A/B is clean. The lock-in is the adapter.

## Open

- Does the [[iter3-rules-adapter]] (90% bot-match Burl winner, less aggressive distillation, on HuggingFace at `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules`) suffer the same lock-in, or is it more steerable than e1-rank16? Not yet tested locally — the iter3-rules adapter must be downloaded from HuggingFace first. Worth running through the same A/B in [[burl-chat]].
- Would a stronger system-prompt override at chat time ("the decision is over, this is now a review session, do not output tool calls") help on top of the primer? Untested. Plausibly closes some of the gap with adapters that are merely steered, not welded.
- Can a smaller chat-only LoRA be trained quickly enough to be worth it as a separate session-time adapter, swapped in over the play adapter? Open.

## Meta-layer corroboration (2026-05-01)

Second-session evidence (see [[burl-chat-spike]] §"Session 2"): even when explicitly invited to chat about itself ("describe the tool you'd want"), Burl produces **structured tool-spec plans**, not prose. It cannot drop the play-decision shape; it just maps it onto the meta-conversation. The play-decision protocol is: read state → think → call tools → commit. The meta-protocol Burl produces is: read request → think → propose tool with description+parameters → end.

Three meta-asks across two decisions, all in the same shape: "I would want a tool called X. **Purpose:** … **Input Parameters:** … **Output Format:** …" — wax_museum tool-declaration style, in prose. The lock-in is not just at `commit_play`; it is at the structural level of "the next assistant turn is a tool-call plan."

This corroborates the original A/B but extends the claim: the lock-in survives even when there is no decision to make. It is not an artifact of in-context recency to a play state; it is encoded in the weights as "what an assistant turn looks like." The post-commit-Q&A adapter must train against this structurally, not just contextually. See [[burl-tool-wishlist]] for the productive read of this finding — the meta-asks are themselves signal about which tools would help, even though the asker is welded.

## Related

- [[burl-chat]] — workbench
- [[burl-chat-spike]] — A/B session
- [[chat-mode-primer]] — what it can't unlock
- [[improvised-tools]] — meta-layer corroboration surface
- [[burl-tool-wishlist]] — the productive read of the meta-layer lock-in
- [[post-commit-q-and-a]] — research direction this constrains
- [[topics/conditional-outcome-structural-nonuse]] — adjacent finding: the same training pipeline produces structural non-use of certain tool calls, also an artifact of distillation distribution.
