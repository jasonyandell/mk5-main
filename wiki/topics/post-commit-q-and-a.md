---
title: Post-commit Q&A — talking with Burl after a hand
kind: topic
first_seen: cba521d
last_updated: cba521d
status: retired
---

## What

A research direction and product slot: after [[burl]] commits a play, open a conversational channel where a human (or a teacher model) asks Burl questions about that decision. The reply is plain prose, in the family-game register, grounded in the actual tool calls and reasoning Burl produced during the decision.

This is distinct from:
- **Play-time tool use** — the existing [[topics/tool-orchestration]] / [[topics/rules-as-tools]] surface.
- **Pre-game commentary** — the explanation-sketcher slot originally framed as the [[lem]] product slot.
- **Self-rationalization for STaR** — the [[r1-rationalization]] mechanism, which re-grounds a chosen play to a target answer.

Post-commit Q&A is a **fourth surface**, oriented at the human player's seat, not at the training loop.

## Why

[[gus]] was conceived to be the belief brain that informs the LLM at decision time. The natural product above that is talking with the player about decisions — the way a teammate would, after the trick.

The closest published precedent is chapters 2–8 of Roberson's *Winning 42*, which is structurally a worked-example dialogue: "here is a hand, here are the trumps and offs, here is the at-risk-point analysis, here is the bid." Each "HAND N" is a (state, declaration, bid, prose-justification) tuple authored by a champion. That's exactly the corpus shape post-commit Q&A wants — hands paired with conversational reasoning traces — and the project already owns the canonical text via the user's family heritage.

## Three cuts at the corpus

### 1. Reference-trace corpus (Roberson-grounded)

Take Roberson's worked hands from chapters 2-8. For each: extract the (hand, declaration, bid) triple, run [[forge]] to get the oracle E[Q] for each candidate play, run [[burl]]'s tool surface to produce a reasoning trace, then have a teacher model (Haiku 4.5, or eventually the [[iter3-rules-adapter]]) produce a Q&A continuation in Roberson's prose style.

Sources of voice anchor: Roberson chapters + Flemmons foreword + transcribed dialogue passages.

### 2. Spike-mode corpus (workbench-grounded)

Use [[burl-chat]] interactively. For each `BURL_BREAKS_CONSENSUS` / `BURL_INDEPENDENT_RIGHT` / `ALL_AGREE_CORRECT` decision the user finds interesting, ask Burl a question, capture the Q+A pair into JSONL. Hand-curated, low volume, high quality. The [[burl-chat-spike]] session shows base Gemma + chat-mode primer can produce real critique without any training — those rows are corpus-quality already.

### 3. Auto-generated corpus (Haiku 4.5 reference traces)

Reuse the [[haiku-4-5]] reference-trace pattern from Burl's iter-2 prep ([[topics/reference-trace-distillation]]). Pipe each harvested decision through Haiku 4.5 with the chat-mode primer + a Roberson primer. Generate N questions per decision and Haiku's prose answers. High volume, machine-quality, suitable for SFT on top of an existing Burl adapter.

## Hard constraints

### Distillation short-circuit

The original [[burl]] design forbade evaluative tools (`get_eq`, `best_move`, `simulate_plan`) because they leak the answer. Post-commit Q&A faces a similar trap: the corpus must teach "I committed 14 because the at-risk-points framework points there," NOT "I committed 14 because the oracle said so." Roberson's framework is the firewall — concrete, articulable, and not a direct oracle echo. See [[topics/conditional-outcome-structural-nonuse]] for the related concern about evaluative tool calls.

### Adapter lock-in

Stacking a Q&A adapter on top of an existing play-decision adapter does not work cleanly: [[play-adapter-lock-in]] shows e1-rank16 ([[experiments/iter5-e1-rank-sweep]]) cannot be talked out of `commit_play` even with explicit prompting + a chat-mode primer. The post-commit Q&A adapter has to be co-trained with play decisions (multi-task) or trained from base, not stacked. Open question whether the [[iter3-rules-adapter]] (90% bot-match winner, less aggressive distillation) is more steerable than e1-rank16; not yet tested.

### Voice authenticity

The model must sound like someone who's actually played a thousand hands of 42, not like an LLM explaining 42. Roberson's transcribed prose has the cadence; the user's family is the authenticity check. A native Texan player can hear a fake explanation in two sentences. This is a real verification signal — anything in the corpus that doesn't pattern-match Roberson's voice is suspect.

## Open questions

- Does the [[iter3-rules-adapter]] (90% bot-match winner, on HuggingFace) suffer the same lock-in as e1-rank16, or is its less-aggressive distillation more steerable? Pull the adapter and A/B in the workbench.
- Does a Roberson primer (Flemmons foreword + chapter 2 paragraphs in the system prompt) shift the model's vocabulary toward "off / walker / at-risk points" without retraining? Cheap test in the workbench.
- Does bucket category (agreement / disagreement / forced / illegal) map to qualitatively different self-critique shapes? Need ~3-4 decisions per bucket sampled before this is meaningful.
- Can [[gus]]'s `belief_trajectory` be exposed as a chat-time tool ("what did Gus think the partner held when you committed?") in the conversational surface, not just at decision time?

## Status

None of the three proposed corpus cuts were ever executed. The [[burl-chat]]/adapter
line has been dormant since 2026-05-07, and the project's current answer ([[jud]])
doesn't route through an LLM adapter at all — post-commit Q&A has no active carrier
architecture.

## Related

- [[burl-chat]] — the workbench
- [[burl-chat-spike]] — first session and findings
- [[chat-mode-primer]] — the priming technique
- [[play-adapter-lock-in]] — the A/B that says you can't talk to the play adapter
- the user's 42 family heritage (project memory) — the family-game authenticity standard
