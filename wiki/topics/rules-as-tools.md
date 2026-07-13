---
title: Rules as Tools (primer content as callable tools)
kind: topic
first_seen: 2026-04-19
last_updated: 2026-07-13
status: retired
---

## Overview

Rules-as-tools is an alternative to the [[primer-tradeoff]] approach of baking rules into a system-prompt primer. Instead, rule content — count values, trump relationships, void rules, contract progress — is exposed as explicit callable tools that the model queries on demand. The model must learn to ask rather than know (b3a27e2).

## Four tools

Implemented in `burl/tools/rules.py` as thin wrappers over `forge/oracle/tables` helpers (no reimplemented logic):

| Tool | Wraps | Purpose |
|---|---|---|
| `count_dominoes_remaining` | `DOMINO_COUNT_POINTS` | Count values for remaining dominoes |
| `trick_winner_if` | `resolve_trick`, `trick_rank` | Who wins a trick under a hypothetical |
| `what_beats_what` | `can_follow`, `is_in_called_suit` | Suit/trump ranking in current context |
| `contract_progress` | `led_suit_for_lead_domino` | Bid, tricks taken, points needed |

(b3a27e2)

## Trade-off vs primer

The primer suppresses tool-use breadth — iter-0 showed that a ~2,700-token primer made the model `is_legal`-heavy and `eq_outcome_distribution`-shy ([[tool-orchestration]] "Primer trade-off"). Rules-as-tools replaces the 1,549-word primer with a 645-byte preamble, shifting rule queries to explicit tool calls. The model must call `what_beats_what` to know trump relationships rather than retrieving the answer from attention on the primer text. Whether this produces better tool-use breadth than the trimmed-primer approach is an open question (b3a27e2, 80704f0).

## Wiring and status

Wired behind `enable_rules_tools=False` flag in 80704f0. When `True`, the four tools are registered in `build_tool_registry` and published in `TOOL_SCHEMAS`, and the compact preamble replaces the trimmed primer. The 42-aware framing block is unchanged (iter-1 evidence confirmed it is net positive regardless). iter-2's verbosity-blend experiment keeps the flag `False` to hold the experiment to a single variable; iter-3 is planned to flip it (80704f0).

## Validated (iter-3-rules)

iter-3-rules trained with `enable_rules_tools=True` and primer removed. Results (dbadb5f):

- **90% bot-match** — the highest recorded for any Burl adapter.
- 0 retry-exhausted.
- 100% first-legal.

Key behavioral finding: `trick_winner_if` tool USAGE INCREASED after SFT. The adapter learned to call the rules-tools more aggressively rather than internalizing the rules as weights. "Tools replace memorization." (dbadb5f)

The [[primer-tradeoff]] tradeoff resolves at this frontier: "rules-as-tools + no primer" is the winning configuration within the Burl adapter lineage (dbadb5f).

## Status (dormant since 2026-05)

This 90% win is real and stands as measured, but the whole [[burl]] line has had zero
commits since 2026-05-07 and was superseded by [[champion]] / [[jud]] (pure NN bid/play
nets) around 2026-06-09. "Current Pareto frontier for Burl" no longer describes an
active program — there is no active Burl frontier to be current for. Separately,
[[gemma-tool-response-shape]] found that every rollout through iter-3-rules
ran with tool responses invisible to the model (a chat-template bug fixed at `54f7776`,
after this adapter was trained); the 90% number was never re-measured under a working
tool-response harness. See [[iter3-rules-adapter]].

## Links

[[burl]] [[tool-orchestration]] [[primer-tradeoff]] [[iter3-rules-adapter]]
