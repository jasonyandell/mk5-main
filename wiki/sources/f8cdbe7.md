---
title: "Source digest: f8cdbe7 — Kerry Q&A generator + 15k corpus for Stage 0 v2"
kind: source
first_seen: f8cdbe7
last_updated: f8cdbe7
status: active
---

## Commit

- **SHA:** f8cdbe7fe30b2ce6dde92ebf9cecaca3c845f14f
- **Date:** 2026-04-11
- **Author:** Jason Yandell

> feat(lem): Kerry-structured Q&A generator + 15k corpus for Stage 0 v2
>
> Curriculum based on Kerry Newberry's Learner's Guide, engine-scaled:
>   A. Getting to Know the Dominoes (2250 examples, 15%)
>   B. Understanding the Game (3000 examples, 20%)
>   C. Following Suit (6000 examples, 40%) — the hard part, weighted heaviest
>   D. Who Wins the Trick? (3750 examples, 25%)
>
> Section C deliberately generates tricky hands where trump membership is
> confusing (e.g., 5-3 when fives are trump can't follow threes). This is
> the exact stumbling block for both humans and Gemma.
>
> Training target: B200, adapter to jasonyandell/gemma-4-e2b-texas42-stage0-kerry

## Files introduced / modified

| Path | Change |
|---|---|
| `lem/rules/generate_qa_v2.py` | New, 484 LOC. Kerry-structured Q&A generator: 4 sections, engine-scaled to 15k examples |
| `lem/gemma_star/train_stage0.py` | Modified: minor updates for v2 corpus path |

## Kerry curriculum breakdown

| Section | Topic | Examples | Weight |
|---|---|---|---|
| A | Getting to Know the Dominoes | 2250 | 15% |
| B | Understanding the Game | 3000 | 20% |
| C | Following Suit | 6000 | 40% |
| D | Who Wins the Trick? | 3750 | 25% |

Section C is weighted heaviest because following suit — particularly with trump — is the exact stumbling block observed in first and second contact. It deliberately generates tricky hands where trump membership is non-obvious (e.g., 5-3 when fives are trump belongs to the trump suit, not threes, and cannot follow a threes lead).

Total corpus: 15,000 examples at `lem/rules/qa_corpus_kerry.jsonl`.

## Target adapter

`jasonyandell/gemma-4-e2b-texas42-stage0-kerry` ([[kerry-adapter]]). Trained in [[sources/43009a4]].

## Related pages

[[kerry-curriculum]] · [[kerry-adapter]] · [[rules-adapter]] · [[lem]] · [[sources/43009a4]]
