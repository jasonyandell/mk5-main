---
title: "Source digest: edf86e9 — SFTConfig max_seq_length=4096, thought-bearing rows no longer truncate"
kind: source
first_seen: edf86e9
last_updated: edf86e9
status: active
---

## Commit

- **SHA:** edf86e981c92dcef48d75cf6a894c6b4dd9a6b84
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> fix(burl): SFTConfig max_seq_length=4096 — thought-bearing rows no longer truncate
>
> Burl's preserve_thoughts corpus has median 2054, max 4210 tokens per row.
> TRL's SFTConfig defaults to 1024, silently truncating exactly the thought
> regions the recipe is meant to train on.
>
> Very likely the root cause of the iter-4-thoughts byte-identical result
> (previously diagnosed as a LoRA capacity ceiling). No Burl adapter before
> iter-5 was trained on complete thought-to-tool-call traces.
>
> Local path (star_mlx.py) was already fixed. This applies the equivalent
> fix to the Modal recipe (star.py).

Applies `max_seq_length=4096` to the Modal SFT recipe (`star.py`), matching the local path fix already in [[sources/6fea6ab]]'s `star_mlx.py`. Reframes the [[experiments/iter4-null-preserve-thoughts]] byte-identical A/B as a truncation artifact rather than a LoRA capacity ceiling. See [[decisions/sft-max-seq-length]] for the generalizable principle.

## Related pages

[[decisions/sft-max-seq-length]] · [[preserve-thoughts]] · [[experiments/iter4-null-preserve-thoughts]] · [[burl]] · [[sources/6fea6ab]]
