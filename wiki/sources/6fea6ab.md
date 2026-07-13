---
title: "Source digest: 6fea6ab — Burl MLX-LM local path (Apple Silicon)"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** 6fea6ab536be822b284f69df12e19c7cebab3ba1
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): local MLX-LM path — training + inference on Apple Silicon
>
> Moves Burl's training + rollout loops off Modal and onto any M-series Mac.
> 1.86× wall speedup, up to 2 concurrent processes per spike measurements.
> Weight mmap shares cleanly across processes up to 5 workers on 48GB.
>
> - burl/modal/gemma_local.py — in-process MLX-LM-backed Gemma 4 E2B with
>   same NativeModelCallable interface as gemma_serve_native.py.
> - burl/train/star_mlx.py — preserve_thoughts bypass intact. Peak ~10 GB
>   @ rank 4 with grad_checkpoint, ~26 GB @ rank 16 with max_seq_length=4096.
> - CLI: --local flag on rollout/spike; --model-source local on spike.
> - test_gemma_local.py + test_star_mlx.py pin boundary-token atomicity
>   and <|channel>thought round-trip invariants.
> - burl/requirements-mlx.txt: mlx>=0.29, mlx-lm>=0.29.

Closes the corpus-generation + training + eval loop as a fully local pipeline on M-series Macs. `gemma_local.py` implements the same `NativeModelCallable` interface as the [[modal]] path so callers are model-source-agnostic. `star_mlx.py` ports the `preserve_thoughts` bypass and includes `max_seq_length=4096` (the local path already had the fix that [[sources/edf86e9]] later applies to the Modal path).

## Related pages

[[mlx-lm]] · [[burl]] · [[gemma-4-e2b]] · [[modal]] · [[preserve-thoughts]] · [[sources/edf86e9]]
