---
title: "Source digest: eebcae5 — iter-2 training launcher + chat-template schema surprise"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** eebcae5882d5bd22d17a7e34c8406dd24780c438
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): iter-2 prep — training launcher + chat-template schema surprise
>
> burl/train/star_iter2.py — thin launcher over train_iter0 with iter-2
> defaults (118-row blended corpus, burl-iter2 adapter name, B200).
>
> Schema surprise: Gemma 4 E2B's chat_template.jinja strips
> <|channel>thought blocks before tokenization. The trainer never sees
> thought prose. The T4 blend's benefit is (1) coverage and (2)
> data-augmentation regularization — not verbosity reduction. Inference
> rambly thoughts are base-model reflex, not trained-in behavior.
> Pinned as regression test test_chat_template_strips_thought_blocks.
>
> Part of burl-iter2-prep team (T6).

Training launcher for iter-2 (`burl/train/star_iter2.py`) plus a significant schema discovery: Gemma 4 E2B's chat template strips `<|channel>thought` blocks before tokenization, so SFTTrainer never trains on thought prose. This reframes [[ls-mixture]]'s verbosity blend — the benefit is corpus coverage and regularization, not verbosity shaping. Also explains why iter-0/iter-1 still ramble at inference: rambly thoughts are base-model reflexes, not trained behaviors.

## Related pages

[[ls-mixture]] · [[burl]] · [[lora-unsloth]] · [[3414507]]
