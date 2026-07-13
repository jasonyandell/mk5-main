---
title: At-risk points — Roberson's bidding framework
kind: topic
first_seen: 2026-04-30
last_updated: 2026-07-13
status: complete
---

## What

Roberson's framework for [[texas-42]] bidding analysis, taught explicitly in chapters 2-8 of *Winning 42*. Bid prediction is computed **backwards**: enumerate the count dominoes and tricks you might lose, sum them, subtract from 42.

A hand has **trumps** (≥3 dominoes of one suit, your declaration), **doubles** (always high in their suit), and **offs** (any non-trump non-double — belongs to two suits and could lose tricks on either). Each off has a "high-side risk" and a "low-side risk." The biggest dangers are 4-offs and 5-offs because they can cost 15 count on one trick (4-6 + 4-1 or 5-5 + 5-0).

Two key rules from the chapter:

- **Offs make or break the bid, not trumps.** A hand with all 7 sixes for trumps but a 5-off can still set itself; a hand with 3 trumps and protected offs can confidently bid 35.
- **Double ahead of your off.** Holding the double-six in front of a 6-trey off reduces the risk on the high side; the double pulls the dangerous 6-4 onto a trick you control before the off has to lead.

## Why it matters here

Roberson's vocabulary is the **canonical 42 voice**, owned by the project via the user's family heritage (family heritage: Roberson's book names the user's actual relatives). Any post-commit Q&A corpus ([[post-commit-q-and-a]]) needs to talk in this register, not in Gus/forge vocabulary (Q axis, mean shifts, catalyst dominoes).

The framework also doubles as a [[topics/rules-as-tools]] target: tool outputs surfacing "you have two offs at five-deuce and ace-blank, your at-risk-points come to 12" would replace raw E[Q] histograms with a Burl-native interpretation layer the model already wants ([[burl-chat-spike]] surfaced this as spontaneous product feedback).

## Worked-example corpus

Each "HAND N" in chapters 2-8 is structurally `(hand, declaration, bid)` plus prose justification — directly usable as a labeled corpus for either:

- **Bidding eval:** does [[forge]]'s oracle agree with Roberson on the bid? Cheap third-party calibration.
- **Q&A reference traces:** Roberson's prose IS the ground-truth answer to "why did you bid 35 here?" Train a teacher (Haiku 4.5 or eventually [[iter3-rules-adapter]]) to extend the same prose style to arbitrary harvested decisions.

## Related

- [[texas-42]] — the game
- [[post-commit-q-and-a]] — research direction this is the voice anchor for
- [[topics/rules-as-tools]] — analogous framing applied to evaluative tools
- [[burl-chat-spike]] — Gemma spontaneously suggested the "why over how" framing that matches this
