---
title: count-vs-pip-sum-confusion — Burl conflates pip-sum with count value
kind: topic
first_seen: 2026-05-01
last_updated: 2026-05-01
status: active
---

## What

Burl reads "count value" as **pip-sum** (the dots on the domino added together) instead of as the official Texas 42 count: only 5-5 (10), 6-4 (10), 5-0 (5), 4-1 (5), 3-2 (5) carry points; all 23 other dominoes are 0-count. The two coincide for the five count-carrying dominoes (and only those), so the mistake is invisible whenever Burl picks one of them. It bites silently the rest of the time.

## Where it surfaced

[[experiments/burl-chat-spike]] §"Session 2" — rerun-fresh on harvest_batched_20260425_072910 decision #1, defense, position 2/4 in trick 1, lead 14(4-4), led suit 4s. Burl needed to pick a 4-x to follow:

> If I play 4(2-1), it is a low count domino (2 points).
> If I play 19(5-4), it is medium count (5 points).
> If I play 25(6-4), it is high count (10 points).

The actual count values are 0 / 0 / 10. Burl is computing pip-sum: 2+1=3 (he wrote "2 points" — possibly reading the high pip alone, possibly miscomputing), 5+4=9 ("5 points"), 6+4=10 ("10 points"). He landed on 25(6-4) being expensive (correct, by accident — count 10 = pip-sum 10 for that one) and 4(2-1)/19(5-4) being cheap (also correct, by accident — they are 0-count, just not for the reason Burl gave). Decision-1 played fine because the heuristic happened to align; the next decision where the mismatch matters will not.

## Where it bites

Of the 28 dominoes, the 5 count-carriers are accidentally well-classified by pip-sum (all map to "count = pip-sum"). The 7 trumps in pip-suit-trump games confuse things further (5-0 is a count domino but trump under blanks-trump, no longer a member of fives). The remaining 16 zero-count non-trump dominoes are misclassified by pip-sum-as-proxy in proportion to their pip total — so a 6-2 (pip-sum 8, count 0) gets read as if it were nearly as expensive as a 5-5, a 6-3 (pip-sum 9, count 0) more so, etc. Worst-case examples:

| Domino | Count value | Pip-sum (Burl's read) | Worst-case mistake |
|---|---|---|---|
| 6-2 | 0 | 8 | refuses to play it on a defensive trick assuming it's a "high count" cost |
| 6-3 | 0 | 9 | same |
| 5-4 | 0 | 9 | refuses to lay it for partner; saves it inappropriately |
| 5-3 | 0 | 8 | same |
| 4-3 | 0 | 7 | same |
| 6-1 | 0 | 7 | same |

## Why it happens

A plausible read: the rules primer ([[topics/rules-adapter]]) listed count dominoes as a numeric set, and the post-train distribution learned "domino has high pips → domino is expensive." That's a usable shortcut for laymen and it is right on average for the count-bearing five — but it is the wrong abstraction for Texas 42's two-tier scoring system, where the 5 named count dominoes are categorical and everything else is zero. The model never internalized the discrete label; it kept the continuous proxy.

## How to detect

Look in any Burl reasoning trace for a sentence pattern: *"Domino X has high/medium/low count (Y points)"* where Y equals the pip-sum, not the count value. Counts are always 0, 5, or 10. Anything that says "2 points" or "9 points" or "8 points" is the bug.

## How to fix

Three layers, ranked cheap → expensive:

1. **Improvised tool** — `count_ledger`: for each domino in hand, emit `count_value: 0` or `count_value: 5/10`, with the loose-count budget. The same level of tool intervention as [[improvised-tools]] for legality. Prevents the bug at the surface; doesn't unlearn the bad heuristic.
2. **System-prompt patch** — add an explicit "Count value vs pip-sum" sentence to the rules section, naming all five count-bearers and stating that nothing else is count. Costs a paragraph of context; reaches all decisions.
3. **Adapter retraining** — add a count-categorical drill to the [[topics/rules-adapter]] / [[topics/trump-drilling]] corpus. Most expensive, most durable.

Order of operations should be: ship (1), confirm the error pattern disappears in rerun-fresh, then do (2) once the fix is needed across the whole harness; consider (3) only if (2) fails to transfer.

## Status

Identified 2026-05-01 in [[burl-chat]] rerun-fresh comparison. Not yet patched. Burl has not asked for the count-ledger tool — but the [[burl-tool-wishlist]] frame says we should let him ask first; if he doesn't and the bug bites a future rerun, build it then.

## Related

- [[burl-chat-spike]] — where it was first observed
- [[burl-tool-wishlist]] — the frame for whether to wait for Burl to ask
- [[topics/rules-adapter]] — likely site of the proxy heuristic's origin
- [[topics/learned-by-playing]] — adjacent: rules learn by playing, not by tabulation
