---
title: "Rematch: Zeb 0.4.1 (the overcall) against the overcalling Demon"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

**STRATEGY-FUSION FENCE.** The Demon ([[the-demon]]) is seated as a sanctioned
opponent only; its values never enter Zeb. This is a measurement of Zeb.

## Question

[[demon-overcalling-2026-10-15]] confirmed that the Demon, once it contests an
opponent's point bid, beats native Zeb 0.4.0's match-aware bidder: 65.14% [61.52,
68.85] of games (99.17% interval; chance is 50%). Zeb 0.4.1 ships the overcall,
tuned for a table that contests ([[match-aware-bidding-2026-10-14]], #170). Does
0.4.1, in its shipped configuration, win more games against that Demon, and by how
much more than 0.4.0 does on the same deals?

## Pre-registration (written before any rematch game)

**Opponent.** The frozen confirmed Demon participant from #165,
`demon-pimc-n100-bidc102a137-straight-2e233d29` (source `2e233d29`, bidder
`pimc-over:64:0.875:0.5`, native binary SHA-256 `1c2ccee8…`), unchanged.

**Zeb 0.4.1.** Participant `zeb-0.4.1-contested`, pinned to tag `zeb-v0.4.1` at
`0b57a8f5dff9fc7b5249c01be88927b7c8c7ecd0`: `chooseBid` with `match` and
`overcall: true` (the contested constants), the seven pip declarations as measured,
160 worlds, cutoff 0.8, seed 0, partner pass, shipped soft reading and exact
margin ties for play. The Rust player is the same compiled-native player the Demon
trials used (trees `zeb/src`, `zeb/net`, `scenario/core` are byte-identical in
0.4.0 and 0.4.1, so one native binary, SHA-256 `c76303d0…`, seats both). The
package's JavaScript is read from the tag because the npm tarball
`@jasonyandell/zeb@0.4.1` was not yet on the registry when the plan was sealed;
its wasm is 0.4.0's, unchanged, and the native host never runs it. The shipped
`index.js`, `match.js` and the wasm are hashed in the sealed plan.

**Reference arm.** `zeb-0.4.0-match`: 0.4.0 match-aware, pip declarations as
measured, same n160, cutoff 0.8, seed 0, same native player and host. It plays the
same Demon on the same deals as 0.4.1 (the two faceoffs share one seed, so pair
`p` is the same deals in both), which measures 0.4.1's gain over 0.4.0 on equal
deals.

**Design.** Two faceoffs on the `marks7` ladder (first to seven marks, straight
42), each **256 duplicate pairs / 512 games**, seats swapped inside every pair.
Faceoff seed `44b264ca537d025d` (fresh: drawn once from the system random
source, never used before; each pair's base seed is the tournament's
`sha256(seed/pair)`). **Fixed size: no early stopping, no dropped games.** A game cut
off by the slot's time cap is set aside and replayed from the start, with a line in
`restarts.log`; nothing is discarded or re-rolled.

**Primary.** Zeb 0.4.1's game-win share against the Demon, with a 95% pair
bootstrap interval (resample the 256 pairs, 40,000 resamples, seed 666). Chance is
50%.

**Secondary (not gated).** Marks per game (zero is even, same interval); the
0.4.1 minus 0.4.0 differences in win share and marks/game, paired on equal deals
(the same resampled pair indices in both arms); runtime ratios as reported, the
Demon's total seat CPU and wall divided by Zeb's (parity is 1), plus Zeb 0.4.1's
cost over 0.4.0's. Runtime includes each seat's startup, bids, declarations,
plays, close, IPC and Python adapter on this machine (Apple M5 Max), with the
Node host's waited native child counted; it is shared with other sessions through
`scripts/cpu-slot` (up to two slots), so it is not an isolated timing and is not a
criterion.

**Method.** Runner, host and analysis are sealed with the plan
(`plan.json`, SHA-256 `120b6713714b9d3d2d67ee76ae1aacc2294543eb8aa35b0f17bb526b349740c4`;
`run_rematch.py` `ceb49f97…`; `report.py` `f4b18d51…`). The faceoff records are assembled by the tournament code and appended to
the store (`hf:jasonyandell/texas-42-tournament`). Evidence goes to
`jasonyandell/texas-42-zeb-research-2026-10`, a new folder, with `SHA256SUMS`.

One smoke pair (seed `0x3039`, a throwaway outside this study) ran both arms
before sealing: it checks that the host runs, that 0.4.1's overcall fires
(7 of 8 and 6 of 10 opponent-held turns) while 0.4.0's seldom does, and that both
arms see identical deals. Those four games are not part of the study.

## Results

(Written after all 1,024 games finish.)

## Links

[[the-demon]] · [[demon-overcalling-2026-10-15]] · [[match-aware-bidding-2026-10-14]] · [[tournament]] ·
[[strategy-fusion-fence]] · [[book-authority]]
