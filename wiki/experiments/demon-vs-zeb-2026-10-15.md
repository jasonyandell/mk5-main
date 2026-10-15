---
title: "Demon versus Zeb at comparable total runtime"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

**STRATEGY-FUSION FENCE.** The [[the-demon]] remains a sanctioned tournament
opponent. The objective is a measured win against current pinned Zeb at
comparable total seat runtime, on complete games of straight 42.

## Declaration audit

The initial development run exposed a domain error in the PIMC bidder:
it could declare 8, doubles as their own suit, the physics used for nello.
The general arena accepted that physics under a points contract. Shipped
Zeb correctly refuses it in straight 42. The straight declarations are
`[0,1,2,3,4,5,6,7,9]`; nello is outside this tournament's rules.

All four initial development arms were stopped and invalidated. Completed
and partial records remain public; none enter selection. The corrected
arms restart on the same development seeds and retain their fixed sizes.
No confirmation had started. The new participant factory records the allowed
declarations explicitly and uses a new identity. Workers exclude nonstraight
PIMC records from scheduling and ratings, and preserve incompatible pending
claims under `unsupported/` without continuing them. Historical participant
records and raw game bytes remain unchanged. Source: [[1273bc97]].

The earlier [[demon-next-form-2026-10-14]] bidder confirmation used declaration
8 in **87 of 5,620 hands**. Its 324–188 result describes the ten-declaration
experimental arena; it is **not a valid straight-42 strength benchmark**.
Removing those hands or games would not repair the comparison, because
declaration eligibility also affects bids that do not win the auction.

[Public audit and invalid trial records](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/25320e15068b745a13959542aeea8167ada602ef/demon/2026-10-15-vs-zeb/declaration-audit)
include the declaration census, complete and partial initial games, original
and corrected protocols, adapter checks and source provenance. The census, one
raw game and the manifest were freshly downloaded and hash-verified.

## Frozen opponents and success criterion

The npm registry's latest release was 0.4.0 when checked. Its integrity is
`sha512-7DhOCStdZbGEkh+RBwdDWOYgptwS+OYEPoGLF3BVJwCs/A0rmMNdq+X3fBrgQ2C55WWRXb3dNVmFY9jucFDKDA==`.
Both opponents use the shipped play defaults at 160 worlds, seed 0, with
reading and exact margin ties enabled:

- Primary: opt-in match-aware bidding, pip declarations only as measured in
  [[match-aware-bidding-2026-10-14]], cutoff 0.8.
- Secondary: the standard [[tournament]] seat, hand-only bidding behind
  partner pass, nine default forecast declarations, cutoff 0.8.

The research host's standard mode is held to the existing tournament host;
its match-aware choices are held to direct calls to the pinned package.
The 27-case check covers seat orientation, bid holder, and declarations.
Only own tiles and public history/auction/scores enter either seat.

Calibration and two budget-development arms run 16 duplicate pairs each.
The budget arms separately raise play from 100 to 800 worlds or bidding
from 16 to 64 common worlds; the bid cutoff stays 0.875. Selection follows
completed paired marks and runtime measurements, never an interim win count.

The selected configuration is frozen for 256 fresh duplicate pairs per
opponent, with no early stopping or dropped games. Success requires a win
fraction interval wholly above the 50% chance baseline and runtime-ratio
upper bounds at most **1.20** for both total seat wall time and CPU.
Faster is acceptable. Intervals resample whole pairs 40,000 times; marks/game
is secondary. If confirmation is repeated after further development, attempt
`r` uses two-sided alpha `0.05/(r*(r+1))`; every attempt is retained.

Total seat time covers initialization, bidding, declaration, play, close,
IPC and Python adapter work. CPU is child-process user+system time plus the
in-process seat calls. Downloads/builds and the shared referee are excluded.
Startup and per-operation latency distributions are reported separately.
Each CPU-slot batch runs one serial duplicate pair on the Apple M5 Max;
processor-specific tuning must be labeled and preserve exact search values.

The protocol is recorded before confirmation in
[issue #165](https://github.com/jasonyandell/mk5-main/issues/165).
Corrected development and profiling are in progress. No win against Zeb or
runtime-qualified promotion is claimed yet.

## Links

[[the-demon]] · [[tournament]] · [[scenario-player]] ·
[[demon-next-form-2026-10-14]] · [[strategy-fusion-fence]]
