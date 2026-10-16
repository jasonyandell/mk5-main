---
title: "Demon exact root reuse: the same values with fewer retired instructions"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
---

**STRATEGY-FUSION FENCE.** [[the-demon]] remains a fenced measuring stick.
This is an exact-policy cost replay after [[demon-ascent-2026-10-16]], not a
new strength trial or an instruction-primary comparison against Zeb.
Work: [#186](https://github.com/jasonyandell/mk5-main/issues/186); standard
roster heads-up: [#187](https://github.com/jasonyandell/mk5-main/issues/187).

## Result and scope

On Apple M5 Max, the first 64 duplicate pairs (128 games) of the prior
confirmation supply **18,125 native requests**: 16,772 plays, 1,207 support
forecasts, 18 best-declaration forecasts and 128 hellos. Every semantic
response is identical to the saved reference: tiles, per-move make counts
and margins, sample sizes, mass, exactness, proposals and bidding certificates.
Only search-node counts and elapsed time may change for root reuse. No game,
request or failed run was dropped; all 128 replays completed successfully.

| Implementation | Native instructions / original, 95% CI | Nodes / original |
|---|---:|---:|
| Cached immutable declaration tables | 0.998676 [0.998617, 0.998733] | 1.000000 |
| Cache plus equivalent root values | **0.966388 [0.960189, 0.971834]** | 0.968482 |

The selected implementation saves **3.36% [2.82%, 3.98%]** of native
instructions. It also beats cache alone: ratio **0.967669
[0.961450, 0.973148]**. Original and selected totals are
2,304,984,979,370 and 2,227,508,975,551 instructions.

The meter reads the complete native process lifetime, including startup and
shutdown, while the exited process is still a zombie (`arena.cost.wait`). Each
game starts a fresh native process and replays all candidate requests in their
original order. Python replay-driver work and Zeb are outside this native
scope. Every replay runs through `scripts/cpu-slot --threads 1`; no exclusive
reservation or quiet-machine wait is used. CPU and wall are descriptive:
native CPU 161.080 → 154.805 s, ratio 0.961044 [0.953718, 0.968333]; shared wall
172.682 → 166.156 s, ratio 0.962213 [0.945332, 0.979702].

The public protocol fixes the input prefix before replay, rotates the three
arm orders by game index, requires exact reference agreement and an instruction
ratio upper bound below one, and reports 40,000 whole-pair bootstrap resamples
(seed 666, 95% percentile intervals). All three arms remain public. The
prior win/CPU confirmation retains its original criterion; these figures do
not relabel it as an instruction-primary confirmation.

## Why the values stay exact

[[d48909c4]] adds two forms of reuse within the Demon only:

- One immutable solver template per declaration, initialized on first use.
  Each world still receives a fresh transposition table, ownership scope and
  node counter. No hidden-world bounds cross worlds or requests.
- One evaluation per class of interchangeable root tiles. Two tiles in the
  mover's hand may share a value only when their count, led context, follow
  membership and rank comparisons against every outside tile agree. The
  outside set includes the open trick and all hidden tiles. It does not
  depend on a sampled ownership assignment.

Swapping the two tile names gives a bijection of complete legal continuations
with identical trick winners and points. The two tiles cannot occur in one
trick, since they belong to the same seat. Their per-world make and margin
values therefore agree under minimax; weighted sums and the original lowest-id
tie rule agree too. Values are expanded back to every legal tile in the reply.

Independent unpruned minimax checks 10,000 two-trick positions across all ten
physics and all four open-trick prefix lengths; every prefix exercises a root
swap. Boundary tests retain distinctions caused by count, follow membership,
led suit or an intervening outside trump. Existing full-fiber tests also check
make counts and margins against independent minimax, including exact weighted
posteriors. The full gate passes 21 native tests, 106 tournament tests and eight
fence tests, plus licenses and conflict guards.

## Source and immutable participant

The new record is
`demon-pimc-n100-bid9dee8eeb-straight-bounds-support-bait40-d48909c4`.
Its configuration equals the independently confirmed early-40 pin: 100 play
worlds, exact margin ties, `pimc-bait:64:0.875:0.875`, bounded and support
forecasts, early point bid 40, and declarations 0–7 and 9. Opponent-specific
strength evidence remains in [[demon-ascent-2026-10-16]].

Source `d48909c485dc647ac6bada3577000b5602b95f64`, trees:

- Silas `fad0fc47d19b57c796d182657fb9cf404984a438` (unchanged).
- scenario/core `84714a99f9ff1d1ba532f9983e74f73c83cafe8e` (unchanged).
- tournament/demon `26dd409acd3e71ac4bcab367fcd46906ffdea4f3`.

The measured checkout binary has SHA-256
`f0d4ba6ed49738b112585ce60b94faa1a2054bf343b3f02249190dbd5dcbf4e6`.
A separate build from the archived public trees has SHA-256
`693f2b4106c4a883454ddc59afe826ab201d810feff7335ee5fc08758634386b`
and independently matches all 181 responses in the first saved game. The
receipt distinguishes these two binaries; it does not assert binary identity.

The new canonical participant has SHA-256
`4a77ad47425c948762b5788073ddd68c045ac90f98bb3b4497183bb32f223eb6`.
It is appended to `ROSTER`; all 14 prior records remain byte-identical. The
public registry append at
[8617f41e](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/8617f41e3ae4d1d308c31215fc560a83d2f93a06/participants)
preserves every earlier participant blob, and a fresh download matches the
canonical bytes.

## Public evidence

All artifacts are CC0-1.0 in the existing
[monthly research dataset](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

- [Frozen protocol, inputs and scripts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/588755009818d480cdfc46ba636ac7b5a84a05f6/demon/2026-10-16-instructions/protocol).
  Plan SHA-256 `ccfe61d247ff633a8a87fa25c632f8e973ce28e197c95e9af500d0d602d8ca58`.
- [All replay responses, costs, logs, report and source-build verification](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/78f9586b9372ce0a4979719877ffe80b56478e82/demon/2026-10-16-instructions/results).
  Fresh plan, manifest, first raw input/result and report downloads match
  local bytes. Compiler and all three measured binary hashes are in the plan.

- [Registry integration and quality-gate receipts](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b97653381ab9c59b3fce722afdfc2cb2e34db5cc/demon/2026-10-16-instructions/integration).
  Fresh manifest, registry receipt and canonical participant downloads match.

Binaries, build caches, repository copies and credentials are excluded from
publication. Strength generalization and a complete instruction-primary
Demon-versus-Zeb trial remain separate questions.
