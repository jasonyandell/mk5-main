---
title: "Gus G0–G20 tournament registration and bounded bid-30 pilot"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-20
status: complete
---

## Outcome and publication boundary

The [[gus-tiny-gpt]] depth campaign now has **21 immutable tournament participant
records, G0 through G20**, plus a common-bidder Zeb reference. A bounded pilot
completed **15 duplicate faceoffs, 30 games**, with **zero timeouts or incomplete
games**. Every faceoff took under 60 seconds. It uses the existing [[tournament]]
`GitStore`, `FaceoffSpec`, replay records and Bradley–Terry fit, under the separate
`marks7-bid30` ladder.

The complete records and audit are public in the existing
[research archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/984edd66744d0c6917293fb821ebf55480db1bad/lem/gus-tournament-pilot-2026-10-18).
**Canonical tournament Hub synchronization is complete (2026-10-20, #216).** All 22
participant records and all 15 faceoffs are in `jasonyandell/texas-42-tournament` at
[`9d02d5a3`](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/9d02d5a376b00b29187a22d12b4bb2e441956ab5):
one commit added the 31 missing files. The 6 already there (G0–G4 and the Zeb anchor) were
byte-identical. The faceoffs sit in the per-claim-day layout,
`faceoffs/marks7-bid30/20261018/`. Every one of the 37 files was then downloaded anonymously
at that revision and matched its manifest hash, and the table built from exactly those
downloads reproduces the table below line for line. The batch's 27 `experiments/` files
were not added to the store: 26 are already in the research archive above, and the 27th is
the batch builder. Before that, the first attempt on 2026-10-18 returned HTTP 429 (repository
commit creation quota **128/hour**), and so did one retry at 14:39 UTC; only G0–G4's five
participant files had landed.
[Retry receipt and log](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/4d3d70ed1de11806127249d08d757453fb773104/lem/gus-splat-pretraining-2026-10-18/previous-tournament-retry.json)
record that failure.

## Contract, identity and runtime

The October observer explicitly rejects contracts other than bid 30. Both
partnerships therefore use the pinned arena `Bid30Bidder`: the seat left of the
shaker opens 30, the others pass, and the bidder declares its best pip trump
(0–6). Games finish at seven marks. This is play under a common bidder, not a
learned Gus auction. Ordinary-auction participants and restricted-contract
participants cannot silently enter each other's ladders.

Each team fields two copies of its policy. Gus receives only its original own
seven tiles and the public bid-30, relative bidder, trump and play prefix. It
uses CPU float32, one Torch thread, one forward pass and legal argmax; exact
logit ties choose the lowest tile id. PyTorch's runtime version is recorded in
each faceoff. The observer adapter, encoder, forward pass, declaration bidder
and supporting rule files are SHA256-pinned; source or cached-weight mismatch
fails closed.

The anchor is **`zeb-0.4.1-contested-bid30` = 1500**, using the standing native
Zeb 0.4.1 release's immutable source trees, JavaScript and native artifact
verification, with 160 worlds and seed 0. The common bidder replaces its
contested auction. This is a different pinned Zeb configuration from the
campaign's training opponent. No numerical comparison with the ordinary
`marks7` ratings is valid: these are separate likelihoods and reference scales.

All release tags were resolved to immutable Hub commits and exact `gus.pt`
hashes. IDs are `gus-gNNN-bid30-<checkpoint-sha256-prefix>`; the complete IDs,
source pins and configs are in
[participants.json](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/984edd66744d0c6917293fb821ebf55480db1bad/lem/gus-tournament-pilot-2026-10-18/participants.json).
The adapter source commit is `08c45eac`; game harness commit is `507637ba`.
The campaign-selected identity-grown G10 function was not entered a second
time. G17 here is the fully trained fifteen-layer generation, not G17's
identity initialization ([[gus-population-depth-2026-10-18]]).

## Pilot results

The coverage was fixed before games: G4, G8, G10, G12, G16, G17 and G20 each
against G0 and Zeb, plus G0 against Zeb. Each matchup is **one duplicate pair**.
G0–G20 were all registered locally; thirteen intermediate generations have no
games in this pilot. Their prior-only fitted numbers are not measured ratings.

Bradley–Terry MAP, the tournament's existing weak prior and cluster-robust 95%
intervals, anchored at the common-bidder Zeb. The chance baseline is 50% wins.
Small-cluster intervals can be too narrow; this is initial ladder connectivity
and integration evidence, not a generation ranking or promotion decision.

| Policy | Games | Wins | Elo | 95% interval |
|---|---:|---:|---:|---:|
| Zeb bid30 | 16 | 14 | 1500 | anchor |
| G4 | 4 | 2 | 1274 | [908, 1641] |
| G12 | 4 | 2 | 1274 | [943, 1605] |
| G16 | 4 | 2 | 1274 | [943, 1605] |
| G17 | 4 | 2 | 1274 | [943, 1605] |
| G8 | 4 | 1 | 1091 | [798, 1385] |
| G10 | 4 | 1 | 1091 | [798, 1385] |
| G20 | 4 | 1 | 1091 | [798, 1385] |
| G0 | 16 | 5 | 1088 | [802, 1375] |

The three 2–0 results against G0 and the mostly 0–2 results against Zeb are
only two games each. One duplicate pair cannot estimate between-pair
uncertainty. The G0–Zeb direct pair split 1–1; wider conclusions require more
independent pairs. No interpretability or native-neuron claim follows.

| Gus | Versus G0, wins | Outer seconds | Versus Zeb, wins | Outer seconds |
|---|---:|---:|---:|---:|
| G4 | 1–1 | 0.73 | 1–1 | 5.24 |
| G8 | 1–1 | 0.74 | 0–2 | 4.63 |
| G10 | 1–1 | 0.85 | 0–2 | 5.22 |
| G12 | 2–0 | 0.85 | 0–2 | 4.39 |
| G16 | 2–0 | 0.90 | 0–2 | 7.12 |
| G17 | 2–0 | 1.99 | 0–2 | 38.32 |
| G20 | 1–1 | 5.57 | 0–2 | 7.31 |
| G0 | — | — | 1–1 | 8.90 |

Outer wall time includes CPU-slot waiting and process startup: **92.75 seconds
for all fifteen invocations**. Every invocation used `scripts/cpu-slot
--threads 2 --timeout 60`, plus an outer 60-second process-group kill that also
covers descendants. No player used time as its thinking budget. Model download
and artifact setup preceded the caps; publication followed them. The slot cost
receipts report **53.20 CPU seconds** and **154.65 billion supervisor-process
instructions**; those instruction counts exclude native child-seat instructions,
which are recorded separately in the games. Wall/CPU times are descriptive,
not strength or efficiency comparisons across policies.

## Replay and verification

The independent audit replayed **326 hands / 177 unique physical deals**:
every play legal, every hand's points reconstructed, every completed game's
winner consistent. All deals were checked against **582,688** original source,
reward-training and reserved-panel physical deals, with **zero overlap**.
[Audit and fitted rows](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/984edd66744d0c6917293fb821ebf55480db1bad/lem/gus-tournament-pilot-2026-10-18/audit.json);
[exact seeds, durations and completion states](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/984edd66744d0c6917293fb821ebf55480db1bad/lem/gus-tournament-pilot-2026-10-18/attempts.json).

The six new tournament tests cover the contract/ladder boundary, source and
cached-weight corruption, original-adapter action parity, hidden-hand
invariance, exact repeat and duplicate-game parity, resumption filters, and
optional Torch runtime metadata. The complete gate was GREEN: 112 tournament
and 59 Lem tests plus the applicable TypeScript, Rust, Python, wiki and license
checks. CUDA forge was unavailable; the superseded Burl suite was not run.
The missing `safetensors` clean-environment dependency was added.

## Completing canonical publication

The preserved batch retains one immutable file per participant and faceoff,
with canonical bytes and SHA256 hashes. Its uploader checks the server's
not-before time, refuses any changed local bytes or conflicting remote bytes,
adds only absent paths in **one batch commit**, then freshly downloads every
file anonymously and checks its hash. It edits no existing record or shared
index. This pilot's batch publication differs from the standing worker's usual
one-file-per-commit operation; the rating input format is unchanged. Retrying
that batch after an unknown outcome is idempotent. No new games are needed. The
[portable upload package, manifest and runner](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/60eb95dc32cce574e550b8a20de568bf0a7d46f7/lem/gus-tournament-pilot-2026-10-18)
contain 64 files / 859,603 bytes; `upload-package.tar.gz` is 90,001 bytes,
SHA256 `0e650c9cd5e84eddd7fac907594584d1f780203b7aa92fcda63471f528a34a6f`.
Fresh anonymous verification matched every original evidence file and the
supplemental upload package. Canonical synchronization completed under #216 (above).

The retained local store already contains all 22 participants and 15 faceoffs.
The existing command reproduces the table immediately from that store:

```bash
python -u -m tournament --store git:scratch/gus-pilot/records table --ladder marks7-bid30
```

That path belongs to the retained `gus-tournament-pilot` worktree. The printed
table includes prior-only rows with zero games; those rows are not strength
measurements.
