---
title: "Gus contingent decisions — mask all-action-tie teaching loss"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Jason selected this comparison after [[gus-contingent-practice-recovery-2026-10-20]]: positions with no teacher-valued preference dominate the population but need not dominate learning. The prize remains discovering better contingent decisions with the continuation machinery, then teaching Gus those decisions.

For a node with exact legal-action values Q, ordinary teacher-action cross entropy requires one chosen optimal action. The new arm multiplies that row loss by `1[max Q > min Q]`, comparing rational values exactly. All-action-tie rows contribute zero; every nonflat row keeps its original target and CE. The full minibatch denominator, root-balanced sampler, optimizer schedule and update count are unchanged. No resampling, retained-row renormalization or additional actor input is introduced. An all-flat batch still performs its scheduled AdamW step; momentum and decay may move weights despite zero actor gradient.

This isolates loss masking from the earlier cost-sensitive comparison, which also changed nonflat gradient shape and scale. It does not assume ties caused failed transfer. A teacher tie assumes optimal future continuation; a self-following student can encounter different later mistakes after different actions. Complete-policy success remains decisive.

## Matched training and selection

Reuse the same 64 TRAIN roots / 1356 rows and the verified ordinary-CE512 trajectory. Of those rows, 1151 are all-action ties; the other 205 retain ordinary CE. The new masked arm starts from the original parent with the same 512 root-balanced minibatches, seed42, batch32, AdamW learning rate1e-4, betas(.9,.95), matrix decay.1, vector decay0 and gradient norm cap1. Original clean own-hand/public-history tokens and legal support remain unchanged.

Both arms offer steps 0/128/256/384/512. Fresh32 development roots select by exact complete-policy success, earliest exact tie. Only then do64 untouched finite-test roots evaluate selected masked, selected historical CE and parent. The finite gate requires at least0.5 percentage points over BOTH parent and CE, with both paired95% lower bounds positive. Only a pass opens128 actual full hands under separate2pp-parent/1pp-control positive-lower gates. No confirmation reselection or automatic promotion. One training seed, reused control and narrow finite endgame support limit conclusions.

Preserve fixed G0 partner and pinned Zeb, uniform legal-consistent root prior carried through future observed field actions, and SUM indistinguishable worlds before one shared learner MAX. There is no search inside deployed Gus. Teacher gain, TRAIN acquisition, unseen finite transfer and actual full-hand strength remain separate claims.

## Registration and execution

All 617035 prior reserved deals are excluded. The new frozen pools reserve 3712 additional potential deals (1536 development, 2048 test, 128 conditional full hands), so all 620747 become future training exclusions, including rejected or unplayed candidates. Stage caps are 24 collection / 2 training / 4 development / 4 test / 8 conditional full-hand segments, each 45 seconds cooperative and 90 seconds outer. Failures, unclosed attempts and exhausted caps stop for inspection.

Nine synthetic checks passed, including exact mask/gradient tests, nonzero ties, tiny rational differences, full-batch scaling, all-flat AdamW progression and serialized 512-step uninterrupted/resumed parity. A pre-registration Path-boundary error was preserved and repaired before the final tests; no scientific attempt failed. Metadata-only checks verified original parent tensors, exact TRAIN provenance and all 512 historical minibatches. No old TRAIN roots were resimulated.

The capability refresh found unchanged Lem #207, pending Book #121 and open Silas #166. The new Lem field-net unit-paper result at `6e0aa141` concerns another architecture and supplies no new requirement for this comparison. Existing information-cell teacher and evaluation tools suffice. No field, actor input or interpreter scope changes follow.

[Public registration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/57d5dcc53094e4024cca1e0fa414ff740ba4935a/lem/gus-contingent-flatmask-2026-10-20/registration) `57d5dcc53094e4024cca1e0fa414ff740ba4935a` freshly verified 11 files / 54 members before launch. Registration SHA `81f77f834e7c81ae4117874b2b8573e86dbcd6e14ee198b092b8be036739015e`. Parent and independent source reviews cleared. Controller launched2026-10-20T20:31:51Z after the pinned runtime check; every short segment repeats it. Current receipts: `scratch/lem/gus-contingent-flatmask-v1/CURRENT.md`. No measured outcome is claimed at launch.

## Result

All512 masked updates and the registered test completed. Fresh32 development selected masked128 and historical ordinary-CE384. On64 untouched finite roots, complete-policy success was masked42.3121%, CE40.3714%, parent39.5703%. Masked minus parent was +2.7418 percentage points, paired95% [-1.5812,+8.2025]; masked minus CE was +1.9407pp [-1.8630,+6.9416]. Both means exceed0.5pp, but neither lower bound is positive. **The finite improvement gate failed.** No full-hand test, reselection or promotion.

Against each comparator, masked improves5 roots, worsens7 and has equal values on52. The favorable mean is magnitude-driven; equal values do not imply identical actions. Against parent, two roots each gain1 in conditional success, while one loses23/45. These post-readout examples describe concentration, not a selected target set or a causal account of masking. Teacher flatness assumes optimal future play; students can make different later mistakes. One seed, reused control and narrow finite-endgame scope limit inference. These are exact remaining-hand values under the declared model, not full-game win rates.

## Audit and delivery

Twelve scientific segments closed: collection9, training1, development1, test1, fullhand0. No scientific failure or budget growth. Independent saved-data audit passed1,041,834 checks over160 selected roots,2071 dataset rows and102352 final snapshot entries. Original/historical initialization tensors, exact mask,512 shared batches, reuse ancestry, exclusions, selection, value sums, paired bootstrap and closed-output identities passed. This is not independent native forward, gradient replay or transition replay.

Source review corrected audit-only ancestry/count bookkeeping before its first execution. A post-registration explanatory-note edit was restored byte-for-byte from the registered archive; the newer implementation notes and restoration receipt remain preserved. Scientific sources and criteria did not change. Pre-registration preparation failures remain in the original registration archive.

[New evidence and independent reviews](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1630bd6b59ba6933d720b9c6fb1b8233b9fa7c8f/lem/gus-contingent-flatmask-2026-10-20/evidence) `1630bd6b59ba6933d720b9c6fb1b8233b9fa7c8f` verified11 files/76716 members after fresh anonymous download. Unchanged TRAIN bytes and historical CE assets are referenced by immutable public anchors rather than uploaded again. All620747 reserved physical deals remain future training exclusions, including rejected and unplayed candidates. The same owner-private [growing report](https://gus-first-look.psychometrix.chatgpt.site/#flatmask) presents the comparison and limits.

The collaborator proposes a separately registered TRAIN-only occupancy/value diagnosis of fixed masked/CE policies: inspect acquisition of known consequential corrections and errors on paths actually visited. This has not run and cannot replace fresh transfer confirmation. No next experiment is selected by this result.
