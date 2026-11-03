---
title: "Zeb weight-space audit: untrained bid inputs, one unpromoted edit"
kind: experiment
first_seen: 2026-11-02
last_updated: 2026-11-02
status: complete
---

## Question and scope

Can direct inspection and editing of the neural weights inside [[scenario-player]]
find a failure and repair it without training? This exploratory audit follows
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
It uses architecture, training metadata and code as well as weights; it is not a
blinded weights-only test, a clean-neuron study, or a full-player strength trial.

The installed Plunge package reports **Zeb 0.4.1**, net SHA-256
`6fa3723dcae04a2f30839efb091969830c8cc65a7668c3526bd47c9fde0275d5`.
The source blob `zeb/net/margin_d250k_h128x64.fnet` and original training artifact
match that identity. The integer `field-v2` MLP is **343 → 128 → 64 → 28**,
54,108 weights and biases. It models other seats inside Zeb's planner; it is not
the complete player.

## Finding and edit

[[lem-field-net-audit-2026-10-15]] already records that all five training pours
use bid 30. The input nevertheless has 13 bid-specific rows. Rows for bids 31–42
receive no data gradient (`x.T @ d` in `scenario/field_train.py`), though weight
decay and quantization act on them. The shipped rows correlate **0.9999555** with
their seed-0 random initialization. `scenario/core/src/planner.rs` passes the
actual contract to this field net, so these rows are reachable during play.

One candidate copies the trained bid-30 row to those twelve rows: **1,536 weight
edits**, no optimizer, architecture change, or other tensor change. Its SHA-256 is
`af3524f1a6b1c75d39ab0154e9a48cc1d0a146d72716ea019cf1f0707e998a5e`.
The candidate was fixed before its behavioral probes; there was no search over
candidates. Need/slack features retain their actual-contract values, including
their remaining extrapolation beyond the training distribution.

## Measurements

- All **434,535** held-out bid-30 feature rows retain all 28 scores bit-for-bit.
  More strongly, every bid-30 feature vector is invariant because the edited
  rows are inactive. This is a preservation theorem, not an improvement result.
- A fixed random subset of **32,768** held-out positions is replayed at each
  contract 31–42, recomputing need/slack. The edit changes **38,275 / 393,216
  (9.73%)** field decisions. Each bid changes 9.03–10.74% of its probes. These
  are counterfactual contracts on bid-30 trajectories, not on-policy games.
- In **154** position-contract probes where the fourth play necessarily settles
  the contract and legal actions include both a win and a loss, the original
  makes **2** mistakes and the edit makes **2**: one fixed, one introduced.
  Counts are descriptive; rows and contracts are dependent, not independent
  tournament trials. No net tactical gain is established.
- Python integer inference and the Rust core reproduce both changed witnesses
  exactly. The full 160-world planner chooses the winning action on both, under
  both networks. The installed WASM package independently confirms both original
  full-player decisions. Field-policy errors must not be reported as live-player
  errors.

The repaired witness bids 31 on twos: playing 2-2 banks 33–8 and makes; 6-6
banks 27–14 and is set. The regression defends a bid of 40 on threes: playing
6-3 banks 35–6 and sets; 6-6 banks 41–0 and lets the bidder make. Only the
actor's hand and public record are needed to settle these outcomes; no hidden
hand or clairvoyant continuation is read.

**Disposition:** the weight edit is real and its causal effect is measured;
playing-strength improvement is not established. The candidate is archived,
not promoted. Plunge and the shipped Zeb net remain unchanged. Whether repairing
contract-conditioning improves full-player outcomes remains open.

## Evidence

Run at mk5-main `7fb5ef06b10392efe2bc3ef518c50008c8842f4b`. The held-out sample
SHA-256 is `2dfebe0e4eb69d60f769399112873e9c3e5321cfe14a3a20b0f145eceb783cb9`.
The existing October research archive holds this November follow-up to its
field-net audit: [code, candidate, counts, public-history witnesses and exact
replays](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/576492b5a4291671609c9c08e858fe967c5a99b5/weight-space/2026-11-02).
The nine remote filenames match the staged set; fresh downloads of the candidate
and results match their local bytes. The artifact README gives reproduction
commands and scope limits.
