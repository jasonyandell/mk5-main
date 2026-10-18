<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## 2026-10-18 — Gus frozen games and native representation benchmark

[[gus-frozen-benchmark-2026-10-18]] freezes 18 duplicate pairs, 36 games and
11,116 clean actor decisions, with 471 exact finite-model endgames and all 234
potential scheduled deals reserved from future training. Four checkpoint
captures preserve logits, residuals, MLP activations and attention; scoring takes
2.9–4.4 seconds on one CPU thread. G0 reproduces every recorded G0 decision and
both games of its repeated anchor exactly. Runtime/asset hashes prevent silent
H2H protocol changes. Static regression, conditional exact regret and actual
game outcomes retain separate scopes.

The public CC0 HF study is pinned at `df8277f60d2c95ce16dfca9473b8eed58a696c40`:
119 files, 410,609,771 bytes, all freshly downloaded anonymously and hash
verified. `gus-training` now carries the fixed-panel workflow, exclusions,
matched policy-head comparisons and the `clean-neuron` representation handoff.
The affected-suite gate passes all 102 Lem tests plus wiki, license and conflict
checks. Implementation: `cbc47a83`; results and immutable links live on the
experiment page.
