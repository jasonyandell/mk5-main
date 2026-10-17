<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Doing:** #198, the first causal test on the 24-neuron nets.

**Where the skill failed:** the card says to test a name by steering the unit, but neither the
card nor the skill says how much to steer. I clamped each unit at its 99th-percentile value. That
changed 46% of all moves and raised regret about 0.017 everywhere, which breaks the net rather
than steering it. The steering prediction failed for a reason that had nothing to do with the
unit.

**What I did instead:** reported the failure as rung 1 (tools) and kept #198 open for a
dose-response clamp.

**Change:**
- Add to the kit table: `causal` (ablate a `+` pole, clamp a `−` pole), compared among the rows
  where the unit fires, so the audit's own selection isn't circular.
- Add to the ladder's tools rung: steer by small, matched increments, and report a
  dose-response curve, never a single extreme clamp.
