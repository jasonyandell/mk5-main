<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Doing:** the first all-neuron audit of Gus, a transformer, against list v2.

**Where the skill failed:** rung 1 gives the floor per list (v1 0.11–0.28, v2 0.28–0.39), but those
are field-net numbers. Untrained Gus scored 0.91–0.94: its decision token reads the position, and
the play count fixes the seat in the trick. I registered a "floor + 0.20" prediction from the
field-net habit, and it broke on the instrument.

**What I did instead:** reported it as rung 1 and ran an exploratory arm with the play-count-fixed
columns blanked. The floor stayed at 0.43–0.47, because position leaks through correlated bits.
Then I filed a position-conditioned audit.

**Change:** the floor belongs to the list *and the architecture*. Before registering a share
prediction on a new architecture, audit its untrained twins first. For any sequence model, read
units conditioned on position (z-score within the play count, or take the top 1% per stratum).
