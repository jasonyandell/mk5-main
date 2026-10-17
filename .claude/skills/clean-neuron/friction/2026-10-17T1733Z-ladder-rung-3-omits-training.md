<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Doing:** grading #196. Two of three 4×2 nets were clean at 0.125 and 0.333, below the floor.

**Where the skill failed:** rung 3 covers data and inputs only. The essay's rung is "bad learning
(data, architecture)", and the cause here was training: both nets closed only 0.2 of the gap, no
more than two neurons do. The skill's ladder had no rung for "this net did not learn the play".

**What I did instead:** checked the gap closed beside the clean share. The low shares sat
exactly on the nets that failed to learn.

**Change:** add training to rung 3 ("bad learning: data, inputs, training, architecture").
Report a unit's net's gap closed beside its share, so a net that failed to learn is visible
before its units are read.
