<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Doing:** grading #197, list v2 against v1 on the 21 drop-out nets and their 21 untrained twins.

**Where the skill failed:** the tools rung gives a single floor, "0.15–0.28 clean". That number
belongs to list v1. Under v2 (70 concepts) the same untrained nets score 0.28–0.39, while the
trained nets barely move (+0.00 to +0.04). Quoting a v2 share against the skill's floor would
have overstated the margin by about 0.15.

**What I did instead:** re-audited the floor nets with the same list as the trained nets, and
quoted each share against its own list's floor.

**Change:** rung 1 should say "the untrained floor *for the same list*, audited alongside", and
give both numbers (v1 0.11–0.28; v2 0.28–0.39). The kit's runner pattern (audit `floor/` beside
`nets/`) belongs in the kit table.
