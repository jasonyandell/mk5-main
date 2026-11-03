## [2026-11-02 | claude | Lem: a field net written by hand, graded]

**Touched pages:** [[lem-hand-written-field-net-2026-11-02]], [[clean-neurons-42]], [[index-experiments]]

`lem/handnet.py` writes Zeb's field net (field-v2) rule by rule: 150 × 124 boolean units, every weight
chosen, no gradient. Held-out regret 0.0143 [0.0141, 0.0145] (the zero-forward-pass arm 0.0170;
shipped 0.0084; random 0.0413). In the arena, hand-field Zeb scores −0.55 marks/game [−0.76, −0.34]
against shipped-field Zeb (384 games). The audit reads the all-rules net at 0.360 clean, below a random
conjunction net's 0.378, and calls an unrelated-OR decoy clean: the instrument's blind spots, measured.
Exploratory: zeroing the shipped net's void block changes 12.8% of its moves where a void is shown.
Page: [[lem-hand-written-field-net-2026-11-02]]; data on the Hub.
