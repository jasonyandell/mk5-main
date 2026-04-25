# §22 Real Drama Examples — Corrected Definition

**Drama definition**: `marginal_eq_gap <= 1.0` AND `count_unplayed >= 15` AND (mid-game or endgame phase).

Marginal E[Q] gap = top1 - top2 E[Q] over legal actions — when this is small, the player genuinely can't tell which action is best from oracle data alone.

---

## Mid-Game Drama (d_idx 10-18, dominoes_in_hand 4-6)

### Mid-Game Example 1
```
Seed 900010, d_idx=10, Player 3 (decl_id=0)
  marginal_eq_gap: 0.0000  |  marginal_pmake_gap: 0.0000
  count_unplayed: 35  |  dominoes_in_hand: 5
  Player 3's current hand: 6-5, 6-3, 6-2, 5-5[10pt], 5-0[5pt]
  Played so far (10 plays):
    Trick 1: P0:3-3, P1:5-1, P2:5-3, P3:3-1
    Trick 2: P0:1-1, P1:3-0, P2:6-1, P3:2-1
    Trick 3: P1:0-0, P2:1-0
  Legal actions with oracle stats:
    slot_2 (5-0[5pt]): E[Q]=+18.46, p_make=0.805, Q_std=22.11, world_argmax=100.00% <-- oracle MODE <-- GUS <-- PLAYED
  SUMMARY: P3 plays 3rd trick 3, holds 6-5, 6-3, 6-2, 5-5[10pt], 5-0[5pt] — E[Q] gap is 0.000 pts between top-2 options, 35 count pip value still unplayed
```

### Mid-Game Example 2
```
Seed 900004, d_idx=10, Player 2 (decl_id=4)
  marginal_eq_gap: 0.0345  |  marginal_pmake_gap: 0.0009
  count_unplayed: 35  |  dominoes_in_hand: 5
  Player 2's current hand: 6-2, 3-3, 3-2[5pt], 3-0, 2-0
  Played so far (10 plays):
    Trick 1: P0:4-4, P1:4-0, P2:4-2, P3:5-4
    Trick 2: P0:6-6, P1:6-1, P2:6-0, P3:6-5
    Trick 3: P0:1-1, P1:4-3
  Legal actions with oracle stats:
    slot_0 (2-0): E[Q]=-27.52, p_make=0.145, Q_std=21.43, world_argmax=15.69%
    slot_1 (3-0): E[Q]=-27.15, p_make=0.157, Q_std=21.83, world_argmax=14.12% <-- oracle MODE <-- PLAYED
    slot_2 (3-2[5pt]): E[Q]=-30.32, p_make=0.082, Q_std=17.86, world_argmax=24.63%
    slot_3 (3-3): E[Q]=-27.61, p_make=0.154, Q_std=21.32, world_argmax=19.84%
    slot_6 (6-2): E[Q]=-27.18, p_make=0.158, Q_std=21.73, world_argmax=25.72% <-- GUS
  SUMMARY: P2 plays 3rd trick 3, holds 6-2, 3-3, 3-2[5pt], 3-0, 2-0 — E[Q] gap is 0.035 pts between top-2 options, 35 count pip value still unplayed
```

### Mid-Game Example 3
```
Seed 900011, d_idx=10, Player 3 (decl_id=1)
  marginal_eq_gap: 0.3329  |  marginal_pmake_gap: 0.0056
  count_unplayed: 35  |  dominoes_in_hand: 5
  Player 3's current hand: 6-6, 6-5, 6-3, 6-2, 4-3
  Played so far (10 plays):
    Trick 1: P0:1-1, P1:1-0, P2:5-1, P3:6-1
    Trick 2: P0:2-2, P1:3-1, P2:4-2, P3:2-0
    Trick 3: P1:0-0, P2:6-0
  Legal actions with oracle stats:
    slot_1 (4-3): E[Q]=-5.00, p_make=0.389, Q_std=28.19, world_argmax=29.19%
    slot_3 (6-2): E[Q]=-4.66, p_make=0.395, Q_std=28.20, world_argmax=30.91% <-- oracle MODE <-- GUS <-- PLAYED
    slot_4 (6-3): E[Q]=-5.60, p_make=0.386, Q_std=28.03, world_argmax=7.41%
    slot_5 (6-5): E[Q]=-5.81, p_make=0.381, Q_std=27.21, world_argmax=24.78%
    slot_6 (6-6): E[Q]=-5.93, p_make=0.376, Q_std=27.94, world_argmax=7.72%
  SUMMARY: P3 plays 3rd trick 3, holds 6-6, 6-5, 6-3, 6-2, 4-3 — E[Q] gap is 0.333 pts between top-2 options, 35 count pip value still unplayed
```

### Mid-Game Example 4
```
Seed 900004, d_idx=11, Player 3 (decl_id=4)
  marginal_eq_gap: 0.5732  |  marginal_pmake_gap: 0.0038
  count_unplayed: 35  |  dominoes_in_hand: 5
  Player 3's current hand: 5-5[10pt], 5-2, 3-1, 2-1, 0-0
  Played so far (11 plays):
    Trick 1: P0:4-4, P1:4-0, P2:4-2, P3:5-4
    Trick 2: P0:6-6, P1:6-1, P2:6-0, P3:6-5
    Trick 3: P0:1-1, P1:4-3, P2:3-0
  Legal actions with oracle stats:
    slot_1 (2-1): E[Q]=+10.28, p_make=0.649, Q_std=26.51, world_argmax=49.59% <-- GUS
    slot_2 (3-1): E[Q]=+10.85, p_make=0.645, Q_std=25.95, world_argmax=50.41% <-- oracle MODE <-- PLAYED
  SUMMARY: P3 plays 4th trick 3, holds 5-5[10pt], 5-2, 3-1, 2-1, 0-0 — E[Q] gap is 0.573 pts between top-2 options, 35 count pip value still unplayed
```

### Mid-Game Example 5
```
Seed 900010, d_idx=11, Player 0 (decl_id=0)
  marginal_eq_gap: 0.0000  |  marginal_pmake_gap: 0.0000
  count_unplayed: 30  |  dominoes_in_hand: 5
  Player 0's current hand: 5-4, 5-2, 4-2, 4-1[5pt], 2-0
  Played so far (11 plays):
    Trick 1: P0:3-3, P1:5-1, P2:5-3, P3:3-1
    Trick 2: P0:1-1, P1:3-0, P2:6-1, P3:2-1
    Trick 3: P1:0-0, P2:1-0, P3:5-0[5pt]
  Legal actions with oracle stats:
    slot_1 (2-0): E[Q]=-31.81, p_make=0.039, Q_std=14.53, world_argmax=100.00% <-- oracle MODE <-- GUS <-- PLAYED
  SUMMARY: P0 plays 4th trick 3, holds 5-4, 5-2, 4-2, 4-1[5pt], 2-0 — E[Q] gap is 0.000 pts between top-2 options, 30 count pip value still unplayed
```

---

## Endgame Drama (dominoes_in_hand <= 3)

### Endgame Example 1
```
Seed 900014, d_idx=18, Player 3 (decl_id=4)
  marginal_eq_gap: 0.0185  |  marginal_pmake_gap: 0.0275
  count_unplayed: 30  |  dominoes_in_hand: 3
  Player 3's current hand: 6-5, 5-3, 0-0
  Played so far (18 plays):
    Trick 1: P0:4-4, P1:4-2, P2:4-0, P3:4-1[5pt]
    Trick 2: P0:3-3, P1:4-3, P2:3-1, P3:3-0
    Trick 3: P1:2-2, P2:2-0, P3:1-0, P0:5-2
    Trick 4: P1:6-6, P2:6-0, P3:6-1, P0:6-3
    Trick 5: P1:5-1, P2:1-1
  Legal actions with oracle stats:
    slot_4 (5-3): E[Q]=-17.24, p_make=0.216, Q_std=20.32, world_argmax=35.42% <-- oracle MODE <-- PLAYED
    slot_6 (6-5): E[Q]=-17.26, p_make=0.189, Q_std=19.35, world_argmax=64.58% <-- GUS
  SUMMARY: P3 plays 3rd trick 5, holds 6-5, 5-3, 0-0 — E[Q] gap is 0.019 pts between top-2 options, 30 count pip value still unplayed
```

### Endgame Example 2
```
Seed 900014, d_idx=16, Player 1 (decl_id=4)
  marginal_eq_gap: 0.1874  |  marginal_pmake_gap: 0.0362
  count_unplayed: 30  |  dominoes_in_hand: 3
  Player 1's current hand: 6-2, 5-1, 2-1
  Played so far (16 plays):
    Trick 1: P0:4-4, P1:4-2, P2:4-0, P3:4-1[5pt]
    Trick 2: P0:3-3, P1:4-3, P2:3-1, P3:3-0
    Trick 3: P1:2-2, P2:2-0, P3:1-0, P0:5-2
    Trick 4: P1:6-6, P2:6-0, P3:6-1, P0:6-3
  Legal actions with oracle stats:
    slot_0 (2-1): E[Q]=-10.08, p_make=0.362, Q_std=24.63, world_argmax=24.56% <-- oracle MODE
    slot_4 (5-1): E[Q]=-10.26, p_make=0.322, Q_std=23.78, world_argmax=48.00% <-- GUS <-- PLAYED
    slot_5 (6-2): E[Q]=-11.11, p_make=0.325, Q_std=24.37, world_argmax=27.44%
  SUMMARY: P1 leads trick 5, holds 6-2, 5-1, 2-1 — E[Q] gap is 0.187 pts between top-2 options, 30 count pip value still unplayed
```

### Endgame Example 3
```
Seed 900001, d_idx=16, Player 3 (decl_id=1)
  marginal_eq_gap: 0.0827  |  marginal_pmake_gap: 0.0000
  count_unplayed: 25  |  dominoes_in_hand: 3
  Player 3's current hand: 6-4[10pt], 5-5[10pt], 5-0[5pt]
  Played so far (16 plays):
    Trick 1: P0:1-1, P1:6-1, P2:4-1[5pt], P3:2-1
    Trick 2: P0:5-1, P1:6-0, P2:3-1, P3:2-0
    Trick 3: P0:1-0, P1:6-2, P2:5-3, P3:3-2[5pt]
    Trick 4: P0:6-5, P1:0-0, P2:6-3, P3:6-6
  Legal actions with oracle stats:
    slot_3 (5-0[5pt]): E[Q]=-6.66, p_make=0.387, Q_std=25.50, world_argmax=15.62%
    slot_4 (5-5[10pt]): E[Q]=+31.90, p_make=1.000, Q_std=5.94, world_argmax=52.28% <-- PLAYED
    slot_5 (6-4[10pt]): E[Q]=+31.98, p_make=1.000, Q_std=5.71, world_argmax=32.09% <-- oracle MODE <-- GUS
  SUMMARY: P3 leads trick 5, holds 6-4[10pt], 5-5[10pt], 5-0[5pt] — E[Q] gap is 0.083 pts between top-2 options, 25 count pip value still unplayed
```

### Endgame Example 4
```
Seed 900004, d_idx=18, Player 3 (decl_id=4)
  marginal_eq_gap: 0.0000  |  marginal_pmake_gap: 0.0000
  count_unplayed: 20  |  dominoes_in_hand: 3
  Player 3's current hand: 5-5[10pt], 5-2, 0-0
  Played so far (18 plays):
    Trick 1: P0:4-4, P1:4-0, P2:4-2, P3:5-4
    Trick 2: P0:6-6, P1:6-1, P2:6-0, P3:6-5
    Trick 3: P0:1-1, P1:4-3, P2:3-0, P3:3-1
    Trick 4: P1:6-4[10pt], P2:3-3, P3:2-1, P0:4-1[5pt]
    Trick 5: P1:2-2, P2:2-0
  Legal actions with oracle stats:
    slot_3 (5-2): E[Q]=+23.56, p_make=0.900, Q_std=12.12, world_argmax=100.00% <-- oracle MODE <-- GUS <-- PLAYED
  SUMMARY: P3 plays 3rd trick 5, holds 5-5[10pt], 5-2, 0-0 — E[Q] gap is 0.000 pts between top-2 options, 20 count pip value still unplayed
```

### Endgame Example 5
```
Seed 900013, d_idx=17, Player 2 (decl_id=3)
  marginal_eq_gap: 0.0000  |  marginal_pmake_gap: 0.0000
  count_unplayed: 20  |  dominoes_in_hand: 3
  Player 2's current hand: 6-5, 4-0, 0-0
  Played so far (17 plays):
    Trick 1: P0:5-5[10pt], P1:5-2, P2:5-0[5pt], P3:5-4
    Trick 2: P0:2-2, P1:6-2, P2:2-1, P3:4-2
    Trick 3: P0:3-0, P1:6-3, P2:3-3, P3:3-1
    Trick 4: P2:6-6, P3:6-1, P0:6-0, P1:4-3
    Trick 5: P1:4-4
  Legal actions with oracle stats:
    slot_3 (4-0): E[Q]=-13.80, p_make=0.196, Q_std=17.37, world_argmax=100.00% <-- oracle MODE <-- GUS <-- PLAYED
  SUMMARY: P2 plays 2nd trick 5, holds 6-5, 4-0, 0-0 — E[Q] gap is 0.000 pts between top-2 options, 20 count pip value still unplayed
```
