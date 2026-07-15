# Otis W5 — 3-2 fate ledger cards

> Regenerated under strict valid-world filtering (issue #52): clusters are built from ONLY sampler-consistent worlds (full 28-domino deals; ≥40 valid worlds required per decision). Supersedes the pre-filter cards. Each card's `n_valid` is the surviving world count.

The 10 highest-mass bimodal P6 decisions (belief-weighted), from the world-bank instrument (`otis/analysis/worldbank.py`). Belief posterior: `gus/adapters/v3_consistency_10000g.pt`. Each card shows the public context, the context clusters of sampled worlds, and why the 3-2's fate value (oracle Q at a* = argmax E[Q]) differs across contexts.

## Card 1 — corpus_v2_train_10-19_d0-9:g63:s16:d3

- Declaration id: 3 · trick 0 (decision idx 2) · actor seat 2
- Worlds: 65 valid of 200 sampled (32%) · ESS=49.0 · a*=slot 5 · action taken=slot 5
- Bimodal split: low mass 79% / high mass 21% · gap 37.8 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=right opp | 6.1% | 4.6% | -32.8 | -42.0 | -11.1 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp | 5.5% | 3.1% | -31.4 | -37.5 | -15.6 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=partner | 3.9% | 3.1% | -27.7 | -41.4 | 13.2 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=right opp | 8.2% | 7.7% | -26.5 | -42.0 | 6.5 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=partner | 5.5% | 7.7% | -23.7 | -42.0 | 6.1 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=right opp | 9.4% | 9.2% | -19.0 | -42.0 | 35.7 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner | 4.4% | 6.2% | -9.6 | -41.6 | 24.5 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=left opp | 3.3% | 3.1% | -7.8 | -27.6 | 7.2 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=right opp | 5.1% | 3.1% | -6.7 | -31.7 | 17.6 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=partner | 2.4% | 1.5% | -5.7 | -5.7 | -5.7 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=left opp | 8.7% | 4.6% | -4.5 | -27.3 | 29.9 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=left opp | 3.5% | 4.6% | -2.1 | -41.9 | 40.4 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=right opp | 2.1% | 3.1% | 1.4 | -1.7 | 13.7 |
| other (pooled minor contexts) | 7.8% | 12.3% | 5.4 | -18.7 | 41.1 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 3.3% | 4.6% | 13.8 | -1.8 | 41.2 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=left opp | 9.4% | 7.7% | 15.9 | 4.0 | 40.4 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp | 2.1% | 1.5% | 22.2 | 22.2 | 22.2 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=partner | 4.9% | 7.7% | 32.8 | 20.4 | 40.0 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=right opp | 4.1% | 4.6% | 33.8 | 28.8 | 39.8 |

> Story: the 3-2 line is worth ~-33 pts when [5-5=your hand, 6-4=partner, 5-0=partner, 4-1=right opp] but ~34 pts when [5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=right opp] — a 67-point swing the belief-averaged marginal cannot see.

## Card 2 — corpus_v2_train_0-9_d0-9:g82:s8:d2

- Declaration id: 2 · trick 0 (decision idx 0) · actor seat 0
- Worlds: 41 valid of 200 sampled (20%) · ESS=37.3 · a*=slot 5 · action taken=slot 5
- Bimodal split: low mass 78% / high mass 22% · gap 33.7 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=partner | 5.7% | 4.9% | -15.3 | -23.6 | -5.0 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=right opp | 6.5% | 4.9% | -9.9 | -38.4 | 11.2 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=left opp | 4.0% | 4.9% | -3.5 | -19.0 | 31.0 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=right opp | 2.9% | 2.4% | -0.5 | -0.5 | -0.5 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=right opp | 13.1% | 9.8% | 1.5 | -35.5 | 39.6 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=partner | 3.2% | 4.9% | 3.0 | 1.8 | 4.9 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=right opp | 6.4% | 7.3% | 3.1 | -33.2 | 29.0 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner | 4.3% | 2.4% | 3.2 | 3.2 | 3.2 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=partner | 4.7% | 4.9% | 3.7 | -10.3 | 29.9 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=left opp | 2.2% | 2.4% | 5.7 | 5.7 | 5.7 |
| other (pooled minor contexts) | 9.8% | 14.6% | 7.3 | -15.0 | 33.2 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=partner | 5.5% | 4.9% | 11.2 | -1.6 | 23.5 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=left opp | 4.4% | 4.9% | 12.5 | -13.2 | 26.7 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=right opp | 5.0% | 4.9% | 21.8 | 20.6 | 22.7 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 2.3% | 2.4% | 22.7 | 22.7 | 22.7 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=left opp | 5.4% | 7.3% | 36.3 | 34.8 | 38.3 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp | 3.0% | 2.4% | 38.0 | 38.0 | 38.0 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=left opp | 5.8% | 4.9% | 38.5 | 38.4 | 38.8 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp | 5.6% | 4.9% | 39.6 | 38.8 | 40.3 |

> Story: the 3-2 line is worth ~-15 pts when [5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=partner] but ~40 pts when [5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp] — a 55-point swing the belief-averaged marginal cannot see.

## Card 3 — corpus_v2_train_20-29_d0-9:g46:s24:d6

- Declaration id: 6 · trick 0 (decision idx 1) · actor seat 1
- Worlds: 92 valid of 200 sampled (46%) · ESS=33.9 · a*=slot 1 · action taken=slot 1
- Bimodal split: low mass 32% / high mass 68% · gap 30.9 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=partner, 6-4=partner, 5-0=left opp, 4-1=your hand | 2.6% | 2.2% | -41.8 | -41.8 | -41.7 |
| 5-5=left opp, 6-4=left opp, 5-0=partner, 4-1=your hand | 6.3% | 5.4% | -41.2 | -42.0 | 3.5 |
| 5-5=left opp, 6-4=right opp, 5-0=left opp, 4-1=your hand | 3.0% | 2.2% | -34.2 | -41.3 | -30.1 |
| 5-5=right opp, 6-4=left opp, 5-0=partner, 4-1=your hand | 3.4% | 7.6% | -32.7 | -42.0 | -16.4 |
| 5-5=partner, 6-4=right opp, 5-0=partner, 4-1=your hand | 3.0% | 3.3% | -32.2 | -37.0 | -26.4 |
| 5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=your hand | 9.8% | 9.8% | -31.2 | -42.0 | 33.6 |
| 5-5=left opp, 6-4=right opp, 5-0=partner, 4-1=your hand | 3.7% | 4.3% | -29.6 | -41.9 | -19.7 |
| 5-5=partner, 6-4=left opp, 5-0=right opp, 4-1=your hand | 6.4% | 3.3% | -18.6 | -30.2 | 19.9 |
| 5-5=partner, 6-4=partner, 5-0=partner, 4-1=your hand | 3.5% | 5.4% | -12.4 | -42.0 | 41.5 |
| 5-5=left opp, 6-4=right opp, 5-0=right opp, 4-1=your hand | 6.9% | 4.3% | -11.7 | -41.9 | -1.6 |
| other (pooled minor contexts) | 9.6% | 28.3% | -10.0 | -42.0 | 41.2 |
| 5-5=left opp, 6-4=left opp, 5-0=left opp, 4-1=your hand | 9.3% | 3.3% | -5.0 | -42.0 | 28.3 |
| 5-5=right opp, 6-4=left opp, 5-0=right opp, 4-1=your hand | 4.8% | 2.2% | -3.9 | -25.9 | -3.2 |
| 5-5=left opp, 6-4=partner, 5-0=left opp, 4-1=your hand | 10.7% | 5.4% | -0.1 | -41.9 | 38.5 |
| 5-5=left opp, 6-4=partner, 5-0=partner, 4-1=your hand | 8.2% | 6.5% | 2.2 | -41.0 | 9.0 |
| 5-5=right opp, 6-4=partner, 5-0=right opp, 4-1=your hand | 4.0% | 5.4% | 6.4 | -41.5 | 41.4 |
| 5-5=right opp, 6-4=right opp, 5-0=right opp, 4-1=your hand | 4.8% | 1.1% | 25.2 | 25.2 | 25.2 |

> Story: the 3-2 line is worth ~-42 pts when [5-5=partner, 6-4=partner, 5-0=left opp, 4-1=your hand] but ~25 pts when [5-5=right opp, 6-4=right opp, 5-0=right opp, 4-1=your hand] — a 67-point swing the belief-averaged marginal cannot see.

## Card 4 — corpus_v2_train_10-19_d0-9:g16:s11:d6

- Declaration id: 6 · trick 0 (decision idx 1) · actor seat 1
- Worlds: 200 valid of 200 sampled (100%) · ESS=99.7 · a*=slot 6 · action taken=slot 6
- Bimodal split: low mass 79% / high mass 21% · gap 29.9 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner | 5.8% | 4.5% | -26.1 | -42.0 | 18.0 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=partner | 7.7% | 6.5% | -25.6 | -42.0 | 39.5 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=partner | 9.8% | 7.5% | -23.3 | -42.0 | 39.2 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp | 5.4% | 2.0% | -22.0 | -41.7 | 14.8 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp | 6.1% | 3.5% | -19.1 | -41.6 | 40.0 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 4.1% | 3.0% | -14.4 | -27.6 | 8.0 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=partner | 3.3% | 4.0% | -13.7 | -33.3 | 40.8 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=right opp | 6.0% | 3.5% | -11.7 | -42.0 | 41.5 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=left opp | 2.1% | 2.5% | -10.2 | -17.8 | 30.7 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=partner | 6.1% | 6.5% | -8.2 | -37.4 | 19.8 |
| other (pooled minor contexts) | 7.6% | 12.0% | -7.7 | -41.9 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=right opp | 3.4% | 5.0% | -1.5 | -41.9 | 41.4 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=left opp | 4.7% | 4.5% | -0.5 | -41.7 | 29.4 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=left opp | 2.8% | 4.0% | 0.4 | -41.6 | 15.4 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=left opp | 2.1% | 3.5% | 4.6 | -21.7 | 19.8 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=left opp | 2.0% | 4.0% | 5.9 | -36.5 | 38.0 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=left opp | 3.6% | 4.0% | 7.2 | -41.9 | 41.3 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=left opp | 3.1% | 6.0% | 9.9 | -41.6 | 41.1 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=right opp | 5.6% | 5.0% | 16.4 | -26.7 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=right opp | 5.6% | 4.5% | 16.4 | -42.0 | 41.5 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=right opp | 3.1% | 4.0% | 30.5 | 5.7 | 41.3 |

> Story: the 3-2 line is worth ~-26 pts when [5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner] but ~31 pts when [5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=right opp] — a 57-point swing the belief-averaged marginal cannot see.

## Card 5 — corpus_v2_train_10-19_d0-9:g62:s16:d2

- Declaration id: 2 · trick 0 (decision idx 2) · actor seat 2
- Worlds: 66 valid of 200 sampled (33%) · ESS=16.2 · a*=slot 5 · action taken=slot 5
- Bimodal split: low mass 20% / high mass 80% · gap 22.9 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp | 20.4% | 3.0% | 15.4 | 14.9 | 41.5 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=left opp | 2.7% | 3.0% | 29.0 | 28.7 | 29.7 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner | 7.5% | 4.5% | 34.2 | 19.4 | 41.4 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=partner | 6.2% | 6.1% | 34.7 | 27.0 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=left opp | 7.3% | 10.6% | 35.6 | 30.8 | 41.6 |
| other (pooled minor contexts) | 10.3% | 27.3% | 37.3 | 19.8 | 41.6 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=left opp | 6.9% | 9.1% | 38.7 | 20.8 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=right opp | 5.1% | 3.0% | 40.1 | 33.1 | 41.3 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=left opp | 4.3% | 4.5% | 40.5 | 39.6 | 41.4 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=right opp | 3.8% | 4.5% | 40.7 | 39.9 | 41.5 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 4.7% | 1.5% | 41.2 | 41.2 | 41.2 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=left opp | 8.1% | 6.1% | 41.2 | 40.9 | 41.4 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=right opp | 2.3% | 6.1% | 41.2 | 40.3 | 41.5 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=partner | 7.2% | 6.1% | 41.4 | 41.3 | 41.6 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=left opp | 3.4% | 4.5% | 41.5 | 41.4 | 41.5 |

> Story: the 3-2 line is worth ~15 pts when [5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp] but ~41 pts when [5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=left opp] — a 26-point swing the belief-averaged marginal cannot see.

## Card 6 — corpus_v2_train_10-19_d0-9:g91:s19:d1

- Declaration id: 1 · trick 0 (decision idx 2) · actor seat 2
- Worlds: 64 valid of 200 sampled (32%) · ESS=22.4 · a*=slot 3 · action taken=slot 0
- Bimodal split: low mass 23% / high mass 77% · gap 21.6 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=partner, 5-0=partner, 4-1=your hand | 9.3% | 3.1% | 2.1 | 0.9 | 30.4 |
| 5-5=right opp, 6-4=partner, 5-0=right opp, 4-1=your hand | 2.2% | 1.6% | 15.0 | 15.0 | 15.0 |
| 5-5=right opp, 6-4=right opp, 5-0=right opp, 4-1=your hand | 4.4% | 3.1% | 23.0 | 19.3 | 40.8 |
| 5-5=right opp, 6-4=right opp, 5-0=left opp, 4-1=your hand | 7.5% | 7.8% | 24.6 | -21.8 | 41.4 |
| 5-5=left opp, 6-4=partner, 5-0=right opp, 4-1=your hand | 3.2% | 6.2% | 27.7 | 1.4 | 41.3 |
| 5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=your hand | 4.1% | 7.8% | 29.4 | 12.7 | 38.5 |
| 5-5=right opp, 6-4=right opp, 5-0=partner, 4-1=your hand | 6.9% | 3.1% | 30.0 | 28.4 | 31.4 |
| 5-5=right opp, 6-4=left opp, 5-0=left opp, 4-1=your hand | 3.9% | 7.8% | 31.0 | 7.0 | 41.5 |
| 5-5=left opp, 6-4=right opp, 5-0=right opp, 4-1=your hand | 5.8% | 6.2% | 34.8 | 22.0 | 41.3 |
| 5-5=right opp, 6-4=left opp, 5-0=partner, 4-1=your hand | 4.5% | 4.7% | 34.9 | 27.7 | 41.3 |
| 5-5=left opp, 6-4=right opp, 5-0=left opp, 4-1=your hand | 9.5% | 10.9% | 36.3 | 22.3 | 40.7 |
| other (pooled minor contexts) | 7.6% | 17.2% | 38.0 | 31.2 | 41.6 |
| 5-5=partner, 6-4=right opp, 5-0=left opp, 4-1=your hand | 16.7% | 4.7% | 38.5 | 38.3 | 41.4 |
| 5-5=partner, 6-4=left opp, 5-0=right opp, 4-1=your hand | 3.4% | 4.7% | 40.6 | 37.0 | 41.5 |
| 5-5=left opp, 6-4=right opp, 5-0=partner, 4-1=your hand | 3.0% | 4.7% | 41.0 | 40.7 | 41.5 |
| 5-5=partner, 6-4=left opp, 5-0=left opp, 4-1=your hand | 5.3% | 4.7% | 41.0 | 40.8 | 41.5 |
| 5-5=partner, 6-4=right opp, 5-0=right opp, 4-1=your hand | 2.6% | 1.6% | 41.4 | 41.4 | 41.4 |

> Story: the 3-2 line is worth ~2 pts when [5-5=right opp, 6-4=partner, 5-0=partner, 4-1=your hand] but ~41 pts when [5-5=partner, 6-4=right opp, 5-0=right opp, 4-1=your hand] — a 39-point swing the belief-averaged marginal cannot see.

## Card 7 — corpus_v2_train_10-19_d0-9:g46:s14:d6

- Declaration id: 6 · trick 0 (decision idx 2) · actor seat 2
- Worlds: 105 valid of 200 sampled (52%) · ESS=72.3 · a*=slot 4 · action taken=slot 4
- Bimodal split: low mass 24% / high mass 76% · gap 20.0 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=partner, 5-0=partner, 4-1=left opp | 2.0% | 2.9% | -39.5 | -41.8 | -36.9 |
| 5-5=right opp, 6-4=right opp, 5-0=right opp, 4-1=partner | 5.0% | 2.9% | -19.5 | -27.0 | -13.5 |
| 5-5=left opp, 6-4=right opp, 5-0=left opp, 4-1=right opp | 4.6% | 1.9% | -11.6 | -29.2 | 0.5 |
| 5-5=left opp, 6-4=left opp, 5-0=left opp, 4-1=right opp | 3.1% | 1.9% | -10.0 | -16.2 | -5.0 |
| 5-5=left opp, 6-4=right opp, 5-0=left opp, 4-1=partner | 2.1% | 1.0% | -6.3 | -6.3 | -6.3 |
| 5-5=right opp, 6-4=left opp, 5-0=partner, 4-1=left opp | 2.3% | 1.9% | 1.0 | -2.3 | 9.6 |
| 5-5=left opp, 6-4=left opp, 5-0=right opp, 4-1=right opp | 4.4% | 2.9% | 2.5 | -7.9 | 41.5 |
| 5-5=right opp, 6-4=left opp, 5-0=left opp, 4-1=right opp | 4.9% | 2.9% | 4.4 | -3.9 | 14.4 |
| other (pooled minor contexts) | 41.0% | 60.0% | 5.4 | -40.8 | 41.5 |
| 5-5=left opp, 6-4=right opp, 5-0=right opp, 4-1=partner | 2.2% | 1.0% | 5.6 | 5.6 | 5.6 |
| 5-5=partner, 6-4=right opp, 5-0=left opp, 4-1=right opp | 2.4% | 2.9% | 7.9 | -27.0 | 18.9 |
| 5-5=left opp, 6-4=left opp, 5-0=left opp, 4-1=partner | 2.4% | 1.9% | 10.7 | -7.4 | 27.1 |
| 5-5=left opp, 6-4=left opp, 5-0=right opp, 4-1=partner | 3.6% | 2.9% | 10.9 | -38.0 | 32.7 |
| 5-5=left opp, 6-4=left opp, 5-0=left opp, 4-1=left opp | 5.0% | 2.9% | 11.5 | 6.1 | 17.9 |
| 5-5=left opp, 6-4=right opp, 5-0=right opp, 4-1=right opp | 4.1% | 1.9% | 12.9 | -1.6 | 41.5 |
| 5-5=left opp, 6-4=partner, 5-0=right opp, 4-1=right opp | 2.1% | 1.0% | 16.0 | 16.0 | 16.0 |
| 5-5=right opp, 6-4=left opp, 5-0=right opp, 4-1=partner | 3.0% | 1.9% | 18.4 | -28.4 | 41.4 |
| 5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=partner | 3.4% | 3.8% | 19.9 | -2.7 | 36.7 |
| 5-5=left opp, 6-4=partner, 5-0=left opp, 4-1=partner | 2.3% | 1.9% | 30.3 | 18.5 | 41.4 |

> Story: the 3-2 line is worth ~-40 pts when [5-5=right opp, 6-4=partner, 5-0=partner, 4-1=left opp] but ~30 pts when [5-5=left opp, 6-4=partner, 5-0=left opp, 4-1=partner] — a 70-point swing the belief-averaged marginal cannot see.

## Card 8 — corpus_v2_train_20-29_d0-9:g3:s20:d3

- Declaration id: 3 · trick 0 (decision idx 3) · actor seat 3
- Worlds: 200 valid of 200 sampled (100%) · ESS=115.6 · a*=slot 6 · action taken=slot 6
- Bimodal split: low mass 79% / high mass 21% · gap 19.4 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=partner, 6-4=your hand, 5-0=partner, 4-1=partner | 7.1% | 4.0% | -26.1 | -41.8 | -14.5 |
| 5-5=right opp, 6-4=your hand, 5-0=partner, 4-1=right opp | 2.4% | 5.0% | -20.6 | -41.0 | 0.6 |
| 5-5=right opp, 6-4=your hand, 5-0=partner, 4-1=partner | 7.6% | 4.5% | -18.7 | -36.6 | -1.5 |
| 5-5=left opp, 6-4=your hand, 5-0=partner, 4-1=left opp | 3.9% | 4.0% | -15.1 | -38.4 | -1.5 |
| 5-5=partner, 6-4=your hand, 5-0=partner, 4-1=left opp | 7.2% | 6.0% | -14.7 | -29.9 | 18.1 |
| 5-5=left opp, 6-4=your hand, 5-0=partner, 4-1=right opp | 2.1% | 4.0% | -12.6 | -32.4 | 17.5 |
| 5-5=right opp, 6-4=your hand, 5-0=partner, 4-1=left opp | 8.5% | 5.5% | -11.1 | -37.2 | 32.6 |
| 5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=partner | 8.2% | 4.5% | -6.9 | -37.6 | 41.2 |
| other (pooled minor contexts) | 12.4% | 22.5% | -5.5 | -37.9 | 39.0 |
| 5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=partner | 6.6% | 3.0% | -4.1 | -14.3 | 4.3 |
| 5-5=partner, 6-4=your hand, 5-0=partner, 4-1=right opp | 4.2% | 6.5% | -3.7 | -32.0 | 41.2 |
| 5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=left opp | 9.4% | 6.5% | -2.0 | -40.2 | 41.0 |
| 5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=right opp | 4.0% | 5.5% | 4.3 | -24.8 | 41.3 |
| 5-5=partner, 6-4=your hand, 5-0=right opp, 4-1=right opp | 2.3% | 6.0% | 5.3 | -39.3 | 41.2 |
| 5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=right opp | 3.9% | 4.5% | 6.7 | -40.3 | 39.0 |
| 5-5=partner, 6-4=your hand, 5-0=right opp, 4-1=left opp | 3.6% | 3.5% | 7.0 | -27.7 | 26.1 |
| 5-5=right opp, 6-4=your hand, 5-0=right opp, 4-1=left opp | 2.6% | 2.5% | 7.0 | -32.5 | 36.9 |
| 5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=left opp | 4.2% | 2.0% | 19.2 | -8.6 | 27.9 |

> Story: the 3-2 line is worth ~-26 pts when [5-5=partner, 6-4=your hand, 5-0=partner, 4-1=partner] but ~19 pts when [5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=left opp] — a 45-point swing the belief-averaged marginal cannot see.

## Card 9 — corpus_v2_train_20-29_d0-9:g79:s27:d9

- Declaration id: 9 · trick 0 (decision idx 3) · actor seat 3
- Worlds: 200 valid of 200 sampled (100%) · ESS=130.4 · a*=slot 1 · action taken=slot 1
- Bimodal split: low mass 21% / high mass 79% · gap 16.1 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=right opp, 5-0=your hand, 4-1=left opp | 8.4% | 7.5% | -18.2 | -41.9 | 40.6 |
| 5-5=your hand, 6-4=right opp, 5-0=your hand, 4-1=partner | 7.3% | 7.5% | -17.0 | -34.1 | 9.4 |
| 5-5=your hand, 6-4=right opp, 5-0=your hand, 4-1=right opp | 5.1% | 10.0% | -7.2 | -32.5 | 34.7 |
| 5-5=your hand, 6-4=partner, 5-0=your hand, 4-1=left opp | 10.5% | 9.0% | -3.7 | -39.0 | 38.8 |
| 5-5=your hand, 6-4=left opp, 5-0=your hand, 4-1=left opp | 11.9% | 10.5% | -2.5 | -41.9 | 40.8 |
| 5-5=your hand, 6-4=left opp, 5-0=your hand, 4-1=partner | 19.3% | 13.0% | -0.9 | -41.8 | 28.7 |
| 5-5=your hand, 6-4=left opp, 5-0=your hand, 4-1=right opp | 10.4% | 15.0% | 1.4 | -41.2 | 40.9 |
| 5-5=your hand, 6-4=partner, 5-0=your hand, 4-1=right opp | 10.7% | 14.0% | 4.1 | -41.9 | 39.9 |
| 5-5=your hand, 6-4=partner, 5-0=your hand, 4-1=partner | 16.4% | 13.5% | 6.4 | -38.7 | 30.6 |

> Story: the 3-2 line is worth ~-18 pts when [5-5=your hand, 6-4=right opp, 5-0=your hand, 4-1=left opp] but ~6 pts when [5-5=your hand, 6-4=partner, 5-0=your hand, 4-1=partner] — a 25-point swing the belief-averaged marginal cannot see.

## Card 10 — corpus_v2_train_20-29_d0-9:g82:s28:d2

- Declaration id: 2 · trick 0 (decision idx 0) · actor seat 0
- Worlds: 90 valid of 200 sampled (45%) · ESS=67.5 · a*=slot 3 · action taken=slot 1
- Bimodal split: low mass 69% / high mass 31% · gap 15.3 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=your hand | 13.2% | 14.4% | -22.6 | -41.9 | 16.0 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=your hand | 22.4% | 18.9% | -9.4 | -41.3 | 36.5 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=your hand | 13.3% | 11.1% | -7.6 | -41.4 | 38.2 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=your hand | 7.9% | 8.9% | -4.8 | -41.8 | 22.5 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=your hand | 11.8% | 10.0% | -1.5 | -41.9 | 40.8 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=your hand | 4.7% | 6.7% | 2.7 | -25.5 | 39.6 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=your hand | 12.9% | 11.1% | 4.9 | -29.9 | 40.7 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=your hand | 4.2% | 7.8% | 6.7 | -41.7 | 40.5 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=your hand | 9.6% | 11.1% | 7.6 | -20.3 | 40.1 |

> Story: the 3-2 line is worth ~-23 pts when [5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=your hand] but ~8 pts when [5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=your hand] — a 30-point swing the belief-averaged marginal cannot see.
