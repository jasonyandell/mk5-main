# Otis W5 — 3-2 fate ledger cards

The 10 highest-mass bimodal P6 decisions (belief-weighted), from the world-bank instrument (`otis/analysis/worldbank.py`). Belief posterior: `gus/adapters/v3_consistency_10000g.pt`. Each card shows the public context, the context clusters of sampled worlds, and why the 3-2's fate value (oracle Q at a* = argmax E[Q]) differs across contexts.

## Card 1 — corpus_v2_train_0-9_d0-9:g41:s4:d1

- Declaration id: 1 · trick 0 (decision idx 1) · actor seat 1
- Worlds M=200 · ESS=14.4 · a*=slot 4 · action taken=slot 4
- Bimodal split: low mass 26% / high mass 74% · gap 35.8 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=left opp | 2.0% | 3.0% | -40.7 | -42.0 | -25.1 |
| 5-5=left opp, 6-4=right opp, 5-0=right opp, 4-1=left opp | 3.0% | 0.5% | -39.0 | -39.0 | -39.0 |
| 5-5=right opp, 6-4=left opp, 5-0=left opp, 4-1=your hand | 3.0% | 0.5% | -39.0 | -39.0 | -39.0 |
| 5-5=partner, 6-4=partner, 5-0=left opp, 4-1=right opp | 4.4% | 2.5% | -29.6 | -41.6 | -12.4 |
| 5-5=left opp, 6-4=partner, 5-0=right opp, 4-1=partner | 6.0% | 3.0% | -28.1 | -28.7 | -4.7 |
| 5-5=partner, 6-4=right opp, 5-0=left opp, 4-1=partner | 7.8% | 0.5% | -17.8 | -17.8 | -17.8 |
| 5-5=left opp, 6-4=partner, 5-0=your hand, 4-1=partner | 3.4% | 0.5% | -9.4 | -9.4 | -9.4 |
| other (pooled minor contexts) | 23.9% | 85.0% | -5.8 | -42.0 | 41.3 |
| 5-5=left opp, 6-4=your hand, 5-0=left opp, 4-1=partner | 8.5% | 0.5% | -5.2 | -5.2 | -5.2 |
| 5-5=left opp, 6-4=right opp, 5-0=partner, 4-1=partner | 18.3% | 1.0% | 13.2 | 7.6 | 38.0 |
| 5-5=partner, 6-4=right opp, 5-0=partner, 4-1=partner | 15.0% | 2.0% | 23.2 | -2.6 | 40.7 |
| 5-5=right opp, 6-4=right opp, 5-0=left opp, 4-1=partner | 4.7% | 1.0% | 31.0 | -17.6 | 39.3 |

> Story: the 3-2 line is worth ~-41 pts when [5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=left opp] but ~31 pts when [5-5=right opp, 6-4=right opp, 5-0=left opp, 4-1=partner] — a 72-point swing the belief-averaged marginal cannot see.

## Card 2 — corpus_v2_train_0-9_d0-9:g93:s9:d3

- Declaration id: 3 · trick 0 (decision idx 3) · actor seat 3
- Worlds M=200 · ESS=20.6 · a*=slot 2 · action taken=slot 2
- Bimodal split: low mass 22% / high mass 78% · gap 35.4 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=left opp, 5-0=left opp, 4-1=your hand | 6.2% | 2.0% | -26.8 | -27.8 | 41.2 |
| 5-5=right opp, 6-4=partner, 5-0=partner, 4-1=your hand | 8.3% | 3.0% | -26.4 | -27.6 | -2.1 |
| 5-5=right opp, 6-4=your hand, 5-0=partner, 4-1=your hand | 5.1% | 0.5% | -16.8 | -16.8 | -16.8 |
| 5-5=left opp, 6-4=partner, 5-0=partner, 4-1=your hand | 2.1% | 3.0% | -15.7 | -29.2 | 5.9 |
| 5-5=right opp, 6-4=your hand, 5-0=right opp, 4-1=your hand | 5.7% | 0.5% | -13.4 | -13.4 | -13.4 |
| 5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=your hand | 10.9% | 0.5% | 2.2 | 2.2 | 2.2 |
| 5-5=right opp, 6-4=right opp, 5-0=partner, 4-1=your hand | 3.4% | 4.0% | 9.5 | -24.5 | 37.7 |
| 5-5=right opp, 6-4=right opp, 5-0=left opp, 4-1=your hand | 5.0% | 3.0% | 10.7 | -28.3 | 41.4 |
| 5-5=partner, 6-4=left opp, 5-0=partner, 4-1=your hand | 3.6% | 5.0% | 11.0 | -23.7 | 41.5 |
| 5-5=left opp, 6-4=left opp, 5-0=partner, 4-1=your hand | 3.2% | 4.0% | 11.1 | -23.4 | 41.2 |
| 5-5=partner, 6-4=left opp, 5-0=left opp, 4-1=your hand | 16.3% | 3.5% | 11.5 | 5.8 | 41.5 |
| 5-5=partner, 6-4=partner, 5-0=right opp, 4-1=your hand | 2.8% | 5.0% | 11.9 | -5.3 | 41.4 |
| 5-5=partner, 6-4=right opp, 5-0=left opp, 4-1=your hand | 2.7% | 3.5% | 14.6 | -14.1 | 35.9 |
| other (pooled minor contexts) | 15.2% | 58.5% | 17.5 | -29.3 | 41.5 |
| 5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=your hand | 2.4% | 0.5% | 21.1 | 21.1 | 21.1 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=your hand | 4.0% | 1.0% | 35.0 | 35.0 | 35.1 |
| 5-5=partner, 6-4=right opp, 5-0=right opp, 4-1=your hand | 3.0% | 2.5% | 39.9 | 23.2 | 41.5 |

> Story: the 3-2 line is worth ~-27 pts when [5-5=right opp, 6-4=left opp, 5-0=left opp, 4-1=your hand] but ~40 pts when [5-5=partner, 6-4=right opp, 5-0=right opp, 4-1=your hand] — a 67-point swing the belief-averaged marginal cannot see.

## Card 3 — corpus_v2_train_0-9_d0-9:g83:s8:d3

- Declaration id: 3 · trick 0 (decision idx 0) · actor seat 0
- Worlds M=200 · ESS=11.9 · a*=slot 2 · action taken=slot 2
- Bimodal split: low mass 24% / high mass 76% · gap 33.8 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=partner | 7.2% | 2.0% | -20.0 | -25.4 | 41.4 |
| 5-5=your hand, 6-4=your hand, 5-0=left opp, 4-1=right opp | 7.4% | 0.5% | -13.9 | -13.9 | -13.9 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=left opp | 9.4% | 3.5% | 8.4 | -14.2 | 41.3 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=left opp | 4.3% | 5.5% | 9.1 | 0.8 | 41.3 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 2.9% | 1.5% | 19.0 | 12.9 | 41.1 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=right opp | 23.9% | 3.0% | 24.0 | 1.5 | 40.9 |
| 5-5=your hand, 6-4=your hand, 5-0=partner, 4-1=right opp | 3.7% | 0.5% | 24.6 | 24.6 | 24.6 |
| other (pooled minor contexts) | 14.8% | 66.0% | 26.8 | -33.5 | 41.5 |
| 5-5=your hand, 6-4=your hand, 5-0=left opp, 4-1=partner | 3.9% | 0.5% | 29.1 | 29.1 | 29.1 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=right opp | 10.7% | 5.0% | 29.3 | 11.4 | 41.2 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=partner | 4.5% | 3.5% | 35.1 | 20.7 | 41.3 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=left opp | 3.5% | 4.5% | 40.1 | 10.8 | 41.5 |
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp | 3.9% | 4.0% | 40.8 | -16.9 | 41.6 |

> Story: the 3-2 line is worth ~-20 pts when [5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=partner] but ~41 pts when [5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp] — a 61-point swing the belief-averaged marginal cannot see.

## Card 4 — corpus_v2_train_20-29_d0-9:g90:s29:d0

- Declaration id: 0 · trick 0 (decision idx 0) · actor seat 0
- Worlds M=200 · ESS=31.3 · a*=slot 3 · action taken=slot 3
- Bimodal split: low mass 22% / high mass 78% · gap 33.3 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp | 2.4% | 4.0% | -24.0 | -33.9 | 40.1 |
| 5-5=your hand, 6-4=left opp, 5-0=left opp, 4-1=right opp | 3.8% | 3.0% | -16.0 | -34.1 | 40.5 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=your hand | 3.1% | 1.0% | -9.2 | -10.6 | 31.0 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=right opp | 10.8% | 1.5% | -7.0 | -9.3 | 41.5 |
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=partner | 2.4% | 5.0% | -1.3 | -41.7 | 25.0 |
| 5-5=your hand, 6-4=your hand, 5-0=partner, 4-1=right opp | 3.1% | 0.5% | 7.9 | 7.9 | 7.9 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=left opp | 3.2% | 4.0% | 9.0 | -35.8 | 40.8 |
| 5-5=your hand, 6-4=right opp, 5-0=left opp, 4-1=right opp | 5.7% | 5.5% | 10.0 | -40.4 | 41.5 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=right opp | 3.0% | 5.0% | 10.9 | -19.6 | 41.6 |
| other (pooled minor contexts) | 16.3% | 43.5% | 14.3 | -39.9 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=right opp | 3.9% | 3.5% | 18.9 | -16.7 | 41.4 |
| 5-5=your hand, 6-4=your hand, 5-0=partner, 4-1=partner | 3.9% | 0.5% | 19.2 | 19.2 | 19.2 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=partner | 4.5% | 4.5% | 21.4 | -27.0 | 40.2 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=right opp | 3.8% | 3.5% | 29.2 | -8.5 | 38.7 |
| 5-5=your hand, 6-4=right opp, 5-0=right opp, 4-1=partner | 7.7% | 3.0% | 32.1 | -13.5 | 40.7 |
| 5-5=your hand, 6-4=your hand, 5-0=left opp, 4-1=partner | 5.5% | 1.0% | 33.2 | 32.0 | 34.2 |
| 5-5=your hand, 6-4=partner, 5-0=right opp, 4-1=partner | 4.9% | 4.5% | 36.9 | -29.1 | 41.5 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=your hand | 4.4% | 0.5% | 37.4 | 37.4 | 37.4 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=left opp | 2.4% | 4.0% | 37.5 | 12.8 | 39.8 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner | 5.1% | 2.0% | 40.8 | 14.5 | 41.2 |

> Story: the 3-2 line is worth ~-24 pts when [5-5=your hand, 6-4=left opp, 5-0=right opp, 4-1=right opp] but ~41 pts when [5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=partner] — a 65-point swing the belief-averaged marginal cannot see.

## Card 5 — corpus_v2_train_10-19_d0-9:g16:s11:d6

- Declaration id: 6 · trick 0 (decision idx 1) · actor seat 1
- Worlds M=200 · ESS=99.7 · a*=slot 6 · action taken=slot 6
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

## Card 6 — corpus_v2_train_20-29_d0-9:g3:s20:d3

- Declaration id: 3 · trick 0 (decision idx 3) · actor seat 3
- Worlds M=200 · ESS=115.6 · a*=slot 6 · action taken=slot 6
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

## Card 7 — corpus_v2_train_20-29_d0-9:g79:s27:d9

- Declaration id: 9 · trick 0 (decision idx 3) · actor seat 3
- Worlds M=200 · ESS=130.4 · a*=slot 1 · action taken=slot 1
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

## Card 8 — corpus_v2_train_20-29_d0-9:g83:s28:d3

- Declaration id: 3 · trick 0 (decision idx 0) · actor seat 0
- Worlds M=200 · ESS=2.4 · a*=slot 3 · action taken=slot 3
- Bimodal split: low mass 77% / high mass 23% · gap 73.5 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=your hand | 61.5% | 13.5% | -38.3 | -38.7 | 41.3 |
| 5-5=your hand, 6-4=partner, 5-0=partner, 4-1=your hand | 9.3% | 12.5% | -34.6 | -40.6 | 40.8 |
| 5-5=your hand, 6-4=right opp, 5-0=partner, 4-1=your hand | 2.4% | 8.5% | -15.1 | -40.4 | 40.9 |
| other (pooled minor contexts) | 4.3% | 56.0% | 18.7 | -41.2 | 41.4 |
| 5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=your hand | 22.6% | 9.5% | 39.5 | -40.5 | 40.9 |

> Story: the 3-2 line is worth ~-38 pts when [5-5=your hand, 6-4=partner, 5-0=left opp, 4-1=your hand] but ~40 pts when [5-5=your hand, 6-4=left opp, 5-0=partner, 4-1=your hand] — a 78-point swing the belief-averaged marginal cannot see.

## Card 9 — corpus_v2_train_20-29_d0-9:g45:s24:d5

- Declaration id: 5 · trick 0 (decision idx 1) · actor seat 1
- Worlds M=200 · ESS=7.4 · a*=slot 5 · action taken=slot 5
- Bimodal split: low mass 26% / high mass 74% · gap 54.5 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=partner, 5-0=right opp, 4-1=your hand | 2.3% | 3.0% | -41.0 | -41.7 | -39.4 |
| 5-5=right opp, 6-4=partner, 5-0=left opp, 4-1=your hand | 7.3% | 2.5% | -40.5 | -41.8 | -33.8 |
| 5-5=left opp, 6-4=partner, 5-0=right opp, 4-1=your hand | 4.5% | 2.5% | -40.4 | -41.9 | -32.0 |
| 5-5=right opp, 6-4=partner, 5-0=partner, 4-1=your hand | 2.9% | 5.5% | -40.3 | -41.6 | -3.7 |
| 5-5=left opp, 6-4=left opp, 5-0=left opp, 4-1=your hand | 9.2% | 2.0% | -30.6 | -41.5 | -30.5 |
| other (pooled minor contexts) | 6.2% | 66.0% | -15.5 | -42.0 | 41.4 |
| 5-5=partner, 6-4=partner, 5-0=left opp, 4-1=your hand | 2.8% | 4.5% | 7.7 | 6.5 | 39.5 |
| 5-5=partner, 6-4=left opp, 5-0=left opp, 4-1=your hand | 21.7% | 8.0% | 19.9 | -13.4 | 40.9 |
| 5-5=partner, 6-4=right opp, 5-0=right opp, 4-1=your hand | 39.9% | 2.5% | 20.2 | 10.4 | 41.2 |
| 5-5=partner, 6-4=right opp, 5-0=partner, 4-1=your hand | 3.2% | 3.5% | 39.1 | 25.5 | 41.3 |

> Story: the 3-2 line is worth ~-41 pts when [5-5=right opp, 6-4=partner, 5-0=right opp, 4-1=your hand] but ~39 pts when [5-5=partner, 6-4=right opp, 5-0=partner, 4-1=your hand] — a 80-point swing the belief-averaged marginal cannot see.

## Card 10 — corpus_v2_train_0-9_d0-9:g29:s2:d9

- Declaration id: 9 · trick 0 (decision idx 2) · actor seat 2
- Worlds M=200 · ESS=3.6 · a*=slot 3 · action taken=slot 3
- Bimodal split: low mass 46% / high mass 54% · gap 54.3 points

| context (other four count tiles) | belief mass | uniform mass | mean Q(3-2 line) | min | max |
|---|---|---|---|---|---|
| 5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=partner | 22.0% | 0.5% | -27.9 | -27.9 | -27.9 |
| 5-5=right opp, 6-4=left opp, 5-0=partner, 4-1=right opp | 2.6% | 2.5% | -10.2 | -19.6 | 41.0 |
| other (pooled minor contexts) | 21.4% | 94.0% | -5.3 | -40.3 | 41.5 |
| 5-5=left opp, 6-4=left opp, 5-0=partner, 4-1=left opp | 3.2% | 0.5% | 26.7 | 26.7 | 26.7 |
| 5-5=partner, 6-4=your hand, 5-0=left opp, 4-1=left opp | 2.3% | 0.5% | 31.5 | 31.5 | 31.5 |
| 5-5=right opp, 6-4=left opp, 5-0=your hand, 4-1=right opp | 48.5% | 2.0% | 38.9 | -29.4 | 39.7 |

> Story: the 3-2 line is worth ~-28 pts when [5-5=right opp, 6-4=your hand, 5-0=left opp, 4-1=partner] but ~39 pts when [5-5=right opp, 6-4=left opp, 5-0=your hand, 4-1=right opp] — a 67-point swing the belief-averaged marginal cannot see.
