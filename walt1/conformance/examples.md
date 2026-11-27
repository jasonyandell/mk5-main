### Example (hand 0, play 0)

```text
walt1 play
decl 6
bid 35
seat 0
bidder 0
hand 1 6 7 8 18 19 26
plays 
n 40
n0 8
seed 1000
```

Reference response:

```json
{"v":1,"kind":"play","choice":19,"forced":false,"raced":false,"opts":[[1,1750],[6,2000],[7,1250],[8,2250],[18,3250],[19,3500],[26,3250]],"leader":0,"points":[0,0]}
```

### Example (hand 0, play 5)

```text
walt1 play
decl 6
bid 35
seat 1
bidder 0
hand 3 4 12 16 22 23 24
plays 0 19 1 16 2 17 3 15 0 8
n 40
n0 8
seed 1000
```

Reference response:

```json
{"v":1,"kind":"play","choice":3,"forced":false,"raced":false,"opts":[[3,250],[4,500],[12,500],[22,687],[23,875],[24,687]],"leader":0,"points":[6,0]}
```

### Example (hand 0, play 13)

```text
walt1 play
decl 6
bid 35
seat 1
bidder 0
hand 3 4 12 16 22 23 24
plays 0 19 1 16 2 17 3 15 0 8 1 3 2 13 3 9 3 5 0 26 1 4 2 21 0 6
n 40
n0 8
seed 1000
```

Reference response:

```json
{"v":1,"kind":"play","choice":12,"forced":false,"raced":false,"opts":[[12,500],[22,1000],[23,1000],[24,750]],"leader":0,"points":[7,6]}
```

### Example (hand 0, play 22)

```text
walt1 play
decl 6
bid 35
seat 3
bidder 0
hand 5 9 11 15 20 25 27
plays 0 19 1 16 2 17 3 15 0 8 1 3 2 13 3 9 3 5 0 26 1 4 2 21 0 6 1 12 2 0 3 11 0 18 1 23 2 2 3 20 1 22 2 10
n 40
n0 8
seed 1000
```

Reference response:

```json
{"v":1,"kind":"play","choice":25,"forced":false,"raced":false,"opts":[[25,0],[27,0]],"leader":1,"points":[13,17]}
```

