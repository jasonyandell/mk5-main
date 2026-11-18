# Walt: the level-1 scenario player for Texas 42

*Jason Yandell. Dedicated to the public domain under CC0 1.0. Made for the love of the game of 42.*

**Specification v1.0 (draft for review), 2026-11-09.** One self-contained document from
which Walt's card play can be rebuilt. It supersedes `SCENARIO-PLAYER.md` v0.1 as a
*reproduction* guide; that document remains the mathematical record (proof obligations,
history, defect log).

Status of the claims here: an implementation built from v0.1 plus the tie protocol of §5.4
played Walt to a statistical draw over 522 full games (rating 1519, 95% interval
1498–1539, against Walt at 1500). Section 8 says how to check your own build.

## 1. What Walt is

**What it is for.** Walt is a fast player that plays about as well as a perfect-information
Monte Carlo (PIMC) player, without cheating, peeking at other hands, or the strategy-fusion
mistakes that make PIMC players uncanny: playing as if it will know the deal later when it
won't. It is a good, fair, fun opponent. It is not claimed to be the strongest possible 42
player.

Walt decides one domino play for one seat in a hand of straight 42, seeing only what that
seat legally sees: its own dealt hand and the public record of plays. It estimates, for each
legal tile, the probability that the bidding team makes its bid, and plays the best tile for
its own team.

The estimate is a best response to a model of the other three seats, computed exactly on a
sample of possible deals:

- **Beliefs:** sample deals uniformly from all deals consistent with what the seat has seen.
- **Field:** each other seat (partner included) is a *level-0 mind*: it samples deals from
  its own point of view and best-responds to opponents who play uniformly at random.
- **Walt itself** is *level 1*: it best-responds to that field. Its own later decisions are
  made without seeing the other hands, so it picks one tile for all deals it still
  considers possible (no strategy fusion).
- **Objective:** P(bidding team banks ≥ bid). Exact rational arithmetic on the sample.

## 2. Game facts used

- 28 tiles of a double-six set, ids `id(hi, lo) = hi·(hi+1)/2 + lo`, hi ≥ lo:
  (0,0)=0, (1,0)=1, (1,1)=2, …, (6,6)=27.
- Seats 0–3 in play order; partners {0,2} and {1,3}. The bidder leads trick 1; each trick's
  winner leads the next. Seven tricks of four plays.
- Declarations: pip trump 0–6 (every tile containing that pip is trump), doubles trump (the
  seven doubles are the trump suit), no-trump (follow-me, doubles high in their suit).
- Led suit: a trump lead calls trump; otherwise the higher pip of the led tile. A seat must
  follow the led suit if it can (a trump tile never counts as the led non-trump suit);
  otherwise it may play anything. Highest trump wins, else the highest tile of the led suit.
  Rank within a suit x (pip trump x, or a non-trump pip suit x): the double x-x highest, then
  the other tiles by their other pip, descending. Doubles trump ranks 6-6 highest down to 0-0.
- Points: 1 per trick, plus count tiles 5-5 and 6-4 (10 each), 5-0, 4-1, 3-2 (5 each).
  42 in all.
- The bidding team **makes** iff its banked points ≥ bid; it is **set** iff the other team's
  banked points > 42 − bid. These are the cutoffs: a node where either holds has value 1 or 0.
- **Voids:** a seat that failed to follow a led suit is void in it; no possible deal may give
  it a tile of that suit (by the same suit membership as above).

Tournament rules for anything not stated: N42PA straight 42.

## 3. State

- **Record** R: the plays so far, `(seat, tile)` in order.
- **Reduced record** κ: (set of played tiles, current leader, plays in the current trick,
  points banked by each team). Completed-trick order is discarded; banked points are NOT
  (two orders of the same tiles can bank differently).
- **Information state** of seat s: (s's remaining tiles, R). Nothing else. No mind, at any
  level, conditions on another seat's tiles.

## 4. Samplers

- **Outer belief (Walt's own):** a world assigns every unseen tile (not played, not in the
  seat's hand) to the other three seats with the hand sizes the record forces. Draw
  uniformly among worlds that respect every void: assign uniformly by sizes, reject any
  world that breaks a void, repeat. Draws are independent (with replacement).
- **Inner belief (a level-0 mind's):** the same, but voids are ignored (sizes only). This is
  a deliberate simplification of the field model.

## 5. The algorithm

### 5.1 The solver: value of a node

`value(κ, alive)` is the probability, over the alive sampled worlds, that the bidding team
makes the bid. `alive` is a set of world indices, initially all `n` worlds.

```
value(κ, alive):
    if bidding team banked ≥ bid:        return 1
    if other team banked > 42 − bid:     return 0
    if (κ, alive) is memoized:           return memo
    s = seat to act at κ
    if s is the deciding seat (Walt's own later turn):
        # one tile for every alive world: Walt does not see the deal
        legal = s's legal tiles (identical in every alive world)
        v = max over t in legal of value(κ after s plays t, alive)   if s on bidding team
            min over t in legal of value(…)                          otherwise
    else:
        # field seat: each world says what this seat plays there
        for each w in alive: m(w) = field move of s in world w at κ
        buckets: alive partitioned by m(w)
        v = Σ over moves m of |bucket_m| / |alive| · value(κ after s plays m, bucket_m)
    memoize and return v
```

Within one sampled world, the deal and the played tiles fix every seat's remaining hand, so
(κ, alive) determines all hands and the memo key needs nothing more. In Walt's own solver
every field seat moves by its level-0 mind (§5.2). Inside a level-0
mind's solver the field seats move by Dice (§5.3). The buckets partition `alive` exactly.

### 5.2 Level-0 mind

A level-0 mind for seat s at κ, holding hand h (its tiles in that world):

```
mind(s, h, κ):                        # cache under PiKey = (s, h, κ); κ includes banked
    draw n0 worlds from the INNER belief of s's information state (h, record)
    run the solver of 5.1 with s as the deciding seat and Dice at every other seat
    return the best tile for s's team (bidding team: highest value; otherwise lowest),
           scanning legal tiles in ascending id and replacing only on strict improvement
```

The result must be a pure function of PiKey: same key, same tile, on every call. That means
its random draws are seeded from the key's contents (and fixed constants), never from call
order. Omitting banked from the key was a real defect (v0.1 §3.4).

### 5.3 Dice

At a Dice seat in sampled world w at κ, the seat plays legal tile number
`SplitMix64(seed_w ⊕ hash(κ)) mod |legal|` (legal tiles in ascending id), where seed_w is a
per-world seed. Keying on the record, not the path, makes this a fixed tape: the same world
at the same record plays the same tile on every branch, so worlds split into buckets rather
than branches multiplying. Every legal tile has positive probability under Dice, so no
lawful world is ever excluded at the bottom of the model.

### 5.4 The decision: evaluate, then refine ties

```
decide(seat, hand, R, n, n0):
    legal = seat's legal tiles
    if |legal| == 1: play it (forced; nothing evaluated)
    best = max if seat on bidding team else min
    W = n worlds from the OUTER belief
    for t in legal: val[t] = value(κ after t, all of W)         # field = level-0 minds
    size = n
    while size < 16·n and at least two tiles share the best value:
        size = 4·size
        W' = a FRESH outer sample of `size` worlds
        for t in the tied tiles only: val[t] = value(κ after t, all of W')   # REPLACE
    play the best tile; a tie still standing goes to the lowest tile id
```

- **This step is not optional.** It is what separates a reproduction from a near-miss: an
  implementation that breaks top ties by tile id plays about 35 Elo weaker. At n = 40, values
  are multiples of 1/40, so ties at the top are routine, mostly at certainty (40/40 or 0/40).
- Any tie at the top counts, not only certainty ties. Only the tied tiles are re-evaluated;
  the others keep their n-world values. Values are replaced, never pooled. With the default
  n = 40 that is at most two rounds: 40 → 160 → 640 worlds.
- Reported values are the final ones, so they mix sample sizes.

### 5.5 Randomness

Use SplitMix64. Derive every stream from the request seed and structural coordinates (seat,
hand, record, round), never from a clock, thread order or global state. The reference keeps
the level-0 minds on a fixed seed independent of the request seed (the field is one fixed
policy); seeding them from the request instead made no measurable difference. Exact seed
formulas are not part of this specification: two correct implementations draw different
worlds and agree statistically, not bit for bit.

## 6. Interface: the `walt1 play` request

One request in, one JSON line out. Request lines are `key values…`, unsigned integers; the
first line is `walt1 play`.

| key | meaning |
|---|---|
| `decl` | 0–6 pip trump, 7 doubles trump, 9 no-trump |
| `bid` | 30–42 |
| `seat` | the seat to decide for; must be the seat to act |
| `bidder` | the seat that won the bid (leads trick 1) |
| `hand` | the seat's 7 originally dealt tile ids |
| `plays` | `seat tile` pairs in order; may be empty |
| `n`, `n0`, `seed` | defaults 40, 8, 0xB7E151628AED2A6B |

Response:

```json
{"v":1,"kind":"play","choice":19,"forced":false,"raced":false,"opts":[[1,1750],[6,2000]],"leader":0,"points":[0,0]}
```

- `opts`: `[tile, bp]` for every legal tile, ascending id; bp = value × 10000 truncated.
  Empty when forced.
- `choice`: the tile from §5.4. `forced`: exactly one legal tile. `raced`: always false.
- `leader`, `points`: the current trick's leader (the last trick's winner when the record
  ends on a completed trick) and the banked points of teams {0,2} and
  {1,3} after replaying `plays`. These must be exact.
- Any bad input (malformed line, illegal play in the record, wrong seat to act):
  `{"v":1,"error":"…"}`.

## 7. Cost

Each tile at the root costs one solve over n worlds; each field decision inside it costs a
level-0 mind (a solve over n0 worlds), cached per PiKey. A careful implementation answers the
defaults in tens of milliseconds on one core (the fastest rebuild: about 50 ms per decision
including refinement, against about 280 ms for the reference).

## 8. Checking a build

Walt's numbers are sampled estimates, so a correct build is judged statistically, against
the reference at the same `n` and `n0`.

1. **Exact fields:** `leader`, `points`, `forced` and legality must match on every request.
2. **Move agreement:** on held-out positions, the share of moves equal to the reference's,
   averaged over several seeds, should match the reference's agreement with itself under
   new seeds (about 73% at n = 40). A build that breaks ties by tile id lands near 68%.
3. **Strength:** full games against the reference on a fair ladder (both sides bid 30;
   duplicate deals with seats swapped). A correct build's 95% rating interval lies within
   ±50 Elo of the reference.

The kit (`conformance/`): `corpus.jsonl` (168 reference decisions from six self-played
hands; hand 0 is for examples, hands 1–5 are held out), `grade.mjs` (runs a binary over the
held-out requests and reports exact fields, agreement and value gap), and the reseeded
reference numbers to compare with.

## 9. What is deliberately left out

Bidding and trump declaration (`walt1 bid`, `walt1 declare`), level 2 and higher (a
level-k field best-responds to level-(k−1) minds), the race-then-refine mode, the
cross-seat review fields, parallelism, and the native per-move deadline.
