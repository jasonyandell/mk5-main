# Zeb

Zeb plays Texas 42. To make a decision, a seat samples deals of the tiles it
cannot see that fit everything the table has shown so far: the plays, and
the suits each seat has failed to follow. It then plans one line of play
across all of those deals at once and plays the plan's first move. The plan
is contingent: the seat chooses once for every deal it cannot tell apart, so
it never gets credit for knowing which deal is the real one. At the bottom
of the plan, the other three seats are played by the *field net*, a small
integer network trained on Zeb's own play. Under a pip trump the seat also
reads how the others have played: deals in which the net would have made
those plays count for more.

This package is that player compiled to WebAssembly, for browsers, Web
Workers and Node. It does three things: it forecasts a hand for bidding, it
picks a seat's play, and it explains a play for hints. **The planner never
sees another seat's hand.** You give it the seat's own seven tiles and the
public plays, and nothing else exists for it.

```sh
npm install @jasonyandell/zeb
```

## Loading

The calls are synchronous and take about a second at the default 160
worlds, so run them in a Web Worker:

```js
// worker.js
import init, { forecast, play, analyze } from '@jasonyandell/zeb';

await init(); // fetches zeb_bg.wasm from beside the glue
```

Some bundlers pre-bundle dependencies in a way that loses the
`import.meta.url` the default loader resolves against. In that case, hand
`init` the file's URL yourself. With Vite:

```js
import init from '@jasonyandell/zeb';
import wasmUrl from '@jasonyandell/zeb/zeb_bg.wasm?url';

await init({ module_or_path: wasmUrl });
```

The other standard fix is to keep the package out of Vite's pre-bundling:
`optimizeDeps: { exclude: ['@jasonyandell/zeb'] }` in `vite.config.js`.

In Node:

```js
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { initSync } from '@jasonyandell/zeb';

initSync({ module: readFileSync(fileURLToPath(import.meta.resolve('@jasonyandell/zeb/zeb_bg.wasm'))) });
```

## Conventions

- **Tiles** are ids 0..27 in triangular order. (high, low) is
  `high * (high + 1) / 2 + low`, so 0-0 is 0, 1-0 is 1, 1-1 is 2, and 6-6
  is 27.
- **Seats** are 0..3. Seats 0 and 2 are partners, and so are 1 and 3. The
  bidder leads the first trick.
- **Declarations**: 0..6 makes that pip suit trump, 7 makes the doubles
  trump, and 9 is no-trump. Declaration 8 (doubles as their own suit)
  belongs to nello, which Zeb does not play, so it is refused.
- **A contract** (`bid`) is 30..41 points, or 42 for a mark bid, where the
  bidding side has to take all 42 points.
- **Probabilities are always the bidding side's.** `q` is the probability
  that the bidding side makes its contract, and `pmake[i]` is the
  probability that it banks at least 30 + i points. A defender's best play
  is the one with the *smallest* `q`.

## Bidding: `forecast` and `chooseBid`

```js
import { forecast, chooseBid } from '@jasonyandell/zeb';

const hand = [27, 26, 24, 20, 17, 12, 2]; // 6-6 6-5 6-3 5-5 5-2 4-2 1-1
const f = forecast({ hand });
// [{ decl: 0, mean: -4.15, pmake: [0.2, 0.16, 0.15, ...] },
//  ...
//  { decl: 6, mean: 31.95, pmake: [0.9, 0.83, 0.8, 0.8, 0.8, 0.78, 0.68, ...] },
//  ...]   (numbers rounded)

chooseBid(f, { need: 30 }); // { bid: 34, decl: 6, pmake: 0.8 }; null would mean pass
```

`forecast({ hand, declarations?, worlds?, seed? })` returns one entry per
declaration. By default it covers the nine straight declarations: 0..6, 7
and 9. Each entry answers one question: if this hand wins a 30 bid and
declares this trump, how does the bidding side fare? That takes one plan
from the bidder's chair before the opening lead. `mean` is the bidding
side's expected final margin (its points minus the defenders'), and
`pmake[i]` is the probability that it banks at least 30 + i points
(targets 30..42). Both come from the planned opening lead.

`chooseBid(forecast, { need, cutoff = 0.8, legalDeclarations? })` finds the
highest target at or above `need` that some declaration makes with
probability at least `cutoff`. At that target it picks the declaration most
likely to make it, and on a tie, the one listed first. It returns null to
pass. It is a plain function of the forecast. The auction's own rules, such
as passing behind a partner or a forced 30, are yours to apply around it.

This rule is the one Plunge's frozen bid book uses. The forecast was graded
against that book (4,680 panels; 512 worlds for the pip trumps, 160 for
doubles and no-trump). Its mean distance from the book's make rate at 30
was 0.052, smaller than the book's own sampling error of 0.066. It is
calibrated through the middle of the range. At the top it runs 3 to 5
points bold (a forecast of 0.85 makes 0.82 of the time, and 0.95 makes
0.90), so with the 0.8 cutoff it bids one target higher than the book on
about 7% of hands. A cutoff of 0.85 gives back the book's temperament.

The default of 160 worlds is a mobile budget, not the graded setting. At
160 each tail is noisier, and `chooseBid` takes the best of nine
declarations at thirteen targets, so the noise leans toward bolder bids on
top of the measured 3 to 5 points. Pass `worlds: 512` to bid as graded, or
raise the cutoff a little.

### Match-aware bidding

`chooseBid` also takes an optional `match`. With it, the bid is the hand-only
call plus one more option: a mark bid (84, or one mark over an 84) at the
declaration most likely to take all 42 points. The mark bid wins when its
chance of winning the *game* beats the hand-only call's, by a match-equity
table solved exactly to `marksToWin`. Every contract from 30 to 42 pays one
mark, made or set, so the score only matters where the marks at stake differ.

```js
chooseBid(f, {
  need: 30,
  match: {
    us: 5, them: 3,        // marks
    marksToWin: 7,         // default
    weShakeNext: false,    // our side shakes first on the next hand
    holder: 'none',        // who holds the high bid: 'none' | 'partner' | 'opp'
    passesAfter: 2,        // seats still to speak after you, 0..3
    maxBid: 84,            // highest legal bid; a mark bid needs 84 or more
    // pointsToWin: true   // a points-to-win game has no mark option
  },
}); // e.g. { bid: 84, decl: 6, pmake: 0.31 }, or the hand-only call, or null to pass
```

Without `match`, `chooseBid` is the hand-only rule above, as in 0.3.0. A
tie keeps the hand-only call. Behind a partner's bid (`holder: 'partner'`)
the hand-only call is a pass, and the mark bid is still priced. With `match`,
`need` may exceed 42, for a seat facing an 84. The pricing uses the hand law
and the auction odds fitted to the deployed field (Zeb bidding at 160 worlds
and playing both sides). Over 4,096 duplicate games to 7 marks it won 53.37%
of games [52.47, 54.27] against the hand-only bidder (a coin is 50%); see
`wiki/experiments/match-aware-bidding-2026-10-14.md` in the repository.
`matchPrice` returns the two match equities and the call.

### Overcalling an opponent's bid (0.4.1)

Add `overcall: true` to the options (it needs `match`) and, when an opponent
holds the high bid, two things change. The forecast is calibrated (an
opponent's bid shows less than the hand-only forecast thinks, and over one it
runs hot), and where the hand-only rule would pass, `chooseBid` bids the
one-mark call, `need` to 42, whose chance of our side taking the hand's mark
beats passing's. Every call from pass to 42 moves one mark, so this choice
does not read the score. The prices use constants fitted to a table that
overcalls back, which is the tradeoff: **it is tuned for opponents who
contest openings.**

```js
chooseBid(f, { need: 30, legalDeclarations: [0, 1, 2, 3, 4, 5, 6], match: { ...state, holder: 'opp' }, overcall: true });
```

Measured against a table that contests (games to 7 marks, duplicate deals),
`over` and `cal` are worth **+5.6 [+1.7, +9.4] points** of game share, and
0.4.0's constants win only **43.5% [40.8, 46.2]** of games against these
(chance is 50%). Against a table that mostly passes over an opening, these
constants give back **3.4 [0.8, 6.0] points** compared with 0.4.0's, and both
still beat the hand-only bidder by 17 points or more. If your opponents rarely
contest, leave `overcall` off. Without it, `chooseBid` is 0.4.0's, byte for
byte. The measured bidder used the seven pip trumps only (see
`legalDeclarations`), 160 worlds, and
`wiki/experiments/match-aware-bidding-2026-10-14.md` (#170) has the numbers.

## Playing: `play` and `analyze`

```js
import { play, analyze } from '@jasonyandell/zeb';

// Seat 0 bid 32, called sixes and led 6-6, which took the first trick.
// Now it leads again.
const position = {
  decl: 6,
  bidder: 0,
  bid: 32,
  seat: 0,                                    // the seat to play
  hand: [27, 25, 23, 21, 20, 11, 8],          // as dealt: 6-6 6-4 6-2 6-0 5-5 4-1 3-2
  plays: [[0, 27], [1, 22], [2, 0], [3, 24]], // every play so far, [seat, tile]
};

play(position);
// { tile: 11, forced: false,
//   q: [[8, 0.38], [11, 0.43], [20, 0.29], [21, 0.29], [23, 0.33], [25, 0]] }   (rounded)

analyze(position);
// the same, plus pmake: [[8, [0.48, 0.38, 0.38, ...]], [11, [0.56, 0.45, 0.43, ...]], ...]
```

`play` returns the seat's move. `q` lists every legal tile in ascending
order, each with the probability that the bidding side makes its contract
if that tile is played. `analyze` returns the same move and adds each legal
tile's full tail at every target, which is what a hint needs. When only one
tile is legal, nothing is planned: `forced` is true, and `q` and `pmake` are
empty.

A position request also takes `worlds` and `seed`, as `forecast` does, and
`reading` (default `true`; see *Reading the plays* below). `reading: false`
plans over deals uniform on what the table has shown, as 0.2.0 did.

`hand` is what the seat was *dealt*. Its own plays are taken off as the
record is replayed. Every play is checked during the replay: the right seat
in turn, a tile not yet played, the seat's own tiles following suit, no
other seat playing a tile this seat holds, and no seat following a suit it
has already shown it lacks. A request that fails any of these throws an
`Error` that says which play is wrong and why.

## Plans and latency

Under a pip trump, the live seat plans alone against the field net. In
`scenario/` notation that is `walt:k0,n<worlds>,F=<net>`, with the cell
floor at 16 (applied only while the deals are a sample; once every deal
consistent with the play fits in `worlds`, the plan is exact), and it is the cheap player that the 250,000-deal pour was
poured with. The net only knows pip trumps. Under doubles and no-trump, the
other seats are instead modeled by a smaller Walt one rung down
(`walt:k1,n<worlds>,m8`), which costs several times more per declaration.
`worlds` (1..1024, default 160) is the number of deals planned over.
Raising it costs wall time and reduces sampling noise.

Measured in Node 24 on an Apple M5 Max (one thread, median and slowest of
nine opening hands; `npm run bench`):

| call | worlds 160 | worlds 512 |
|---|---|---|
| `forecast`, 7 pip declarations | 301 ms (393 ms) | 1.16 s (1.75 s) |
| `forecast`, doubles-trump alone | 298 ms (344 ms) | 1.01 s (1.43 s) |
| `forecast`, no-trump alone | 233 ms (272 ms) | 0.75 s (0.99 s) |
| `play`, an opening lead, sixes | 44 ms (57 ms) | 154 ms (221 ms) |

A full default forecast covers all nine declarations, so at 160 worlds it
takes about 0.8 s. Later plays get cheaper as the hand shrinks. (Measured on
0.2.0; 0.3.0's defaults leave these calls' cost unchanged, since no play has
been made to read. Reading's own cost is below.)

## Reading the plays, and ties

Two defaults arrived in 0.3.0.

**Reading.** Under a pip trump, `play` and `analyze` draw their deals from a
posterior over the other seats' plays instead of uniformly
(`d=posterior,r=uniform:4194304:0.28` in `scenario/` notation). Each play of
one of n legal tiles weighs ε/n + (1 − ε)/|A| if it lies in A, the legal
tiles the field net scores within 2 points of its best, and ε/n otherwise.
ε = 0.28 was fitted to Zeb's own play, so no deal is ever ruled out. Doubles
and no-trump are not read, because the net does not play them. A forecast is
unchanged, because it plans before any play. `reading: false` turns it off.

**Ties.** Among the tiles tied exactly for the best `q`, Zeb plays the one
with the best mean final margin under its plan (the bidding side the
largest, a defender the smallest), then the lowest id (`t=margin`). It never
trades an estimated make for points. It applies under every declaration.

The evidence, all against 0.2.0's plan in the arena (`bid30` bidders, games
to 7 marks, each deal played from both sides, 95% intervals over the paired
games):

| change | marks a game | other |
|---|---|---|
| reading alone, n512, a pre-registered replication on fresh deals | +0.37 [+0.24, +0.51] | +14.7 settled points a game |
| ties alone, n512 | +0.01 [−0.04, +0.07] | +4.55 points a hand |
| **both, n160 (0.3.0 as shipped), pre-registered** | **+0.21 [+0.08, +0.34]** | A/A control +0.06 [−0.07, +0.18] |

The write-ups are in the repository's wiki:
[reading](https://github.com/jasonyandell/mk5-main/blob/main/wiki/experiments/reads-plays-soft-2026-10-14.md),
[ties](https://github.com/jasonyandell/mk5-main/blob/main/wiki/experiments/tie-margin-2026-10-14.md)
and [the 0.3.0 confirmation](https://github.com/jasonyandell/mk5-main/blob/main/wiki/experiments/zeb-030-confirmation-2026-10-14.md).

Reading costs the most at tricks 1 to 3, where the posterior is drawn by
rejection. CPU milliseconds per `play` under sixes in Node 24 on an Apple M5
Max, median / 90th percentile, with reading on and off, over the nine
opening hands and every unforced decision of six deals (`npm run bench`).
The machine was busy with other work, which inflates every cell (the
opening lead took 70 ms here against 44 ms quiet), so read the ratios:

| | 160 worlds, on | off | 512 worlds, on | off |
|---|---|---|---|---|
| opening lead | 70 / 83 | 71 / 84 | 266 / 346 | 264 / 365 |
| trick 1 | 72 / 156 | 53 / 113 | 204 / 476 | 162 / 405 |
| trick 2 | 72 / 171 | 39 / 64 | 216 / 372 | 132 / 296 |
| trick 3 | 78 / 193 | 31 / 59 | 162 / 286 | 96 / 188 |
| trick 4 | 19 / 40 | 16 / 29 | 36 / 99 | 41 / 95 |
| trick 5 | 3 / 14 | 2 / 11 | 3 / 24 | 2 / 23 |
| trick 6 | 1 / 2 | 0 / 1 | 1 / 2 | 0 / 1 |

The opening lead costs the same either way. At tricks 1 to 3 reading costs
1.3 to 2.5 times the unread median. Scaled to a quiet machine by the
opening lead, a trick-2 or trick-3 decision at 160 worlds is about 50 ms
median and 120 ms at the 90th percentile.

## Determinism and receipts

An answer is a pure function of the request. Together with `version()`,
`netId()`, the `seed` (default 0; a number up to 2^53 − 1 or any bigint
below 2^64) and `reading`, the request is the receipt: the same receipt
gives the same answer, bit for bit, on every platform. No clock, thread or
ambient randomness enters the answer. It is also the answer that the Python
referee, `scenario/planner.py`, gives under the same plan. The package's
tests hold the WebAssembly to the referee to the last bit, on 38 positions
across every declaration (three with reading off), for each scenario's
weight and final margin. `netId()` is the
sha256 of the embedded net, `net/margin_d250k_h128x64.fnet`.

## Provenance

The pour came first. The cheap player planned 250,000 deals of self-play,
which gave 4.34 million positions, each with every legal move's outcome in
every sampled deal. The pour is published, CC0, as
[jasonyandell/texas-42-zeb-pour](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour)
(the first 50,000 deals) and
[jasonyandell/texas-42-zeb-pour-200k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-200k)
(the next 200,000). The net was trained on the pour (`scenario/field_train.py`,
margin target, 343 → 128 → 64 → 28, int16 weights). Then the net was
exported into this package (`scripts/export_net.py`). With that net as its
field, the cheap player is level with the two-rung champion
(`walt:k1,n512,m8`): +0.11 marks a game, with a 95% interval of
[−0.19, +0.41] over 384 mirrored games, at about a quarter of the cost.

## Building from source

The package lives at `zeb/` in [mk5-main](https://github.com/jasonyandell/mk5-main).
It needs rustup and wasm-pack 0.15.0 exactly (`build.sh` checks: wasm-pack
picks the wasm-opt that shapes the bytes). The compiler is pinned in
`rust-toolchain.toml`.

```sh
npm run build   # scripts/build.sh: checkout -> pkg/, the same bytes every time
npm test        # builds, then node --test: parity, determinism, errors, chooseBid
cargo test      # the same player natively, down to every scenario's outcome
```

`python -u -m zeb.scripts.make_fixtures` (run from the repository root)
rewrites the parity fixtures from the Python referee.
`python -u -m zeb.scripts.export_net NET.npz` exports a trained net.

## License

MIT (`LICENSE`), by Jason Yandell. Walt, the planner underneath, is CC0 in the
[source repository](https://github.com/jasonyandell/mk5-main), and so are the
field net's weights and the pour they were trained on. The code is free (CC0 / MIT). The names Walt, Zeb,
and Plunge are not licensed as the name of your product.

## Powered by Walt

![Powered by Walt](https://raw.githubusercontent.com/jasonyandell/mk5-main/main/docs/powered-by-walt.svg)

If Zeb plays in your game, a credit line is welcome and never required:

```
Powered by Walt — Jason Yandell's scenario player for Texas 42
```

Shipping something with Zeb? [Say hi](https://github.com/jasonyandell/mk5-main/issues).
