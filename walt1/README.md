<!--
SPDX-FileCopyrightText: Jason Yandell
SPDX-License-Identifier: CC0-1.0
-->

# Walt

Walt plays Texas 42. To pick a tile, a seat samples deals of the tiles it
cannot see that fit everything the table has shown: the plays, and the
suits each seat has failed to follow. In each deal it gives the other three
seats a simple mind of their own: a player who sees only its own hand,
imagines a few deals itself, and plays its best tile against opponents who
play at random. Then the seat plans one line of play across all of its
deals at once and plays the plan's first tile. The plan is contingent: the
seat chooses once for every deal it cannot tell apart, so it never gets
credit for knowing which deal is the real one. **The planner never sees
another seat's hand.**

There is no neural network and nothing to download but 85 KB of
WebAssembly. Walt is an algorithm, written down in full in
[WALT.md](./WALT.md), and this package is one implementation of that
specification, built from the document alone.

```sh
npm install @jasonyandell/walt
```

Walt plays the hand; it does not bid. For a stronger player that also bids,
see [`@jasonyandell/zeb`](https://www.npmjs.com/package/@jasonyandell/zeb),
which is built on Walt's planner. `play` takes the same request and returns
the same move in both packages, so a game can seat either one.

## Speed

Over the 112 unforced decisions of six full hands, in Node on an Apple M
series laptop (`npm run bench`): **mean 24 ms, median 4 ms, 90% under
75 ms, slowest 395 ms**. The slow decisions are the first two tricks. A
forced tile returns at once. The call is synchronous, so in a browser put
it in a Web Worker if a 400 ms pause would show.

## Loading

```js
import init, { play } from '@jasonyandell/walt';

await init(); // fetches walt_bg.wasm from beside the glue
```

Some bundlers pre-bundle dependencies in a way that loses the
`import.meta.url` the default loader resolves against. In that case, hand
`init` the file's URL yourself. With Vite:

```js
import init from '@jasonyandell/walt';
import wasmUrl from '@jasonyandell/walt/walt_bg.wasm?url';

await init({ module_or_path: wasmUrl });
```

In Node:

```js
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { initSync } from '@jasonyandell/walt';

initSync({ module: readFileSync(fileURLToPath(import.meta.resolve('@jasonyandell/walt/walt_bg.wasm'))) });
```

## Conventions

- **Tiles** are ids 0 to 27. The tile with high pip `h` and low pip `l` is
  `h * (h + 1) / 2 + l`: the double blank is 0, the 1-0 is 1, the double
  one is 2, the double six is 27.
- **Seats** are 0 to 3 around the table; 0 and 2 are partners.
- **Declarations**: 0 to 6 are the pip trumps, blanks to sixes; 7 is
  doubles as trumps; 9 is no trump (follow-me, doubles high). 8 is not
  used.
- **The bid** is 30 to 42: the points the bidding side needs.

## Playing a tile

```js
const move = play({
  decl: 6,                           // sixes are trumps
  bidder: 0,                         // seat 0 won the bid and leads
  bid: 35,
  seat: 0,                           // the seat to play; must be the seat to act
  hand: [1, 6, 7, 8, 18, 19, 26],    // the seat's seven dealt tiles
  plays: [],                         // every play so far, in order: [seat, tile]
});
// {
//   tile: 19,
//   forced: false,
//   q: [[1, 0.225], [6, 0.3], [7, 0.375], [8, 0.275], [18, 0.375], [19, 0.425], [26, 0.325]],
//   leader: 0,
//   points: [0, 0],
// }
```

`q` gives, for each legal tile, the estimated chance that **the bidding
side** makes its bid if the seat plays that tile. The bidding side plays
the largest; the defense plays the smallest. `leader` is who led the
current trick and `points` is each side's count so far, so a game can
check its own bookkeeping against Walt's.

`hand` is always the seven tiles the seat was dealt, including the ones it
has already played. Walt works out what is left from `plays`.

Three optional knobs:

| Field | Default | Meaning |
| --- | --- | --- |
| `worlds` | 40 | Deals the seat samples. When the best tiles tie, Walt resamples them at 4 times and then 16 times this number. |
| `minds` | 8 | Deals each of the other seats' minds imagines. |
| `seed` | fixed | A number or bigint below 2^64. The same request with the same seed is the same move, on every machine. Vary it between decisions if you want Walt's play to vary. |

A request that is not a lawful position (a tile played twice, a seat out of
turn, a play that failed to follow suit from the seat's own hand) throws an
`Error` that names the play. A value of the wrong type or range throws a
`TypeError` or `RangeError` before Walt runs.

## The text protocol

Underneath, the package speaks the `walt1 play` request of WALT.md: a few
lines of text in, one line of JSON out. `requestText(request)` builds the
text and `ask(text)` sends it, which is useful for checking another
implementation against this one:

```js
import { ask, requestText } from '@jasonyandell/walt';

ask(requestText({ decl: 6, bidder: 0, bid: 35, seat: 0, hand: [1, 6, 7, 8, 18, 19, 26] }));
// { v: 1, kind: 'play', choice: 19, forced: false, raced: false,
//   opts: [[1, 2250], [6, 3000], ...], leader: 0, points: [0, 0] }
```

## How good is it

Walt is a quick, honest opponent that is fun to play against. It is about
as strong as a player that solves sampled deals with every hand face up,
without that player's habit of assuming it will know the cards later. It is
not the strongest player here; Zeb is.

WALT.md is the definition, and `conformance/` in the repository holds the
grader that any implementation can be run against. This build answers every
exact field of the 168-request corpus correctly and picks the same tile as
the reference implementation on 72% of the unforced decisions; the
reference agrees with itself, run on a different seed, on 74%. Sampled
estimates differ between implementations by design, so the specification
does not ask for identical choices.

## License

[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). Walt, its
specification and this code are in the public domain. Use them for
anything.
