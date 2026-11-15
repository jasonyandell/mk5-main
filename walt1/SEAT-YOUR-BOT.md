<!--
SPDX-FileCopyrightText: Jason Yandell
SPDX-License-Identifier: CC0-1.0
-->

# Seat your own bot

You have a Texas 42 bot and want to know how it plays against Walt or Zeb,
on enough hands to trust the answer. This page is the short path.

The idea is one small agreement. A player is any program that reads a
position and names a tile:

```
walt1 play
decl 6
bid 35
seat 0
bidder 0
hand 1 6 7 8 18 19 26
plays 0 19 1 16
```

```json
{"v":1,"kind":"play","choice":26,"forced":false,"leader":0,"points":[0,0]}
```

One request on stdin, one line of JSON on stdout, then exit. The request
holds the seat's own seven dealt tiles and the public plays, as
`seat tile` pairs in order. It never holds another seat's hand, so nothing
seated this way can peek. The full definition is in [WALT.md](./WALT.md).

## What a reply must get right

The referee checks every reply and stops the match on the first wrong one:

| Field | Must be |
| --- | --- |
| `choice` | a legal tile id for the seat |
| `forced` | `true` exactly when the seat has one legal tile |
| `leader` | the seat that led the current trick |
| `points` | each side's count so far, `[seats 0 and 2, seats 1 and 3]`, a point per trick included |

The last three are there to catch a bot whose rules differ from the
table's before a single game is scored. Walt also returns `opts`, its
estimate for each tile. Your bot does not have to.

The request may carry `n`, `n0` and `seed` lines. They are Walt's knobs.
Ignore them, or use `seed` as your random seed: a bot that answers the same
request the same way every time gives a match anyone can replay.

**No clocks.** Think on a fixed budget (samples, nodes), never on elapsed
time. A bot on a timer plays differently on a busy machine, and then the
result belongs to the machine.

## The conventions

- **Tiles** are ids 0 to 27. The tile with high pip `h` and low pip `l` is
  `h * (h + 1) / 2 + l`.
- **Seats** are 0 to 3 around the table; 0 and 2 are partners. The bidder
  leads the first trick.
- **`decl`**: 0 to 6 are the pip trumps, blanks to sixes. 7 is doubles as
  trumps. 9 is no trumps, where a double is the highest tile of its own
  suit.
- A led tile that is not a trump calls for its high pip.

Not covered: bidding, nello, and bids in marks. The seat plays the hand
only. The table gives both sides the same simple auction, so the match
compares card play and nothing else.

## A template

[`examples/seat-your-bot/bot.mjs`](./examples/seat-your-bot/bot.mjs) is a
complete seat in about 80 lines of Node with no dependencies. It parses the
request, replays the tricks, works out the legal tiles, and calls one
function:

```js
function choose(p) {
  return p.legal[0];   // your bot goes here
}
```

`p` holds the contract, the seat's remaining tiles, the legal ones, the
finished tricks, the trick in progress and the count. Wrap your bot there,
or use the file as a guide for a seat in any other language.

## Check it before you play

```sh
node conformance/grade.mjs examples/seat-your-bot/bot.mjs
```

This sends 140 recorded positions and compares the bookkeeping. You want
`errors`, `timeouts`, `badJson` and `illegalChoice` at 0, and `leader`,
`points` and `forced` at 100%. The template scores that as shipped.
`choiceMatch` says how often your bot picked the tile Walt picked. It is
not a grade: two good players disagree often. A bot that plays the lowest
legal tile matches Walt about half the time on this set.

## Play the match

The table is the tournament in this repository
([wiki/topics/tournament.md](../wiki/topics/tournament.md)). Each deal is
played twice with the sides swapped, so the luck of the tiles cancels.
Games go to seven marks. Every hand is bid at 30 by the seat left of the
shaker and played in that seat's best pip trump.

```sh
# register your seat and a Walt to play against (each is pinned by its sha256)
python -u -m tournament --store git:my-store add-walt1 mybot path/to/bot.mjs path/to/bot-dir
python -u -m tournament --store git:my-store add-walt1 walt walt1/target/release/walt-play walt1

# play: each run takes the next unplayed faceoffs; run several at once on a big machine
python -u -m tournament --store git:my-store work --ladder marks7-bid30 \
    --anchor walt1-walt-<sha8> --only walt1-walt-<sha8>,walt1-mybot-<sha8> --budget-s 1200

# read
python -u -m tournament --store git:my-store table --ladder marks7-bid30 --anchor walt1-walt-<sha8>
```

Today these commands live on the `bench/42-bench-ladder` branch, and the
setup is `scripts/check`'s (node, cargo, uv). That branch has not merged
yet. If you would rather not build any of it, send the seat and we will
run it.

Against Walt, a faceoff (two games) takes a minute or two on two cores.
Three hundred faceoffs is an hour or two on a laptop running six at once.

## Read the result

The table gives each player an Elo against the anchor at 1500, with a 95%
interval.

- **Decide when you will look before you start.** We use: read once, at
  the first check with 600 games or more. Looking early and stopping when
  the number is pleasing finds differences that are not there.
- **Read the interval, not the middle.** If the whole interval sits inside
  1500 ± 50, the two players are equivalent at this table. If it sits
  wholly above or below 1500, one is stronger. Anything else is "not
  known yet": play more.
- **Fifty games tell you almost nothing.** At 600 games the interval is
  still about ± 20 Elo wide for two players this close.

For scale, on this table Zeb's card play sits near 1586 [1552, 1621]
against Walt at 1500, and the public records are at
[jasonyandell/texas-42-bench](https://huggingface.co/datasets/jasonyandell/texas-42-bench).

## What the number does not say

It says how the two play the hand against each other, at a bid of 30, in
pip trumps, with partners who are copies of themselves. It says nothing
about bidding, about people, or about a table of mixed partners. A bot that
solves sampled deals with every hand face up will do well here and can
still make a habit of plays that only work when it knows the cards; Walt
is built not to. Both kinds are welcome at the table. That difference is
one of the things the table is for.
