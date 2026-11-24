# Texas 42

Free, open tools and research for the Texas domino game 42: computer players that never look
at another seat's hand, a language for asking exact questions about a position, a public
ladder, and a long research log on what it takes to learn the game. Everything runs offline.
Built on weekends for the love of the game. CC0 / MIT.

Sister repositories: [texas-42](https://github.com/jasonyandell/texas-42) (the mathematics of
42 as an imperfect-information game, proved in Lean) ·
[plunge](https://github.com/jasonyandell/plunge) (the free mobile web game) ·
[the-table](https://github.com/jasonyandell/the-table) (where players from different projects
meet).

## Building a 42 game? Start here

| You want | Take | Where |
|----------|------|-------|
| A computer partner or opponent | **Zeb**: bids, plays and explains a play from one seat's seven tiles and the public record. WebAssembly, for browsers, Web Workers and Node. MIT. | `npm install @jasonyandell/zeb` · [`zeb/`](zeb/README.md) |
| The algorithm, to write your own | **Walt**, the scenario player under Zeb: sample the deals the seat cannot tell apart, plan one line of play across all of them, play its first move. CC0. | [`scenario/`](scenario/) · [scenario-player](wiki/entities/scenario-player.md) |
| Exact answers about a position | **Silas**: a typed query language over the hidden deals. The four-fives 84 of *Winning 42* p. 113 comes out as exactly 33/95. CC0. | [`silas/`](silas/README.md) |
| To measure your own player | **The ladder**: an append-only, Bradley–Terry-rated record of every player here, and a one-line-per-decision seat protocol for bringing yours. | [tournament](wiki/topics/tournament.md) · [records](https://huggingface.co/datasets/jasonyandell/texas-42-tournament) · [the-table](https://github.com/jasonyandell/the-table) |
| The rules, written down completely | Straight 42 and the family variants. | [rules-of-42](wiki/topics/rules-of-42.md) |
| Data and trained nets | Every dataset and model the project has generated, public. | [huggingface-assets](wiki/topics/huggingface-assets.md) |
| To see it played | A free mobile web game, no login. | [plunge](https://github.com/jasonyandell/plunge) |

## What is being worked on

The north star is to learn 42 as completely as it can be learned: policies, beliefs and
dynamics, from players that only ever see what a seat can see. Several streams run side by
side, each with its own pages in the [wiki](wiki/index.md) and its data on
[Hugging Face](wiki/topics/huggingface-assets.md).

**The players: Walt and Zeb.** Walt samples the deals a seat cannot tell apart and plans one
line across all of them; Zeb is that player with a small integer field net playing the other
seats, compiled to WebAssembly. Current work is Walt bidding for itself
([walt-bidder](wiki/experiments/walt-bidder-2026-11-02.md)).
[`scenario/`](scenario/) · [`zeb/`](zeb/README.md) · [zeb](wiki/entities/zeb.md)

**Silas and the exact player.** Silas is a typed relational language over the worlds a seat
cannot tell apart, in Rust. On top of it sits an exact information-state player: under a fixed
model of how the other three play, what is the best move here, computed exactly rather than
sampled. `silas-metal/` moves the counting onto Apple GPUs, byte-equal to the CPU; the
trick-1 ladder is the current frontier.
[`silas/`](silas/README.md) · [`silas-metal/`](silas-metal/) ·
[silas](wiki/entities/silas.md) · [exact-gpu](wiki/experiments/exact-gpu-2026-10-22.md) ·
[trick-1 ladder](wiki/experiments/trick1-ladder-2026-10-22.md)

**Lem: interpretability.** Can a net that plays 42 well be read? Lem trains small nets and
reads them: neuron audits of Zeb's field net, a field net written by hand where every weight
is a rule, and blind weight reads with predictions sealed before grading. Negative results are
kept and published.
[`lem/`](lem/) · [lem-the-teacher](wiki/topics/lem-the-teacher.md) ·
[blind read](wiki/experiments/lem-blind-read-2026-11-03.md) ·
[hand-written net](wiki/experiments/lem-hand-written-field-net-2026-11-02.md)

**The Book.** Dennis Roberson's *Winning 42* is treated as the authority: when the book's
advice fails to measure as better, the measurement is the suspect. This stream turns the
book's claims into detectors and played-out lines, and covers the assumptions the book leaves
implicit. [`w42/`](w42/README.md) · [book-authority](wiki/topics/book-authority.md) ·
[book-assumptions](wiki/topics/book-assumptions.md)

**The ladder and the Demon.** Every player here is seated on an append-only, Bradley–Terry
ladder with paired deals and pre-registered trials. The Demon is a CPU sampling player that
is allowed to reason per-deal; it is fenced off as a measuring stick and never becomes part
of Walt, Zeb or Silas ([the fence](wiki/topics/strategy-fusion-fence.md)).
[`tournament/`](tournament/) · [tournament](wiki/topics/tournament.md) ·
[the-demon](wiki/topics/the-demon.md)

**History.** Earlier eras live on in the tree and the wiki: a GPU oracle solver and E[Q]
labels (`forge/`), the Gus transformer (`gus/`), the Champion and Jud belief-state players
(`champion/`, `arena/`), Otis (`otis/`), and LLM players on small open models (`burl/`). Start
at [the-wall](wiki/topics/the-wall.md) for the central question, or [the log](wiki/log.md)
for what happened when.

## Repository map

```
src/            TypeScript game engine: pure core, rule layers, multiplayer, Svelte UI
scenario/       Walt, the scenario player (CC0) and Zeb's layer on it (MIT)
zeb/            Zeb: Rust → WebAssembly, published as @jasonyandell/zeb
silas/          Silas: the query language and the exact player (Rust)
silas-metal/    Silas counting on Apple GPUs (Rust + Metal)
lem/            Interpretability: probes, audits, hand-written nets
w42/            Winning 42 claim validation
tournament/     The ladder, seats, the fenced Demon
arena/          Full-game evaluation (auctions, marks to 7)
walt/ forge/ gus/ champion/ otis/ burl/   earlier research lines
wiki/           The knowledge wiki: start at wiki/index.md
docs/           Operational references and dated evidence
```

## The engine

The game engine is built around one transformation, `STATE → ACTION → NEW STATE`.

- **Event sourcing.** `state = replayActions(config, history)`. Every game is replayable,
  shareable by URL, and deterministically testable.
- **Rule layers.** Composable layers supply both execution rules and legal actions: Nello
  overrides trick completion for three players, Tournament filters special bids. Executors
  hold no variant conditionals.
- **Capability-based multiplayer.** Permission tokens (`act-as-player`, `observe-hands`)
  instead of identity checks; the client receives a pre-computed view and imports no game
  logic, and guardrail tests check no hidden state leaks to it.

Deep dives: [engine-architecture](wiki/topics/engine-architecture.md) ·
[layer-system](wiki/topics/layer-system.md) ·
[multiplayer-pattern](wiki/topics/multiplayer-pattern.md) ·
[suit-algebra-spec](wiki/topics/suit-algebra-spec.md)

```bash
npm install
npm run dev        # development server
npm test           # unit tests (Vitest)
npm run test:e2e   # Playwright
npm run test:all   # typecheck, lint, everything
cargo test --manifest-path silas/Cargo.toml   # Silas
```

**Stack:** TypeScript, Svelte 5, Vite · Rust, WebAssembly, Metal · Python, PyTorch ·
Hugging Face for data and models · Lean next door in
[texas-42](https://github.com/jasonyandell/texas-42).

## License

The science is public domain (CC0 1.0): the engine, Walt (the scenario player), Silas, the
arena, the wiki, the data and everything else, by default. Zeb, the player built on Walt
(`zeb/` and Zeb's layer in `scenario/`), is MIT. Each file says which in its SPDX header or in
[`REUSE.toml`](REUSE.toml); [`LICENSE`](LICENSE) draws the line in words.

The code is free (CC0 / MIT). The names Walt, Zeb, and Plunge are not licensed as the name of
your product.

## Powered by Walt

![Powered by Walt](docs/powered-by-walt.svg)

If you build on Walt, credit is welcome and never required. Copy and paste:

```
Powered by Walt — Jason Yandell's scenario player for Texas 42
```

The badge is [`docs/powered-by-walt.svg`](docs/powered-by-walt.svg) (CC0). Shipping something
with Walt? [Say hi](https://github.com/jasonyandell/mk5-main/issues).
