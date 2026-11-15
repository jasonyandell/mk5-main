# Texas 42

Free, open tools for the Texas domino game 42: computer players that never look at another
seat's hand, a language for asking exact questions about a position, a public ladder, and the
research behind them. Everything runs offline. Built on weekends for the love of the game.

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

The proofs live next door: [texas-42](https://github.com/jasonyandell/texas-42) holds the
mathematics of 42 as an imperfect-information game, checked in Lean.

The rest of this page describes the repository itself: the TypeScript game engine and the
research history, including the finished oracle era.

## Quick Start

```bash
npm install
npm run dev       # Development server
npm test          # Unit tests (Vitest)
npm run test:e2e  # Playwright E2E tests
npm run typecheck # TypeScript validation
```

## The Knowledge Wiki

The project's full history and current state — every architecture decision, experiment,
and design concept — lives in a backlinked knowledge wiki under [`wiki/`](wiki/index.md).
Start at [the-wall](wiki/topics/the-wall.md) (the project's central question) or the hubs:
[texas-42](wiki/entities/texas-42.md) (the game itself),
[engine](wiki/entities/engine.md), [forge](wiki/entities/forge.md),
[champion](wiki/entities/champion.md).

## Architecture

The entire system is built around one transformation:

```
STATE → ACTION → NEW STATE
```

**Event sourcing** — `state = replayActions(config, history)`. Actions are truth, state is
derived. Every game is replayable, shareable via compressed URL, and deterministically
testable.

**Unified Layer system** — composable layers provide both execution rules and action
generation on two orthogonal surfaces. Nello overrides `isTrickComplete` for 3-player
tricks; Tournament filters special bids; Speed annotates forced moves. Zero conditional
logic in executors — they delegate to `rules.method()` and trust the result.

**Capability-based multiplayer** — permission tokens (`act-as-player`, `observe-hands`)
replace identity checks. The server validates everything; the client is intentionally dumb —
it receives a pre-computed `GameView` and displays it without importing game logic.

Deep dives: [engine-architecture](wiki/topics/engine-architecture.md) ·
[layer-system](wiki/topics/layer-system.md) ·
[multiplayer-pattern](wiki/topics/multiplayer-pattern.md) ·
[suit-algebra-spec](wiki/topics/suit-algebra-spec.md) (the algebra that makes the rules fast)

## The ML Pipeline

| Stage | What | Where |
|-------|------|-------|
| **Oracle** | GPU backward-induction solver — exact minimax values for the full play phase of any deal | `forge/oracle/`, [the-oracle](wiki/topics/the-oracle.md) |
| **E[Q]** | Expectation over sampled hidden worlds — value under imperfect information | `forge/eq/`, [expected-q-value](wiki/topics/expected-q-value.md) |
| **Gus** | Multi-head transformer distilled from the oracle: policy, value, belief | `gus/`, [gus](wiki/entities/gus.md) |
| **Champion** | The unification target: one belief-state player that bids and plays full games to 7 marks | `champion/`, `arena/`, [champion](wiki/entities/champion.md), [jud](wiki/entities/jud.md) |

The [w42](wiki/entities/w42.md) workstream validates the strategy claims of Dennis
Roberson's *Winning 42* against oracle ground truth.

## Project Structure

```
src/                 TypeScript game engine (pure core, layers, multiplayer, UI)
forge/               ML pipeline: oracle solver, E[Q], training, analysis, models
gus/                 Oracle-distillation student (multi-head transformer)
champion/            Unified belief-state player + evidence artifacts
arena/               Full-game evaluation harness (auctions, marks to 7)
w42/                 Winning 42 book-validation artifacts
wiki/                The knowledge wiki (start at wiki/index.md)
docs/                Operational references and dated evidence artifacts
scratch/             Gitignored scratch space
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Svelte 5, TypeScript, Vite, Tailwind CSS, DaisyUI |
| **Architecture** | Event sourcing, immutable state, pure functions |
| **Testing** | Vitest, Playwright, architectural guardrail tests |
| **ML** | PyTorch, Lightning, HuggingFace Hub, Weights & Biases |
| **Compute** | Apple Silicon (MPS), Modal (cloud GPU), Vast.ai |

## Testing

```bash
npm test              # Unit tests
npm run test:e2e      # Playwright E2E tests
npm run typecheck     # TypeScript strict mode
npm run check         # Svelte type checking
npm run lint          # ESLint
npm run test:all      # Everything
```

Architectural guardrail tests enforce invariants automatically — no-bypass (imports can't
skip the GameRules interface), projection-security (no hidden state leaks to clients),
rule-contracts, and no-backwards-compat (no `@deprecated`, no legacy shims — this is a
greenfield project and we keep it that way). See
[engine-testing-patterns](wiki/topics/engine-testing-patterns.md).

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
