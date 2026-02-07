# Texas 42

A web implementation of the classic Texas trick-taking dominoes game — event-sourced architecture, composable layer system, capability-based multiplayer, and a ground-up ML pipeline that solves the game from perfect play to self-play reinforcement learning.

Built on weekends for the love of building. 8th major overhaul and counting.

## Quick Start

```bash
npm install
npm run dev       # Development server
npm test          # 1,045 unit tests
npm run test:e2e  # Playwright E2E tests
npm run typecheck # TypeScript validation
```

## Architecture

The entire system is built around one transformation:

```
STATE → ACTION → NEW STATE
```

**Event sourcing** — `state = replayActions(config, history)`. Actions are truth, state is derived. Every game is replayable, shareable via compressed URL, and deterministically testable.

**Unified Layer system** — 10 composable layers provide both execution rules and action generation on two orthogonal surfaces. Nello overrides `isTrickComplete` for 3-player tricks. Tournament filters special bids. Speed annotates forced moves with `autoExecute`. Zero conditional logic in executors — they delegate to `rules.method()` and trust the result.

**Capability-based multiplayer** — Permission tokens (`act-as-player`, `observe-hands`) replace identity checks. Server validates everything; the client is intentionally dumb — it receives a pre-computed `GameView` and displays it without importing game logic.

**Dumb client pattern** — `buildKernelView()` computes derived fields server-side. The client never imports from `src/game/core/`. 43-line `GameClient` class, fire-and-forget actions, updates via subscription.

For the full architecture deep-dive, see [docs/ORIENTATION.md](docs/ORIENTATION.md).

## Crystal Forge (ML Pipeline)

A 3-stage ML pipeline that attacks Texas 42 from perfect information to realistic play:

| Stage | What | Result |
|-------|------|--------|
| **Oracle** | GPU minimax solver — exact Q-values for every state (~50K per deal) | 215GB dataset, 11.24M training examples |
| **E[Q]** | Sample N opponent hands → query oracle → average under uncertainty | 12,325x GPU speedup, Bayesian posterior weighting |
| **Zeb** | AlphaZero-style MCTS self-play with oracle leaf evaluation | 1.5M games, 70% win rate in 5 days |

The recommended model (3.3M-parameter DominoTransformer) achieves **99.4% zero-regret rate** vs the oracle with Q-gap 0.074.

Distributed training runs on Vast.ai — 4 RTX 3060 workers at $0.05/hr, learner on a local 3050 Ti. An 8-hour session costs ~$2.88.

See [forge/README.md](forge/README.md) for the full pipeline narrative, or [forge/ORIENTATION.md](forge/ORIENTATION.md) for operational reference.

## Project Structure

```
src/
├── game/
│   ├── core/            Pure game engine (state machine, executors, rules)
│   ├── layers/          10 composable layers (nello, splash, plunge, sevens,
│   │                      tournament, oneHand, hints, speed, consensus, base)
│   ├── ai/              AI strategies, hand analysis, Monte Carlo, game simulator
│   ├── types.ts         Core types (GameState, GameAction, Domino, Bid)
│   └── view-projection.ts  Pure state → UI-ready data transformation
├── kernel/              Pure helpers (executeKernelAction, buildKernelView)
├── server/              Room orchestrator, HeadlessRoom, AI lifecycle
├── multiplayer/         Protocol, authorization, capabilities, GameClient
├── stores/              Svelte reactive state (gameStore)
├── lib/                 UI components (Svelte 5)
├── styles/              Tailwind/DaisyUI theming
└── tests/
    ├── unit/            Game logic tests
    ├── e2e/             Playwright browser tests
    ├── guardrails/      Architectural enforcement (no-bypass, projection security)
    └── helpers/         StateBuilder, deal constraints, test context

forge/
├── oracle/              Stage 1: GPU minimax solver (backward induction)
├── ml/                  PyTorch Lightning training (DominoTransformer, tokenizer)
├── eq/                  Stage 2: E[Q] imperfect-info pipeline
├── zeb/                 MCTS self-play + distributed Vast.ai workers
├── bidding/             Monte Carlo P(make) estimation
├── analysis/            25-module scientific analysis (98 Jupyter notebooks)
├── models/              Pre-trained model catalog (5 checkpoints)
├── flywheel/            Automated generate→tokenize→train→evaluate loop
└── cli/                 Command-line interfaces

docs/
├── ORIENTATION.md       Primary architecture guide (start here)
├── MULTIPLAYER.md       Socket/GameClient/Room pattern
├── rules.md             Official Texas 42 game rules
├── ARCHITECTURE_PRINCIPLES.md  Design philosophy
└── CONCEPTS.md          Complete implementation reference
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Svelte 5, TypeScript, Vite 7, Tailwind CSS 3, DaisyUI 4 |
| **Architecture** | Event sourcing, immutable state, pure functions |
| **Testing** | Vitest (1,045 unit tests), Playwright (E2E), architectural guardrails |
| **ML Training** | PyTorch, Lightning, HuggingFace Hub, Weights & Biases |
| **Compute** | Vast.ai (distributed self-play), Modal (cloud GPU), Lambda Labs |

## Testing

```bash
npm test              # Unit tests (1,045 across 99 files)
npm run test:e2e      # Playwright E2E tests
npm run typecheck     # TypeScript strict mode
npm run check         # Svelte type checking
npm run lint          # ESLint
npm run test:all      # Everything
```

Architectural guardrail tests enforce invariants automatically:
- **no-bypass** — prevents imports that skip the GameRules interface
- **projection-security** — verifies no hidden state leaks to clients
- **rule-contracts** — base + special contract conformance
- **no-backwards-compat** — no `@deprecated`, no legacy shims

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/ORIENTATION.md](docs/ORIENTATION.md) | Architecture deep-dive — start here |
| [docs/MULTIPLAYER.md](docs/MULTIPLAYER.md) | Multiplayer pattern (Socket/GameClient/Room) |
| [docs/rules.md](docs/rules.md) | Official Texas 42 game rules |
| [docs/ARCHITECTURE_PRINCIPLES.md](docs/ARCHITECTURE_PRINCIPLES.md) | Design philosophy and invariants |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Complete implementation reference |
| [forge/README.md](forge/README.md) | Crystal Forge ML pipeline |
| [forge/ORIENTATION.md](forge/ORIENTATION.md) | Forge operational reference (1,100 lines) |

## License

MIT
