# Texas 42

Web implementation of Texas 42 dominoes game with complete rule enforcement and scoring.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Run E2E tests
npm run test:e2e

# Build for production
npm run build
```

## Tech Stack

- **Frontend**: Svelte 5 + TypeScript + Vite + Tailwind CSS
- **Architecture**: Event sourcing with immutable state
- **Testing**: Vitest (unit) + Playwright (E2E)
- **Mobile-first**: Responsive design optimized for touch

## Game Rules

Tournament-compliant Texas 42:
- **Bidding**: 30-42 points or marks (1 mark = 42 points)
- **Scoring**: 42 points per hand (7 tricks + 35 counting dominoes)
- **Winning**: First team to 7 marks

## Project Structure

```
src/
├── game/           # Core game logic (types, state machine)
├── stores/         # Reactive state management
├── components/     # UI components
└── tests/          # Test suite
```

## Testing

```bash
npm test              # Unit tests
npm run test:e2e      # E2E tests  
npm run check         # TypeScript validation
```

### Debug Tools

- **URL Replay**: `node scripts/replay-from-url.js "<url>" --generate-test`
- **State Injection**: Load specific game scenarios via URL parameters
- **Test Helpers**: Comprehensive utilities in `src/tests/e2e/helpers/`

## License

MIT