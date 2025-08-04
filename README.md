# Texas 42

A TypeScript implementation of the classic Texas domino game with tournament-compliant rules and comprehensive testing.

## Features

### Core Game Engine
- **Pure functional state machine** with immutable state transitions
- **Comprehensive rule validation** including tournament compliance
- **Clean separation** between game logic and UI concerns
- **Full TypeScript implementation** for compile-time safety

### Professional Development Experience
- **Modern tooling**: TypeScript + Svelte 5 + Vite
- **Comprehensive testing**: Unit tests with Vitest + E2E tests with Playwright
- **Performance optimizations**: Hardware acceleration, CSS containment
- **Accessibility features**: Reduced motion support, proper ARIA attributes

### Testing & Debug Infrastructure
- **Comprehensive test helper** with state injection utilities
- **Automatic bug report generation** with reproducible test cases
- **Tournament compliance validation** suite
- **Mathematical constant verification** (42-point system)
- **URL parameter state injection** for debugging specific scenarios
- **Debug UI** for real-time game state inspection

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

## Architecture

### Game State Management
The game uses a pure functional state machine where:
- All state is immutable
- State transitions are explicit and predictable
- `getNextStates()` returns all valid actions as state descriptors
- No side effects in game logic

```typescript
import { createInitialState, getNextStates } from './game';

const state = createInitialState();
const availableActions = getNextStates(state);
// Execute action to get new state
const newState = availableActions[0].newState;
```

### UI Components
Built with Svelte 5 using modern runes syntax:
- Reactive stores for game state
- Performance-optimized components with CSS containment
- Responsive design for all screen sizes
- Hardware-accelerated rendering

### Testing Strategy
Comprehensive testing at multiple levels:

1. **Unit Tests** - Pure function testing of game logic
2. **Integration Tests** - Component interaction testing
3. **E2E Tests** - Full gameplay scenarios
4. **Property-based Testing** - Rule validation across state space

## Game Rules

Implements official Texas 42 tournament rules:

### Bidding
- Opening bids: 30-41 points or 1-2 marks
- Progressive bidding with 42-point equivalency
- Tournament mode: No special contracts (Nello, Splash, Plunge)
- Mark progression: 3+ marks only after 2-mark bid

### Gameplay
- 7 tricks per hand, 7 dominoes per player
- Must follow suit when possible

### Scoring
- Counting dominoes: 5-5 (10), 6-4 (10), 5-0 (5), 4-1 (5), 3-2 (5)
- Each trick worth 1 point (7 total) + counting dominoes (35 total) = 42 points
- First team to 7 marks wins
- Failed bids award marks to opponents

## Development

### Project Structure
```
src/
├── game/              # Pure game logic
│   ├── core/         # State management & rules
│   ├── types.ts      # TypeScript definitions
│   └── constants.ts  # Game constants
├── ui/               # Svelte components
│   ├── components/   # Reusable UI components
│   ├── stores/       # Reactive state stores
│   └── styles/       # Performance-optimized CSS
├── debug/            # Debug UI components (display-only)
└── tests/            # Comprehensive test suite
    ├── unit/         # Pure function tests
    ├── e2e/          # End-to-end scenarios
    └── helpers/      # Test utilities
```

### Debug Features
- **Debug Panel**: Real-time state inspection (read-only display)
- **State Injection**: Load specific game scenarios for testing
- **Bug Reports**: Automatic generation with reproduction steps
- **Rule Validation**: Live validation of game state integrity
- **Previous Tricks Display**: Shows completed tricks with led suit information

### Performance Features
- **CSS Containment**: Optimized rendering with `contain` property
- **Hardware Acceleration**: GPU-accelerated animations
- **Code Splitting**: Lazy loading for optimal bundle size
- **Reduced Motion**: Respects user accessibility preferences

## Testing

### Unit Tests
```bash
npm test
# Run specific test file
npm test src/tests/unit/bidding.test.ts
```

### E2E Tests
```bash
npm run test:e2e
# Run in headed mode
npm run test:e2e -- --headed
# Run specific test
npm run test:e2e tests/basic-gameplay.spec.ts
```

### Test Utilities
The `GameTestHelper` class provides extensive utilities:

```typescript
import { GameTestHelper } from './tests/helpers/gameTestHelper';

// Create specific game scenarios
const biddingState = GameTestHelper.createBiddingScenario(0, []);
const playingState = GameTestHelper.createPlayingScenario(3, 0);

// Validate game rules
const errors = GameTestHelper.validateGameRules(state);

// Generate bug reports
const report = GameTestHelper.generateBugReport(
  state, 
  "Expected behavior",
  "Actual behavior", 
  ["Step 1", "Step 2"]
);
```

## Contributing

1. Follow existing code patterns and conventions
2. Ensure all tests pass: `npm test && npm run test:e2e`
3. Validate TypeScript: `npm run check`
4. Format code: `npm run format`

## License

MIT License - See LICENSE file for details.

---

Built with a focus on correctness, testability, and adherence to official Texas 42 tournament rules.