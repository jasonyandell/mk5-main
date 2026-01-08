import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';

export default [js.configs.recommended, {
  files: ['**/*.ts'],
  languageOptions: {
    parser: typescriptParser,
    parserOptions: {
      ecmaVersion: 2022,
      sourceType: 'module'
    },
    globals: {
      console: 'readonly',
      document: 'readonly',
      window: 'readonly',
      setTimeout: 'readonly',
      clearTimeout: 'readonly',
      queueMicrotask: 'readonly',
      CSSStyleRule: 'readonly',
      URLSearchParams: 'readonly',
      URL: 'readonly',
      atob: 'readonly',
      btoa: 'readonly',
      Buffer: 'readonly',
      Element: 'readonly',
      HTMLElement: 'readonly',
      requestAnimationFrame: 'readonly',
      cancelAnimationFrame: 'readonly',
      navigator: 'readonly',
      ShareData: 'readonly'
    }
  },
  plugins: {
    '@typescript-eslint': typescript
  },
  rules: {
    ...typescript.configs.recommended.rules,
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/no-explicit-any': 'error',
    'no-redeclare': 'error',
    'no-case-declarations': 'error',
    'no-restricted-imports': ['error', {
      paths: [{
        name: '../game/types/execution',
        importNames: ['createExecutionContext'],
        message: 'createExecutionContext is restricted. Use Room or HeadlessRoom for composition. Only allowed in: Room.ts, HeadlessRoom.ts, test helpers (src/tests/helpers/), and test files (*.test.ts, *.spec.ts).'
      }, {
        name: '../../game/types/execution',
        importNames: ['createExecutionContext'],
        message: 'createExecutionContext is restricted. Use Room or HeadlessRoom for composition. Only allowed in: Room.ts, HeadlessRoom.ts, test helpers (src/tests/helpers/), and test files (*.test.ts, *.spec.ts).'
      }, {
        name: '../../../game/types/execution',
        importNames: ['createExecutionContext'],
        message: 'createExecutionContext is restricted. Use Room or HeadlessRoom for composition. Only allowed in: Room.ts, HeadlessRoom.ts, test helpers (src/tests/helpers/), and test files (*.test.ts, *.spec.ts).'
      }],
      patterns: [{
        group: ['@/game/types/execution'],
        importNames: ['createExecutionContext'],
        message: 'createExecutionContext is restricted. Use Room or HeadlessRoom for composition. Only allowed in: Room.ts, HeadlessRoom.ts, test helpers (src/tests/helpers/), and test files (*.test.ts, *.spec.ts).'
      }]
    }]
  }
}, {
  files: ['**/*.test.ts', '**/*.spec.ts', '**/tests/**/*.ts'],
  languageOptions: {
    globals: {
      console: 'readonly',
      document: 'readonly',
      window: 'readonly',
      test: 'readonly',
      expect: 'readonly',
      describe: 'readonly',
      it: 'readonly',
      beforeEach: 'readonly',
      afterEach: 'readonly',
      URL: 'readonly',
      atob: 'readonly',
      btoa: 'readonly',
      Buffer: 'readonly',
      Element: 'readonly',
      HTMLElement: 'readonly'
    }
  }
}, {
  // Allow createExecutionContext in specific files
  files: [
    'src/server/Room.ts',
    'src/server/HeadlessRoom.ts',
    'src/tests/helpers/**/*.ts',
    '**/*.test.ts',
    '**/*.spec.ts'
  ],
  rules: {
    'no-restricted-imports': 'off'
  }
}, {
  files: ['**/*.svelte'],
  rules: {
    // Skip linting Svelte files for now
  }
}, {
  ignores: ['node_modules/', 'dist/', 'build/', '**/*.svelte']
}];