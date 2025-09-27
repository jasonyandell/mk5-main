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
    'no-case-declarations': 'error'
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
  files: ['**/*.svelte'],
  rules: {
    // Skip linting Svelte files for now
  }
}, {
  ignores: ['node_modules/', 'dist/', 'build/', '**/*.svelte']
}];