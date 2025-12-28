import { writable } from 'svelte/store';
import type { SeedProgress } from '../game/ai/gameSimulator';

interface SeedFinderState {
  isSearching: boolean;
  currentSeed: number;
  seedsTried: number;
  gamesPlayed: number;
  totalGames: number;
  currentWinRate: number;
  cancelled: boolean;
  useBest: boolean;
  bestSeed: number | null;
  bestWinRate: number;
  foundSeed: number | null;
  foundWinRate: number;
  waitingForConfirmation: boolean;
}

const INITIAL_STATE: SeedFinderState = {
  isSearching: false,
  currentSeed: 0,
  seedsTried: 0,
  gamesPlayed: 0,
  totalGames: 100,
  currentWinRate: 0,
  cancelled: false,
  useBest: false,
  bestSeed: null,
  bestWinRate: 1.0,
  foundSeed: null,
  foundWinRate: 0,
  waitingForConfirmation: false
};

function createSeedFinderStore() {
  const { subscribe, set, update } = writable<SeedFinderState>(INITIAL_STATE);

  return {
    subscribe,

    startSearch: () => {
      set({
        ...INITIAL_STATE,
        isSearching: true
      });
    },
    
    updateProgress: (progress: SeedProgress) => {
      update(state => ({
        ...state,
        currentSeed: progress.currentSeed,
        seedsTried: progress.seedsTried,
        gamesPlayed: progress.gamesPlayed,
        totalGames: progress.totalGames,
        currentWinRate: progress.currentWinRate,
        bestSeed: progress.bestSeed || state.bestSeed,
        bestWinRate: progress.bestWinRate || state.bestWinRate
      }));
    },
    
    stopSearch: () => {
      update(state => ({
        ...state,
        isSearching: false
      }));
    },
    
    cancel: () => {
      update(state => ({
        ...state,
        cancelled: true,
        isSearching: false,
        waitingForConfirmation: false
      }));
    },

    useBest: () => {
      update(state => ({
        ...state,
        useBest: true,
        cancelled: true,  // Stop current evaluation immediately
        isSearching: false
      }));
    },
    
    setSeedFound: (seed: number, winRate: number) => {
      update(state => ({
        ...state,
        isSearching: false,
        foundSeed: seed,
        foundWinRate: winRate,
        waitingForConfirmation: true
      }));
    },
    
    clearFoundSeed: () => {
      update(state => ({
        ...state,
        foundSeed: null,
        foundWinRate: 0,
        waitingForConfirmation: false
      }));
    },
    
    confirmSeed: () => {
      update(state => ({
        ...state,
        waitingForConfirmation: false
      }));
    }
  };
}

export const seedFinderStore = createSeedFinderStore();