import { writable } from 'svelte/store';
import type { SeedProgress } from '../game/core/seedFinder';

interface SeedFinderState {
  isSearching: boolean;
  currentSeed: number;
  seedsTried: number;
  gamesPlayed: number;
  totalGames: number;
  currentWinRate: number;
  cancelled: boolean;
  foundSeed: number | null;
  foundWinRate: number;
  waitingForConfirmation: boolean;
}

function createSeedFinderStore() {
  const { subscribe, set, update } = writable<SeedFinderState>({
    isSearching: false,
    currentSeed: 0,
    seedsTried: 0,
    gamesPlayed: 0,
    totalGames: 100,
    currentWinRate: 0,
    cancelled: false,
    foundSeed: null,
    foundWinRate: 0,
    waitingForConfirmation: false
  });

  return {
    subscribe,
    
    startSearch: () => {
      set({
        isSearching: true,
        currentSeed: 0,
        seedsTried: 0,
        gamesPlayed: 0,
        totalGames: 100,
        currentWinRate: 0,
        cancelled: false,
        foundSeed: null,
        foundWinRate: 0,
        waitingForConfirmation: false
      });
    },
    
    updateProgress: (progress: SeedProgress) => {
      update(state => ({
        ...state,
        currentSeed: progress.currentSeed,
        seedsTried: progress.seedsTried,
        gamesPlayed: progress.gamesPlayed,
        totalGames: progress.totalGames,
        currentWinRate: progress.currentWinRate
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
    },
    
    isCancelled: (): boolean => {
      let cancelled = false;
      const unsubscribe = subscribe(state => {
        cancelled = state.cancelled;
      });
      unsubscribe();
      return cancelled;
    },
    
    getFoundSeed: (): number | null => {
      let seed: number | null = null;
      const unsubscribe = subscribe(state => {
        seed = state.foundSeed;
      });
      unsubscribe();
      return seed;
    }
  };
}

export const seedFinderStore = createSeedFinderStore();