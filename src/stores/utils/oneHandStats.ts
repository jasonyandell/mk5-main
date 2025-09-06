// One Hand attempts tracking per seed (persisted)
const ONE_HAND_STATS_KEY = 'oneHandStatsV1';

type OneHandStats = Record<string, { attempts: number; lastWinAttempts?: number }>;

function loadOneHandStats(): OneHandStats {
  try {
    if (typeof window !== 'undefined') {
      const raw = window.localStorage.getItem(ONE_HAND_STATS_KEY);
      if (raw) return JSON.parse(raw) as OneHandStats;
    }
  } catch {
    // Ignore localStorage errors
  }
  return {};
}

function saveOneHandStats(stats: OneHandStats) {
  try {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(ONE_HAND_STATS_KEY, JSON.stringify(stats));
    }
  } catch {
    // Ignore localStorage errors
  }
}

let oneHandStats: OneHandStats = loadOneHandStats();

export function getAttempts(seed: number): number {
  const key = String(seed);
  return oneHandStats[key]?.attempts ?? 0;
}

export function incrementAttempts(seed: number): number {
  const key = String(seed);
  const current = oneHandStats[key]?.attempts ?? 0;
  const next = current + 1;
  oneHandStats[key] = { ...(oneHandStats[key] || {}), attempts: next };
  saveOneHandStats(oneHandStats);
  return next;
}

export function recordWin(seed: number): number {
  const key = String(seed);
  const attempts = oneHandStats[key]?.attempts ?? 0;
  oneHandStats[key] = { attempts: 0, lastWinAttempts: attempts };
  saveOneHandStats(oneHandStats);
  return attempts;
}