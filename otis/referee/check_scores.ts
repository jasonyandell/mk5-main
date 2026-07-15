/**
 * Otis cross-engine referee (P1).
 *
 * Reads the fixed JSONL interchange format (one game per line) and replays each
 * game through the project's TypeScript engine (pure functional, event-sourced),
 * then emits one JSON line per game to stdout:
 *
 *   {"game_id": str, "team02_points": int, "team13_points": int}
 *
 * This tool is an INDEPENDENT referee: it never re-implements trick-winner or
 * point logic. It seeds the exact deal, runs a minimal auction so the given
 * bidder wins, declares trump, plays the 28 dominoes in order, and reads the
 * engine's own accumulated `teamScores`. Any illegal play, out-of-order seat,
 * or points that don't sum to 42 is a stop-the-line P1 failure and throws.
 *
 * Teams: {0,2} => teamId 0 (team02), {1,3} => teamId 1 (team13).
 *
 * Usage:
 *   npx tsx otis/referee/check_scores.ts <path-to.jsonl>
 */

import { readFileSync } from 'node:fs';

import type { Domino, TrumpSelection, GameAction, GameState } from '../../src/game/types';
import { createInitialState } from '../../src/game/core/state';
import { executeAction } from '../../src/game/core/actions';
import { composeRules, baseLayer } from '../../src/game/layers';
import type { Layer } from '../../src/game/layers';

// The forge eq-corpus is a FULL 28-play playout: every trick is played and all 42
// points are distributed across the seven tricks. The base engine, however,
// short-circuits to the scoring phase as soon as the bid outcome is
// mathematically decided (`checkHandOutcome`, src/game/core/handOutcome.ts) —
// after which it refuses further plays and `teamScores` holds only a PARTIAL
// count. To referee the corpus's full point split we must replay all seven
// tricks, so this layer overrides `checkHandOutcome` to never terminate early.
// The 7-trick terminal transition (src/game/core/actions.ts:327) is unaffected,
// so the hand still ends after all 28 dominoes are down. This changes only WHEN
// scoring happens, never HOW points are counted — the engine's own
// `calculateTrickWinner` / `calculateTrickPoints` still do all the arithmetic.
const fullPlayoutLayer: Layer = {
  name: 'otis-full-playout',
  rules: {
    checkHandOutcome: () => ({ isDetermined: false }),
  },
};

// Base rules + full-playout override: no special contracts, no consensus layer.
// `complete-trick` and `score-hand` execute immediately via the pure core executor.
const rules = composeRules([baseLayer, fullPlayoutLayer]);

// ---------------------------------------------------------------------------
// Interchange record shape (see otis/referee context contract)
// ---------------------------------------------------------------------------

interface PlayRecord {
  seat: number;
  domino: string;
}

interface GameRecord {
  game_id: string;
  hands: string[][]; // [seat 0..3][7] of pip strings
  decl_id?: number;
  decl_name?: string;
  bidder: number;
  bid_value: number;
  plays: PlayRecord[]; // 28 plays in exact play order
}

interface RefereeOutput {
  game_id: string;
  team02_points: number;
  team13_points: number;
}

// ---------------------------------------------------------------------------
// Pip-string <-> engine domino id translation
// ---------------------------------------------------------------------------

/**
 * Normalize a pip string "a-b" to the engine's canonical "hi-lo" id and Domino.
 * Accepts either order ("4-6" or "6-4"); engine ids are always high>=low.
 */
function parseDomino(pip: string): Domino {
  const parts = pip.split('-');
  if (parts.length !== 2) {
    throw new Error(`Malformed domino string: ${JSON.stringify(pip)}`);
  }
  const a = Number(parts[0]);
  const b = Number(parts[1]);
  if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || a > 6 || b < 0 || b > 6) {
    throw new Error(`Domino pips out of range 0..6: ${JSON.stringify(pip)}`);
  }
  const high = Math.max(a, b);
  const low = Math.min(a, b);
  return { high, low, id: `${high}-${low}` };
}

/** Canonical engine id ("hi-lo") for a pip string. */
function dominoId(pip: string): string {
  return String(parseDomino(pip).id);
}

// ---------------------------------------------------------------------------
// Declaration -> engine TrumpSelection
//
// Forge declaration ids (forge/oracle/declarations.py):
//   0..6  pip trumps (blanks..sixes)         -> { type: 'suit', suit }
//   7     doubles-trump (doubles have power)  -> { type: 'doubles' }
//   8     doubles-suit  (absorb, no power)    -> NOT representable in base rules
//   9     notrump                             -> { type: 'no-trump' }
// ---------------------------------------------------------------------------

const PIP_NAME_TO_SUIT: Record<string, 0 | 1 | 2 | 3 | 4 | 5 | 6> = {
  blanks: 0,
  ones: 1,
  twos: 2,
  threes: 3,
  fours: 4,
  fives: 5,
  sixes: 6,
};

function declToTrump(declName: string | undefined, declId: number | undefined): TrumpSelection {
  const name = declName?.trim().toLowerCase();

  if (name !== undefined && name in PIP_NAME_TO_SUIT) {
    return { type: 'suit', suit: PIP_NAME_TO_SUIT[name as keyof typeof PIP_NAME_TO_SUIT]! };
  }
  if (name === 'doubles-trump' || name === 'doubles' || name === 'dt') {
    return { type: 'doubles' };
  }
  if (name === 'notrump' || name === 'no-trump' || name === 'nt') {
    return { type: 'no-trump' };
  }
  if (name === 'doubles-suit' || name === 'ds') {
    throw new Error(
      "decl 'doubles-suit' (forge id 8) has no base-engine equivalent " +
        '(absorptionId=7 with powerId=8 is only reachable via the nello contract, ' +
        'which changes scoring); cannot referee this declaration.'
    );
  }

  // Fall back to forge decl_id when the name is absent/unrecognized.
  if (declId !== undefined) {
    if (declId >= 0 && declId <= 6) {
      return { type: 'suit', suit: declId as 0 | 1 | 2 | 3 | 4 | 5 | 6 };
    }
    if (declId === 7) return { type: 'doubles' };
    if (declId === 9) return { type: 'no-trump' };
    if (declId === 8) {
      throw new Error(
        "decl_id 8 ('doubles-suit') has no base-engine equivalent; cannot referee."
      );
    }
  }

  throw new Error(`Unknown declaration: name=${JSON.stringify(declName)} id=${String(declId)}`);
}

// ---------------------------------------------------------------------------
// Bid value -> engine bid action
// ---------------------------------------------------------------------------

/**
 * Build the bid action for the winning bidder. The bid value does not affect
 * per-team point totals (it only decides marks), but a legal auction is
 * required to reach the playing phase.
 *
 * Points bids: 30..41 (engine MIN_BID..MAX_BID). Values >= 42 are treated as
 * marks bids (42 => 1 mark, 84 => 2 marks, ...). The bid-30-fixed corpus only
 * exercises the points path.
 */
function bidAction(bidder: number, bidValue: number): GameAction {
  if (bidValue >= 30 && bidValue <= 41) {
    return { type: 'bid', player: bidder, bid: 'points', value: bidValue };
  }
  if (bidValue >= 42) {
    const marks = Math.max(1, Math.round(bidValue / 42));
    return { type: 'bid', player: bidder, bid: 'marks', value: marks };
  }
  throw new Error(`bid_value ${bidValue} is below the engine minimum of 30`);
}

// ---------------------------------------------------------------------------
// Replay one game through the engine
// ---------------------------------------------------------------------------

function refereeGame(game: GameRecord): RefereeOutput {
  const ctx = `game ${JSON.stringify(game.game_id)}`;

  // --- validate the deal shape up front -------------------------------------
  if (!Array.isArray(game.hands) || game.hands.length !== 4) {
    throw new Error(`${ctx}: hands must be 4 seats, got ${game.hands?.length}`);
  }
  const initialHands: Domino[][] = game.hands.map((hand, seat) => {
    if (!Array.isArray(hand) || hand.length !== 7) {
      throw new Error(`${ctx}: seat ${seat} must have 7 dominoes, got ${hand?.length}`);
    }
    return hand.map(parseDomino);
  });

  // --- seed the exact deal --------------------------------------------------
  let state: GameState = createInitialState({
    dealOverrides: { initialHands },
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    shuffleSeed: 0,
  });

  const exec = (action: GameAction): void => {
    state = executeAction(state, action, rules);
  };
  // `state` is reassigned inside the `exec` closure, so TS control-flow
  // narrowing of `state.phase` goes stale after any exec. Read the phase
  // through this accessor to keep the full-union type at each check.
  const phase = (): GameState['phase'] => state.phase;

  // --- auction: given bidder wins at bid_value, everyone else passes --------
  let guard = 0;
  while (state.phase === 'bidding') {
    if (guard++ > 8) throw new Error(`${ctx}: auction did not resolve`);
    const p = state.currentPlayer;
    if (p === game.bidder) {
      exec(bidAction(game.bidder, game.bid_value));
    } else {
      exec({ type: 'pass', player: p });
    }
  }

  // --- trump declaration by the winning bidder ------------------------------
  if (state.phase !== 'trump_selection') {
    throw new Error(`${ctx}: expected trump_selection after auction, got ${state.phase}`);
  }
  if (state.winningBidder !== game.bidder) {
    throw new Error(
      `${ctx}: winning bidder mismatch (engine=${state.winningBidder}, record=${game.bidder})`
    );
  }
  exec({
    type: 'select-trump',
    player: state.currentPlayer,
    trump: declToTrump(game.decl_name, game.decl_id),
  });

  // Bid winner must lead trick 1.
  if (state.currentPlayer !== game.bidder) {
    throw new Error(
      `${ctx}: expected bidder ${game.bidder} to lead trick 1, engine says ${state.currentPlayer}`
    );
  }

  // --- play the 28 dominoes in exact order ----------------------------------
  if (!Array.isArray(game.plays) || game.plays.length !== 28) {
    throw new Error(`${ctx}: expected 28 plays, got ${game.plays?.length}`);
  }

  game.plays.forEach((play, i) => {
    if (state.phase !== 'playing') {
      throw new Error(`${ctx}: play ${i} but phase is ${state.phase}`);
    }
    // Cross-check turn order: the record's seat must match the engine's turn.
    if (play.seat !== state.currentPlayer) {
      throw new Error(
        `${ctx}: play ${i} seat mismatch (record=${play.seat}, engine expects ${state.currentPlayer})`
      );
    }
    exec({ type: 'play', player: play.seat, dominoId: dominoId(play.domino) });

    // Close the trick once four dominoes are down.
    if (state.currentTrick.length === 4) {
      exec({ type: 'complete-trick' });
    }
  });

  // --- read the engine's own scoring ----------------------------------------
  if (phase() !== 'scoring') {
    throw new Error(`${ctx}: expected scoring phase after 28 plays, got ${state.phase}`);
  }
  const team02 = state.teamScores[0];
  const team13 = state.teamScores[1];

  // P1 invariant: per-team points (count + tricks) must sum to 42.
  if (team02 + team13 !== 42) {
    throw new Error(
      `${ctx}: P1 invariant violated: team02(${team02}) + team13(${team13}) = ${team02 + team13} != 42`
    );
  }

  return { game_id: game.game_id, team02_points: team02, team13_points: team13 };
}

// ---------------------------------------------------------------------------
// CLI entry
// ---------------------------------------------------------------------------

function main(): void {
  const path = process.argv[2];
  if (!path) {
    console.error('usage: npx tsx otis/referee/check_scores.ts <path-to.jsonl>');
    process.exit(2);
  }

  const raw = readFileSync(path, 'utf8');
  const lines = raw.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);

  for (const line of lines) {
    let game: GameRecord;
    try {
      game = JSON.parse(line) as GameRecord;
    } catch (e) {
      throw new Error(`Invalid JSON line: ${(e as Error).message}\n  ${line.slice(0, 200)}`);
    }
    const out = refereeGame(game);
    process.stdout.write(JSON.stringify(out) + '\n');
  }
}

main();
