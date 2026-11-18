// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

export { default, initSync, version } from './pkg/walt.js';
export type { InitInput, InitOutput, SyncInitInput } from './pkg/walt.js';

/** Tile ids are 0..27 in triangular order: (high, low) is high·(high+1)/2 + low, so 0-0 is 0 and 6-6 is 27. */
export type Tile = number;

/** Declarations: 0..6 makes that pip trump, 7 makes doubles trump, 9 is no-trump. */
export type Declaration = number;

/**
 * A seat's position: the hand it was dealt and every play so far. The same
 * shape as `@jasonyandell/zeb`'s `PositionRequest`, so a game can seat either.
 */
export interface PositionRequest {
  decl: Declaration;
  /** The seat that won the auction; it leads the first trick. Seats 0..3; 0 and 2 are partners. */
  bidder: number;
  /** The contract: 30..41 points, or 42 for a mark bid (all 42 points). */
  bid: number;
  /** The seat to play. */
  seat: number;
  /** The seven tiles the seat was dealt (its plays included). */
  hand: readonly Tile[];
  /** Every play so far as [seat, tile], from the opening lead; default none. */
  plays?: readonly (readonly [number, Tile])[];
  /** Deals sampled for the first look, 1..4096; default 40. Tied best tiles are looked at again on 4× and 16× as many. */
  worlds?: number;
  /** Deals each imagined other seat samples for itself, 1..4096; default 8. */
  minds?: number;
  /** Seeds the sampled deals (a non-negative integer below 2^64); default 0xB7E151628AED2A6B. */
  seed?: number | bigint;
}

/** A seat's move. */
export interface Move {
  tile: Tile;
  /** Only one tile was legal; nothing was planned and q is empty. */
  forced: boolean;
  /**
   * [tile, p] for each legal tile, ascending: p is the estimated probability the
   * bidding side makes its contract if the tile is played, to four places. The
   * bidding side plays the largest, the defenders the smallest; a tie that
   * survives two closer looks goes to the lowest tile.
   */
  q: [Tile, number][];
  /** The current trick's leader (the last trick's winner between tricks). */
  leader: number;
  /** Points banked so far by seats {0,2} and by seats {1,3}. */
  points: [number, number];
}

/** The raw response of WALT.md §6. */
export interface Response {
  v: 1;
  kind: 'play';
  choice: Tile;
  forced: boolean;
  raced: false;
  /** [tile, value × 10000 truncated] for each legal tile, ascending; empty when forced. */
  opts: [Tile, number][];
  leader: number;
  points: [number, number];
}

/**
 * The seat's move. Walt sees only the seat's own tiles and the public plays.
 * Throws a RangeError or TypeError for a malformed request and an Error for a
 * record that breaks the rules (a play out of turn, a tile played twice, a
 * seat following a suit it showed void in).
 */
export function play(request: PositionRequest): Move;

/** The request as `walt1 play` text. */
export function requestText(request: PositionRequest): string;

/** One `walt1 play` request as text in, the response line parsed; an error line throws. */
export function ask(text: string): Response;
