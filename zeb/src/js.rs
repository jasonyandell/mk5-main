//! The JavaScript surface. Requests are read field by field with plain
//! errors (thrown as JS `Error`s); results come back as plain objects and
//! arrays. Nothing here panics on what a caller passes; should the planner
//! ever panic, the panic hook logs why to the console before the trap.

use std::ops::RangeInclusive;

use js_sys::{Array, Object, Reflect};
use scenario_core::planner::MAX_WORLDS;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::position::refuse;
use crate::{Error, Player, Position, DECLARATIONS, DEFAULT_WORLDS};

#[wasm_bindgen(typescript_custom_section)]
const TYPES: &'static str = r#"
/** Tile ids are 0..27 in triangular order: (high, low) is high·(high+1)/2 + low, so 0-0 is 0 and 6-6 is 27. */
export type Tile = number;

/**
 * Declarations: 0..6 makes that pip trump, 7 makes doubles trump, 9 is no-trump.
 * (8, doubles as their own suit, is nello's and is refused.)
 */
export type Declaration = number;

/** A bidder's seven tiles, before the auction closes. */
export interface ForecastRequest {
  /** The bidder's seven tiles. */
  hand: readonly Tile[];
  /** The declarations to forecast, distinct; default all nine: 0..7 and 9. */
  declarations?: readonly Declaration[];
  /** Scenarios planned per declaration, 1..1024; default 160. */
  worlds?: number;
  /** Seeds every random draw (a non-negative integer below 2^64); default 0. */
  seed?: number | bigint;
}

/** How the bidder fares under one declaration, having bid 30 and led the planned tile. */
export interface Forecast {
  decl: Declaration;
  /** The bidding side's expected final margin: its points less the defenders'. */
  mean: number;
  /** pmake[i]: the probability the bidding side banks at least 30 + i points (targets 30..42). */
  pmake: number[];
}

/** A seat's position: the hand it was dealt and every play so far. */
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
  /** Every play so far as [seat, tile], from the opening lead. */
  plays: readonly (readonly [number, Tile])[];
  /** Scenarios planned, 1..1024; default 160. */
  worlds?: number;
  /** Seeds every random draw (a non-negative integer below 2^64); default 0. */
  seed?: number | bigint;
}

/** A seat's move. */
export interface Move {
  tile: Tile;
  /** Only one tile was legal; nothing was planned and q is empty. */
  forced: boolean;
  /**
   * [tile, p] for each legal tile, ascending: p is the probability the bidding side
   * makes its contract if the tile is played. The bidding side plays the largest,
   * the defenders the smallest.
   */
  q: [Tile, number][];
}

/** A seat's move, opened up for a hint. */
export interface Hint extends Move {
  /**
   * [tile, pmake] for each legal tile, ascending: pmake[i] is the probability the
   * bidding side banks at least 30 + i points if the tile is played. Empty when forced.
   */
  pmake: [Tile, number[]][];
}
"#;

/// When the module loads: should the planner ever panic, log why to the
/// console before the trap.
#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// One plan per declaration from the bidder's chair before the opening
/// lead: how the bidder fares if it wins a 30 bid and declares each. Pip
/// declarations take tens of milliseconds at 160 worlds; doubles and
/// no-trump, which the field net does not play, five to seven times that.
#[wasm_bindgen(unchecked_return_type = "Forecast[]")]
pub fn forecast(
    #[wasm_bindgen(unchecked_param_type = "ForecastRequest")] request: JsValue,
) -> Result<JsValue, JsError> {
    let r = Request::new(request, &["hand", "declarations", "worlds", "seed"])?;
    let hand = r.ints("hand", 27)?;
    let decls = match r.get("declarations")? {
        Some(_) => r.ints("declarations", 9)?,
        None => DECLARATIONS.to_vec(),
    };
    to_js(&r.player()?.forecast(&hand, &decls)?)
}

/// The seat's move: the first move of the plan.
#[wasm_bindgen(unchecked_return_type = "Move")]
pub fn play(#[wasm_bindgen(unchecked_param_type = "PositionRequest")] request: JsValue) -> Result<JsValue, JsError> {
    let (player, position) = read_position(request)?;
    to_js(&player.play(&position))
}

/// The seat's move with each legal tile's tails, for a hint.
#[wasm_bindgen(unchecked_return_type = "Hint")]
pub fn analyze(#[wasm_bindgen(unchecked_param_type = "PositionRequest")] request: JsValue) -> Result<JsValue, JsError> {
    let (player, position) = read_position(request)?;
    to_js(&player.analyze(&position))
}

/// The package version.
#[wasm_bindgen]
pub fn version() -> String {
    crate::version().to_string()
}

/// The embedded field net's sha256, lowercase hex: with the version and the
/// seed, the receipt for any answer.
#[wasm_bindgen(js_name = netId)]
pub fn net_id() -> String {
    crate::net_id().to_string()
}

fn read_position(request: JsValue) -> Result<(Player, Position), Error> {
    let r = Request::new(request, &["decl", "bidder", "bid", "seat", "hand", "plays", "worlds", "seed"])?;
    let position = Position::new(
        r.int("decl", 0..=9)?,
        r.int("bidder", 0..=3)?,
        r.int("bid", 30..=42)?,
        r.int("seat", 0..=3)?,
        &r.ints("hand", 27)?,
        &r.plays()?,
    )?;
    Ok((r.player()?, position))
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsError::new(&e.to_string()))
}

/// A request object, read field by field.
struct Request(Object);

impl Request {
    /// The request, refusing anything but a plain object of the named fields.
    fn new(value: JsValue, fields: &[&str]) -> Result<Request, Error> {
        if !value.is_object() || Array::is_array(&value) {
            refuse!("a request is an object with the fields {}", fields.join(", "));
        }
        let obj: Object = value.unchecked_into();
        for key in Object::keys(&obj) {
            let key = key.as_string().unwrap_or_default();
            if !fields.contains(&key.as_str()) {
                refuse!("unknown field {key:?}; a request has the fields {}", fields.join(", "));
            }
        }
        Ok(Request(obj))
    }

    /// The field's value; None when it is absent or undefined. A getter
    /// that throws is the caller's error, reported as such.
    fn get(&self, name: &str) -> Result<Option<JsValue>, Error> {
        match Reflect::get(&self.0, &JsValue::from_str(name)) {
            Ok(v) => Ok(Some(v).filter(|v| !v.is_undefined())),
            Err(_) => refuse!("reading {name} threw"),
        }
    }

    fn required(&self, name: &str) -> Result<JsValue, Error> {
        self.get(name)?.ok_or_else(|| Error(format!("{name} is required")))
    }

    fn int(&self, name: &str, range: RangeInclusive<u8>) -> Result<u8, Error> {
        int(&self.required(name)?, name, range)
    }

    /// An array of integers in 0..=max.
    fn ints(&self, name: &str, max: u8) -> Result<Vec<u8>, Error> {
        array(&self.required(name)?, name)?
            .iter()
            .enumerate()
            .map(|(i, v)| int(&v, &format!("{name}[{i}]"), 0..=max))
            .collect()
    }

    /// `plays`, as (seat, tile) pairs.
    fn plays(&self) -> Result<Vec<(u8, u8)>, Error> {
        array(&self.required("plays")?, "plays")?
            .iter()
            .enumerate()
            .map(|(i, pair)| {
                let what = format!("plays[{i}]");
                let pair = array(&pair, &what)?;
                if pair.length() != 2 {
                    refuse!("{what} must be a [seat, tile] pair");
                }
                Ok((
                    int(&pair.get(0), &format!("{what}[0]"), 0..=3)?,
                    int(&pair.get(1), &format!("{what}[1]"), 0..=27)?,
                ))
            })
            .collect()
    }

    /// The player `worlds` and `seed` ask for.
    fn player(&self) -> Result<Player, Error> {
        let worlds = match self.get("worlds")? {
            None => DEFAULT_WORLDS,
            Some(v) => match v.as_f64() {
                Some(x) if x.fract() == 0.0 && (0.0..=u32::MAX as f64).contains(&x) => x as usize,
                _ => refuse!("worlds must be an integer from 1 to {MAX_WORLDS}"),
            },
        };
        let seed = match self.get("seed")? {
            None => 0,
            Some(v) if v.is_bigint() => u64::try_from(v).map_err(|_| Error(SEED.into()))?,
            Some(v) => match v.as_f64() {
                Some(x) if x.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER).contains(&x) => x as u64,
                _ => refuse!("{SEED}"),
            },
        };
        Player::new(worlds, seed)
    }
}

/// The largest integer a JS number holds exactly, 2^53 − 1.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

const SEED: &str = "seed must be a non-negative integer below 2^64: a number up to 2^53 − 1, or a bigint";

fn array(v: &JsValue, what: &str) -> Result<Array, Error> {
    match v.dyn_ref::<Array>() {
        Some(a) => Ok(a.clone()),
        None => refuse!("{what} must be an array"),
    }
}

fn int(v: &JsValue, what: &str, range: RangeInclusive<u8>) -> Result<u8, Error> {
    match v.as_f64() {
        Some(x) if x.fract() == 0.0 && (*range.start() as f64..=*range.end() as f64).contains(&x) => Ok(x as u8),
        _ => refuse!("{what} must be an integer from {} to {}", range.start(), range.end()),
    }
}
