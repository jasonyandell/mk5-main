"""Void-creation paired contrast probe.

Tests ch05-void-creation book claim: setters should deliberately void themselves
in an off-suit by leading their last tile from that suit, so they can later trump
in or punish a count tile led to that suit.

Each snapshot shows a SETTER player leading a trick (they hold >= 2 distinct suits,
at least 1 of which is a singleton — the void-creation candidate).

For each snapshot, the probe evaluates two contrasting continuations:

  A (preserve suit)  : lead a tile from a suit where the setter has >= 2 tiles.
  B (create void)    : lead the last tile from a singleton suit (creating a void).

The contrast B - A measures the incremental value of deliberately voiding.

Metrics compared (per pair):
  - EV delta:         E[Q](B) - E[Q](A)  — positive = create-void better
  - CVaR_10 delta:    downside risk of each action
  - p_set delta:      P(bidder fails to make bid) under each alternative
                      (positive = void creation helps setter)
  - threshold_mass delta: P(Q >= bid_value) — setter maximizes failure of bidder

Sliced by:
  - suit_type: trump-adjacent vs off-suit void
  - phase: early (tricks 1-2), mid (3-5), late (6-7)
  - bidder_count_exposure: does bidder hold count in the voided suit?

Usage:
    python run_void_creation_probe.py \\
        [--snapshots snapshots.jsonl] \\
        [--checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt] \\
        [--samples 100] [--device mps] [--output-dir .]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.eq.game_tensor import GameStateTensor
from forge.oracle.declarations import PIP_TRUMP_IDS, DOUBLES_TRUMP, DOUBLES_SUIT, NOTRUMP
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_IS_DOUBLE,
    DOMINO_COUNT_POINTS,
    led_suit_for_lead_domino,
)

TRUMP_SUIT = 7  # sentinel used by led_suit_for_lead_domino


# ---------------------------------------------------------------------------
# Suit and void-creation helpers
# ---------------------------------------------------------------------------

def domino_suit(did: int, decl_id: int) -> int:
    """Return the suit ID (0-6 = pip suit, 7 = trump) when this domino leads."""
    return led_suit_for_lead_domino(did, decl_id)


def hand_suit_groups(hand: list[int], decl_id: int) -> dict[int, list[int]]:
    """Map suit -> [domino_ids] for tiles still in hand (id >= 0)."""
    groups: dict[int, list[int]] = {}
    for did in hand:
        if did < 0:
            continue
        s = domino_suit(did, decl_id)
        groups.setdefault(s, []).append(did)
    return groups


def find_void_creation_slot(snap: dict) -> int | None:
    """Return the hand slot of the void-creating tile.

    Policy: lowest-sum tile from the smallest singleton non-trump suit.
    If no singleton exists, return None (skip snapshot).
    """
    # Setter is player 1 or 3. Bidder is player 0 in legacy corpus.
    bidder = snap["bidder"]
    setter_player = _infer_current_player(snap)
    if setter_player is None:
        return None
    if setter_player == bidder or (setter_player % 2) == (bidder % 2):
        return None  # not a setter

    decl_id = snap["decl_id"]
    hand = snap["hands"][setter_player]
    legal = snap["_legal_mask"]
    groups = hand_suit_groups(hand, decl_id)

    # Singleton off-suit groups only
    singleton_suits = sorted(
        [s for s, tiles in groups.items() if len(tiles) == 1 and s != TRUMP_SUIT]
    )
    if not singleton_suits:
        return None

    # Pick the smallest singleton suit; break ties by domino pip sum
    best_did = None
    for s in singleton_suits:
        tile = groups[s][0]
        pip_sum = DOMINO_HIGH[tile] + DOMINO_LOW[tile]
        if best_did is None or pip_sum < DOMINO_HIGH[best_did] + DOMINO_LOW[best_did]:
            best_did = tile

    if best_did is None:
        return None

    # Map domino id to slot index
    for slot, did in enumerate(hand):
        if did == best_did and legal[slot]:
            return slot

    return None  # tile not legal


def find_preserve_suit_slot(snap: dict, void_slot: int) -> int | None:
    """Return a hand slot from a suit with >= 2 tiles (NOT the void-creating suit).

    Avoids trump to keep the contrast clean.
    """
    setter_player = _infer_current_player(snap)
    if setter_player is None:
        return None

    decl_id = snap["decl_id"]
    hand = snap["hands"][setter_player]
    legal = snap["_legal_mask"]
    groups = hand_suit_groups(hand, decl_id)

    void_did = hand[void_slot]
    void_suit = domino_suit(void_did, decl_id)

    # Multi-tile off-suit groups
    multi_suits = {
        s: tiles
        for s, tiles in groups.items()
        if len(tiles) >= 2 and s != void_suit and s != TRUMP_SUIT
    }
    if not multi_suits:
        # Fall back to any suit (even trump) with >= 2 tiles, excluding void suit
        multi_suits = {
            s: tiles
            for s, tiles in groups.items()
            if len(tiles) >= 2 and s != void_suit
        }
    if not multi_suits:
        return None

    # Choose highest-count suit; break ties by lowest pip-sum tile
    best_suit = max(multi_suits, key=lambda s: len(multi_suits[s]))
    tiles = multi_suits[best_suit]
    # Pick lowest pip-sum tile in that suit (most conservative play)
    best_tile = min(tiles, key=lambda d: DOMINO_HIGH[d] + DOMINO_LOW[d])

    for slot, did in enumerate(hand):
        if did == best_tile and legal[slot] and slot != void_slot:
            return slot

    return None


def _infer_current_player(snap: dict) -> int | None:
    """Return which player is to move (0-3), or None if indeterminate.

    All void_creation snapshots are at trick starts (trick_plays empty).
    The current player is the winner of the previous trick.
    """
    history = snap["history"]
    plays = [h for h in history if h[0] >= 0]
    n = len(plays)

    if n == 0:
        return int(snap["leader"])  # very first trick

    if n % 4 != 0:
        # Mid-trick — determine by rotation
        trick_start = (n // 4) * 4
        leader = plays[trick_start][0]
        current = (leader + (n % 4)) % 4
        return current

    # Start of a new trick: determine winner of last completed trick
    last_trick = plays[-4:]
    decl_id = snap["decl_id"]
    lead_did = last_trick[0][1]
    led_suit = led_suit_for_lead_domino(lead_did, decl_id)

    best_player = last_trick[0][0]
    best_did = lead_did
    best_rank = _trick_rank(best_did, led_suit, decl_id)

    for play in last_trick[1:]:
        p, did, _ = play
        rank = _trick_rank(did, led_suit, decl_id)
        if rank > best_rank:
            best_rank = rank
            best_player = p
            best_did = did

    return best_player


def _trick_rank(did: int, led_suit: int, decl_id: int) -> int:
    """Simplified trick rank — matches forge.oracle.tables.trick_rank logic."""
    from forge.oracle.tables import is_in_called_suit, can_follow, has_trump_power

    if has_trump_power(decl_id) and is_in_called_suit(did, decl_id):
        base = 200
        if DOMINO_IS_DOUBLE[did]:
            return base + 14
        return base + DOMINO_HIGH[did] + DOMINO_LOW[did]

    if can_follow(did, led_suit, decl_id):
        base = 100
        if DOMINO_IS_DOUBLE[did]:
            return base + 14
        return base + DOMINO_HIGH[did] + DOMINO_LOW[did]

    return 0


def snap_phase(snap: dict) -> str:
    """Return 'early', 'mid', or 'late' based on tricks completed."""
    plays = [h for h in snap["history"] if h[0] >= 0]
    trick_num = len(plays) // 4
    if trick_num <= 2:
        return "early"
    if trick_num <= 5:
        return "mid"
    return "late"


def snap_void_suit_type(snap: dict, void_slot: int) -> str:
    """Return 'trump_adjacent' or 'off_suit' for the suit being voided."""
    setter = _infer_current_player(snap)
    if setter is None:
        return "unknown"
    hand = snap["hands"][setter]
    void_did = hand[void_slot]
    decl_id = snap["decl_id"]
    void_suit = domino_suit(void_did, decl_id)
    if void_suit == TRUMP_SUIT:
        return "trump"
    # Check if trump-adjacent (the void_suit shares a pip with trump pip)
    if decl_id in PIP_TRUMP_IDS:
        trump_pip = decl_id
        if void_suit == trump_pip:
            return "trump_adjacent"  # shouldn't happen — that would be trump
    return "off_suit"


def bidder_has_count_in_suit(snap: dict, void_slot: int) -> bool:
    """Return True if the bidder (player 0) holds a count tile in the voided suit."""
    bidder = snap["bidder"]
    setter = _infer_current_player(snap)
    if setter is None:
        return False
    hand_setter = snap["hands"][setter]
    void_did = hand_setter[void_slot]
    decl_id = snap["decl_id"]
    void_suit = domino_suit(void_did, decl_id)

    bidder_hand = snap["hands"][bidder]
    for did in bidder_hand:
        if did < 0:
            continue
        if domino_suit(did, decl_id) == void_suit:
            if DOMINO_COUNT_POINTS[did] > 0:
                return True
    return False


# ---------------------------------------------------------------------------
# EQ evaluation
# ---------------------------------------------------------------------------

def eq_for_action(
    model,
    snap: dict,
    action_slot: int,
    n_samples: int,
    device: str,
) -> tuple[float, float, torch.Tensor]:
    """Evaluate E[Q] for a specific first action from the snapshot.

    Returns:
        (ev, cvar_10, pdf_tensor) where pdf_tensor has shape (85,) over Q in {-42,..,+42}.
    """
    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    records = generate_eq_from_snapshots(
        model=model,
        snapshots=[snap],
        n_samples=n_samples,
        device=device,
        greedy=True,
    )

    first_dec = records[0].decisions[0]
    ev = float(first_dec.e_q[action_slot].item()) if first_dec.e_q is not None else float("nan")

    pdf = first_dec.e_q_pdf  # [7, 85] or None
    if pdf is not None:
        action_pdf = pdf[action_slot]  # (85,)
        q_values = torch.arange(-42, 43, dtype=torch.float32)
        weights = action_pdf.float()
        cum = torch.cumsum(weights, dim=0) / (weights.sum() + 1e-12)
        mask_10 = cum <= 0.10
        if mask_10.any():
            cvar_10 = float(
                (q_values * weights * mask_10.float()).sum()
                / (weights[mask_10].sum() + 1e-12)
            )
        else:
            cvar_10 = float(q_values[0])
    else:
        action_pdf = torch.zeros(85)
        cvar_10 = float("nan")

    return ev, cvar_10, action_pdf


def threshold_mass(pdf: torch.Tensor, bid_value: int) -> float:
    """P(Q >= bid_value) from a (85,) PDF over Q in {-42,..,+42}."""
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    mask = q_values >= bid_value
    return float(pdf[mask].sum().item())


def p_set(pdf: torch.Tensor, bid_value: int) -> float:
    """P(Q < bid_value) = P(bidder fails to make contract).

    A setter wants to MAXIMIZE p_set, so a positive p_set delta (B-A)
    means create-void is better for setting.
    """
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    mask = q_values < bid_value
    return float(pdf[mask].sum().item())


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(
    snapshots: list[dict],
    model,
    n_samples: int,
    device: str,
) -> list[dict]:
    """Run paired contrasts on all snapshots."""
    rows: list[dict] = []

    for i, snap in enumerate(snapshots):
        void_slot = find_void_creation_slot(snap)
        if void_slot is None:
            print(f"  [skip {i}] no void-creation candidate", flush=True)
            continue

        preserve_slot = find_preserve_suit_slot(snap, void_slot)
        if preserve_slot is None:
            print(f"  [skip {i}] no preserve-suit candidate", flush=True)
            continue

        bid_value = int(snap.get("bid_value", 30))
        decl_id = snap["decl_id"]
        setter = _infer_current_player(snap)
        hand = snap["hands"][setter] if setter is not None else []
        void_did = hand[void_slot] if setter is not None else -1
        preserve_did = hand[preserve_slot] if setter is not None else -1
        void_suit = domino_suit(void_did, decl_id) if void_did >= 0 else -1
        phase = snap_phase(snap)
        suit_type = snap_void_suit_type(snap, void_slot)
        bidder_count = bidder_has_count_in_suit(snap, void_slot)

        print(
            f"  [{i + 1:3d}/{len(snapshots)}] decl={decl_id} bid={bid_value} "
            f"phase={phase} suit_type={suit_type} "
            f"void_slot={void_slot}(did={void_did}) preserve_slot={preserve_slot}(did={preserve_did})",
            end=" ",
            flush=True,
        )

        t0 = time.perf_counter()
        # A: preserve suit (keep the suit intact)
        ev_a, cvar_a, pdf_a = eq_for_action(model, snap, preserve_slot, n_samples, device)
        # B: create void (lead last tile of singleton suit)
        ev_b, cvar_b, pdf_b = eq_for_action(model, snap, void_slot, n_samples, device)
        elapsed = time.perf_counter() - t0

        # The e_q values are from TEAM 0 perspective (V is always Team 0).
        # Setter is on Team 1. For setter, lower Q = better (bidder fails).
        # So setter prefers LOWER ev from Team 0 perspective.
        # EV delta (B-A) from setter's perspective: negative EV delta = better for setter.
        # But we also want to report p_set delta.
        ev_delta_t0 = ev_b - ev_a  # from Team 0 perspective
        ev_delta_setter = -ev_delta_t0  # from setter (Team 1) perspective

        cvar_delta = cvar_b - cvar_a
        tm_a = threshold_mass(pdf_a, bid_value)
        tm_b = threshold_mass(pdf_b, bid_value)
        tm_delta = tm_b - tm_a

        ps_a = p_set(pdf_a, bid_value)
        ps_b = p_set(pdf_b, bid_value)
        ps_delta = ps_b - ps_a  # positive = void creation helps setter

        print(f"EV_delta(setter)={ev_delta_setter:+.2f} p_set_delta={ps_delta:+.4f} t={elapsed:.1f}s", flush=True)

        rows.append({
            "snapshot_idx": i,
            "decl_id": decl_id,
            "bid_value": bid_value,
            "setter_player": setter,
            "phase": phase,
            "suit_type": suit_type,
            "void_suit": void_suit,
            "bidder_count_exposure": bidder_count,
            "void_slot": void_slot,
            "void_did": void_did,
            "preserve_slot": preserve_slot,
            "preserve_did": preserve_did,
            # Team 0 perspective (raw oracle output)
            "ev_a_t0": ev_a,
            "ev_b_t0": ev_b,
            "ev_delta_t0": ev_delta_t0,
            # Setter perspective (negated)
            "ev_delta_setter": ev_delta_setter,
            "cvar_10_a": cvar_a,
            "cvar_10_b": cvar_b,
            "cvar_delta": cvar_delta,
            "p_set_a": ps_a,
            "p_set_b": ps_b,
            "p_set_delta": ps_delta,
            "threshold_mass_a": tm_a,
            "threshold_mass_b": tm_b,
            "threshold_mass_delta": tm_delta,
        })

    return rows


def summarise(rows: list[dict]) -> dict:
    """Compute headline summary and slice breakdowns."""
    if not rows:
        return {}

    ev_deltas = np.array([r["ev_delta_setter"] for r in rows])
    ps_deltas = np.array([r["p_set_delta"] for r in rows])
    cvar_deltas = np.array([r["cvar_delta"] for r in rows])
    tm_deltas = np.array([r["threshold_mass_delta"] for r in rows])
    n = len(rows)

    def ci95(arr: np.ndarray) -> tuple[float, float]:
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)

    ev_lo, ev_hi = ci95(ev_deltas)
    ps_lo, ps_hi = ci95(ps_deltas)

    n_void_better = int((ev_deltas > 0).sum())
    n_pset_higher = int((ps_deltas > 0).sum())

    summary = {
        "N_pairs": n,
        # Headline: p_set delta (primary setter objective)
        "p_set_delta_mean": float(ps_deltas.mean()),
        "p_set_delta_ci95_lo": ps_lo,
        "p_set_delta_ci95_hi": ps_hi,
        "p_set_direction": "void > preserve" if ps_deltas.mean() > 0 else "preserve > void",
        "n_p_set_higher_void": n_pset_higher,
        "pct_p_set_higher_void": float(100 * n_pset_higher / n),
        # EV delta from setter perspective
        "ev_delta_setter_mean": float(ev_deltas.mean()),
        "ev_delta_ci95_lo": ev_lo,
        "ev_delta_ci95_hi": ev_hi,
        "ev_direction": "void > preserve" if ev_deltas.mean() > 0 else "preserve > void",
        "n_void_ev_better": n_void_better,
        "pct_void_ev_better": float(100 * n_void_better / n),
        # CVaR and threshold mass
        "cvar_delta_mean": float(cvar_deltas.mean()),
        "threshold_mass_delta_mean": float(tm_deltas.mean()),
        # Claim verdict
        "claim_ledger_impact": _propose_verdict(ps_deltas, ev_deltas, n),
        "claim_ledger_rationale": _verdict_rationale(ps_deltas, ev_deltas, n),
    }

    # Slice: suit type
    summary["slices"] = {}
    for suit_type in ["off_suit", "trump_adjacent", "trump"]:
        sub = [r for r in rows if r["suit_type"] == suit_type]
        if sub:
            sub_ps = np.array([r["p_set_delta"] for r in sub])
            sub_ev = np.array([r["ev_delta_setter"] for r in sub])
            summary["slices"][f"suit_type={suit_type}"] = {
                "N": len(sub),
                "p_set_delta_mean": float(sub_ps.mean()),
                "ev_delta_setter_mean": float(sub_ev.mean()),
                "pct_void_better": float(100 * (sub_ev > 0).mean()),
            }

    # Slice: phase
    for phase in ["early", "mid", "late"]:
        sub = [r for r in rows if r["phase"] == phase]
        if sub:
            sub_ps = np.array([r["p_set_delta"] for r in sub])
            sub_ev = np.array([r["ev_delta_setter"] for r in sub])
            summary["slices"][f"phase={phase}"] = {
                "N": len(sub),
                "p_set_delta_mean": float(sub_ps.mean()),
                "ev_delta_setter_mean": float(sub_ev.mean()),
                "pct_void_better": float(100 * (sub_ev > 0).mean()),
            }

    # Slice: bidder count exposure
    for exposed in [True, False]:
        sub = [r for r in rows if r["bidder_count_exposure"] == exposed]
        if sub:
            sub_ps = np.array([r["p_set_delta"] for r in sub])
            sub_ev = np.array([r["ev_delta_setter"] for r in sub])
            label = "count_exposed" if exposed else "no_count"
            summary["slices"][f"bidder_count_exposure={label}"] = {
                "N": len(sub),
                "p_set_delta_mean": float(sub_ps.mean()),
                "ev_delta_setter_mean": float(sub_ev.mean()),
                "pct_void_better": float(100 * (sub_ev > 0).mean()),
            }

    return summary


def _propose_verdict(ps_deltas: np.ndarray, ev_deltas: np.ndarray, n: int) -> str:
    """Propose a claim-ledger status.

    The book claims void creation (B) > preserve suit (A) for the setter.
    A positive ps_delta means B improves setting probability.
    A positive ev_delta (setter perspective) means B is better EV for setter.
    'supported'    : both CI lower bounds > 0 (clear benefit for void creation)
    'contradicted' : both CI upper bounds < 0 (clear harm from void creation)
    'context-limited': mixed across slices or CI straddles 0 narrowly
    'underpowered' : n too small or effect too noisy
    """
    if n < 30:
        return "underpowered"
    mean_ps = ps_deltas.mean()
    se_ps = ps_deltas.std(ddof=1) / np.sqrt(n)
    ci_lo_ps = mean_ps - 1.96 * se_ps
    ci_hi_ps = mean_ps + 1.96 * se_ps

    mean_ev = ev_deltas.mean()
    se_ev = ev_deltas.std(ddof=1) / np.sqrt(n)
    ci_lo_ev = mean_ev - 1.96 * se_ev
    ci_hi_ev = mean_ev + 1.96 * se_ev

    # Clear support: CI excludes 0 on the positive side for both metrics
    if ci_lo_ps > 0.005 and ci_lo_ev > 0:
        return "supported"

    # Clear contradiction: CI excludes 0 on the negative side for EITHER metric
    # Use the EV as primary signal (higher N, more granular)
    if ci_hi_ev < 0 and ci_hi_ps < 0:
        return "contradicted"
    if ci_hi_ev < 0 and abs(mean_ps) < 0.01:
        # EV clearly negative but p_set delta noisy/small
        return "contradicted"

    # Context-limited: EV is negative for some sub-slices but not all
    # (would need slice analysis to determine — default to context-limited
    # if global CI straddles zero or is weakly one-sided)
    if abs(mean_ev) < 1.0:
        return "context-limited"

    return "underpowered"


def _verdict_rationale(ps_deltas: np.ndarray, ev_deltas: np.ndarray, n: int) -> str:
    mean_ps = float(ps_deltas.mean())
    mean_ev = float(ev_deltas.mean())
    pct = float(100 * (ev_deltas > 0).mean())
    return (
        f"N={n} paired contrasts from oracle-greedy legacy corpus. "
        f"Mean p_set delta={mean_ps:+.4f}, mean EV delta (setter)={mean_ev:+.2f}, "
        f"{pct:.1f}% of contrasts favor void creation."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots",
        type=str,
        default=str(
            PROJECT_ROOT
            / "w42/book_validation_v1/wave2/snapshots/void_creation/snapshots.jsonl"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
        ),
    )
    parser.add_argument("--samples", type=int, default=100,
                        help="World samples per decision (default: 100)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
    )
    parser.add_argument(
        "--max-snapshots", type=int, default=None,
        help="Cap number of snapshots (for smoke tests)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshots
    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        print(f"Error: {snap_path} not found.", flush=True)
        return 1

    with snap_path.open() as fh:
        snapshots = [json.loads(line) for line in fh if line.strip()]

    if args.max_snapshots:
        snapshots = snapshots[: args.max_snapshots]

    print(f"Loaded {len(snapshots)} snapshots from {snap_path}", flush=True)

    # Validate
    print("Validating snapshots…", flush=True)
    valid = []
    for snap in snapshots:
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid.append(snap)
        except Exception as e:
            print(f"  skip invalid snapshot: {e}", flush=True)
    snapshots = valid
    print(f"Valid snapshots: {len(snapshots)}", flush=True)

    # Pre-filter: must have both void_creation and preserve candidates
    filtered = []
    for snap in snapshots:
        vs = find_void_creation_slot(snap)
        if vs is None:
            continue
        ps = find_preserve_suit_slot(snap, vs)
        if ps is None:
            continue
        filtered.append(snap)
    print(f"Snapshots with valid paired contrast: {len(filtered)}", flush=True)
    snapshots = filtered

    if not snapshots:
        print("No valid paired contrasts. Exiting.", flush=True)
        return 1

    # Device selection
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA unavailable; falling back to MPS.", flush=True)
            device = "mps"
        else:
            print("CUDA unavailable; falling back to CPU.", flush=True)
            device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    # Load model
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {ckpt} on {device}…", flush=True)
    oracle = Stage1Oracle(str(ckpt), device=device, compile=False)
    print("Model loaded.", flush=True)

    # Run probe
    print(
        f"\nRunning paired contrasts ({len(snapshots)} snapshots, "
        f"{args.samples} samples each)…",
        flush=True,
    )
    t_start = time.perf_counter()
    rows = run_probe(snapshots, oracle.model, n_samples=args.samples, device=device)
    t_total = time.perf_counter() - t_start
    print(f"\nProbe complete in {t_total:.1f}s ({len(rows)} valid pairs)", flush=True)

    if not rows:
        print("No valid pairs produced. Exiting.", flush=True)
        return 1

    # Save CSV
    import csv
    csv_path = out_dir / "paired_contrasts.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}", flush=True)

    # Summary
    summary = summarise(rows)

    # Slice breakdown CSV
    slice_rows = []
    for slice_name, stats in summary.get("slices", {}).items():
        slice_rows.append({"slice": slice_name, **stats})
    if slice_rows:
        slice_csv = out_dir / "slice_breakdown.csv"
        with slice_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(slice_rows[0].keys()))
            writer.writeheader()
            writer.writerows(slice_rows)
        print(f"Saved {slice_csv}", flush=True)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {summary_path}", flush=True)

    # Manifest
    import hashlib
    manifest = {
        "bead": "t42-26j8",
        "claim": "ch05-void-creation",
        "wave": "wave2",
        "probe": "void_creation_paired_contrast",
        "snapshots_path": str(snap_path),
        "snapshots_sha256": hashlib.sha256(snap_path.read_bytes()).hexdigest(),
        "checkpoint": str(ckpt),
        "n_snapshots_input": len(valid),
        "n_pairs_run": len(rows),
        "device": device,
        "samples_per_decision": args.samples,
        "runtime_seconds": round(t_total, 1),
        "artifacts": {
            "paired_contrasts": str(csv_path),
            "slice_breakdown": str(out_dir / "slice_breakdown.csv"),
            "summary": str(summary_path),
        },
        "command": (
            f"python run_void_creation_probe.py "
            f"--snapshots {snap_path} "
            f"--checkpoint {ckpt} "
            f"--samples {args.samples} "
            f"--device {device} "
            f"--output-dir {out_dir}"
        ),
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved {manifest_path}", flush=True)

    # Print headline numbers
    print("\n=== HEADLINE NUMBERS ===", flush=True)
    print(f"N pairs:                {summary['N_pairs']}", flush=True)
    print(
        f"p_set delta (mean):     {summary['p_set_delta_mean']:+.4f}  "
        f"95% CI [{summary['p_set_delta_ci95_lo']:+.4f}, {summary['p_set_delta_ci95_hi']:+.4f}]",
        flush=True,
    )
    print(f"p_set direction:        {summary['p_set_direction']}", flush=True)
    print(
        f"EV delta setter (mean): {summary['ev_delta_setter_mean']:+.3f}  "
        f"95% CI [{summary['ev_delta_ci95_lo']:+.3f}, {summary['ev_delta_ci95_hi']:+.3f}]",
        flush=True,
    )
    print(f"% void EV better:       {summary['pct_void_ev_better']:.1f}%", flush=True)
    print(f"CVaR delta (mean):      {summary['cvar_delta_mean']:+.3f}", flush=True)
    print(f"Claim impact:           {summary['claim_ledger_impact']}", flush=True)
    print(f"Rationale:              {summary['claim_ledger_rationale']}", flush=True)

    print("\n=== SLICE BREAKDOWN ===", flush=True)
    for slice_name, stats in summary.get("slices", {}).items():
        print(
            f"  {slice_name}: N={stats['N']} "
            f"p_set_delta={stats['p_set_delta_mean']:+.4f} "
            f"ev_delta={stats['ev_delta_setter_mean']:+.2f} "
            f"pct_void_better={stats['pct_void_better']:.1f}%",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
