"""Void-creation (follow-position) paired contrast probe.

Tests ch05-void-creation book claim in the canonical scenario:
  A setter is FOLLOWING a non-trump lead and cannot follow suit.
  They hold exactly 1 tile in some non-led, non-trump suit ("singleton").
  The book recommends playing the singleton to void that suit,
  creating a future trump-in window when partner/opponent leads to that suit.

For each snapshot, the probe evaluates two contrasting continuations:

  A (preserve): play a tile from a DIFFERENT held suit, keeping the singleton suit alive.
  B (void):     play the singleton tile, voiding that suit.

Metrics compared (per pair, from setter perspective = Team 1):
  - EV delta:           E[Q](B) - E[Q](A) from setter view (positive = void better)
  - p_set delta:        P(bidder fails to make bid) under B vs A (positive = void better)
  - CVaR_10 delta:      downside risk shift
  - threshold_mass delta: P(Q >= bid_value) from Team 0 view

Sliced by:
  - Phase: early (trick 1-2), mid (3-5), late (6-7)
  - Whether void is in a count-bearing suit (bidder holds count in that suit)
  - Whether bidder side is currently winning the trick

Bidder-winning-trick slice: we check if the current trick leader is bidder-side (team 0).

Usage:
    python run_void_creation_follow_probe.py \\
        [--snapshots snapshots.jsonl] \\
        [--checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt] \\
        [--samples 100] [--device mps] [--output-dir .]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.eq.game_tensor import GameStateTensor
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_IS_DOUBLE,
    DOMINO_COUNT_POINTS,
    led_suit_for_lead_domino,
)

TRUMP_SUIT = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def domino_suit(did: int, decl_id: int) -> int:
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


def infer_current_player(snap: dict) -> int | None:
    """Return which player is to move (0-3), or None if indeterminate."""
    history = snap["history"]
    plays = [h for h in history if h[0] >= 0]
    n = len(plays)
    trick_plays = snap["trick_plays"]
    n_trick = sum(1 for d in trick_plays if d >= 0)

    if n == 0 and n_trick == 0:
        return int(snap["leader"])

    # leader of current trick
    if n % 4 == 0:
        # current trick leader: whoever won last completed trick
        if n == 0:
            return int(snap["leader"])
        last_trick = plays[-4:]
        decl_id = snap["decl_id"]
        lead_did = last_trick[0][1]
        led_suit = led_suit_for_lead_domino(lead_did, decl_id)

        best_player = last_trick[0][0]
        best_rank = _trick_rank(lead_did, led_suit, decl_id)
        for play in last_trick[1:]:
            p, did, _ = play
            rank = _trick_rank(did, led_suit, decl_id)
            if rank > best_rank:
                best_rank = rank
                best_player = p
        current_leader = best_player
    else:
        current_leader = plays[-(n % 4)][0]  # start of partial trick
        # Actually leader is whoever led the current partial trick
        trick_start_idx = (n // 4) * 4
        current_leader = plays[trick_start_idx][0]

    current = (current_leader + n_trick) % 4
    return current


def _trick_rank(did: int, led_suit: int, decl_id: int) -> int:
    """Simplified trick rank."""
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


def find_void_and_preserve_slots(snap: dict) -> tuple[int | None, int | None]:
    """
    Identify:
      void_slot:     slot of singleton tile (voiding action B)
      preserve_slot: slot of an alternative legal tile from a different suit (preserving action A)

    Returns (None, None) if contrast cannot be formed.
    """
    setter = infer_current_player(snap)
    if setter is None:
        return None, None

    bidder = snap["bidder"]
    # Setter must be on team 1
    if (setter % 2) == (bidder % 2):
        return None, None

    decl_id = snap["decl_id"]
    hand = snap["hands"][setter]
    legal_mask = snap["_legal_mask"]

    led_suit = snap.get("_led_suit")
    if led_suit is None:
        trick_lead_domino = snap.get("_trick_lead_domino")
        if trick_lead_domino is None or trick_lead_domino < 0:
            return None, None
        led_suit = led_suit_for_lead_domino(trick_lead_domino, decl_id)

    # Legal slots by suit (can't follow, so these are off-suit discards or trump)
    suit_legal: dict[int, list[int]] = {}  # suit -> [slot indices]
    for slot, (legal, did) in enumerate(zip(legal_mask, hand)):
        if not legal or did < 0:
            continue
        s = domino_suit(did, decl_id)
        suit_legal.setdefault(s, []).append(slot)

    # Full hand suit groups (to find singletons)
    full_suit_groups = hand_suit_groups(hand, decl_id)

    # Singleton non-led, non-trump suits with a legal tile
    singleton_candidates: list[tuple[int, int, int]] = []  # (suit, slot, did)
    for suit, tiles in full_suit_groups.items():
        if suit == led_suit or suit == TRUMP_SUIT:
            continue
        if len(tiles) == 1:
            did = tiles[0]
            # Must be legal
            for slot in suit_legal.get(suit, []):
                if hand[slot] == did:
                    singleton_candidates.append((suit, slot, did))
                    break

    if not singleton_candidates:
        return None, None

    # Pick best void candidate: prioritize count-bearing suit, else lowest pip-sum
    def void_priority(item: tuple) -> tuple:
        suit, slot, did = item
        count = DOMINO_COUNT_POINTS[did]
        pip_sum = DOMINO_HIGH[did] + DOMINO_LOW[did]
        # Prefer count-bearing (higher count first), then lower pip sum
        return (-count, pip_sum)

    singleton_candidates.sort(key=void_priority)
    void_suit, void_slot, void_did = singleton_candidates[0]

    # Preserve candidate: legal tile from a DIFFERENT suit (not void_suit, not led_suit)
    # Prefer non-trump to keep the contrast clean
    preserve_slot = None
    # First pass: non-trump, non-void-suit, non-led-suit
    for suit, slots in suit_legal.items():
        if suit == void_suit or suit == led_suit or suit == TRUMP_SUIT:
            continue
        # Pick lowest pip-sum tile (most conservative)
        best = min(slots, key=lambda s: DOMINO_HIGH[hand[s]] + DOMINO_LOW[hand[s]])
        preserve_slot = best
        break

    if preserve_slot is None:
        # Second pass: allow trump (but not void-suit or led-suit)
        for suit, slots in suit_legal.items():
            if suit == void_suit or suit == led_suit:
                continue
            best = min(slots, key=lambda s: DOMINO_HIGH[hand[s]] + DOMINO_LOW[hand[s]])
            preserve_slot = best
            break

    if preserve_slot is None or preserve_slot == void_slot:
        return None, None

    return void_slot, preserve_slot


def snap_phase(snap: dict) -> str:
    plays = [h for h in snap["history"] if h[0] >= 0]
    # Include partial trick plays
    n_trick_plays = sum(1 for d in snap["trick_plays"] if d >= 0)
    trick_num = len(plays) // 4
    if trick_num <= 2:
        return "early"
    if trick_num <= 5:
        return "mid"
    return "late"


def void_suit_has_count_exposure(snap: dict, void_slot: int) -> bool:
    """Return True if bidder holds a count tile in the voided suit."""
    setter = infer_current_player(snap)
    if setter is None:
        return False

    decl_id = snap["decl_id"]
    hand_setter = snap["hands"][setter]
    void_did = hand_setter[void_slot]
    void_suit = domino_suit(void_did, decl_id)

    bidder = snap["bidder"]
    bidder_hand = snap["hands"][bidder]
    for did in bidder_hand:
        if did < 0:
            continue
        if domino_suit(did, decl_id) == void_suit and DOMINO_COUNT_POINTS[did] > 0:
            return True
    return False


def bidder_side_winning_trick(snap: dict) -> bool:
    """Return True if the current trick leader is on bidder's team (team 0)."""
    history = snap["history"]
    plays = [h for h in history if h[0] >= 0]
    n = len(plays)
    trick_plays = snap["trick_plays"]
    n_trick = sum(1 for d in trick_plays if d >= 0)

    if n == 0:
        trick_leader = snap["leader"]
    elif n % 4 == 0:
        # Determine winner of last completed trick
        last_trick = plays[-4:]
        decl_id = snap["decl_id"]
        lead_did = last_trick[0][1]
        led_suit = led_suit_for_lead_domino(lead_did, decl_id)
        best_player = last_trick[0][0]
        best_rank = _trick_rank(lead_did, led_suit, decl_id)
        for play in last_trick[1:]:
            p, did, _ = play
            rank = _trick_rank(did, led_suit, decl_id)
            if rank > best_rank:
                best_rank = rank
                best_player = p
        trick_leader = best_player
    else:
        trick_start_idx = (n // 4) * 4
        trick_leader = plays[trick_start_idx][0]

    bidder = snap["bidder"]
    # Bidder side = team 0 = players 0 and 2
    return (trick_leader % 2) == (bidder % 2)


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
    """Evaluate E[Q] for a specific first action from the snapshot."""
    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    records = generate_eq_from_snapshots(
        model=model,
        snapshots=[snap_clean],
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
    """P(Q < bid_value) = P(bidder fails to make contract)."""
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
    rows: list[dict] = []

    for i, snap in enumerate(snapshots):
        void_slot, preserve_slot = find_void_and_preserve_slots(snap)
        if void_slot is None or preserve_slot is None:
            print(f"  [skip {i}] no valid paired contrast", flush=True)
            continue

        bid_value = int(snap.get("bid_value", 30))
        decl_id = snap["decl_id"]
        setter = infer_current_player(snap)
        if setter is None:
            continue

        hand = snap["hands"][setter]
        void_did = hand[void_slot]
        preserve_did = hand[preserve_slot]
        void_suit = domino_suit(void_did, decl_id)
        preserve_suit = domino_suit(preserve_did, decl_id)
        led_suit = int(snap.get("_led_suit", -1))

        phase = snap_phase(snap)
        count_exposed = void_suit_has_count_exposure(snap, void_slot)
        bidder_winning = bidder_side_winning_trick(snap)

        print(
            f"  [{i + 1:3d}/{len(snapshots)}] decl={decl_id} bid={bid_value} "
            f"phase={phase} led_suit={led_suit} void_suit={void_suit}(did={void_did}) "
            f"preserve_suit={preserve_suit}(did={preserve_did}) "
            f"count_exp={count_exposed} bidder_win={bidder_winning}",
            end=" ",
            flush=True,
        )

        t0 = time.perf_counter()
        # A: preserve suit
        ev_a, cvar_a, pdf_a = eq_for_action(model, snap, preserve_slot, n_samples, device)
        # B: void suit (singleton discard)
        ev_b, cvar_b, pdf_b = eq_for_action(model, snap, void_slot, n_samples, device)
        elapsed = time.perf_counter() - t0

        # E[Q] is from Team 0 perspective. Setter is Team 1, wants lower EV for Team 0.
        ev_delta_t0 = ev_b - ev_a
        ev_delta_setter = -ev_delta_t0  # positive = void is better for setter

        cvar_delta = cvar_b - cvar_a
        tm_a = threshold_mass(pdf_a, bid_value)
        tm_b = threshold_mass(pdf_b, bid_value)
        tm_delta = tm_b - tm_a  # positive = void hurts bidder making contract

        ps_a = p_set(pdf_a, bid_value)
        ps_b = p_set(pdf_b, bid_value)
        ps_delta = ps_b - ps_a  # positive = void helps setter

        print(
            f"EV_delta(setter)={ev_delta_setter:+.2f} p_set_delta={ps_delta:+.4f} t={elapsed:.1f}s",
            flush=True,
        )

        rows.append({
            "snapshot_idx": i,
            "decl_id": decl_id,
            "bid_value": bid_value,
            "setter_player": setter,
            "led_suit": led_suit,
            "void_suit": void_suit,
            "preserve_suit": preserve_suit,
            "phase": phase,
            "count_exposed": count_exposed,
            "bidder_winning_trick": bidder_winning,
            "void_slot": void_slot,
            "void_did": void_did,
            "preserve_slot": preserve_slot,
            "preserve_did": preserve_did,
            # Team 0 perspective
            "ev_a_t0": ev_a,
            "ev_b_t0": ev_b,
            "ev_delta_t0": ev_delta_t0,
            # Setter perspective
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
    if not rows:
        return {"N_pairs": 0, "error": "no pairs"}

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

    verdict = _propose_verdict(ps_deltas, ev_deltas, n)

    summary = {
        "N_pairs": n,
        "p_set_delta_mean": float(ps_deltas.mean()),
        "p_set_delta_ci95_lo": ps_lo,
        "p_set_delta_ci95_hi": ps_hi,
        "p_set_direction": "void > preserve" if ps_deltas.mean() > 0 else "preserve > void",
        "n_p_set_higher_void": n_pset_higher,
        "pct_p_set_higher_void": float(100 * n_pset_higher / n),
        "ev_delta_setter_mean": float(ev_deltas.mean()),
        "ev_delta_ci95_lo": ev_lo,
        "ev_delta_ci95_hi": ev_hi,
        "ev_direction": "void > preserve" if ev_deltas.mean() > 0 else "preserve > void",
        "n_void_ev_better": n_void_better,
        "pct_void_ev_better": float(100 * n_void_better / n),
        "cvar_delta_mean": float(cvar_deltas.mean()),
        "threshold_mass_delta_mean": float(tm_deltas.mean()),
        "claim_ledger_impact": verdict,
        "claim_ledger_rationale": _verdict_rationale(ps_deltas, ev_deltas, n),
    }

    summary["slices"] = {}

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

    # Slice: count_exposed
    for exposed in [True, False]:
        sub = [r for r in rows if r["count_exposed"] == exposed]
        if sub:
            sub_ps = np.array([r["p_set_delta"] for r in sub])
            sub_ev = np.array([r["ev_delta_setter"] for r in sub])
            label = "count_exposed" if exposed else "no_count"
            summary["slices"][f"count_exposure={label}"] = {
                "N": len(sub),
                "p_set_delta_mean": float(sub_ps.mean()),
                "ev_delta_setter_mean": float(sub_ev.mean()),
                "pct_void_better": float(100 * (sub_ev > 0).mean()),
            }

    # Slice: bidder_winning_trick
    for bw in [True, False]:
        sub = [r for r in rows if r["bidder_winning_trick"] == bw]
        if sub:
            sub_ps = np.array([r["p_set_delta"] for r in sub])
            sub_ev = np.array([r["ev_delta_setter"] for r in sub])
            label = "bidder_winning" if bw else "setter_winning"
            summary["slices"][f"trick_lead={label}"] = {
                "N": len(sub),
                "p_set_delta_mean": float(sub_ps.mean()),
                "ev_delta_setter_mean": float(sub_ev.mean()),
                "pct_void_better": float(100 * (sub_ev > 0).mean()),
            }

    return summary


def _propose_verdict(ps_deltas: np.ndarray, ev_deltas: np.ndarray, n: int) -> str:
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

    if ci_lo_ps > 0.005 and ci_lo_ev > 0:
        return "supported"
    if ci_hi_ev < 0 and ci_hi_ps < 0:
        return "contradicted"
    if ci_hi_ev < 0 and abs(mean_ps) < 0.01:
        return "contradicted"
    if abs(mean_ev) < 1.0:
        return "context-limited"
    return "underpowered"


def _verdict_rationale(ps_deltas: np.ndarray, ev_deltas: np.ndarray, n: int) -> str:
    mean_ps = float(ps_deltas.mean())
    mean_ev = float(ev_deltas.mean())
    pct = float(100 * (ev_deltas > 0).mean())
    se_ev = ev_deltas.std(ddof=1) / np.sqrt(n)
    ci_lo = mean_ev - 1.96 * se_ev
    ci_hi = mean_ev + 1.96 * se_ev
    return (
        f"N={n} paired contrasts (follow-position) from oracle-greedy legacy corpus. "
        f"Mean p_set delta={mean_ps:+.4f}, mean EV delta (setter)={mean_ev:+.2f} "
        f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}], "
        f"{pct:.1f}% of contrasts favor void creation."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_snaps = str(
        Path(__file__).parent.parent.parent
        / "snapshots/void_creation_follow/snapshots.jsonl"
    )
    default_ckpt = str(
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
    )
    parser.add_argument("--snapshots", type=str, default=default_snaps)
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--max-snapshots", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print("Validating snapshots...", flush=True)
    valid = []
    for snap in snapshots:
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid.append(snap)
        except Exception as e:
            print(f"  skip invalid: {e}", flush=True)
    snapshots = valid
    print(f"Valid snapshots: {len(snapshots)}", flush=True)

    # Pre-filter: must produce valid contrast pair
    filtered = []
    for snap in snapshots:
        vs, ps = find_void_and_preserve_slots(snap)
        if vs is not None and ps is not None:
            filtered.append(snap)
    print(f"Snapshots with valid paired contrast: {len(filtered)}", flush=True)
    snapshots = filtered

    if not snapshots:
        print("No valid paired contrasts. Exiting.", flush=True)
        return 1

    # Device check
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"CUDA unavailable; falling back to {device}.", flush=True)
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("MPS unavailable; falling back to CPU.", flush=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {ckpt} on {device}...", flush=True)
    oracle = Stage1Oracle(str(ckpt), device=device, compile=False)
    print("Model loaded.", flush=True)

    print(
        f"\nRunning paired contrasts ({len(snapshots)} snapshots, "
        f"{args.samples} samples each)...",
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
    manifest = {
        "bead": "t42-z31l",
        "claim": "ch05-void-creation-follow",
        "wave": "wave2",
        "probe": "void_creation_follow_paired_contrast",
        "snapshots_path": str(snap_path),
        "snapshots_sha256": hashlib.sha256(snap_path.read_bytes()).hexdigest(),
        "checkpoint": str(ckpt),
        "n_snapshots_input": len(valid),
        "n_snapshots_with_contrast": len(filtered),
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
            f"python run_void_creation_follow_probe.py "
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
