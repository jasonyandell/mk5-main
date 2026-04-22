"""LAMIR-1 evaluation harness — three inference modes.

Modes (--mode flag):
  direct        Direct π_me argmax with legal mask. Baseline.
  v-bootstrap   Depth-1 V_head: apply action a → query V_head immediately
                (no opp rollout). Tests whether V_head has good action
                ordering at depth-1 before any opp simulation.
  lamir1        1-ply look-ahead: simulate remaining trick players using
                rotation-equivariant π_me as π_opp, then score with
                V_head at leaf averaged over M corpus worlds.

For v-bootstrap and lamir1, falls back to direct π_me for trick_pos==3
(last player in trick — nothing to roll out).

Batched inference: all M worlds processed in parallel per model call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch import Tensor

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.features import _hand_list
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector
from forge.eq.game_tensor import GameStateTensor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_student(path: str, device: str):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    args = ckpt["args"]
    is_voids = "voids_hidden" in args
    cls = StudentTransformerFullVoids if is_voids else StudentTransformerFull
    kwargs = {
        "d_model": args["d_model"],
        "n_heads": args["n_heads"],
        "n_layers": args["n_layers"],
        "ff_dim": args.get("ff_dim", 256),
        "dropout": 0.0,
        "d_world": args.get("d_world", 64),
        "q_hidden": args.get("q_hidden", 256),
    }
    if is_voids:
        kwargs["voids_hidden"] = args.get("voids_hidden", 64)
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, is_voids


# ---------------------------------------------------------------------------
# World rotation
# ---------------------------------------------------------------------------

def reindex_world_rows(
    world_hands_batch: Tensor,  # [M, 3, 7] — relative to original P
    old_cp: int,
    new_cp: int,
) -> Tensor:
    """Return [M, 3, 7] world tensor for new_cp's POV.

    Row i of result = hand of absolute player (new_cp + i + 1) % 4.
    Row j of source = hand of absolute player (old_cp + j + 1) % 4.
    """
    M = world_hands_batch.shape[0]
    out = torch.zeros_like(world_hands_batch)
    for i in range(3):
        abs_p = (new_cp + i + 1) % 4
        if abs_p == old_cp:
            # new_cp sees old_cp at relative seat i — not in world_hands (known player)
            continue
        src_row = (abs_p - old_cp - 1) % 4
        if src_row < 3:
            out[:, i, :] = world_hands_batch[:, src_row, :]
    return out


def world_batch_to_assignment_vectorized(world_hands_batch: Tensor) -> Tensor:
    """Convert [M, 3, 7] world rows to [M, 28, 3] seat-one-hot (vectorized)."""
    M = world_hands_batch.shape[0]
    assign = torch.zeros(M, 28, 3, dtype=torch.float32, device=world_hands_batch.device)
    flat = world_hands_batch.reshape(M, 3 * 7)  # [M, 21]
    for seat in range(3):
        for slot in range(7):
            d_ids = world_hands_batch[:, seat, slot].long()  # [M]
            valid = (d_ids >= 0) & (d_ids < 28)
            if valid.any():
                m_idx = torch.where(valid)[0]
                assign[m_idx, d_ids[m_idx], seat] = 1.0
    return assign


# ---------------------------------------------------------------------------
# Batched model query
# ---------------------------------------------------------------------------

def query_model_batched(
    model,
    is_voids: bool,
    tokens: Tensor,          # [L, 5] — same for all M worlds
    attn_mask: Tensor,       # [L]
    voids_vec: Tensor,       # [24]
    world_assign: Tensor,    # [M, 28, 3]
    device: str,
) -> dict[str, Tensor]:
    """Run model on M worlds at once. Returns dict with v [M] and pi_me_logits [M, 7]."""
    M = world_assign.shape[0]

    # Expand tokens/mask/voids to [M, ...]
    tokens_b = tokens.unsqueeze(0).expand(M, -1, -1).to(device)   # [M, L, 5]
    mask_b = attn_mask.unsqueeze(0).expand(M, -1).to(device)       # [M, L]
    world_b = world_assign.to(device)                               # [M, 28, 3]

    with torch.no_grad():
        if is_voids:
            voids_b = voids_vec.unsqueeze(0).expand(M, -1).to(device)  # [M, 24]
            out = model(tokens_b, mask_b, world_b, voids_b)
        else:
            out = model(tokens_b, mask_b, world_b)

    return out  # v: [M], pi_me_logits: [M, 7]


# ---------------------------------------------------------------------------
# Per-world game hands
# ---------------------------------------------------------------------------

def _world_game_hands(
    real_hands: list[list[int]],
    world_hands_m: Tensor,   # [3, 7] — relative to P
    P: int,
) -> list[list[int]]:
    """Build 4-player game_hands where P keeps real hand and opps use world hands.

    world_hands_m row i = hand of absolute player (P + i + 1) % 4.
    """
    result: list[list[int]] = [list(real_hands[p]) for p in range(4)]
    for i in range(3):
        abs_p = (P + i + 1) % 4
        result[abs_p] = [int(x) for x in world_hands_m[i].tolist()]
    return result


# ---------------------------------------------------------------------------
# Build tokens+voids for a (game, state) snapshot
# ---------------------------------------------------------------------------

@dataclass
class _FakeDecision:
    """Minimal decision proxy for tokenize_decision reconstruction."""
    player: int
    action_taken: int


def _build_tokens_voids(
    game_hands: list[list[int]],
    decl_id: int,
    orig_decisions: list,
    d_idx: int,
    extra: list[_FakeDecision],
    query_player: int,
) -> tuple[Tensor, Tensor]:
    """Build (tokens [L, 5], voids [24]) for query_player at state = orig[:d_idx] + extra.

    game_hands must be the world-correct 4-player hands for any simulated plays in extra.
    """

    class _Stub:
        def __init__(self, p, a):
            self.player = p
            self.action_taken = a

    synth = list(orig_decisions[:d_idx]) + [_Stub(fd.player, fd.action_taken) for fd in extra]
    synth_d_idx = len(synth)
    synth.append(_Stub(query_player, 0))  # placeholder for current decision

    tokens, attn_mask = tokenize_decision(game_hands, decl_id, synth, synth_d_idx)

    # Build prior plays for voids (original plays + extra)
    plays: list[tuple[int, int]] = []
    for j in range(d_idx):
        dd = orig_decisions[j]
        p = int(dd.player)
        slot = int(dd.action_taken)
        hand = _hand_list(game_hands[p])
        if 0 <= slot < len(hand) and int(hand[slot]) >= 0:
            plays.append((p, int(hand[slot])))
    for fd in extra:
        hand = _hand_list(game_hands[fd.player])
        slot = fd.action_taken
        if 0 <= slot < len(hand) and int(hand[slot]) >= 0:
            plays.append((fd.player, int(hand[slot])))

    voids = voids_feature_vector(plays, decl_id, query_player)

    return tokens, attn_mask, voids


# ---------------------------------------------------------------------------
# Replay state up to decision d_idx
# ---------------------------------------------------------------------------

def _replay_state(game_hands: list[list[int]], decl_id: int, decisions: list, d_idx: int) -> GameStateTensor:
    """Replay the game to the state just before decision d_idx."""
    state = GameStateTensor.from_deals([game_hands], [decl_id], device="cpu")
    for j in range(d_idx):
        slot_j = int(decisions[j].action_taken)
        state = state.apply_actions(torch.tensor([slot_j], dtype=torch.long))
    return state


# ---------------------------------------------------------------------------
# Shared oracle-best helper
# ---------------------------------------------------------------------------

def _oracle_info(decision) -> tuple[float, int, Tensor, list[int]]:
    """Return (oracle_best_eq, oracle_best_action, e_q [7], legal_slots)."""
    legal_mask = decision.legal_mask.bool()
    e_q = decision.e_q.float()
    e_q_legal = e_q.clone()
    e_q_legal[~legal_mask] = float("-inf")
    oracle_best_eq = float(e_q_legal.max().item())
    oracle_best_action = int(e_q_legal.argmax().item())
    legal_slots = legal_mask.nonzero(as_tuple=True)[0].tolist()
    return oracle_best_eq, oracle_best_action, e_q, legal_slots


# ---------------------------------------------------------------------------
# Mode: direct (baseline π_me argmax)
# ---------------------------------------------------------------------------

def direct_decision(
    model,
    is_voids: bool,
    game,
    d_idx: int,
    device: str,
) -> tuple[float, int, float]:
    """Direct π_me argmax. Returns (regret, is_bot_match, oracle_best_eq)."""
    decision = game.decisions[d_idx]
    P = int(decision.player)
    oracle_best_eq, oracle_best_action, e_q, legal_slots = _oracle_info(decision)
    legal_mask = decision.legal_mask.bool()

    tokens, attn_mask, voids = _build_tokens_voids(
        game.hands, int(game.decl_id), game.decisions, d_idx, [], P
    )
    dummy_world = torch.zeros(1, 28, 3)
    out = query_model_batched(model, is_voids, tokens, attn_mask, voids, dummy_world, device)
    pi = out["pi_me_logits"][0]
    pi_masked = pi.masked_fill(~legal_mask.to(device), float("-inf"))
    chosen = int(pi_masked.argmax().item())
    regret = oracle_best_eq - float(e_q[chosen].item())
    return regret, int(chosen == oracle_best_action), oracle_best_eq


# ---------------------------------------------------------------------------
# Mode: v-bootstrap (depth-1 V_head, no opp rollout)
# ---------------------------------------------------------------------------

def v_bootstrap_decision(
    model,
    is_voids: bool,
    game,
    d_idx: int,
    device: str,
    world_cap: int = 200,
) -> tuple[float, int, float]:
    """Depth-1 V_head: apply a → query V_head immediately, no opp simulation.

    For each legal action a:
      extra = [a]  (P played a)
      tokens/voids = P's view after playing a
      score[a] = mean_m V_head(tokens, voids, world_m)
    chosen = argmax score

    Falls back to direct π_me for trick_pos==3.
    Returns (regret, is_bot_match, oracle_best_eq).
    """
    decision = game.decisions[d_idx]
    P = int(decision.player)
    oracle_best_eq, oracle_best_action, e_q, legal_slots = _oracle_info(decision)
    legal_mask = decision.legal_mask.bool()
    trick_pos = d_idx % 4

    # trick_pos==3: last play, nothing to evaluate ahead — direct fallback
    if trick_pos == 3:
        return direct_decision(model, is_voids, game, d_idx, device)

    world_hands_all = decision.world_hands  # [M, 3, 7]
    M = min(world_hands_all.shape[0], world_cap)
    world_hands = world_hands_all[:M]
    world_assign_P = world_batch_to_assignment_vectorized(world_hands)  # [M, 28, 3]

    # After P plays, the leaf current_player is (P+1)%4.
    # V_head is trained in leaf player's team frame. If leaf player is on opp team,
    # the value must be negated to convert to P's team frame.
    leaf_cp = (P + 1) % 4
    sign = 1 if (leaf_cp % 2) == (P % 2) else -1

    action_scores: dict[int, float] = {}
    for a_slot in legal_slots:
        extra = [_FakeDecision(player=P, action_taken=a_slot)]
        tokens, attn_mask, voids = _build_tokens_voids(
            game.hands, int(game.decl_id), game.decisions, d_idx, extra, P
        )
        out = query_model_batched(
            model, is_voids, tokens, attn_mask, voids, world_assign_P, device
        )
        action_scores[a_slot] = sign * float(out["v"].mean().item())

    chosen = max(action_scores, key=lambda a: action_scores[a])
    regret = oracle_best_eq - float(e_q[chosen].item())
    return regret, int(chosen == oracle_best_action), oracle_best_eq


# ---------------------------------------------------------------------------
# LAMIR-1 per-decision
# ---------------------------------------------------------------------------

def lamir1_decision(
    model,
    is_voids: bool,
    game,
    d_idx: int,
    device: str,
    world_cap: int = 200,
) -> tuple[float, int, float]:
    """1-ply rollout. Returns (regret, is_bot_match, oracle_best_eq).

    Falls back to direct π_me for trick_pos==3.
    """
    decision = game.decisions[d_idx]
    P = int(decision.player)
    oracle_best_eq, oracle_best_action, e_q, legal_slots = _oracle_info(decision)
    legal_mask = decision.legal_mask.bool()

    trick_pos = d_idx % 4
    n_remaining = 3 - trick_pos

    if trick_pos == 3:
        return direct_decision(model, is_voids, game, d_idx, device)

    # Cap worlds for speed
    world_hands_all = decision.world_hands  # [M, 3, 7]
    M_full = world_hands_all.shape[0]
    M = min(M_full, world_cap)
    world_hands = world_hands_all[:M]  # [M, 3, 7]

    # Replay state to just before decision d_idx
    state_at_d = _replay_state(game.hands, int(game.decl_id), game.decisions, d_idx)

    # Precompute per-world 4-player game_hands (world m's opps replace real deal).
    world_game_hands: list[list[list[int]]] = [
        _world_game_hands(game.hands, world_hands[m], P)
        for m in range(M)
    ]
    world_assign_P = world_batch_to_assignment_vectorized(world_hands)  # [M, 28, 3]

    action_scores: dict[int, float] = {}

    for a_slot in legal_slots:
        # Step state: P plays slot a_slot (same for all worlds — P's hand is real)
        state_after_a = state_at_d.apply_actions(torch.tensor([a_slot], dtype=torch.long))
        extra_a = _FakeDecision(player=P, action_taken=a_slot)

        # Per-world states and per-world extra lists
        states = [state_after_a] * M
        extras = [[extra_a] for _ in range(M)]

        for step in range(n_remaining):
            p_next = (P + 1 + step) % 4

            # Rotate world tensor for p_next's POV: [M, 3, 7]
            world_rotated = reindex_world_rows(world_hands, old_cp=P, new_cp=p_next)
            world_assign_pnext = world_batch_to_assignment_vectorized(world_rotated)

            # Per-world tokens for p_next (extras differ per world after step 0)
            tok_list, mask_list, void_list = [], [], []
            for m in range(M):
                t, am, v = _build_tokens_voids(
                    world_game_hands[m], int(game.decl_id), game.decisions, d_idx,
                    extras[m], p_next,
                )
                tok_list.append(t)
                mask_list.append(am)
                void_list.append(v)

            with torch.no_grad():
                tok_d = torch.stack(tok_list).to(device)   # [M, L, 5]
                msk_d = torch.stack(mask_list).to(device)  # [M, L]
                wa_d = world_assign_pnext.to(device)
                if is_voids:
                    vd_d = torch.stack(void_list).to(device)  # [M, 24]
                    out_pnext = model(tok_d, msk_d, wa_d, vd_d)
                else:
                    out_pnext = model(tok_d, msk_d, wa_d)

            pi_pnext = out_pnext["pi_me_logits"]  # [M, 7]

            new_states = []
            for m in range(M):
                state_m = states[m]
                trick_len_m = int((state_m.trick_plays[0] >= 0).sum().item())
                if trick_len_m == 0 and step > 0:
                    new_states.append(state_m)
                    continue

                opp_legal_m = state_m.legal_actions()[0].to(device)
                pi_m = pi_pnext[m].masked_fill(~opp_legal_m, float("-inf"))
                a_next_m = int(pi_m.argmax().item())

                new_states.append(state_m.apply_actions(torch.tensor([a_next_m], dtype=torch.long)))
                extras[m].append(_FakeDecision(player=p_next, action_taken=a_next_m))

            states = new_states

        # Query V_head from P's POV at leaf — per-world tokens using world-correct hands
        all_tokens, all_masks, all_voids = [], [], []
        for m in range(M):
            t, am, v = _build_tokens_voids(
                world_game_hands[m], int(game.decl_id), game.decisions, d_idx,
                extras[m], P,
            )
            all_tokens.append(t)
            all_masks.append(am)
            all_voids.append(v)

        with torch.no_grad():
            tokens_d = torch.stack(all_tokens).to(device)   # [M, L, 5]
            masks_d = torch.stack(all_masks).to(device)     # [M, L]
            world_d = world_assign_P.to(device)
            if is_voids:
                voids_d = torch.stack(all_voids).to(device)  # [M, 24]
                out_v = model(tokens_d, masks_d, world_d, voids_d)
            else:
                out_v = model(tokens_d, masks_d, world_d)

        action_scores[a_slot] = float(out_v["v"].mean().item())

    chosen = max(action_scores, key=lambda a: action_scores[a])
    regret = oracle_best_eq - float(e_q[chosen].item())
    return regret, int(chosen == oracle_best_action), oracle_best_eq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MODES = ("direct", "v-bootstrap", "lamir1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="gus/adapters/v3_consistency_10000g.pt")
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None,
                        help="JSON output path (default: scratch/lamir1_<mode>.json)")
    parser.add_argument("--world-cap", type=int, default=200)
    parser.add_argument("--mode", choices=_MODES, default="lamir1",
                        help="Inference mode (default: lamir1)")
    args = parser.parse_args()

    device = args.device or _pick_device()
    out_path = args.out or f"scratch/lamir1_{args.mode.replace('-', '_')}.json"
    print(
        f"Mode: {args.mode}  adapter: {args.adapter}  "
        f"device: {device}  world-cap: {args.world_cap}",
        flush=True,
    )

    model, is_voids = _load_student(args.adapter, device)
    print(f"Model loaded, voids={is_voids}", flush=True)

    ds = JointWorldFullDataset(args.eval, seed=42)
    print(f"Eval corpus: {len(ds.games)} games, qualifying decisions: {len(ds.index)}", flush=True)

    # Select inference function
    if args.mode == "direct":
        def run_decision(game, d_idx):
            return direct_decision(model, is_voids, game, d_idx, device)
    elif args.mode == "v-bootstrap":
        def run_decision(game, d_idx):
            return v_bootstrap_decision(model, is_voids, game, d_idx, device, args.world_cap)
    else:  # lamir1
        def run_decision(game, d_idx):
            return lamir1_decision(model, is_voids, game, d_idx, device, args.world_cap)

    total_regret = 0.0
    total_items = 0
    bot_matches = 0
    negligible_regret = 0
    bucket_regret: dict[int, list[float]] = defaultdict(list)
    bucket_match: dict[int, list[int]] = defaultdict(list)
    # trick_pos buckets: 0=leader, 1=2nd, 2=3rd, 3=last
    pos_regret: dict[int, list[float]] = defaultdict(list)
    pos_match: dict[int, list[int]] = defaultdict(list)
    per_decision_results: list[dict] = []

    t_start = time.time()

    for g_idx, game in enumerate(ds.games):
        for d_idx, dec in enumerate(game.decisions):
            if dec.world_hands is None or dec.q_per_world is None:
                continue

            regret, is_match, oracle_best = run_decision(game, d_idx)
            trick_pos = d_idx % 4

            total_regret += regret
            total_items += 1
            bot_matches += is_match
            if regret < 0.5:
                negligible_regret += 1
            bucket_regret[d_idx].append(regret)
            bucket_match[d_idx].append(is_match)
            pos_regret[trick_pos].append(regret)
            pos_match[trick_pos].append(is_match)
            per_decision_results.append({
                "game": g_idx,
                "d_idx": d_idx,
                "trick_pos": trick_pos,
                "regret": regret,
                "bot_match": is_match,
                "oracle_best_eq": oracle_best,
            })

        elapsed = time.time() - t_start
        n_done = total_items
        print(
            f"  game {g_idx+1}/{len(ds.games)}  decisions={n_done}  "
            f"regret={total_regret/max(n_done,1):.3f}  "
            f"bot-match={bot_matches/max(n_done,1):.3%}  "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    elapsed_total = time.time() - t_start
    mean_regret = total_regret / max(total_items, 1)
    bot_match_rate = bot_matches / max(total_items, 1)
    near_tie_rate = negligible_regret / max(total_items, 1)

    print(flush=True)
    print(f"=== [{args.mode}] Summary over {total_items} decisions ===")
    print(f"  Bot-match rate:            {bot_match_rate:.3%}")
    print(f"  Mean regret (Q-points):    {mean_regret:.3f}")
    print(f"  Decisions with regret<0.5: {negligible_regret}/{total_items} "
          f"= {near_tie_rate:.1%} (near-ties)")
    print(f"  Wall-clock time:           {elapsed_total:.1f}s")
    print(f"  Baseline regret:           0.551  bot-match: 76.07%")
    print(flush=True)

    print("=== Per-trick-pos breakdown ===")
    print(f"{'pos':>3s}  {'n':>4s}  {'bot':>6s}  {'regret':>8s}  note")
    pos_labels = {0: "leads", 1: "2nd", 2: "3rd", 3: "last(fallback)"}
    for pos in range(4):
        if not pos_regret[pos]:
            continue
        rs = pos_regret[pos]
        ms = pos_match[pos]
        n = len(rs)
        print(
            f"  {pos}  {n:>4d}  {sum(ms)/n:>6.2%}  {sum(rs)/n:>8.3f}"
            f"  {pos_labels[pos]}"
        )

    print(flush=True)
    print("=== Per-decision-slot regret + bot-match ===")
    print(f"{'dec':>3s}  {'bot':>6s}  {'regret':>8s}  {'near-tie':>8s}  {'n':>3s}")
    for d in sorted(bucket_regret.keys()):
        regrets = bucket_regret[d]
        matches = bucket_match[d]
        n = len(regrets)
        mean_r = sum(regrets) / n
        match_rate = sum(matches) / n
        near_ties = sum(1 for r in regrets if r < 0.5) / n
        print(f"{d:>3d}  {match_rate:>6.2%}  {mean_r:>8.2f}  {near_ties:>8.1%}  {n:>3d}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "device": device,
        "world_cap": args.world_cap,
        "n_decisions": total_items,
        "mean_regret": mean_regret,
        "bot_match_rate": bot_match_rate,
        "near_tie_rate": near_tie_rate,
        "wall_clock_s": elapsed_total,
        "baseline_regret": 0.551,
        "baseline_bot_match": 0.7607,
        "by_trick_pos": {
            str(pos): {
                "n": len(pos_regret[pos]),
                "mean_regret": sum(pos_regret[pos]) / max(len(pos_regret[pos]), 1),
                "bot_match": sum(pos_match[pos]) / max(len(pos_match[pos]), 1),
            }
            for pos in range(4) if pos_regret[pos]
        },
        "per_decision": per_decision_results,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
